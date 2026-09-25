"""FlashMoE and DuoServe-MoE, reimplemented from their papers on the same
transformers stack as PHASOR-HF, ZipMoE and MoE-Infinity (neither released code).

Both keep the path their papers describe: experts are read from the checkpoint
into host memory and copied to the GPU before the matmul.  With a budget below
the model, what their papers keep in host RAM is a budget-bounded host cache
backed by the SSD.

FlashMoE (Kim et al., arXiv 2601.17063):
  per-layer GPU cache with a fixed number of expert slots; on a miss in a full
  layer the expert a 3-layer FFN over (1/recency, frequency/max) scores highest
  is evicted (trained on Belady labels, scripts/train_flashmoe.py); prefill
  loads each required expert once and only fills free slots.
DuoServe-MoE (Zhang et al., arXiv 2509.07379):
  experts live in host memory; the GPU holds only the experts in use.  Prefill:
  expert-by-expert pipeline on two CUDA streams -- the next expert is copied
  while the current one computes.  Decode: a predictor trained offline from
  activation traces (popularity, inter-layer affinity and the experts chosen so
  far) names the next layer's top-k, which are copied ahead on the copy stream.
"""
import json
import os
import struct
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def safetensors_index(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    return {k: (base + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], v["shape"])
            for k, v in hdr.items() if k != "__metadata__"}


class ExpertReader:
    """Reads one expert's three bf16 matrices from the checkpoint into pinned
    host tensors (buffered pread, as torch/safetensors loaders do)."""

    def __init__(self, ckpt, moe_attr, names, L, E):
        shards = sorted(os.path.join(ckpt, f) for f in os.listdir(ckpt) if f.endswith(".safetensors"))
        self.fds = [os.open(s, os.O_RDONLY) for s in shards]
        idx = {}
        for fi, s in enumerate(shards):
            for k, v in safetensors_index(s).items():
                idx[k] = (fi,) + v
        self.loc = {}
        for l in range(L):
            for e in range(E):
                self.loc[(l, e)] = [idx[f"model.layers.{l}.{moe_attr}.experts.{e}.{nm}.weight"] for nm in names]
        self.unit_bytes = sum(v[2] for v in self.loc[(0, 0)])
        self.bytes_read = 0

    def read(self, l, e):
        out = []
        for fi, off, ln, shape in self.loc[(l, e)]:
            buf = torch.empty(ln // 2, dtype=torch.bfloat16, pin_memory=True)
            mv = memoryview(buf.view(torch.uint8).numpy())
            got = os.preadv(self.fds[fi], [mv], off)
            assert got == ln, (got, ln)
            self.bytes_read += ln
            out.append(buf.view(shape[0], shape[1]))
        return out


def expert_ffn(x, g, u, d):
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


# ----------------------------------------------------------------- FlashMoE*
class FlashMoECache:
    def __init__(self, reader, L, E, slots_per_layer, weights_path):
        self.r, self.L, self.E, self.S = reader, L, E, max(1, slots_per_layer)
        self.cache = [OrderedDict() for _ in range(L)]      # e -> gpu tensors
        self.last = np.full((L, E), -1e9); self.freq = np.zeros((L, E)); self.step = 0
        self.net = self._load(weights_path)
        self.hits = self.misses = 0
        self.dec_hits = self.dec_misses = 0     # decode steps only (diagnostics)

    def _load(self, p):
        lines = open(p).read().split("\n"); k = 0; nl = int(lines[k]); k += 1; Ws = []
        for _ in range(nl):
            r, c = map(int, lines[k].split()); k += 1
            W = np.array(lines[k].split(), float).reshape(r, c); k += 1
            b = np.array(lines[k].split(), float); k += 1; Ws.append((W, b))
        return Ws

    def _score(self, X):
        for i, (W, b) in enumerate(self.net):
            X = X @ W.T + b
            if i < len(self.net) - 1: X = X / (1 + np.exp(-X))
        return X[:, 0]

    def tick(self): self.step += 1

    def get(self, l, experts, prefill):
        out = {}
        for e in experts:
            self.last[l, e] = self.step; self.freq[l, e] += 1
        for e in experts:
            c = self.cache[l]
            if e in c:
                self.hits += 1; out[e] = c[e]
                if not prefill: self.dec_hits += 1
                continue
            self.misses += 1
            if not prefill: self.dec_misses += 1
            w = [t.to("cuda", non_blocking=True) for t in self.r.read(l, e)]
            out[e] = w
            if len(c) < self.S:
                c[e] = w
            elif not prefill:
                cand = [x for x in c if x not in experts]
                if cand:
                    fm = max(self.freq[l].max(), 1)
                    X = np.array([[1.0 / (self.step - self.last[l, x] + 1), self.freq[l, x] / fm] for x in cand])
                    v = cand[int(np.argmax(self._score(X)))]
                    del c[v]; c[e] = w
        return out


# ----------------------------------------------------------------- DuoServe*
class DuoPredictor(nn.Module):
    """Multi-layer MLP over [experts chosen at layers < l (multi-hot, one row
    per layer summed), popularity of layer l, affinity from layer l-1]."""
    def __init__(self, E, hidden=256, depth=7):
        super().__init__()
        layers, d = [], 3 * E
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
        layers.append(nn.Linear(d, E))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class DuoServeCache:
    def __init__(self, reader, L, E, K, host_units, predictor, pop, aff):
        self.r, self.L, self.E, self.K = reader, L, E, K
        self.host = OrderedDict()                              # (l,e) -> pinned tensors (LRU)
        self.host_cap = max(1, host_units)
        self.pred, self.pop, self.aff = predictor, pop, aff    # pop [L,E], aff [L-1,E,E]
        self.copy = torch.cuda.Stream()
        self.prefetched = {}                                   # (l,e) -> (gpu tensors, event)
        self.hits = self.misses = self.pred_hits = self.pred_total = 0
        self.chosen = np.zeros(E)

    def tick(self): self.chosen[:] = 0

    def _host(self, l, e):
        k = (l, e)
        if k in self.host:
            self.host.move_to_end(k); self.hits += 1; return self.host[k]
        self.misses += 1
        w = self.r.read(l, e)
        self.host[k] = w
        while len(self.host) > self.host_cap: self.host.popitem(last=False)
        return w

    def _to_gpu(self, l, e):
        with torch.cuda.stream(self.copy):
            w = [t.to("cuda", non_blocking=True) for t in self._host(l, e)]
            ev = torch.cuda.Event(); ev.record(self.copy)
        return w, ev

    def gpu(self, l, e):
        k = (l, e)
        if k in self.prefetched:
            w, ev = self.prefetched.pop(k); self.pred_hits += 1
        else:
            w, ev = self._to_gpu(l, e)
        torch.cuda.current_stream().wait_event(ev)
        return w

    def after_layer(self, l, experts):
        """Decode: predict layer l+1 and copy its experts ahead."""
        self.prefetched = {k: v for k, v in self.prefetched.items() if k[0] > l}
        for e in experts: self.chosen[e] += 1
        if l + 1 >= self.L: return
        with torch.no_grad():
            a = self.aff[l][experts].sum(0) if len(experts) else np.zeros(self.E)
            x = np.concatenate([self.chosen / max(self.chosen.max(), 1), self.pop[l + 1] / max(self.pop[l + 1].max(), 1),
                                a / max(a.max(), 1)])
            s = self.pred(torch.tensor(x, dtype=torch.float32, device="cuda")[None])[0]
            top = torch.topk(s, self.K).indices.tolist()
        self.pred_total += self.K
        for e in top:
            if (l + 1, e) not in self.prefetched:
                self.prefetched[(l + 1, e)] = self._to_gpu(l + 1, e)


class BaselineMoE(nn.Module):
    def __init__(self, gate, layer, E, K, norm, cache, kind, returns_logits):
        super().__init__()
        self.gate, self.layer, self.E, self.K, self.norm = gate, layer, E, K, norm
        self.cache, self.kind, self.returns_logits = cache, kind, returns_logits

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        logits = self.gate(x)
        w = F.softmax(logits, dim=1, dtype=torch.float)
        w, sel = torch.topk(w, self.K, dim=-1)
        if self.norm: w = w / w.sum(dim=-1, keepdim=True)
        w = w.to(x.dtype)
        prefill = s > 1                     # a batched decode step is decode
        experts = torch.unique(sel).tolist()
        out = torch.zeros_like(x)
        if self.kind == "flashmoe":
            ws = self.cache.get(self.layer, experts, prefill)
            for e in experts:
                tok, kk = torch.where(sel == e)
                out.index_add_(0, tok, expert_ffn(x[tok], *ws[e]) * w[tok, kk, None])
        else:  # duoserve: copy of the next expert overlaps this expert's matmuls
            for e in experts:
                g, u, d = self.cache.gpu(self.layer, e)
                tok, kk = torch.where(sel == e)
                out.index_add_(0, tok, expert_ffn(x[tok], g, u, d) * w[tok, kk, None])
            if not prefill:
                self.cache.after_layer(self.layer, experts)
        out = out.view(b, s, h)
        return (out, logits) if self.returns_logits else out


def build(ckpt, kind, budget_gib, weights=None, predictor=None, traces=()):
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open
    cfg = AutoConfig.from_pretrained(ckpt)
    arch = cfg.architectures[0].lower()
    if "qwen3moe" in arch:
        L, E, K, norm, moe_attr, names, rl = cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok, cfg.norm_topk_prob, "mlp", ("gate_proj", "up_proj", "down_proj"), False
    elif "mixtral" in arch:
        L, E, K, norm, moe_attr, names, rl = cfg.num_hidden_layers, cfg.num_local_experts, cfg.num_experts_per_tok, True, "block_sparse_moe", ("w1", "w3", "w2"), True
    else:
        raise RuntimeError(arch)
    reader = ExpertReader(ckpt, moe_attr, names, L, E)
    units = int(budget_gib * (1 << 30) // reader.unit_bytes)
    if kind == "flashmoe":
        cache = FlashMoECache(reader, L, E, units // L, weights)
    else:
        pop = np.zeros((L, E)); aff = np.zeros((max(L - 1, 1), E, E))
        for p in traces:
            d = np.load(p, allow_pickle=True); tags = list(d["tags"])
            dm = np.isin(d["tag"], [i for i, t in enumerate(tags) if t.endswith("/decode")])
            lay, ex, pos, tg = d["layer"][dm].astype(int), d["expert"][dm].astype(int), d["pos"][dm], d["tag"][dm]
            for l, row in zip(lay, ex):
                for e in row:
                    if e >= 0: pop[l, e] += 1
            key = {}
            for t, p_, l, row in zip(tg, pos, lay, ex): key[(t, p_, l)] = row
            for (t, p_, l), row in key.items():
                if l == 0 or (t, p_, l - 1) not in key: continue
                for a in key[(t, p_, l - 1)]:
                    for e in row:
                        if a >= 0 and e >= 0: aff[l - 1, a, e] += 1
        ck = torch.load(predictor, map_location="cuda")
        pred = DuoPredictor(E, hidden=ck["hidden"]).cuda().eval()
        pred.load_state_dict(ck["state"])
        cache = DuoServeCache(reader, L, E, K, units, pred, pop, aff)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    for l, layer in enumerate(model.model.layers):
        old = getattr(layer, moe_attr)
        gate = nn.Linear(old.gate.in_features, old.gate.out_features, bias=False, device="cuda", dtype=torch.bfloat16)
        setattr(layer, moe_attr, BaselineMoE(gate, l, E, K, norm, cache, kind, rl))
    for sp in sorted(os.path.join(ckpt, f) for f in os.listdir(ckpt) if f.endswith(".safetensors")):
        with safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if ".experts." in k: continue
                set_module_tensor_to_device(model, k, "cuda", value=f.get_tensor(k), dtype=torch.bfloat16)
    for name, buf in list(model.named_buffers()):
        mod = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        mod._buffers[name.rsplit(".", 1)[-1]] = buf.to("cuda")
    model.eval()
    model.register_forward_pre_hook(lambda m, a: cache.tick())
    return model, AutoTokenizer.from_pretrained(ckpt), cache
