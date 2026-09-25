"""PHASOR inside Hugging Face transformers.

The model is built without its routed experts, the non-expert weights are
loaded onto the GPU, and every MoE block is replaced by PhasorMoE, which asks
the C++ engine (phasor_ext.cpp) for the weights of the experts its router
chose.  Everything else -- attention, norms, sampling, generate() -- is the
stock transformers code, the same stack MoE-Infinity and ZipMoE run on.

Supported: Qwen3-MoE (Qwen3-30B-A3B) and Mixtral, bf16 safetensors.
"""
import json
import os
import struct

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "lib")
_ext = None


def ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="phasor_ext",
            sources=[os.path.join(HERE, "phasor_ext.cpp")],
            extra_include_paths=[LIB, "/usr/local/cuda-13.0/targets/sbsa-linux/include"],
            extra_cflags=["-O3", "-std=c++17"],
            extra_ldflags=[os.path.join(LIB, "libgtier.a"), "-L/usr/local/cuda-13.0/targets/sbsa-linux/lib", "-Wl,-rpath,/usr/local/cuda-13.0/targets/sbsa-linux/lib",
                           "-lcudart", "-luring", "-lcufile"],
            build_directory=os.path.join(HERE, "build"),
            verbose=False,
        )
    return _ext


def safetensors_index(path):
    """name -> (absolute byte offset, length, shape) for one shard."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out[k] = (base + a, b - a, v["shape"], v["dtype"])
    return out


# E5: when set to a dict, every MoE layer adds its time in seconds, split by
# phase, into PROFILE[phase][part]: "wait" = collect (expert bytes not yet
# resident), "expert" = expert matmuls, "moe" = the whole block.  Profiling
# synchronizes around each part, so it runs separately from the timed matrix.
PROFILE = None


def _pf(phase, part, dt):
    d = PROFILE.setdefault(phase, {})
    d[part] = d.get(part, 0.0) + dt


class PhasorMoE(nn.Module):
    """Drop-in for Qwen3MoeSparseMoeBlock / MixtralSparseMoeBlock."""

    def __init__(self, gate, layer, n_experts, top_k, norm_topk, engine, chunk, returns_logits, pipeline=True):
        super().__init__()
        self.gate = gate
        self.layer, self.E, self.k, self.norm = layer, n_experts, top_k, norm_topk
        self.engine, self.chunk, self.returns_logits = engine, chunk, returns_logits
        self.pipeline = pipeline

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        logits = self.gate(x)
        w = F.softmax(logits, dim=1, dtype=torch.float)
        w, sel = torch.topk(w, self.k, dim=-1)
        if self.norm:
            w = w / w.sum(dim=-1, keepdim=True)
        w = w.to(x.dtype)
        # a decode step of a batch is still decode: b sequences, one token each
        prefill = s > 1
        prof = PROFILE is not None
        if prof:
            torch.cuda.synchronize(); t_blk = time.perf_counter(); phase = "prefill" if prefill else "decode"
        if prefill:
            self.engine.note_prefill(self.layer, torch.bincount(sel.flatten(), minlength=self.E))
        experts = torch.unique(sel).tolist()
        out = torch.zeros_like(x)
        chunks = [experts[i:i + self.chunk] for i in range(0, len(experts), self.chunk)]
        # Causal pipeline within the layer: the next chunk is submitted before
        # the current one is collected, so its reads overlap this chunk's
        # matmuls; the synchronize before a submit keeps a slot from being
        # refilled while a kernel still reads it.
        handles = [self.engine.submit(self.layer, chunks[0])] if chunks else []
        for i, ch in enumerate(chunks):
            if not self.pipeline:          # ablation: read a chunk only after the last one computed
                if i >= 1:
                    torch.cuda.synchronize()
                    handles.append(self.engine.submit(self.layer, ch))
            elif i + 1 < len(chunks):
                if i >= 1:
                    torch.cuda.synchronize()
                handles.append(self.engine.submit(self.layer, chunks[i + 1]))
            if prof:
                torch.cuda.synchronize(); t1 = time.perf_counter()
            if prof:
                st0 = self.engine.stats()
            ws = self.engine.collect(handles[i], prefill)
            if prof:
                t2 = time.perf_counter(); _pf(phase, "wait", t2 - t1)
                st1 = self.engine.stats()
                _pf(phase, "hits", st1[0] - st0[0]); _pf(phase, "misses", st1[1] - st0[1])
                _pf(phase, "read_gib", (st1[2] - st0[2]) / 2**30)
            for e, (g, u, d) in zip(ch, ws):
                tok, kk = torch.where(sel == e)
                xe = x[tok]
                y = F.linear(F.silu(F.linear(xe, g)) * F.linear(xe, u), d)
                out.index_add_(0, tok, y * w[tok, kk, None])
            if prof:
                torch.cuda.synchronize(); _pf(phase, "expert", time.perf_counter() - t2)
        torch.cuda.synchronize()
        if prof:
            _pf(phase, "moe", time.perf_counter() - t_blk); _pf(phase, "layers", 1)
        out = out.view(b, s, h)
        return (out, logits) if self.returns_logits else out


def build(ckpt, budget_gib, window_gib=0.5, slot_mib=4, policy="phasor", mix=0.5,
          rec_half=8.0, w_rec=1.0, chunk=None, pipeline=True):
    """Returns (model, tokenizer, engine).  budget_gib covers the arena and the
    staging window; the non-expert weights and the KV cache sit on the GPU as in
    every transformers-based system."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(ckpt)
    arch = cfg.architectures[0].lower()
    if "qwen3moe" in arch:
        L, E, K = cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok
        norm, moe_attr, names, ret_logits = cfg.norm_topk_prob, "mlp", ("gate_proj", "up_proj", "down_proj"), False
    elif "mixtral" in arch:
        L, E, K = cfg.num_hidden_layers, cfg.num_local_experts, cfg.num_experts_per_tok
        norm, moe_attr, names, ret_logits = True, "block_sparse_moe", ("w1", "w3", "w2"), True
    else:
        raise RuntimeError(f"unsupported architecture {arch}")

    shards = sorted(os.path.join(ckpt, f) for f in os.listdir(ckpt) if f.endswith(".safetensors"))
    idx = {}
    for fi, sp in enumerate(shards):
        for k, v in safetensors_index(sp).items():
            idx[k] = (fi,) + v
    arena_gib = max(0.0, budget_gib - window_gib)
    eng = ext().Engine(shards, L, E, window_gib, slot_mib, arena_gib, policy, mix, rec_half, w_rec)
    for l in range(L):
        for e in range(E):
            projs = []
            for nm in names:
                key = f"model.layers.{l}.{moe_attr}.experts.{e}.{nm}.weight"
                fi, off, ln, shape, dt = idx[key]
                assert dt == "BF16", dt
                projs.append([fi, off, ln, shape[0], shape[1]])
            eng.add_unit(l, e, projs)
    eng.finalize()

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    ranges_per_ticket = max(4, int(window_gib * 1024 / slot_mib) // 4)
    chunk = chunk or max(1, ranges_per_ticket // 3)
    for l, layer in enumerate(model.model.layers):
        old = getattr(layer, moe_attr)
        gate = nn.Linear(old.gate.in_features, old.gate.out_features, bias=False,
                         device="cuda", dtype=torch.bfloat16)
        setattr(layer, moe_attr, PhasorMoE(gate, l, E, K, norm, eng, chunk, ret_logits, pipeline))

    from safetensors import safe_open
    for sp in shards:
        with safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if ".experts." in k:
                    continue
                set_module_tensor_to_device(model, k, "cuda", value=f.get_tensor(k),
                                            dtype=torch.bfloat16)
    for name, buf in list(model.named_buffers()):
        mod = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        mod._buffers[name.rsplit(".", 1)[-1]] = buf.to("cuda")
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.eval()
    # one serving step per forward: a prefill, or one decode token
    model.register_forward_pre_hook(lambda m, a: eng.step())
    tok = AutoTokenizer.from_pretrained(ckpt)
    return model, tok, eng
