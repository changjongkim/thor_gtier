"""MoE-APEX* (ASPLOS'26), reimplemented on the transformers stack from its
preprint HOBBIT (Tang et al., arXiv 2411.01433), since no code is released.

Three mechanisms, as the paper describes them:
  1. Token-level dynamic expert loading (Sec. 3.2).  The K selected experts
     are ranked by their normalised gate weight ||G(x)_e||; the unimportance
     score of the i-th is s_i = sum_{j<i} ||G(x)_j|| (s_0 = 0).  On a cache
     miss an expert with s <= T1 is loaded in high precision (bf16), with
     T1 < s <= T2 in low precision (int4), and with s > T2 skipped.  T1 and T2
     come from profiling the score distribution (the paper's Mixtral split is
     67% high / 30% low / 3% skip; we take the same quantiles on held-out
     prompts).  Applied to decode tokens; prefill loads every expert it needs
     in high precision.
  2. Layer-level adaptive prefetching (Sec. 3.3).  The current layer's gating
     input is fed to the next layers' gates (stacked); the predicted experts of
     layer l+1 that are not cached are loaded ahead with the precision their
     predicted score gives, and if all are cached the predictor looks one layer
     further (up to p = 2).  Predicted experts are masked from eviction.
  3. Sequence-level multidimensional caching (Sec. 3.4).  Separate high- and
     low-precision caches; the resident with the lowest priority
        p = w_lru R/T + w_lfu F/T + w_lhu H/T + w_fld (1 - ((l_t - l_i + L) % L) / L)
     is evicted (R last-use step, F use count, H high-precision use count in the
     current sequence, T the step; records reset per sequence).  The paper
     tunes the four weights on a calibration set; we use equal weights.
Lossy: low-precision and skipped experts change the output; the runner records
the generated tokens so the agreement with bf16 can be reported.
"""
import json, os, struct
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

GROUP = 32          # blocks of 32, as llama.cpp Q4_0 (HOBBIT is built on llama.cpp)


def quantize_int4(w):
    """w [rows, cols] bf16 (cuda) -> packed uint8 [rows, cols/2], scale bf16 [rows, cols/GROUP].
    Symmetric int4 in blocks of GROUP values (levels -7..7, one bf16 scale per block)."""
    r, c = w.shape
    g = w.float().view(r, c // GROUP, GROUP)
    scale = g.abs().amax(-1).clamp_min(1e-8) / 7.0
    q = torch.round(g / scale[..., None]).clamp(-7, 7).to(torch.int8).view(r, c) + 8     # 1..15
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale.to(torch.bfloat16)


def dequant_int4(packed, scale, rows, cols):
    lo = (packed & 0xF).to(torch.int8) - 8
    hi = (packed >> 4).to(torch.int8) - 8
    q = torch.stack((lo, hi), -1).view(rows, cols).to(torch.bfloat16)
    return (q.view(rows, cols // GROUP, GROUP) * scale[..., None]).view(rows, cols)


def build_int4_store(reader, out_dir, L, E):
    """One file of every expert's three matrices in int4 (+ scales) and an index."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "experts_int4.bin"); idx = {}
    off = 0
    with open(path, "wb") as f:
        for l in range(L):
            for e in range(E):
                mats = []
                for (fi, o, ln, shape), w in zip(reader.loc[(l, e)], reader.read(l, e)):
                    rows, cols = shape
                    p, s = quantize_int4(w.view(rows, cols).cuda())
                    pb, sb = p.cpu().numpy().tobytes(), s.view(torch.int16).cpu().numpy().tobytes()
                    f.write(pb); f.write(sb)
                    mats.append([off, len(pb), len(sb), rows, cols]); off += len(pb) + len(sb)
                idx[f"{l},{e}"] = mats
    json.dump(idx, open(os.path.join(out_dir, "index.json"), "w"))
    return path


class Int4Reader:
    def __init__(self, store_dir):
        self.idx = json.load(open(os.path.join(store_dir, "index.json")))
        self.fd = os.open(os.path.join(store_dir, "experts_int4.bin"), os.O_RDONLY)
        self.bytes_read = 0
        m = self.idx["0,0"]; self.unit_bytes = sum(x[1] + x[2] for x in m)

    def read(self, l, e):
        out = []
        for off, pl, sl, rows, cols in self.idx[f"{l},{e}"]:
            buf = torch.empty(pl + sl, dtype=torch.uint8, pin_memory=True)
            got = os.preadv(self.fd, [memoryview(buf.numpy())], off)
            assert got == pl + sl
            self.bytes_read += pl + sl
            out.append((buf[:pl].view(rows, cols // 2), buf[pl:].view(torch.bfloat16).view(rows, cols // GROUP), rows, cols))
        return out


class ApexCache:
    """High- and low-precision expert caches on the GPU with MoE-APEX's
    loading, prefetching and eviction.  budget: bytes for both caches; the
    high-precision cache gets `high_frac` of it (the paper keeps it larger)."""

    def __init__(self, reader, lowreader, L, E, K, budget_bytes, gates, t1=0.6, t2=0.9,
                 high_frac=0.8, w=(0.25, 0.25, 0.25, 0.25), p_ahead=2):
        self.r, self.lr, self.L, self.E, self.K = reader, lowreader, L, E, K
        self.hcap = max(1, int(budget_bytes * high_frac // reader.unit_bytes))
        self.lcap = max(1, int(budget_bytes * (1 - high_frac) // lowreader.unit_bytes))
        self.high, self.low = {}, {}                 # (l,e) -> list of tensors
        self.gates = gates                           # list of nn.Linear, per layer
        self.t1, self.t2, self.w, self.p = t1, t2, w, p_ahead
        self.pool = ThreadPoolExecutor(4); self.copy = torch.cuda.Stream()
        self.pending = {}                            # (l,e,prec) -> future
        self.mask = set()
        self.reset(); self.hits = self.misses = self.skips = self.low_loads = 0
        self.calib = None                            # list collecting s values when profiling

    def reset(self):
        self.T = 0
        self.R = {}; self.F = {}; self.H = {}

    def tick(self): self.T += 1

    # ---------------------------------------------------------------- loading
    def _load_high(self, l, e):
        with torch.cuda.stream(self.copy):
            w = [t.to("cuda", non_blocking=True) for t in self.r.read(l, e)]
            ev = torch.cuda.Event(); ev.record(self.copy)
        return w, ev

    def _load_low(self, l, e):
        with torch.cuda.stream(self.copy):
            w = [(p.to("cuda", non_blocking=True), s.to("cuda", non_blocking=True), r, c)
                 for p, s, r, c in self.lr.read(l, e)]
            ev = torch.cuda.Event(); ev.record(self.copy)
        return w, ev

    def _prio(self, key, cur_layer):
        T = max(self.T, 1); l = key[0]
        fld = 1 - ((l - cur_layer + self.L) % self.L) / self.L
        a, b, c, d = self.w
        return a * self.R.get(key, 0) / T + b * self.F.get(key, 0) / T + c * self.H.get(key, 0) / T + d * fld

    def _admit(self, cache, cap, key, val, cur_layer, used):
        if key in cache: return
        while len(cache) >= cap:
            cand = [k for k in cache if k not in used and k not in self.mask]
            if not cand: return
            v = min(cand, key=lambda k: self._prio(k, cur_layer))
            del cache[v]
        cache[key] = val

    def _fetch(self, l, e, prec):
        key = (l, e, prec)
        if key in self.pending:
            w, ev = self.pending.pop(key).result()
        else:
            w, ev = (self._load_high if prec == "h" else self._load_low)(l, e)
        torch.cuda.current_stream().wait_event(ev)
        return w

    # ------------------------------------------------------------- selection
    def scores(self, weights):
        """weights: normalised gate weights of the selected experts, any order ->
        (order by decreasing weight, unimportance score per position)."""
        order = np.argsort(-weights)
        s = np.concatenate([[0.0], np.cumsum(weights[order])[:-1]])
        return order, s

    def get(self, l, experts, weights, prefill):
        """experts: ids; weights: their normalised gate weights (decode: one token).
        Returns {e: ("h", [g,u,d]) | ("l", [(p,s,r,c)...]) } without skipped experts."""
        out = {}
        used = {(l, e) for e in experts}
        if prefill:
            for e in experts:
                self._touch(l, e, True)
                k = (l, e)
                if k in self.high: self.hits += 1; out[e] = ("h", self.high[k]); continue
                self.misses += 1
                w = self._fetch(l, e, "h"); out[e] = ("h", w)
                self._admit(self.high, self.hcap, k, w, l, used)
            return out
        order, s = self.scores(weights)
        if self.calib is not None: self.calib.extend(s[1:].tolist())
        for pos, si in zip(order, s):
            e = experts[pos]; k = (l, e)
            if k in self.high:
                self.hits += 1; self._touch(l, e, True); out[e] = ("h", self.high[k]); continue
            if si <= self.t1:
                self.misses += 1; self._touch(l, e, True)
                w = self._fetch(l, e, "h"); out[e] = ("h", w)
                self._admit(self.high, self.hcap, k, w, l, used)
            elif si <= self.t2:
                self._touch(l, e, False)
                if k in self.low: self.hits += 1; out[e] = ("l", self.low[k]); continue
                self.misses += 1; self.low_loads += 1
                w = self._fetch(l, e, "l"); out[e] = ("l", w)
                self._admit(self.low, self.lcap, k, w, l, used)
            else:
                self.skips += 1
        return out

    def _touch(self, l, e, high):
        k = (l, e); self.R[k] = self.T; self.F[k] = self.F.get(k, 0) + 1
        if high: self.H[k] = self.H.get(k, 0) + 1

    # ------------------------------------------------------------- prefetch
    def prefetch(self, l, x, norm):
        """After layer l's routing (decode): predict the next layers' experts from
        the same gating input x [1, h] and load the missing ones ahead."""
        self.mask = set()
        for d in range(1, self.p + 1):
            n = l + d
            if n >= self.L: break
            with torch.no_grad():
                pr = torch.softmax(self.gates[n](x).float(), -1)
                w, sel = torch.topk(pr, self.K, -1)
                if norm: w = w / w.sum(-1, keepdim=True)
            sel = sel[0].tolist(); w = w[0].cpu().numpy()
            order, s = self.scores(w)
            missing = False
            for pos, si in zip(order, s):
                e = sel[pos]; k = (n, e); self.mask.add(k)
                if k in self.high: continue
                if si <= self.t1: prec = "h"
                elif si <= self.t2:
                    if k in self.low: continue
                    prec = "l"
                else: continue
                missing = True
                key = (n, e, prec)
                if key not in self.pending:
                    fn = self._load_high if prec == "h" else self._load_low
                    self.pending[key] = self.pool.submit(fn, n, e)
            if missing: break          # adaptive: look further only if layer n is covered
        # drop stale prefetches of layers already passed
        for key in [k for k in self.pending if k[0] <= l]:
            self.pending.pop(key)
