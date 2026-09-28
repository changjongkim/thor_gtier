"""MoE-APEX* (Tang et al., ASPLOS'26, doi 10.1145/3779212.3790187), reimplemented on
the transformers stack; no code is released (the paper builds on llama.cpp).

Following the paper:
  1. Token-level dynamic expert loading (Sec. 3.2).  The K selected experts
     are ranked by normalised gate weight ||G(x)_e||; the cumulative score of
     the i-th is s_i = sum_{j<i} ||G(x)_j|| (s_0 = 0, eq. 2).  On a cache miss
     s <= T1 loads the high-precision expert (bf16 here, float16 there),
     T1 < s <= T2 the low-precision one (int2, as the paper's Float16+Int2),
     s > T2 skips it; the top-1 expert is always high precision.  T1/T2 come
     from profiling the score distribution; for Mixtral the paper's T1=0.6,
     T2=0.9 give 67% high / 30% low / 3% skip, and we take those quantiles on
     held-out prompts for each model.  Applied to decode tokens (batch 1, the
     paper's setting); prefill loads the experts it needs in high precision.
  2. Layer-level adaptive prefetching (Sec. 3.3).  The current gating input
     goes through the next layers' gates (stacked); if every predicted expert of
     layer l+1 is cached the predictor moves to l+2, up to p = 2 (the paper's
     optimum is 2-4); missing predicted experts are loaded ahead in the
     precision their score gives, and predicted experts are masked from
     eviction.  (The paper cuts a mispredicted load short block by block; here
     a started load completes.)
  3. Sequence-level cost-aware caching, LCU (Sec. 3.4).  Separate high- and
     low-precision caches.  Cost C_t = H_t + (B_l/B_h) L_t (eq. 3) with H/L the
     sequence's high/low-precision use counts and B_l/B_h = 2/16; priority
     p_t = C_t/T + 1/D_t^i if t was used in the last forward pass, else C_t/T
     (eq. 4), D_t^i = (l_t - l_i + l_n) % l_n + 0.1; the lowest-priority resident
     is evicted.  Records reset at each new sequence.
Modes.  "bf16" (the evaluation's): precision adaptation off -- every expert is
loaded in bf16 and none is skipped, so outputs equal the original model like
every other system here; caching (LCU) and prefetching are the paper's.
"mixed": the paper's Float16+Int2 loading (lossy), kept for reference.
"""
import json, os, struct
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

GROUP = 64          # 2-bit codes + a bf16 scale and min per 64 weights = 2.5 bits/weight (llama.cpp Q2_K: 2.56)
BITS_LOW, BITS_HIGH = 2, 16


def quantize_int2(w):
    """w [rows, cols] -> packed uint8 [rows, cols/4], scale and min bf16 [rows, cols/GROUP].
    Asymmetric 2-bit per block; the block's range is shrunk by the factor (of
    five) with the least squared error, as k-quants search their scales."""
    r, c = w.shape
    g = w.float().view(r, c // GROUP, GROUP)
    lo0, hi0 = g.amin(-1), g.amax(-1)
    best = None
    for f in (1.0, 0.9, 0.8, 0.7, 0.6):
        mid = (lo0 + hi0) / 2; half = (hi0 - lo0) / 2 * f
        lo, hi = mid - half, mid + half
        sc = ((hi - lo) / 3).clamp_min(1e-8)
        q = torch.round((g - lo[..., None]) / sc[..., None]).clamp(0, 3)
        err = ((q * sc[..., None] + lo[..., None] - g) ** 2).sum(-1)
        if best is None: best = [err, q, sc, lo]
        else:
            m = err < best[0]
            best[0] = torch.where(m, err, best[0]); best[1] = torch.where(m[..., None], q, best[1])
            best[2] = torch.where(m, sc, best[2]); best[3] = torch.where(m, lo, best[3])
    q = best[1].to(torch.uint8).view(r, c)
    packed = q[:, 0::4] | (q[:, 1::4] << 2) | (q[:, 2::4] << 4) | (q[:, 3::4] << 6)
    return packed, best[2].to(torch.bfloat16), best[3].to(torch.bfloat16)


def dequant_int2(packed, scale, mn, rows, cols):
    q = torch.stack([(packed >> sh) & 3 for sh in (0, 2, 4, 6)], -1).view(rows, cols).to(torch.bfloat16)
    return (q.view(rows, cols // GROUP, GROUP) * scale[..., None] + mn[..., None]).view(rows, cols)


def build_low_store(reader, out_dir, L, E):
    """One file of every expert's three matrices in int2 (+ scales, mins) and an index."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "experts_int2.bin"); idx = {}
    off = 0
    with open(path, "wb") as f:
        for l in range(L):
            for e in range(E):
                mats = []
                for (fi, o, ln, shape), w in zip(reader.loc[(l, e)], reader.read(l, e)):
                    rows, cols = shape
                    p, sc, mn = quantize_int2(w.view(rows, cols).cuda())
                    pb = p.cpu().numpy().tobytes()
                    sb = torch.cat([sc, mn], -1).view(torch.int16).cpu().numpy().tobytes()
                    f.write(pb); f.write(sb)
                    mats.append([off, len(pb), len(sb), rows, cols]); off += len(pb) + len(sb)
                idx[f"{l},{e}"] = mats
    json.dump(idx, open(os.path.join(out_dir, "index.json"), "w"))
    return path


class LowReader:
    def __init__(self, store_dir):
        self.idx = json.load(open(os.path.join(store_dir, "index.json")))
        self.fd = os.open(os.path.join(store_dir, "experts_int2.bin"), os.O_RDONLY)
        self.bytes_read = 0
        m = self.idx["0,0"]; self.unit_bytes = sum(x[1] + x[2] for x in m)

    def read(self, l, e):
        out = []
        for off, pl, sl, rows, cols in self.idx[f"{l},{e}"]:
            buf = torch.empty(pl + sl, dtype=torch.uint8, pin_memory=True)
            got = os.preadv(self.fd, [memoryview(buf.numpy())], off)
            assert got == pl + sl
            self.bytes_read += pl + sl
            sm = buf[pl:].view(torch.bfloat16).view(rows, 2 * (cols // GROUP))
            out.append((buf[:pl].view(rows, cols // 4), sm[:, :cols // GROUP], sm[:, cols // GROUP:], rows, cols))
        return out


class ApexCache:
    """High- and low-precision expert caches on the GPU with MoE-APEX's loading,
    prefetching and LCU eviction.  budget: bytes for both caches (bf16 mode: all
    of it for the high-precision cache; mixed: `high_frac` of it)."""

    def __init__(self, reader, lowreader, L, E, K, budget_bytes, gates, t1=0.6, t2=0.9,
                 high_frac=0.8, p_ahead=2, mixed=False):
        self.r, self.lr, self.L, self.E, self.K = reader, lowreader, L, E, K
        self.mixed = mixed
        hf = high_frac if mixed else 1.0
        self.hcap = max(1, int(budget_bytes * hf // reader.unit_bytes))
        self.lcap = max(1, int(budget_bytes * (1 - hf) // lowreader.unit_bytes)) if mixed else 0
        self.high, self.low = {}, {}                 # (l,e) -> tensors
        self.gates = gates                           # nn.Linear per layer
        self.t1, self.t2, self.p = t1, t2, p_ahead
        self.cost_low = BITS_LOW / BITS_HIGH         # B_l / B_h
        self.pool = ThreadPoolExecutor(4); self.copy = torch.cuda.Stream()
        self.pending = {}                            # (l,e,prec) -> future
        self.mask = set()
        self.reset(); self.hits = self.misses = self.skips = self.low_loads = 0
        self.calib = None                            # list collecting s values when profiling

    def reset(self):                                 # a new sequence: LR, HR, FR records
        self.T = 0
        self.Hc = {}; self.Lc = {}; self.F_prev = set(); self.F_cur = set()

    def tick(self):                                  # one forward pass (token)
        self.T += 1; self.F_prev = self.F_cur; self.F_cur = set()

    # ---------------------------------------------------------------- loading
    def _load_high(self, l, e):
        with torch.cuda.stream(self.copy):
            w = [t.to("cuda", non_blocking=True) for t in self.r.read(l, e)]
            ev = torch.cuda.Event(); ev.record(self.copy)
        return w, ev

    def _load_low(self, l, e):
        with torch.cuda.stream(self.copy):
            w = [tuple(x.to("cuda", non_blocking=True) if torch.is_tensor(x) else x for x in t)
                 for t in self.lr.read(l, e)]
            ev = torch.cuda.Event(); ev.record(self.copy)
        return w, ev

    def _prio(self, key, cur_layer):
        """LCU, eq. (3)-(4): C_t/T (+ 1/D_t^i if t was used in the last forward pass)."""
        T = max(self.T, 1)
        c = self.Hc.get(key, 0) + self.cost_low * self.Lc.get(key, 0)
        p = c / T
        if key in self.F_prev:
            p += 1.0 / ((key[0] - cur_layer + self.L) % self.L + 0.1)
        return p

    def _admit(self, cache, cap, key, val, cur_layer, used):
        if key in cache or cap <= 0: return
        while len(cache) >= cap:
            cand = [k for k in cache if k not in used and k not in self.mask]
            if not cand: return
            del cache[min(cand, key=lambda k: self._prio(k, cur_layer))]
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
        if not self.mixed:                            # bf16 mode: decode like prefill
            return self.get(l, experts, None, True)
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
        k = (l, e); self.F_cur.add(k)
        if high: self.Hc[k] = self.Hc.get(k, 0) + 1
        else: self.Lc[k] = self.Lc.get(k, 0) + 1

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
                if not self.mixed or si <= self.t1: prec = "h"
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
