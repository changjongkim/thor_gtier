#!/usr/bin/env python3
"""Decode hit rate of LRU, PHASOR's estimate, FlashMoE-style per-layer LFU/LRU
mix and DuoServe-style prefetch against Belady, for one trace and cache size;
plus routing statistics.  CPU only; used alongside the serving runs.

usage: sim_all.py <trace.npz> <resident-fraction> [held-out.npz ...]
"""
import sys, json, numpy as np
d = np.load(sys.argv[1], allow_pickle=True); frac = float(sys.argv[2])
tags = list(d["tags"]); L = int(d["n_layers"]); E = int(d["n_experts"])
tag, lay, pos, ex = d["tag"], d["layer"].astype(int), d["pos"], d["expert"].astype(int)
N = L * E; C = max(1, int(frac * N))
reqs = {}
for i, t in enumerate(tags):
    n, ph = t.rsplit("/", 1); reqs.setdefault(n, {})[ph] = i
R = []
for n, ph in reqs.items():
    if "prefill" not in ph or "decode" not in ph: continue
    pm = tag == ph["prefill"]; dm = tag == ph["decode"]
    ids = (lay[pm][:, None] * E + ex[pm]).ravel(); ids = ids[ex[pm].ravel() >= 0] if False else ids
    u = np.zeros(N, bool); u[ids] = True
    pf = np.bincount(ids, minlength=N).astype(float)
    dp = pos[dm]; toks = []
    for p in np.unique(dp):
        m = dp == p
        toks.append(np.unique((lay[dm][m][:, None] * E + ex[dm][m]).ravel()))
    R.append((u, pf, toks))
R2 = R * 2

def run(score):
    res = np.zeros(N, bool); last = np.full(N, -1e9); hist = np.zeros(N)
    now = 0; hit = tot = 0
    def keep(cand, s):
        idx = np.flatnonzero(cand)
        if len(idx) <= C: r = np.zeros(N, bool); r[idx] = True; return r
        top = idx[np.argpartition(-s[idx], C - 1)[:C]]
        r = np.zeros(N, bool); r[top] = True; return r
    for (u, pf, toks) in R2:
        now += 1; pfn = pf / max(pf.max(), 1)
        last[u] = now
        res = keep(res | u, score(last, now, pfn, hist))
        for t in toks:
            now += 1
            hit += res[t].sum(); tot += len(t)
            last[t] = now; hist[t] += 1
            need = np.zeros(N, bool); need[t] = True
            res = keep(res | need, score(last, now, pfn, hist))
    return hit / tot

lru = lambda last, now, pfn, hist: last
def phasor(a=0.5, H=8, wr=1.0):
    return lambda last, now, pfn, hist: (wr * np.exp2(-(now - last) / H) + a * pfn
                                         + (1 - a) * hist / max(hist.max(), 1e-12))
lfu = lambda last, now, pfn, hist: hist + 1e-6 * last

def belady():
    seq = []
    for (u, pf, toks) in R2:
        seq.append((True, np.flatnonzero(u)))
        for t in toks: seq.append((False, t))
    T = len(seq); INF = 10**9
    nxt = np.full(N, INF); nexts = [None] * T
    for i in range(T - 1, -1, -1):
        nexts[i] = nxt.copy()
        if not seq[i][0]: nxt[seq[i][1]] = i
    res = np.zeros(N, bool); hit = tot = 0
    for i, (isp, ids) in enumerate(seq):
        if not isp: hit += res[ids].sum(); tot += len(ids)
        cand = res.copy(); cand[ids] = True
        idx = np.flatnonzero(cand)
        if len(idx) > C:
            sc = -nexts[i][idx].astype(float)
            top = idx[np.argpartition(-sc, C - 1)[:C]]
            res = np.zeros(N, bool); res[top] = True
        else: res = cand
    return hit / tot

# routing statistics
union_frac = np.mean([u.sum() / N for (u, _, _) in R])
cons = []
for (_, _, toks) in R:
    for a, b in zip(toks, toks[1:]):
        cons.append(len(np.intersect1d(a, b)) / max(len(a), 1))
k25 = int(0.25 * N); pred = []
for (u, pf, toks) in R:
    df = np.zeros(N)
    for t in toks: df[t] += 1
    top = np.argsort(-pf)[:k25]; pred.append(df[top].sum() / max(df.sum(), 1))
out = {"trace": sys.argv[1], "frac": frac, "C": C, "N": N,
       "prefill_union_frac": round(float(union_frac), 4),
       "consecutive_decode_overlap": round(float(np.mean(cons)), 4) if cons else None,
       "prompt_top25_covers_decode": round(float(np.mean(pred)), 4),
       "LRU": round(run(lru), 4), "LFU": round(run(lfu), 4),
       "PHASOR": round(run(phasor()), 4), "Belady": round(belady(), 4)}
sens = {}
for a in (0, 0.25, 0.5, 0.75, 1):
    for H in (2, 8, 32):
        sens[f"a{a}_H{H}"] = round(run(phasor(a=a, H=H)), 4)
out["sensitivity"] = sens
print(json.dumps(out))
