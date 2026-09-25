#!/usr/bin/env python3
"""Decode miss rate against resident capacity for LRU (access-level time),
LFU (decode history), PHASOR and Belady, by trace replay.  CPU only.
usage: sim_sweep.py <trace.npz> <fractions comma-separated>"""
import json, sys
import numpy as np
d = np.load(sys.argv[1], allow_pickle=True); fracs = [float(x) for x in sys.argv[2].split(",")]
tags = list(d["tags"]); L = int(d["n_layers"]); E = int(d["n_experts"])
tag, lay, pos, ex = d["tag"], d["layer"].astype(int), d["pos"], d["expert"].astype(int)
N = L * E; layer_of = np.arange(N) // E; rng = np.random.default_rng(1)
reqs = {}
for i, t in enumerate(tags):
    n, ph = t.rsplit("/", 1); reqs.setdefault(n, {})[ph] = i
R = []
for n, ph in reqs.items():
    if "prefill" not in ph or "decode" not in ph: continue
    pm = tag == ph["prefill"]; dm = tag == ph["decode"]
    ids = (lay[pm][:, None] * E + ex[pm]).ravel()
    u = np.zeros(N, bool); u[ids] = True
    pf = np.bincount(ids, minlength=N).astype(float)
    dp = pos[dm]; toks = []
    for p in np.unique(dp):
        m = dp == p; toks.append(np.unique((lay[dm][m][:, None] * E + ex[dm][m]).ravel()))
    R.append((u, pf, toks))
R2 = R * 2

def run(kind, C):
    res = np.zeros(N, bool); last = np.full(N, -1e9); hist = np.zeros(N); now = 0.0; hit = tot = 0
    jit = rng.random(N) * 1e-9
    def keep(cand, pfn):
        idx = np.flatnonzero(cand)
        if len(idx) <= C: r = np.zeros(N, bool); r[idx] = True; return r
        if kind == "lru": s = last
        elif kind == "lfu": s = hist
        else: s = np.exp2(-(now - last) / 8) + 0.5 * pfn + 0.5 * hist / max(hist.max(), 1e-12)
        s = s + jit
        top = idx[np.argpartition(-s[idx], C - 1)[:C]]
        r = np.zeros(N, bool); r[top] = True; return r
    for (u, pf, toks) in R2:
        now += 1; pfn = pf / max(pf.max(), 1); ids = np.flatnonzero(u)
        last[ids] = now + layer_of[ids] / L
        res = keep(res | u, pfn)
        for t in toks:
            now += 1; hit += res[t].sum(); tot += len(t)
            last[t] = now + layer_of[t] / L; hist[t] += 1
            nd = np.zeros(N, bool); nd[t] = True; res = keep(res | nd, pfn)
    return 1 - hit / tot

seq = []
for (u, pf, toks) in R2:
    seq.append((True, np.flatnonzero(u)))
    for t in toks: seq.append((False, t))
T = len(seq); INF = 10**9; nxt = np.full(N, INF); nexts = [None] * T
for i in range(T - 1, -1, -1):
    nexts[i] = nxt.copy()
    if not seq[i][0]: nxt[seq[i][1]] = i
def belady(C):
    res = np.zeros(N, bool); hit = tot = 0
    for i, (isp, ids) in enumerate(seq):
        if not isp: hit += res[ids].sum(); tot += len(ids)
        cand = res.copy(); cand[ids] = True; idx = np.flatnonzero(cand)
        if len(idx) > C:
            top = idx[np.argpartition(nexts[i][idx].astype(float), C - 1)[:C]]
            res = np.zeros(N, bool); res[top] = True
        else: res = cand
    return 1 - hit / tot

out = {"trace": sys.argv[1], "points": []}
for f in fracs:
    C = max(1, int(f * N))
    out["points"].append({"frac": f, "lru": round(run("lru", C), 4), "lfu": round(run("lfu", C), 4),
                          "phasor": round(run("phasor", C), 4), "belady": round(belady(C), 4)})
    print(json.dumps(out["points"][-1]), file=sys.stderr, flush=True)
print(json.dumps(out))
