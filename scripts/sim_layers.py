#!/usr/bin/env python3
"""Per-layer decode miss rate of LRU and PHASOR's estimate at one residency
fraction, and how resident capacity ends up spread across layers.  CPU only.
usage: sim_layers.py <trace.npz> <resident-fraction>"""
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
    ids = (lay[pm][:, None] * E + ex[pm]).ravel()
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
    now = 0; hit = np.zeros(L); tot = np.zeros(L); occ = np.zeros(L); nocc = 0
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
            lt = t // E
            np.add.at(tot, lt, 1); np.add.at(hit, lt, res[t])
            last[t] = now; hist[t] += 1
            need = np.zeros(N, bool); need[t] = True
            res = keep(res | need, score(last, now, pfn, hist))
            occ += res.reshape(L, E).sum(1); nocc += 1
    return (1 - hit / np.maximum(tot, 1)), occ / nocc / E
lru = lambda last, now, pfn, hist: last
ph = lambda last, now, pfn, hist: (np.exp2(-(now - last) / 8) + 0.5 * pfn
                                   + 0.5 * hist / max(hist.max(), 1e-12))
mr_l, oc_l = run(lru); mr_p, oc_p = run(ph)
print(json.dumps({"trace": sys.argv[1], "frac": frac, "L": L,
                  "miss_lru": [round(x, 4) for x in mr_l], "miss_phasor": [round(x, 4) for x in mr_p],
                  "occupancy_lru": [round(x, 4) for x in oc_l], "occupancy_phasor": [round(x, 4) for x in oc_p]}))
