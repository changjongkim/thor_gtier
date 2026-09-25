#!/usr/bin/env python3
"""Residency design points, one at a time, by trace replay (CPU only).

Every policy keeps at most C units.  A policy is (score, admission):
  score      recency (step- or access-level time), frequency (decode-only or
             decode+prefill history), or PHASOR's sum of them
  admission  selective: a passing unit enters only if it outranks the lowest
             resident; admit-all: every unit that is used enters, evicting the
             lowest resident
Reported per policy: decode hit rate, decode miss rate by layer quarter, and
the fraction of each request's prefill union that is not resident when the
request arrives (the prefill read volume the policy leaves).

Timestamps.  step-level: one tick per serving step, so all layers of a token
share a time and ties are broken at random.  access-level: within a step the
layers are touched in order, layer l at time step + l/L, which is what a real
LRU observes.

usage: sim_ablate.py <trace.npz> <resident-fraction> <seed>
"""
import json, sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=True); frac = float(sys.argv[2]); rng = np.random.default_rng(int(sys.argv[3]))
tags = list(d["tags"]); L = int(d["n_layers"]); E = int(d["n_experts"])
tag, lay, pos, ex = d["tag"], d["layer"].astype(int), d["pos"], d["expert"].astype(int)
N = L * E; C = max(1, int(frac * N)); layer_of = np.arange(N) // E
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


def run(kind, time_unit="step", hist_scope="decode", admission="selective", a=0.5, H=8):
    res = np.zeros(N, bool); last = np.full(N, -1e9); hist = np.zeros(N); now = 0.0
    hit = np.zeros(L); tot = np.zeros(L); pre_miss = []; jitter = rng.random(N) * 1e-9

    def stamp(ids):
        if time_unit == "access": last[ids] = now + layer_of[ids] / L
        else: last[ids] = now

    def score(pfn):
        rec = np.exp2(-(now - last) / H)
        hn = hist / max(hist.max(), 1e-12)
        if kind == "lru": s = last.copy()
        elif kind == "lfu": s = hist.copy()
        else: s = rec + a * pfn + (1 - a) * hn
        return s + rng.random(N) * 1e-9 if time_unit == "step" else s + jitter

    def admit(new, pfn):
        nonlocal res
        s = score(pfn)
        if admission == "selective":
            cand = res | new
            idx = np.flatnonzero(cand)
            if len(idx) > C:
                top = idx[np.argpartition(-s[idx], C - 1)[:C]]
                res = np.zeros(N, bool); res[top] = True
            else:
                res = cand
        else:  # admit-all: every used unit enters; the lowest residents make room
            add = new & ~res
            res = res | add
            over = int(res.sum()) - C
            if over > 0:
                old = np.flatnonzero(res & ~new)
                if len(old):
                    drop = old[np.argpartition(s[old], min(over, len(old)) - 1)[:min(over, len(old))]]
                    res[drop] = False
                over = int(res.sum()) - C
                if over > 0:                      # the new set alone exceeds C
                    nn = np.flatnonzero(res)
                    res[nn[np.argpartition(s[nn], over - 1)[:over]]] = False

    for (u, pf, toks) in R2:
        now += 1.0; pfn = pf / max(pf.max(), 1)
        pre_miss.append(float((u & ~res).sum()) / max(u.sum(), 1))
        ids = np.flatnonzero(u); stamp(ids)
        if hist_scope == "all": hist[ids] += 1
        admit(u.copy(), pfn)
        for t in toks:
            now += 1.0
            lt = t // E
            np.add.at(tot, lt, 1); np.add.at(hit, lt, res[t])
            stamp(t); hist[t] += 1
            need = np.zeros(N, bool); need[t] = True
            admit(need, pfn)
    q = L // 4; miss = 1 - hit / np.maximum(tot, 1)
    return {"hit": round(float(hit.sum() / tot.sum()), 4),
            "miss_by_quarter": [round(float(miss[i*q:(i+1)*q].mean()), 4) for i in range(4)],
            "prefill_nonresident": round(float(np.mean(pre_miss)), 4)}


out = {"trace": sys.argv[1], "frac": frac, "C": C, "N": N, "runs": {
    # 2: recency unit and ties
    "lru_step_randomtie": run("lru", "step"),
    "lru_access": run("lru", "access"),
    "phasor_step": run("phasor", "step"),
    "phasor_access": run("phasor", "access"),
    # 1: history scope
    "phasor_hist_decode": run("phasor", hist_scope="decode"),
    "phasor_hist_all": run("phasor", hist_scope="all"),
    "lfu_decode": run("lfu", hist_scope="decode"),
    "lfu_all": run("lfu", hist_scope="all"),
    # 3: admission
    "phasor_selective": run("phasor", admission="selective"),
    "phasor_admit_all": run("phasor", admission="all"),
    "lru_admit_all": run("lru", "access", admission="all"),
}}
print(json.dumps(out))
