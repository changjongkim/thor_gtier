#!/usr/bin/env python3
"""Decode hit rate and per-layer misses for LRU, PHASOR, FlashMoE* and
DuoServe*, same trace replay as sim_all.py.  CPU only.

FlashMoE*: per-layer slots (C/L), eviction by the trained FFN over
  (1/recency, frequency/max); prefill fills free slots only.
DuoServe*: global LRU; each decode token also fetches the predicted top-k of
  every layer (popularity x affinity from the held-out traces, conditioned on
  the experts actually chosen at the previous layer); extra = predicted units
  that were not needed.
usage: sim_sota.py <trace.npz> <frac> <flashmoe-weights> <held-out.npz>...
"""
import sys, json, numpy as np
d = np.load(sys.argv[1], allow_pickle=True); frac = float(sys.argv[2]); wfile = sys.argv[3]
held = sys.argv[4:]
tags = list(d["tags"]); L = int(d["n_layers"]); E = int(d["n_experts"])
N = L * E; C = max(1, int(frac * N)); K = int(d["topk"])

def load(path):
    dd = np.load(path, allow_pickle=True); tg = list(dd["tags"])
    tag, lay, pos, ex = dd["tag"], dd["layer"].astype(int), dd["pos"], dd["expert"].astype(int)
    reqs = {}
    for i, t in enumerate(tg):
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
            sel = np.full((L, K), -1)
            for l, row in zip(lay[dm][m], ex[dm][m]): sel[l, :len(row)] = row
            toks.append((np.unique((lay[dm][m][:, None] * E + ex[dm][m]).ravel()), sel))
        R.append((u, pf, toks))
    return R
R2 = load(sys.argv[1]) * 2

# ---- global keep-top-C policies (LRU, PHASOR) ------------------------------
def run_global(score):
    res = np.zeros(N, bool); last = np.full(N, -1e9); hist = np.zeros(N); now = 0
    hit = np.zeros(L); tot = np.zeros(L)
    def keep(cand, s):
        idx = np.flatnonzero(cand)
        if len(idx) <= C: r = np.zeros(N, bool); r[idx] = True; return r
        top = idx[np.argpartition(-s[idx], C - 1)[:C]]
        r = np.zeros(N, bool); r[top] = True; return r
    for (u, pf, toks) in R2:
        now += 1; pfn = pf / max(pf.max(), 1); last[u] = now
        res = keep(res | u, score(last, now, pfn, hist))
        for t, _ in toks:
            now += 1; lt = t // E
            np.add.at(tot, lt, 1); np.add.at(hit, lt, res[t])
            last[t] = now; hist[t] += 1
            nd = np.zeros(N, bool); nd[t] = True
            res = keep(res | nd, score(last, now, pfn, hist))
    return hit, tot, 0
lru = lambda last, now, pfn, hist: last
phasor = lambda last, now, pfn, hist: (np.exp2(-(now - last) / 8) + 0.5 * pfn
                                       + 0.5 * hist / max(hist.max(), 1e-12))

# ---- FlashMoE* ---------------------------------------------------------------
lines = open(wfile).read().split("\n"); k = 0; nl = int(lines[k]); k += 1; Ws = []
for _ in range(nl):
    r, c = map(int, lines[k].split()); k += 1
    W = np.array(lines[k].split(), float).reshape(r, c); k += 1
    b = np.array(lines[k].split(), float); k += 1; Ws.append((W, b))
def ffn(x):
    for i, (W, b) in enumerate(Ws):
        x = x @ W.T + b
        if i < len(Ws) - 1: x = x / (1 + np.exp(-x))
    return x[:, 0]
def run_flashmoe():
    slots = max(1, C // L)
    cache = [set() for _ in range(L)]; last = np.full(N, -1e9); freq = np.zeros(N); step = 0
    hit = np.zeros(L); tot = np.zeros(L)
    for (u, pf, toks) in R2:
        for idx in np.flatnonzero(u):                 # prefill fills free slots
            l = idx // E
            if len(cache[l]) < slots: cache[l].add(idx)
        for t, _ in toks:
            step += 1
            for idx in t: last[idx] = step; freq[idx] += 1
            for idx in t:
                l = idx // E; tot[l] += 1
                if idx in cache[l]: hit[l] += 1; continue
                if len(cache[l]) < slots: cache[l].add(idx); continue
                cand = [c for c in cache[l] if c not in set(t)]
                if not cand: continue
                fm = max(freq[l*E:(l+1)*E].max(), 1)
                X = np.array([[1.0 / (step - last[c] + 1), freq[c] / fm] for c in cand])
                v = cand[int(np.argmax(ffn(X)))]
                cache[l].discard(v); cache[l].add(idx)
    return hit, tot, 0

# ---- DuoServe* ---------------------------------------------------------------
pop = np.zeros(N); aff = np.zeros((L, E, E))
for hp in held:
    for (_, _, toks) in load(hp):
        for t, sel in toks:
            for l in range(L):
                for e in sel[l]:
                    if e < 0: continue
                    pop[l*E+e] += 1
                    if l: 
                        for p in sel[l-1]:
                            if p >= 0: aff[l-1, p, e] += 1
def run_duoserve():
    res = np.zeros(N, bool); last = np.full(N, -1e9); now = 0; extra = 0
    hit = np.zeros(L); tot = np.zeros(L)
    def keep(cand):
        idx = np.flatnonzero(cand)
        if len(idx) <= C: r = np.zeros(N, bool); r[idx] = True; return r
        top = idx[np.argpartition(-last[idx], C - 1)[:C]]
        r = np.zeros(N, bool); r[top] = True; return r
    for (u, pf, toks) in R2:
        now += 1; last[u] = now; res = keep(res | u)
        for t, sel in toks:
            now += 1
            pred = []
            for l in range(L):
                sc = pop[l*E:(l+1)*E] + 1e-9
                if l:
                    a = sum(aff[l-1, p] for p in sel[l-1] if p >= 0)
                    sc = sc * (a + 1e-9)
                pred += list(l*E + np.argsort(-sc)[:K])
            pred = np.array(pred); need = set(t.tolist())
            fetched_pred = [p for p in pred if p not in need and not res[p]]
            extra += len(fetched_pred)
            lt = t // E; np.add.at(tot, lt, 1); np.add.at(hit, lt, res[t])
            last[t] = now; last[fetched_pred] = now - 0.5
            nd = np.zeros(N, bool); nd[t] = True; nd[fetched_pred] = True
            res = keep(res | nd)
    return hit, tot, extra

out = {"trace": sys.argv[1], "frac": frac, "L": L}
for name, fn in (("LRU", lambda: run_global(lru)), ("PHASOR", lambda: run_global(phasor)),
                 ("FlashMoE", run_flashmoe), ("DuoServe", run_duoserve)):
    h, t, x = fn()
    out[name] = {"hit": round(float(h.sum() / t.sum()), 4),
                 "miss_by_layer": [round(float(1 - a / max(b, 1)), 4) for a, b in zip(h, t)],
                 "extra_fetch_per_access": round(float(x / t.sum()), 4)}
print(json.dumps(out))
