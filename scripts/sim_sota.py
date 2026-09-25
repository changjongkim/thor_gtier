#!/usr/bin/env python3
"""Decode hit rate of the residency policies of the systems that solve the
same problem, by trace replay on one trace and capacity.  CPU only.

Definitions match Fig. 9 / sim_ablate.py and the serving baselines
(baselines_hf/offload_hf.py):
  LRU        global keep-top-C on access-level time (lru_access)
  LFU        every access counted, the prompt per token (lfu_all)
  PHASOR     step-level time (phasor_step)
  FlashMoE*  per-layer slots (C // L); eviction by its trained FFN over
             (1/recency, frequency/max), trained on the held-out workloads at
             exactly C // L slots; prefill fills free slots only
  DuoServe*  host LRU of C units (access-level) plus decode prefetch: after
             layer l of a token, the predictor picks top-K of layer l+1 from
             the experts chosen so far in this token, the popularity of layer
             l+1 and the affinity from layer l's actual choices; the copies
             are assumed to land in time, so a predicted unit counts as a hit
             for layer l+1 of the same token (favours DuoServe).  Predictor:
             the trained MLP used in serving (--predictor .npz), and the
             popularity x affinity score it takes as input, for reference.
Also reported: prefetches of units that were neither needed nor resident per
decode access (extra SSD reads) and predictor top-K precision.

usage: sim_sota.py <trace.npz> <frac> <flashmoe-weights> <duoserve.npz> <held-out.npz>...
"""
import sys, json, numpy as np
d = np.load(sys.argv[1], allow_pickle=True); frac = float(sys.argv[2]); wfile = sys.argv[3]
pfile = sys.argv[4]; held = sys.argv[5:]
L = int(d["n_layers"]); E = int(d["n_experts"])
N = L * E; C = max(1, int(frac * N)); K = int(d["topk"]); layer_of = np.arange(N) // E


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
rng = np.random.default_rng(1)


# ---- global keep-top-C policies: same rules as sim_ablate / sim_sweep --------
def run_global(kind):
    access = kind == "lru"; hist_all = kind == "lfu"
    res = np.zeros(N, bool); last = np.full(N, -1e9); hist = np.zeros(N); now = 0.0
    hit = np.zeros(L); tot = np.zeros(L); jit = rng.random(N) * 1e-9
    def stamp(ids): last[ids] = now + layer_of[ids] / L if access else now
    def keep(cand, pfn):
        idx = np.flatnonzero(cand)
        if len(idx) <= C: r = np.zeros(N, bool); r[idx] = True; return r
        if kind == "lru": s = last.copy()
        elif kind == "lfu": s = hist.copy()
        else: s = np.exp2(-(now - last) / 8) + 0.5 * pfn + 0.5 * hist / max(hist.max(), 1e-12)
        s = s + (jit if access else rng.random(N) * 1e-9)
        top = idx[np.argpartition(-s[idx], C - 1)[:C]]
        r = np.zeros(N, bool); r[top] = True; return r
    for (u, pf, toks) in R2:
        now += 1; pfn = pf / max(pf.max(), 1); ids = np.flatnonzero(u); stamp(ids)
        if hist_all: hist[ids] += pf[ids]
        res = keep(res | u, pfn)
        for t, _ in toks:
            now += 1; lt = t // E
            np.add.at(tot, lt, 1); np.add.at(hit, lt, res[t])
            stamp(t); hist[t] += 1
            nd = np.zeros(N, bool); nd[t] = True; res = keep(res | nd, pfn)
    return hit, tot, {}


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
            ts = set(t.tolist())
            for idx in t:
                l = idx // E; tot[l] += 1
                if idx in cache[l]: hit[l] += 1; continue
                if len(cache[l]) < slots: cache[l].add(idx); continue
                cand = [c for c in cache[l] if c not in ts]
                if not cand: continue
                fm = max(freq[l*E:(l+1)*E].max(), 1)
                X = np.array([[1.0 / (step - last[c] + 1), freq[c] / fm] for c in cand])
                v = cand[int(np.argmax(ffn(X)))]
                cache[l].discard(v); cache[l].add(idx)
    return hit, tot, {"slots_per_layer": slots}


# ---- DuoServe* ---------------------------------------------------------------
pop = np.zeros((L, E)); aff = np.zeros((L - 1, E, E))   # from the held-out decode tokens, as in training
for hp in held:
    for (_, _, toks) in load(hp):
        for _, sel in toks:
            for l in range(L):
                for e in sel[l]:
                    if e >= 0: pop[l, e] += 1
                if l:
                    for a in sel[l - 1]:
                        for e in sel[l]:
                            if a >= 0 and e >= 0: aff[l - 1, a, e] += 1
P = np.load(pfile); nP = len([k for k in P.files if k.startswith("W")])
def mlp(x):
    for i in range(nP):
        x = x @ P[f"W{i}"].T + P[f"b{i}"]
        if i < nP - 1: x = np.maximum(x, 0)
    return x
def run_duoserve(learned):
    res = np.zeros(N, bool); last = np.full(N, -1e9); now = 0.0; jit = rng.random(N) * 1e-9
    hit = np.zeros(L); tot = np.zeros(L); extra = 0; phit = 0; ptot = 0
    def admit(ids, t):
        nonlocal res
        last[ids] = t
        res[ids] = True
        over = int(res.sum()) - C
        if over > 0:
            idx = np.flatnonzero(res)
            res[idx[np.argpartition(last[idx] + jit[idx], over - 1)[:over]]] = False
    for (u, pf, toks) in R2:
        now += 1; ids = np.flatnonzero(u)
        res_ids = ids[np.argsort(layer_of[ids], kind="stable")]
        admit(res_ids, now + layer_of[res_ids] / L)
        for t, sel in toks:
            now += 1; chosen = np.zeros(E); pred = np.array([], int)
            for l in range(L):
                need = l * E + sel[l][sel[l] >= 0]
                ok = res[need] | np.isin(need, pred)
                tot[l] += len(need); hit[l] += ok.sum()
                if len(pred): phit += np.isin(pred, need).sum(); ptot += len(pred)
                admit(need, now + l / L)
                if l + 1 >= L: break
                chosen[sel[l][sel[l] >= 0]] += 1
                a = aff[l][sel[l][sel[l] >= 0]].sum(0)
                if learned:
                    x = np.concatenate([chosen / max(chosen.max(), 1), pop[l + 1] / max(pop[l + 1].max(), 1),
                                        a / max(a.max(), 1)])
                    s = mlp(x[None])[0]
                else:
                    s = (pop[l + 1] + 1e-9) * (a + 1e-9)
                pred = (l + 1) * E + np.argsort(-s)[:K]
                new = pred[~res[pred]]
                nxt = set(((l + 1) * E + sel[l + 1][sel[l + 1] >= 0]).tolist())
                extra += sum(1 for p in new if p not in nxt)
                admit(new, now + l / L + 0.5 / L)       # copied during layer l, before layer l+1
    return hit, tot, {"extra_fetch_per_access": round(extra / tot.sum(), 4),
                      "pred_precision": round(phit / max(ptot, 1), 4)}


out = {"trace": sys.argv[1], "frac": frac, "C": C, "L": L, "flashmoe_weights": wfile, "duoserve_predictor": pfile}
for name, fn in (("LRU", lambda: run_global("lru")), ("LFU", lambda: run_global("lfu")),
                 ("PHASOR", lambda: run_global("phasor")), ("FlashMoE", run_flashmoe),
                 ("DuoServe", lambda: run_duoserve(True)), ("DuoServe_popaff", lambda: run_duoserve(False))):
    h, t, x = fn()
    out[name] = {"hit": round(float(h.sum() / t.sum()), 4),
                 "miss_by_layer": [round(float(1 - a / max(b, 1)), 4) for a, b in zip(h, t)], **x}
    print(name, out[name]["hit"], x, file=sys.stderr, flush=True)
print(json.dumps(out))
