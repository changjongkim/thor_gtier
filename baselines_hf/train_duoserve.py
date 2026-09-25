#!/usr/bin/env python3
"""Train DuoServe-MoE's decode-stage predictor (Zhang et al., arXiv 2509.07379):
an MLP that, after layer l, predicts the experts of layer l+1 from the experts
chosen so far, the popularity of layer l+1 and the affinity from layer l's
choices; multi-label, binary cross-entropy.  Trained on the model's other
workloads, so the evaluated requests are never seen.

usage: train_duoserve.py out.pt held-out.npz ...
"""
import sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from offload_hf import DuoPredictor

out = sys.argv[1]
tokens = []          # per decode token: array [L, K] of experts
L = E = K = None
for p in sys.argv[2:]:
    d = np.load(p, allow_pickle=True); tags = list(d["tags"])
    L, E, K = int(d["n_layers"]), int(d["n_experts"]), int(d["topk"])
    for i, t in enumerate(tags):
        if not t.endswith("/decode"): continue
        m = d["tag"] == i
        lay, ex, pos = d["layer"][m].astype(int), d["expert"][m].astype(int), d["pos"][m]
        for p_ in np.unique(pos):
            mm = pos == p_
            sel = np.full((L, K), -1)
            for l, row in zip(lay[mm], ex[mm]): sel[l, :len(row)] = row
            tokens.append(sel)
pop = np.zeros((L, E)); aff = np.zeros((L - 1, E, E))
for sel in tokens:
    for l in range(L):
        for e in sel[l]:
            if e >= 0: pop[l, e] += 1
        if l:
            for a in sel[l - 1]:
                for e in sel[l]:
                    if a >= 0 and e >= 0: aff[l - 1, a, e] += 1
X, Y = [], []
for sel in tokens:
    chosen = np.zeros(E)
    for l in range(L - 1):
        for e in sel[l]:
            if e >= 0: chosen[e] += 1
        a = aff[l][[e for e in sel[l] if e >= 0]].sum(0)
        x = np.concatenate([chosen / max(chosen.max(), 1), pop[l + 1] / max(pop[l + 1].max(), 1), a / max(a.max(), 1)])
        y = np.zeros(E); y[[e for e in sel[l + 1] if e >= 0]] = 1
        X.append(x); Y.append(y)
X = torch.tensor(np.array(X), dtype=torch.float32); Y = torch.tensor(np.array(Y), dtype=torch.float32)
# hold out 10% to check the predictor against the popularity x affinity score it
# is given as input: a trained predictor must at least match its own inputs
perm0 = torch.randperm(len(X), generator=torch.Generator().manual_seed(1))
nv = len(X) // 10; Xv, Yv = X[perm0[:nv]], Y[perm0[:nv]]; X, Y = X[perm0[nv:]], Y[perm0[nv:]]
print(f"samples {len(X)}  E={E} K={K}", file=sys.stderr)
torch.manual_seed(0)
net = DuoPredictor(E, hidden=512)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
# k of E labels are positive; weight them so the loss does not settle on "none"
lossf = nn.BCEWithLogitsLoss(pos_weight=torch.full((E,), (E - K) / K))
def precision(Xs, Ys):
    with torch.no_grad():
        top = torch.topk(net(Xs), K, dim=1).indices
        return (Ys.gather(1, top).sum() / (K * len(Xs))).item()
def base_precision(Xs, Ys):
    s = Xs[:, E:2 * E] * (Xs[:, 2 * E:] + 1e-9)
    top = torch.topk(s, K, dim=1).indices
    return (Ys.gather(1, top).sum() / (K * len(Xs))).item()
for ep in range(60):
    perm = torch.randperm(len(X)); tot = 0
    for i in range(0, len(X), 1024):
        idx = perm[i:i + 1024]
        opt.zero_grad(); loss = lossf(net(X[idx]), Y[idx]); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    sched.step()
    if ep % 20 == 19:
        print(f"epoch {ep+1} loss {tot/len(X):.4f} held-out top-{K} precision {precision(Xv, Yv):.3f} "
              f"(popularity x affinity {base_precision(Xv, Yv):.3f})", file=sys.stderr)
torch.save({"state": net.state_dict(), "hidden": 512}, out)
print(f"wrote {out}", file=sys.stderr)
