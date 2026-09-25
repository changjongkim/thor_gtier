#!/usr/bin/env python3
"""Train FlashMoE's cache-replacement network (Kim et al., arXiv 2601.17063)
so it can run as a baseline in the serving driver.

Reproduced from the paper, since no code is released:
  * per-layer cache with a fixed number of expert slots per layer
  * features per cached expert: recency 1/r_t and frequency f_t / max f,
    with r_t = 1 on access else r_{t-1} + 1, f_t = f_{t-1} + 1 on access
  * a 3-layer FFN, hidden 128, SiLU, MSE loss, AdamW (lr 1e-3, wd 1e-2)
  * targets from Belady's algorithm: 1 for the expert Belady evicts, 0 for
    the other candidates at that eviction ("masked target")
The paper trains on TriviaQA routing; here the training traces are the same
model's other workloads, so the evaluated requests are never seen.

usage: train_flashmoe.py <out.txt> <slots-per-layer> <train.npz>...
"""
import sys
import numpy as np
import torch
import torch.nn as nn

out, slots = sys.argv[1], int(sys.argv[2])
paths = sys.argv[3:]

def decode_steps(path):
    """Per request, the decode steps as a list of per-layer expert sets."""
    d = np.load(path, allow_pickle=True)
    tags = list(d["tags"]); L = int(d["n_layers"])
    tag, lay, pos, ex = d["tag"], d["layer"].astype(int), d["pos"], d["expert"].astype(int)
    seq = []
    for i, t in enumerate(tags):
        if not t.endswith("/decode"): continue
        m = tag == i
        ps, ls, es = pos[m], lay[m], ex[m]
        for p in np.unique(ps):
            mm = ps == p
            step = [set() for _ in range(L)]
            for l, row in zip(ls[mm], es[mm]):
                step[l].update(int(e) for e in row if e >= 0)
            seq.append(step)
    return seq, L, int(d["n_experts"])

X, Y = [], []
for p in paths:
    seq, L, E = decode_steps(p)
    T = len(seq)
    for l in range(L):
        acc = [seq[t][l] for t in range(T)]
        # next use of each expert after step t
        nxt = [None] * T; nu = {}
        for t in range(T - 1, -1, -1):
            nxt[t] = dict(nu)
            for e in acc[t]: nu[e] = t
        cache = set(); r = np.full(E, 1e9); f = np.zeros(E)
        for t in range(T):
            for e in range(E): r[e] = r[e] + 1
            for e in acc[t]: r[e] = 1; f[e] += 1
            for e in acc[t]:
                if e in cache: continue
                if len(cache) < slots: cache.add(e); continue
                cand = [c for c in cache if c not in acc[t]]
                if not cand: continue
                far = max(cand, key=lambda c: nxt[t].get(c, 10**9))
                fm = max(f.max(), 1.0)
                for c in cand:
                    X.append([1.0 / r[c], f[c] / fm]); Y.append(1.0 if c == far else 0.0)
                cache.discard(far); cache.add(e)
if not X:
    # Every expert fits its layer's slots: nothing is ever evicted, and the
    # network is never consulted.  Write a zero network so the run can start.
    with open(out, "w") as fo:
        fo.write("1\n1 2\n0 0\n0\n")
    print("no evictions at this size; wrote a zero network", file=sys.stderr)
    sys.exit(0)
X = torch.tensor(np.array(X), dtype=torch.float32)
Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(1)
print(f"samples {len(X)}  positive {Y.mean().item():.3f}", file=sys.stderr)

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
lossf = nn.MSELoss()
n = len(X); bs = 4096
for epoch in range(20):
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, bs):
        idx = perm[i:i + bs]
        opt.zero_grad(); loss = lossf(net(X[idx]), Y[idx]); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    if epoch % 5 == 4: print(f"epoch {epoch+1} mse {tot/n:.5f}", file=sys.stderr)

# Text weights: each linear layer as "rows cols" then W row-major then b.
with open(out, "w") as fo:
    lin = [m for m in net if isinstance(m, nn.Linear)]
    fo.write(f"{len(lin)}\n")
    for m in lin:
        W = m.weight.detach().numpy(); b = m.bias.detach().numpy()
        fo.write(f"{W.shape[0]} {W.shape[1]}\n")
        fo.write(" ".join(f"{v:.7g}" for v in W.ravel()) + "\n")
        fo.write(" ".join(f"{v:.7g}" for v in b) + "\n")
print(f"wrote {out}", file=sys.stderr)
