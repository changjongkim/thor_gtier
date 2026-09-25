#!/usr/bin/env python3
"""ZipMoE's planning trace (list[prompt][layer][expert] -> decode activation
count) from our routing captures of the other workloads, so the evaluated
requests are never seen.  usage: zipmoe_trace.py out.pt held-out.npz..."""
import sys, numpy as np, torch
out = []
for p in sys.argv[2:]:
    d = np.load(p, allow_pickle=True); tags = list(d["tags"])
    L, E = int(d["n_layers"]), int(d["n_experts"])
    for i, t in enumerate(tags):
        if not t.endswith("/decode"): continue
        m = d["tag"] == i
        cnt = np.zeros((L, E), int)
        for l, row in zip(d["layer"][m].astype(int), d["expert"][m].astype(int)):
            for e in row:
                if e >= 0: cnt[l, e] += 1
        out.append(cnt.tolist())
torch.save(out, sys.argv[1]); print(f"{len(out)} prompts -> {sys.argv[1]}")
