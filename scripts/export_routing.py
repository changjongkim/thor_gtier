#!/usr/bin/env python3
"""Export the captured routing into a compact binary the C++ driver reads.

The serving driver has to replay exactly the experts the model chose, per
request and per phase, or the residency policies are being compared against
a distribution nobody uses.  npz is convenient for analysis and awkward from
C++, so this writes a flat file instead.

Layout (all little-endian):
  u32 magic 'GRT1' | u32 n_layers | u32 n_experts | u32 topk | u32 n_requests
  per request: u32 name_len, bytes name, u32 n_prefill, u32 n_decode,
               i16 prefill[n_prefill][n_layers][topk],
               i16 decode [n_decode ][n_layers][topk]
"""
import argparse, struct
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="results/SCOPE/routing.npz")
ap.add_argument("--out", default="results/SCOPE/routing.bin")
a = ap.parse_args()

d = np.load(a.npz, allow_pickle=True)
tags = [str(t) for t in d["tags"]]
E, K, L = int(d["n_experts"]), int(d["topk"]), int(d["n_layers"])
tag, lay, pos, exp = d["tag"], d["layer"], d["pos"], d["expert"]

names = sorted({t.split("/")[0] for t in tags})
out = open(a.out, "wb")
out.write(struct.pack("<4sIIII", b"GRT1", L, E, K, len(names)))

for nm in names:
    blocks = {}
    for phase in ("prefill", "decode"):
        ti = tags.index(f"{nm}/{phase}")
        m = tag == ti
        n = int(pos[m].max()) + 1
        # [token][layer][topk]; a token that never reached a layer keeps -1
        arr = np.full((n, L, K), -1, dtype=np.int16)
        arr[pos[m], lay[m]] = exp[m]
        blocks[phase] = arr
    nb = nm.encode()
    out.write(struct.pack("<I", len(nb))); out.write(nb)
    out.write(struct.pack("<II", blocks["prefill"].shape[0], blocks["decode"].shape[0]))
    out.write(blocks["prefill"].tobytes())
    out.write(blocks["decode"].tobytes())
    print(f"{nm}: prefill {blocks['prefill'].shape[0]} decode {blocks['decode'].shape[0]}")
out.close()
import os
print(f"wrote {a.out} ({os.path.getsize(a.out)/2**20:.1f} MiB)")
