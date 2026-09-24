#!/usr/bin/env python3
"""Bytes a decode token touches in a GGUF model: every non-expert tensor plus
K of E slices of each stacked expert tensor.  Used to carry a measured
per-token compute cost from one model to another of the same family."""
import sys
sys.path.insert(0, "/home/thor/skim/llama.cpp/gguf-py")
from gguf import GGUFReader

def active(paths, k, e):
    dense = exps = 0
    for p in paths:
        for t in GGUFReader(p).tensors:
            if "_exps" in t.name: exps += int(t.n_bytes)
            else: dense += int(t.n_bytes)
    return dense + exps * k / e, dense, exps

if __name__ == "__main__":
    k, e = int(sys.argv[1]), int(sys.argv[2])
    a, d, x = active(sys.argv[3:], k, e)
    print(f"{a:.0f} {d} {x}")
