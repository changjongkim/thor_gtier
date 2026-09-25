#!/usr/bin/env python3
"""Slots per layer FlashMoE gets at a budget: what the driver computes as
R / unit / L, with R = budget - dense - window - path overhead."""
import sys
sys.path.insert(0, "/home/thor/skim/llama.cpp/gguf-py")
from gguf import GGUFReader
budget, window, po, L, E = map(float, sys.argv[1:6])
dense = exps = 0
for p in sys.argv[6:]:
    for t in GGUFReader(p).tensors:
        if "_exps" in t.name: exps += int(t.n_bytes)
        else: dense += int(t.n_bytes)
G = 1 << 30
R = budget * G - dense - window * G - po * G
print(max(1, int(R // (exps / (L * E)) // L)))
