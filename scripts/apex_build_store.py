#!/usr/bin/env python3
"""Build MoE-APEX*'s int4 expert store for a checkpoint (usage: apex_build_store.py CKPT [OUT])."""
import os, sys, torch
sys.path.insert(0, "/home/thor/kcj/thor_gtier/baselines_hf")
from transformers import AutoConfig
from offload_hf import ExpertReader
from apex_hf import build_int4_store
ck = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else ck.rstrip("/") + "_int4"
cfg = AutoConfig.from_pretrained(ck); arch = cfg.architectures[0].lower()
if "qwen3moe" in arch: L, E, attr, names = cfg.num_hidden_layers, cfg.num_experts, "mlp", ("gate_proj", "up_proj", "down_proj")
else: L, E, attr, names = cfg.num_hidden_layers, cfg.num_local_experts, "block_sparse_moe", ("w1", "w3", "w2")
p = build_int4_store(ExpertReader(ck, attr, names, L, E), out, L, E)
print("wrote", p, os.path.getsize(p) / 2**30, "GiB")
