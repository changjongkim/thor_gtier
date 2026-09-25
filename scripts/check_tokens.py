#!/usr/bin/env python3
"""Greedy tokens of a system must equal stock transformers on the same prompts.
usage: check_tokens.py <system> <ckpt> <model-type> <budget-gib> <out.json>
system: stock | phasor | zipmoe"""
import sys, json, time, os
system, ck, mt, budget, out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
out = os.path.abspath(out); ck = os.path.abspath(ck)   # the ZipMoE path chdirs into its repo
prompts = ["Explain why the sky is blue in two sentences.",
           "Write a Python function that returns the n-th Fibonacci number."]
import torch
if system == "stock":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ck)
    model = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
elif system == "phasor":
    sys.path.insert(0, "/home/thor/kcj/thor_gtier/phasor_hf"); import phasor_hf
    slot = 128 if mt == "mixtral" else 4; win = 1.5 if mt == "mixtral" else 0.5
    model, tok, eng = phasor_hf.build(ck, budget, window_gib=win, slot_mib=slot)
else:
    sys.path.insert(0, "/home/thor/kcj/ZipMoE"); os.chdir("/home/thor/kcj/ZipMoE")
    from entry.llm_modeling import MoE
    from transformers import AutoTokenizer
    from utils.constants import (List_expert_topk, List_num_elements_per_expert, List_num_tensors_per_expert,
                                 List_num_expert_layers, List_num_experts, List_first_k_dense_replace)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    cfg = {"offload_path": f"/home/thor/kcj/ZipMoE/offload/{mt}/", "caching_algorithm": "ZipMoE", "prefetcher_topk": 4,
           "device_memory_ratio": min(0.95, budget / total), "gpu_pool_ratio": 0.95, "batch_size": 1,
           "code_type": "LZ4HC", "hyperparam_state_margin": 0.1, "num_file_chunks": 3, "num_compute_threads": 6,
           "trace_path": f"/home/thor/kcj/ZipMoE/trace/{mt}_mmlu_heldout.pt",
           "expert_topk": List_expert_topk[mt], "num_elements_per_expert": List_num_elements_per_expert[mt],
           "num_tensors_per_expert": List_num_tensors_per_expert[mt], "num_expert_layers": List_num_expert_layers[mt],
           "num_experts": List_num_experts[mt], "first_k_dense_replace": List_first_k_dense_replace[mt]}
    model = MoE(f"/home/thor/kcj/ZipMoE/models/{mt}/", cfg); tok = AutoTokenizer.from_pretrained(ck)
res = {}
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids.cuda()
    if system == "phasor": eng.begin_request()
    with torch.no_grad():
        o = model.generate(ids, max_new_tokens=24, do_sample=False, attention_mask=torch.ones_like(ids),
                           pad_token_id=tok.eos_token_id)
    res[p] = o[0, ids.shape[1]:].tolist()
    print(system, repr(tok.decode(res[p])))
json.dump(res, open(out, "w"))
# ZipMoE's worker threads do not exit after its task pool is destroyed, which
# leaves the process running until the timeout; the results are written, so leave.
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
