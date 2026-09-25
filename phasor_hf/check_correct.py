"""Greedy tokens from PHASOR must equal those of the stock model."""
import sys, time, json, torch
sys.path.insert(0, "/home/thor/kcj/thor_gtier/phasor_hf")
ck = sys.argv[1]; budget = float(sys.argv[2]); mode = sys.argv[3]
prompts = ["Explain why the sky is blue in two sentences.",
           "Write a Python function that returns the n-th Fibonacci number."]
out = {}
if mode == "phasor":
    import phasor_hf
    t0 = time.time(); model, tok, eng = phasor_hf.build(ck, budget); print("load", time.time() - t0, "s, arena units", eng.arena_units())
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ck)
    model = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids.cuda()
    if mode == "phasor": eng.begin_request()
    t0 = time.time()
    with torch.no_grad():
        o = model.generate(ids, max_new_tokens=24, do_sample=False, attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    out[p] = o[0, ids.shape[1]:].tolist()
    print(f"{mode} {dt:.2f}s :: {tok.decode(out[p])!r}")
if mode == "phasor": print("stats hits,misses,bytes,resident,slots,unit_bytes", eng.stats())
json.dump(out, open(f"/home/thor/kcj/thor_gtier/results/PHASOR_HF/correct_{mode}.json", "w"))
