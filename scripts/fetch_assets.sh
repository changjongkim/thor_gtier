#!/bin/bash
# Models and datasets for the experiment plan.
#
# Mixtral is here for generality: 8 experts with top-2 is a different routing
# shape from Qwen3's 128 with top-8, so it decides whether the per-layer skew
# and the prefill union saturation are structural or particular to one family.
# It is also what Mixtral-offloading and MoE-Infinity report against, so the
# numbers can be compared to published ones.
#
# The datasets each carry one claim.  ShareGPT is the serving workload vLLM,
# SGLang and DistServe all measure on, and its multi-turn conversations make
# prefix reuse arise naturally rather than being staged.  LongBench is the one
# that matters most: prompts average thousands of tokens where the current
# trace tops out at 613, and the claim that prefill dominates has to hold
# there or be withdrawn.  MMLU covers 57 subjects, which is the diversity the
# current single-domain trace lacks.
set -u
R=/home/thor/kcj/thor_gtier
D=/home/thor/kcj/datasets
M=/home/thor/kcj/models
. "$R/scripts/torch_env.sh"
mkdir -p "$D" "$M"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

say "Mixtral-8x7B-Instruct Q4_K_M"
$TORCH_VENV/bin/python - <<'PY'
from huggingface_hub import snapshot_download
import traceback
try:
    p = snapshot_download("TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
                          allow_patterns=["*Q4_K_M*.gguf"],
                          local_dir="/home/thor/kcj/models/mixtral8x7b_q4km",
                          max_workers=4)
    print("DONE", p, flush=True)
except Exception:
    traceback.print_exc()
PY

say "Mixtral-8x7B-Instruct bf16 (라우팅 캡처용 - 큼, 별도 판단)"
# bf16 is ~87 GiB and only needed to hook the router.  Left out by default;
# the GGUF gives the tensor layout and the routing can be captured from a
# smaller sibling if space is short.

say "datasets"
$TORCH_VENV/bin/python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
import traceback, os
D = "/home/thor/kcj/datasets"
jobs = [
    ("ShareGPT", dict(repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
                      filename="ShareGPT_V3_unfiltered_cleaned_split.json",
                      repo_type="dataset", local_dir=f"{D}/sharegpt")),
]
for name, kw in jobs:
    try:
        p = hf_hub_download(**kw); print("DONE", name, p, flush=True)
    except Exception:
        traceback.print_exc()
for name, repo in (("LongBench","THUDM/LongBench"), ("MMLU","cais/mmlu")):
    try:
        p = snapshot_download(repo_id=repo, repo_type="dataset",
                              local_dir=f"{D}/{name.lower()}", max_workers=4)
        print("DONE", name, p, flush=True)
    except Exception:
        traceback.print_exc()
PY
say "sizes"
du -sh "$M"/mixtral8x7b_q4km "$D"/* 2>/dev/null
