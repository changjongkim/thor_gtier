#!/bin/bash
# Paper 4.1: is our ZipMoE setup faithful?  On its own paper's model
# (Qwen1.5-MoE-A2.7B-Chat) and its own harness (evaluation/evaluate.py, 24
# ShareGPT prompts, 64 new tokens, 512-token prompts, 30 s cooling per prompt):
#   (a) its caching (ZipMoE) against its LRU and LFU options at 10/20/30 GB
#       footprints (the presets scaled from the 64 GB Orin to this device) --
#       the paper's claim that its caching beats LRU/LFU;
#   (b) our runner (scripts/zipmoe_serve.py) on the same prompts and memory at
#       20 GB -- equal latency means our harness does not handicap ZipMoE.
# Sourced by stage 6 under the pipeline lock (uses its say/drop).
F=$R/results/ZIPMOE_FIDELITY; mkdir -p $F
if [ ! -f $F/DONE ]; then
  say "=== ZipMoE fidelity (Qwen1.5-MoE-A2.7B-Chat) start ==="
  $ZPY - <<'PY' >> $F/download.log 2>&1
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen1.5-MoE-A2.7B-Chat", local_dir="/home/thor/kcj/ZipMoE/models/qwen",
                  allow_patterns=["*.safetensors", "*.json", "*.txt", "merges.txt", "vocab.json"])
print("downloaded")
PY
  grep -q downloaded $F/download.log || { say "ZipMoE fidelity: download FAILED"; }
  export ZIPMOE_MEM_SCALE=$(python3 -c "print(64/122.8)")
  for fp in 20 10 30; do for alg in ZipMoE LRU LFU; do
    [ -s $F/official_M${fp}_$alg.log ] && grep -q "Save Success" $F/official_M${fp}_$alg.log && continue
    drop
    (cd /home/thor/kcj/ZipMoE && PYTHONPATH=$(pwd) timeout 10800 $R/scripts/in_cgroup.sh fid max $ZPY evaluation/evaluate.py \
      --model_type qwen --memory_footprint $fp --cache_algorithm $alg --num_test_prompts 24 --output_dir $F/official/ ) \
      > $F/official_M${fp}_$alg.log 2>&1
    say "ZipMoE fidelity: official M$fp $alg $(grep -c 'Save Success' $F/official_M${fp}_$alg.log) prompts saved"
  done; done
  # (b) our runner on the same 24 prompts at the 20 GB footprint
  $ZPY - <<'PY'
import json, sys
sys.path.insert(0, "/home/thor/kcj/ZipMoE/evaluation")
from profile_tools import sample_first_prompts
ps = sample_first_prompts("/home/thor/kcj/ZipMoE/evaluation/dataset/sharegpt_gpt4.jsonl", num_samples=24, seed=321, max_candidates=500)
json.dump([{"name": f"sharegpt_zipmoe_{i}", "prompt": p, "max_new": 64} for i, p in enumerate(ps)],
          open("/home/thor/kcj/thor_gtier/results/ZIPMOE_FIDELITY/prompts24.json", "w"))
PY
  total=$($ZPY -c "import torch;print(torch.cuda.get_device_properties(0).total_memory/2**30)")
  b20=$(python3 -c "print(round(0.25*64/122.8*$total, 2))")
  drop
  timeout 10800 scripts/in_cgroup.sh fid max $ZPY scripts/zipmoe_serve.py --model-type qwen --workload $F/prompts24.json \
    --budget-gib $b20 --trace /home/thor/kcj/ZipMoE/trace/qwen_trace.pt --max-prompt 512 --max-new 64 --out $F/ours_M20.json > $F/ours_M20.log 2>&1
  say "ZipMoE fidelity: our runner M20 $(grep -o 'request_s=[0-9.]*' $F/ours_M20.log)"
  python3 scripts/zipmoe_fidelity_table.py > $F/FIDELITY.md 2>>$LOG
  touch $F/DONE
  git add -f $F/*.md $F/*.log $F/*.json $F/official 2>/dev/null
  git commit -q -m "ZipMoE fidelity on Qwen1.5-MoE-A2.7B (its own model and harness)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
  rm -rf /home/thor/kcj/ZipMoE-ICML26/offload/qwen
  say "=== ZipMoE fidelity done ==="
fi
