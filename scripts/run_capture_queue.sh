#!/bin/bash
# Routing captures: every model over every workload, same prompts.
#
# One capture per (model, workload).  The prompts are identical across models,
# so a difference in what comes out is a difference between the models rather
# than between the requests they were given.
set -u
R=/home/thor/kcj/thor_gtier
cd "$R"
. "$R/scripts/torch_env.sh"
. "$R/scripts/memguard.sh"
OUT=$R/results/SCOPE; LOG=$OUT/capture.log
mkdir -p "$OUT"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# name:path:approx-bf16-GiB
MODELS=${MODELS:-"qwen30b:/home/thor/kcj/models/qwen3_30b_a3b:57 mixtral8x7b:/home/thor/kcj/models/mixtral8x7b_bf16:87"}

for spec in $MODELS; do
  mname=${spec%%:*}; rest=${spec#*:}; mpath=${rest%%:*}; msize=${rest##*:}
  [ -d "$mpath" ] || { say "skip $mname (no model)"; continue; }
  # Loading the checkpoint makes it resident; refuse rather than take the
  # host down.
  mg_check "$(awk -v s="$msize" 'BEGIN{printf "%.1f", s+8}')" >/dev/null 2>&1 || {
    say "skip $mname -- memguard refused ($msize GiB + overhead)"; continue; }
  for w in longbench sharegpt mmlu; do
    o=$OUT/rt_${mname}_${w}.npz
    [ -s "$o" ] && { say "skip $mname/$w"; continue; }
    say "capture $mname / $w"
    timeout 14400 $TORCH_VENV/bin/python scripts/capture_workload.py \
      --model "$mpath" --workload "$R/results/WORKLOADS/$w.json" --out "$o" \
      >> "$OUT/capture_${mname}_${w}.out" 2>&1 \
      && say "  ok $(du -h "$o"|cut -f1)" || say "  FAILED $mname/$w"
  done
done
say "=== captures done ==="
cd "$R"; git add -A results/SCOPE results/WORKLOADS >/dev/null 2>&1
git diff --cached --quiet || { git commit -q -m "Routing captures over the three workloads

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q; }
