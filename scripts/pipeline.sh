#!/bin/bash
# The whole experiment pipeline, one step at a time.
#
# Two things repeatedly cost a night's work: steps that overlapped and fought
# over memory or the device, and one configuration that took the host down.
# So every step here runs alone, drops the page cache first, and passes
# through the memory guard; the pipeline is a list of steps with a marker per
# step, so a reboot loses only the step that was running.
set -u
R=/home/thor/kcj/thor_gtier
cd "$R"
. "$R/scripts/torch_env.sh"
. "$R/scripts/memguard.sh"
ST=$R/results/PIPELINE; mkdir -p "$ST"
LOG=$ST/pipeline.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }

# step <name> <need-GiB> -- <command...>
step(){
  local name=$1 need=$2; shift 2; [ "${1:-}" = "--" ] && shift
  if [ -f "$ST/$name.done" ]; then say "skip $name"; return 0; fi
  if ! mg_check "$need" >>"$LOG" 2>&1; then
    say "REFUSED $name (needs $need GiB)"; echo refused > "$ST/$name.refused"; return 0
  fi
  say "run  $name"
  drop
  if "$@" >> "$ST/$name.out" 2>&1; then
    touch "$ST/$name.done"; say "  ok $name"
  else
    say "  FAILED $name (rc=$?)"
  fi
}

# Only one pipeline at a time, and nothing else heavy alongside it.
exec 9>/tmp/gtier_pipeline.lock
flock -n 9 || { echo "pipeline already running"; exit 0; }
echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== pipeline start ==="

# ---- 1. routing captures: every model over every workload ----------------
for spec in "qwen30b:/home/thor/kcj/models/qwen3_30b_a3b:65" \
            "mixtral8x7b:/home/thor/kcj/models/mixtral8x7b_bf16:95"; do
  mn=${spec%%:*}; rest=${spec#*:}; mp=${rest%%:*}; msz=${rest##*:}
  [ -d "$mp" ] || { say "skip $mn (absent)"; continue; }
  for w in longbench sharegpt mmlu; do
    step "cap_${mn}_${w}" "$msz" -- \
      $TORCH_VENV/bin/python scripts/capture_workload.py \
        --model "$mp" --workload "$R/results/WORKLOADS/$w.json" \
        --out "$R/results/SCOPE/rt_${mn}_${w}.npz"
  done
done

# ---- 2. export the traces the serving driver reads ------------------------
for f in "$R"/results/SCOPE/rt_*.npz; do
  [ -s "$f" ] || continue
  b=$(basename "$f" .npz)
  step "exp_$b" 4 -- $TORCH_VENV/bin/python scripts/export_routing.py \
      --npz "$f" --out "$R/results/SCOPE/$b.bin"
done

say "=== pipeline stage 1 done ==="
cd "$R"; git add -A results >/dev/null 2>&1
git diff --cached --quiet || { git commit -q -m "Pipeline: routing captures

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q; }
