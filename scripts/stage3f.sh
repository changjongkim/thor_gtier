#!/bin/bash
# Stage 3f: the real SOTA systems on Qwen3-30B bf16, same prompts, same cap.
#   ZipMoE (ICML'26), ported to Qwen3-MoE (shared expert removed, sm_110 build)
#   MoE-Infinity, its own engine
# Budgets are fractions of the bf16 checkpoint (57 GiB).  The first ZipMoE
# start compresses the checkpoint into its offload store; that is a step of
# its own so a failure there stops the rest.
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. "$R/scripts/torch_env.sh"; . "$R/scripts/memguard.sh"
ST=$R/results/PIPELINE; OUT=$R/results/MATRIX3/qwen30b_bf16; mkdir -p "$OUT"
LOG=$ST/pipeline.log; ZPY=/home/thor/kcj/envs/zipmoe/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
step(){
  local name=$1 need=$2; shift 2; [ "${1:-}" = "--" ] && shift
  [ -f "$ST/$name.done" ] && return 0
  local fails=$(cat "$ST/$name.fail" 2>/dev/null || echo 0)
  if [ "$fails" -ge 2 ]; then say "give up $name (failed $fails times)"; return 1; fi
  if ! mg_check "$need" >>"$LOG" 2>&1; then say "REFUSED $name (needs $need GiB)"; return 1; fi
  say "run  $name"; drop
  if "$@" >> "$ST/$name.out" 2>&1; then touch "$ST/$name.done"; say "  ok $name"; return 0
  else local rc=$?; echo $((fails+1)) > "$ST/$name.fail"; say "  FAILED $name (rc=$rc)"; return 1; fi
}
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== stage 3f (real SOTA, bf16) start ==="
BF=57.0
capb(){ awk -v b="$1" 'BEGIN{printf "%.0f", (b+0.5)*1073741824}'; }
bud(){ awk -v f="$1" 'BEGIN{printf "%.2f", 57.0*f}'; }
# 1. ZipMoE smoke: offload (one-time compression) + 2 prompts at the middle budget
if step zip_smoke 40 -- timeout 7200 $ZPY "$R/scripts/zipmoe_serve.py" --workload "$R/results/WORKLOADS/mmlu.json" \
     --budget-gib $(bud 0.45) --trace /home/thor/kcj/ZipMoE/trace/qwen3_mmlu_heldout.pt --limit 2 \
     --out "$OUT/zip_smoke.json"; then
  for w in longbench sharegpt mmlu; do mkdir -p "$OUT/$w"; for f in 0.25 0.45 0.65; do
    b=$(bud $f)
    step "z_${w}_$f" $(awk -v b=$b 'BEGIN{printf "%.0f", b+8}') -- bash -c "
      timeout 14400 '$R/scripts/in_cgroup.sh' m3 $(capb $b) $ZPY '$R/scripts/zipmoe_serve.py' \
        --workload '$R/results/WORKLOADS/$w.json' --budget-gib $b \
        --trace /home/thor/kcj/ZipMoE/trace/qwen3_${w}_heldout.pt --out '$OUT/$w/zipmoe_$f.json' \
        > '$OUT/$w/zipmoe_$f.txt' 2>&1; grep -q '^RESULT' '$OUT/$w/zipmoe_$f.txt'"
  done; done
fi
# 2. MoE-Infinity, its own engine (converts the checkpoint once into its store)
mkdir -p /home/thor/kcj/offload_tmp/qwen30b
for w in longbench sharegpt mmlu; do mkdir -p "$OUT/$w"; for f in 0.25 0.45 0.65; do
  b=$(bud $f)
  step "mi_${w}_$f" $(awk -v b=$b 'BEGIN{printf "%.0f", b+8}') -- bash -c "
    timeout 14400 '$R/scripts/in_cgroup.sh' m3 $(capb $b) $TORCH_VENV/bin/python '$R/scripts/sota_serve.py' \
      --system moe-infinity --checkpoint /home/thor/kcj/models/qwen3_30b_a3b \
      --workload '$R/results/WORKLOADS/$w.json' --offload-dir /home/thor/kcj/offload_tmp/qwen30b \
      --budget-gib $b --out '$OUT/$w/moeinf_real_$f.json' > '$OUT/$w/moeinf_real_$f.txt' 2>&1; \
    grep -q '^RESULT' '$OUT/$w/moeinf_real_$f.txt'"
done; done
git add -A results/MATRIX3/qwen30b_bf16 scripts >/dev/null 2>&1
git diff --cached --quiet || { git commit -q -m "Real SOTA systems on Qwen3-30B bf16: ZipMoE, MoE-Infinity

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q origin HEAD; }
say "=== stage 3f done ==="
