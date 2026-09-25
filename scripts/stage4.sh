#!/bin/bash
# Stage 4: real systems, one compute stack (transformers), same prompts, same
# memory cap.  Each system runs its own code, data path and scheduling:
#   PHASOR-HF      this work, inside transformers (phasor_hf/)
#   PHASOR-HF/LRU  the same with LRU residency (ablation of the policy)
#   ZipMoE         ICML'26 release, ported to Qwen3-MoE / Mixtral
#   MoE-Infinity   its own engine
# Budgets are fractions of the bf16 checkpoint; 1.08 = the model fits, each
# system's in-memory reference.  Every run under a cgroup cap of budget+0.5 GiB.
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. "$R/scripts/torch_env.sh"; . "$R/scripts/memguard.sh"
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log
ZPY=/home/thor/kcj/envs/zipmoe/bin/python
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
capb(){ awk -v b="$1" 'BEGIN{printf "%.0f", (b+0.5)*1073741824}'; }
need(){ awk -v b="$1" 'BEGIN{printf "%.0f", b+10}'; }
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== stage 4 (real systems, transformers stack) start ==="
FRACS="0.25 0.45 0.65 1.08"

run_model(){  # run_model <tag> <ckpt> <bf16-GiB> <zipmoe-type> <slot-mib> <window-gib>
  local tag=$1 ck=$2 gb=$3 zt=$4 slot=$5 win=$6
  local O=$R/results/MATRIX4/$tag; mkdir -p "$O"
  local b45=$(awk -v g=$gb 'BEGIN{printf "%.2f", g*0.45}')
  # smoke tests: two prompts each; the one-time conversions happen here
  step s4_smoke_phasor_$tag $(need $b45) -- timeout 3600 $ZPY "$R/phasor_hf/phasor_serve.py" --checkpoint $ck \
    --workload "$R/results/WORKLOADS/mmlu.json" --budget-gib $b45 --slot-mib $slot --window-gib $win \
    --limit 2 --out "$O/smoke_phasor.json"
  local zip_ok=1 mi_ok=1
  step s4_smoke_zip_$tag 60 -- timeout 10800 $ZPY "$R/scripts/zipmoe_serve.py" --model-type $zt \
    --workload "$R/results/WORKLOADS/mmlu.json" --budget-gib $b45 \
    --trace /home/thor/kcj/ZipMoE/trace/${zt}_mmlu_heldout.pt --limit 2 --out "$O/smoke_zip.json" || zip_ok=0
  if [ $tag = qwen30b ]; then
    mkdir -p /home/thor/kcj/offload_tmp/$tag
    step s4_smoke_mi_$tag 60 -- timeout 10800 $TORCH_VENV/bin/python "$R/scripts/sota_serve.py" --system moe-infinity \
      --checkpoint $ck --workload "$R/results/WORKLOADS/mmlu.json" --offload-dir /home/thor/kcj/offload_tmp/$tag \
      --budget-gib $b45 --limit 2 --out "$O/smoke_mi.json" || mi_ok=0
  else mi_ok=0; fi
  for w in mmlu sharegpt longbench; do mkdir -p "$O/$w"; for f in $FRACS; do
    local b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}')
    step "s4_${tag}_${w}_phasor_$f" $(need $b) -- bash -c "
      timeout 21600 '$R/scripts/in_cgroup.sh' m4 $(capb $b) $ZPY '$R/phasor_hf/phasor_serve.py' --checkpoint $ck \
        --workload '$R/results/WORKLOADS/$w.json' --budget-gib $b --slot-mib $slot --window-gib $win \
        --out '$O/$w/phasor_$f.json' > '$O/$w/phasor_$f.txt' 2>&1; grep -q '^RESULT' '$O/$w/phasor_$f.txt'"
    step "s4_${tag}_${w}_phasorlru_$f" $(need $b) -- bash -c "
      timeout 21600 '$R/scripts/in_cgroup.sh' m4 $(capb $b) $ZPY '$R/phasor_hf/phasor_serve.py' --checkpoint $ck \
        --workload '$R/results/WORKLOADS/$w.json' --budget-gib $b --slot-mib $slot --window-gib $win --policy lru \
        --out '$O/$w/phasorlru_$f.json' > '$O/$w/phasorlru_$f.txt' 2>&1; grep -q '^RESULT' '$O/$w/phasorlru_$f.txt'"
    [ $zip_ok = 1 ] && step "s4_${tag}_${w}_zipmoe_$f" $(need $b) -- bash -c "
      timeout 21600 '$R/scripts/in_cgroup.sh' m4 $(capb $b) $ZPY '$R/scripts/zipmoe_serve.py' --model-type $zt \
        --workload '$R/results/WORKLOADS/$w.json' --budget-gib $b \
        --trace /home/thor/kcj/ZipMoE/trace/${zt}_${w}_heldout.pt --out '$O/$w/zipmoe_$f.json' \
        > '$O/$w/zipmoe_$f.txt' 2>&1; grep -q '^RESULT' '$O/$w/zipmoe_$f.txt'"
    [ $mi_ok = 1 ] && step "s4_${tag}_${w}_moeinf_$f" $(need $b) -- bash -c "
      timeout 21600 '$R/scripts/in_cgroup.sh' m4 $(capb $b) $TORCH_VENV/bin/python '$R/scripts/sota_serve.py' \
        --system moe-infinity --checkpoint $ck --workload '$R/results/WORKLOADS/$w.json' \
        --offload-dir /home/thor/kcj/offload_tmp/$tag --budget-gib $b \
        --out '$O/$w/moeinf_$f.json' > '$O/$w/moeinf_$f.txt' 2>&1; grep -q '^RESULT' '$O/$w/moeinf_$f.txt'"
  done; done
  git add -A results/MATRIX4 scripts phasor_hf >/dev/null 2>&1
  git diff --cached --quiet || { git commit -q -m "Real-system matrix (transformers stack): $tag

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q origin HEAD; }
}

run_model qwen30b /home/thor/kcj/models/qwen3_30b_a3b 57.0 qwen3 4 0.5
# Mixtral projections are 112 MiB: slots of 128 MiB, a window of 1.5 GiB
run_model mixtral8x7b /home/thor/kcj/models/mixtral8x7b_bf16 87.0 mixtral 128 1.5
say "=== stage 4 done ==="
