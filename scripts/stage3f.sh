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
capb(){ awk -v b="$1" 'BEGIN{printf "%.0f", (b+0.5)*1073741824}'; }
need(){ awk -v b="$1" 'BEGIN{printf "%.0f", b+8}'; }
FRACS="0.25 0.45 0.65 1.08"   # 1.08: the whole model fits (each system's in-memory reference)

zip_runs(){  # zip_runs <model-type> <tag> <bf16-GiB>
  local mt=$1 tag=$2 gb=$3 O=$R/results/MATRIX3/${tag}_bf16
  mkdir -p "$O"
  step zip_smoke_$tag 40 -- timeout 10800 $ZPY "$R/scripts/zipmoe_serve.py" --model-type $mt \
     --workload "$R/results/WORKLOADS/mmlu.json" --budget-gib $(awk -v g=$gb 'BEGIN{printf "%.2f", g*0.45}') \
     --trace /home/thor/kcj/ZipMoE/trace/${mt}_mmlu_heldout.pt --limit 2 --out "$O/zip_smoke.json" || return
  for w in longbench sharegpt mmlu; do mkdir -p "$O/$w"; for f in $FRACS; do
    local b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}')
    step "z_${tag}_${w}_$f" $(need $b) -- bash -c "
      timeout 21600 '$R/scripts/in_cgroup.sh' m3 $(capb $b) $ZPY '$R/scripts/zipmoe_serve.py' --model-type $mt \
        --workload '$R/results/WORKLOADS/$w.json' --budget-gib $b \
        --trace /home/thor/kcj/ZipMoE/trace/${mt}_${w}_heldout.pt --out '$O/$w/zipmoe_$f.json' \
        > '$O/$w/zipmoe_$f.txt' 2>&1; grep -q '^RESULT' '$O/$w/zipmoe_$f.txt'"
  done; done
}
phasor_bf16(){  # phasor_bf16 <tag> <ckpt-dir> <bf16-GiB> <compute-args...>
  local tag=$1 ck=$2 gb=$3; shift 3
  local O=$R/results/MATRIX3/${tag}_bf16
  local sh=""; for f in "$ck"/model-*.safetensors; do sh="$sh --shard $f"; done
  local src=$tag; [ $tag = qwen30b ] || src=mixtral8x7b
  for w in longbench sharegpt mmlu; do mkdir -p "$O/$w"; for f in $FRACS; do
    local b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}')
    step "pb_${tag}_${w}_$f" $(need $b) -- bash -c "
      timeout 14400 '$R/scripts/in_cgroup.sh' m3 $(capb $b) '$R/lib/serve_bench.v3' $sh \
        --trace '$R/results/SCOPE/rt_${src}_${w}.bin' --budget $b --window 0.5 --repeats 2 \
        --policy 12 --backend 0 --path-overhead 0.085 --overlap $* \
        > '$O/$w/ledger_$f.txt' 2>&1; grep -q '^RESULT' '$O/$w/ledger_$f.txt'"
  done; done
}

# 1. Qwen3-30B bf16 (57 GiB): ZipMoE, MoE-Infinity, PHASOR with its real decode kernels
zip_runs qwen3 qwen30b 57.0
mkdir -p /home/thor/kcj/offload_tmp/qwen30b
for w in longbench sharegpt mmlu; do mkdir -p "$OUT/$w"; for f in $FRACS; do
  b=$(awk -v f=$f 'BEGIN{printf "%.2f", 57.0*f}')
  step "mi_${w}_$f" $(need $b) -- bash -c "
    timeout 21600 '$R/scripts/in_cgroup.sh' m3 $(capb $b) $TORCH_VENV/bin/python '$R/scripts/sota_serve.py' \
      --system moe-infinity --checkpoint /home/thor/kcj/models/qwen3_30b_a3b \
      --workload '$R/results/WORKLOADS/$w.json' --offload-dir /home/thor/kcj/offload_tmp/qwen30b \
      --budget-gib $b --out '$OUT/$w/moeinf_real_$f.json' > '$OUT/$w/moeinf_real_$f.txt' 2>&1; \
    grep -q '^RESULT' '$OUT/$w/moeinf_real_$f.txt'"
done; done
phasor_bf16 qwen30b /home/thor/kcj/models/qwen3_30b_a3b 57.0 --compute --prompt-compute-ms 0.81486

# 2. Make room: the offload stores this stage wrote for Qwen3-30B (ours, regenerable)
if ls "$ST"/z_qwen30b_*_1.08.done >/dev/null 2>&1; then
  rm -rf /home/thor/kcj/offload_tmp/qwen30b "/home/thor/kcj/ZipMoE-ICML26/offload/qwen3"
  say "removed the Qwen3-30B offload stores (ZipMoE, MoE-Infinity) to make room for Mixtral"
fi

# 3. Mixtral-8x7B bf16 (87 GiB): ZipMoE and PHASOR.  A Mixtral bf16 projection
#    (112 MiB) exceeds the driver's kernel slot, so PHASOR's compute here is the
#    calibrated per-token cost; compare it by slowdown against its own 1.08 run.
zip_runs mixtral mixtral8x7b 87.0
phasor_bf16 mixtral8x7b /home/thor/kcj/models/mixtral8x7b_bf16 87.0 --compute-ms 38.6528 --prompt-compute-ms 1.68858

git add -A results/MATRIX3/qwen30b_bf16 results/MATRIX3/mixtral8x7b_bf16 scripts >/dev/null 2>&1
git diff --cached --quiet || { git commit -q -m "Real SOTA systems on Qwen3-30B bf16: ZipMoE, MoE-Infinity

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q origin HEAD; }
say "=== stage 3f done ==="
