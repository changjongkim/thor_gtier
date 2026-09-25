#!/bin/bash
# Stage 3b: FlashMoE* on the same prompts, budgets and cap as stage 3.
# Its network is trained per model and budget (slots per layer depend on the
# budget) on the model's other two workloads, never on the evaluated one.
set -u
R=/home/thor/kcj/thor_gtier
cd "$R"
. "$R/scripts/torch_env.sh"
. "$R/scripts/memguard.sh"
ST=$R/results/PIPELINE; OUT=$R/results/MATRIX3; W8=$R/results/FLASHMOE
mkdir -p "$W8"
LOG=$ST/pipeline.log
MD=/home/thor/kcj/models
BIN=$R/lib/serve_bench.v3
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
step(){
  local name=$1 need=$2; shift 2; [ "${1:-}" = "--" ] && shift
  [ -f "$ST/$name.done" ] && return 0
  local fails=$(cat "$ST/$name.fail" 2>/dev/null || echo 0)
  if [ "$fails" -ge 2 ]; then say "give up $name (failed $fails times)"; return 0; fi
  if ! mg_check "$need" >>"$LOG" 2>&1; then say "REFUSED $name (needs $need GiB)"; return 0; fi
  say "run  $name"; drop
  if "$@" >> "$ST/$name.out" 2>&1; then touch "$ST/$name.done"; say "  ok $name"
  else local rc=$?; echo $((fails+1)) > "$ST/$name.fail"; say "  FAILED $name (rc=$rc)"; fi
}
exec 9>/tmp/gtier_pipeline.lock
flock 9
say "=== stage 3b (FlashMoE*) start ==="
declare -A MDIR=( [qwen30b]=$MD/qwen3_30b_q4km [mixtral8x7b]=$MD/mixtral8x7b_q4km [qwen235b]=$MD/moe235b_q4km )
declare -A NL=( [qwen30b]=48 [mixtral8x7b]=32 [qwen235b]=94 )
declare -A NE=( [qwen30b]=128 [mixtral8x7b]=8 [qwen235b]=128 )
ggufs(){ ls "${MDIR[$1]}"/*.gguf 2>/dev/null | grep -v q8_0 | sort; }
gib(){ du -cb $(ggufs "$1") | tail -1 | awk '{printf "%.2f", $1/1073741824}'; }
cal(){ awk -v m="$1" -v c="$2" '$1==m{print $c}' "$R/results/MATRIX/calib.tsv"; }
capb(){ awk -v b="$1" 'BEGIN{printf "%.0f", (b+0.5)*1073741824}'; }
PO=0.59
fm(){
  local m=$1 w=$2 frac=$3 bt=${4:-1}
  local b=$(awk -v t="$(gib $m)" -v f="$frac" 'BEGIN{printf "%.2f", t*f}')
  local slots=$(python3 "$R/scripts/fm_slots.py" $b 0.5 $PO ${NL[$m]} ${NE[$m]} $(ggufs $m))
  local wt=$W8/${m}_${w}_s$slots.txt
  local train=""; for ow in longbench sharegpt mmlu; do [ $ow = $w ] || train="$train $R/results/SCOPE/rt_${m}_$ow.npz"; done
  step "fmtrain_${m}_${w}_s$slots" 8 -- $TORCH_VENV/bin/python "$R/scripts/train_flashmoe.py" "$wt" $slots $train
  [ -s "$wt" ] || return
  local name=flashmoe_$frac; [ "$bt" != 1 ] && name=flashmoe_b${bt}_$frac
  local shards=""; for f in $(ggufs $m); do shards="$shards --shard $f"; done
  local need=$(awk -v b="$b" 'BEGIN{printf "%.0f", b+6}')
  mkdir -p "$OUT/$m/$w"
  step "m3_${m}_${w}_$name" "$need" -- bash -c "
    timeout 14400 '$R/scripts/in_cgroup.sh' m3 $(capb $b) '$BIN' $shards --trace '$R/results/SCOPE/rt_${m}_${w}.bin' \
      --budget $b --window 0.5 --repeats 2 --compute-ms $(cal $m 2) --prompt-compute-ms $(cal $m 3) \
      --policy 13 --fm-weights '$wt' --backend 3 --path-overhead $PO --overlap --batch $bt \
      > '$OUT/$m/$w/$name.txt' 2>&1; \
    awk '{printf \"cgroup_peak_gib=%.2f\\n\", \$1/1073741824}' /sys/fs/cgroup/ledger_bench/m3/memory.peak >> '$OUT/$m/$w/$name.txt'; \
    grep -q '^RESULT' '$OUT/$m/$w/$name.txt'"
}
for m in qwen30b mixtral8x7b qwen235b; do
  [ -n "$(ggufs $m)" ] || continue
  for w in longbench sharegpt mmlu; do
    for frac in 0.25 0.45 0.65; do fm $m $w $frac; done
    fm $m $w 0.45 4
  done
  python3 "$R/scripts/summarize_matrix3.py" > "$OUT/SUMMARY.md"
  git add -A results/MATRIX3 results/FLASHMOE scripts lib/*.cu lib/*.h >/dev/null 2>&1
  git diff --cached --quiet || { git commit -q -m "FlashMoE* baseline: $m

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q origin HEAD; }
done
say "=== stage 3b done ==="
