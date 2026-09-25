#!/bin/bash
# Stage 3e: tight budgets (0.10, 0.15 of the model).  The copy paths charge
# 2.18x the window against the budget, so below some budget they have no room
# left for residency, or cannot start at all; that point is the result.
set -u
R=/home/thor/kcj/thor_gtier
cd "$R"
. "$R/scripts/torch_env.sh"
. "$R/scripts/memguard.sh"
ST=$R/results/PIPELINE; OUT=$R/results/MATRIX3; mkdir -p "$ST" "$OUT"
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
  if ! mg_check "$need" >>"$LOG" 2>&1; then
    say "REFUSED $name (needs $need GiB)"; return 0
  fi
  say "run  $name"; drop
  if "$@" >> "$ST/$name.out" 2>&1; then touch "$ST/$name.done"; say "  ok $name"
  else local rc=$?; echo $((fails+1)) > "$ST/$name.fail"; say "  FAILED $name (rc=$rc)"; fi
}
commit(){
  git add -A results/MATRIX3 scripts tools/*.cpp lib/*.cu lib/*.h >/dev/null 2>&1
  git diff --cached --quiet || { git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q origin HEAD; }
}

exec 9>/tmp/gtier_pipeline.lock
flock 9
say "=== stage 3e (tight budgets) start ==="
declare -A MDIR=( [qwen30b]=$MD/qwen3_30b_q4km [mixtral8x7b]=$MD/mixtral8x7b_q4km [qwen235b]=$MD/moe235b_q4km )
ggufs(){ ls "${MDIR[$1]}"/*.gguf 2>/dev/null | grep -v q8_0 | sort; }
gib(){ du -cb $(ggufs "$1") | tail -1 | awk '{printf "%.2f", $1/1073741824}'; }
cal(){ awk -v m="$1" -v c="$2" '$1==m{print $c}' "$R/results/MATRIX/calib.tsv"; }
capb(){ awk -v b="$1" 'BEGIN{printf "%.0f", (b+0.5)*1073741824}'; }

# sb <model> <workload> <run-name> <budget-frac> <args...>
sb(){
  local m=$1 w=$2 name=$3 frac=$4; shift 4
  local tr=$R/results/SCOPE/rt_${m}_${w}.bin
  [ -s "$tr" ] || { say "skip $m/$w/$name (no trace)"; return; }
  local b=$(awk -v t="$(gib $m)" -v f="$frac" 'BEGIN{printf "%.2f", t*f}')
  local need=$(awk -v b="$b" 'BEGIN{printf "%.0f", b+6}')
  local shards=""; for f in $(ggufs $m); do shards="$shards --shard $f"; done
  mkdir -p "$OUT/$m/$w"
  step "m3_${m}_${w}_${name}" "$need" -- bash -c "
    timeout 14400 '$R/scripts/in_cgroup.sh' m3 $(capb $b) '$BIN' $shards --trace '$tr' \
      --budget $b --window 0.5 --repeats 2 --compute-ms $(cal $m 2) --prompt-compute-ms $(cal $m 3) $* \
      > '$OUT/$m/$w/$name.txt' 2>&1; \
    awk '{printf \"cgroup_peak_gib=%.2f\\n\", \$1/1073741824}' /sys/fs/cgroup/ledger_bench/m3/memory.peak >> '$OUT/$m/$w/$name.txt'; \
    grep -q '^RESULT' '$OUT/$m/$w/$name.txt'"
}
# llama.cpp on the same prompts under the same cap; the TSV is the one the
# routing was captured from (235B: the same first 8 prompts).
ll(){
  local m=$1 w=$2 frac=$3
  local b=$(awk -v t="$(gib $m)" -v f="$frac" 'BEGIN{printf "%.2f", t*f}')
  local need=$(awk -v b="$b" 'BEGIN{printf "%.0f", b+12}')
  local tsv=$R/results/WORKLOADS/$w.tsv
  if [ "$m" = qwen235b ]; then head -n 8 "$tsv" > /tmp/claude-1000/ll_${m}_$w.tsv; tsv=/tmp/claude-1000/ll_${m}_$w.tsv; fi
  mkdir -p "$OUT/$m/$w"
  step "m3_${m}_${w}_llama_$frac" "$need" -- bash -c "
    timeout 21600 '$R/scripts/in_cgroup.sh' m3 $(capb $b) '$R/tools/llama_serve' \
      '$(ggufs $m | head -1)' '$tsv' 12 8192 > '$OUT/$m/$w/llama_$frac.txt' 2>'$OUT/$m/$w/llama_$frac.err'; \
    awk '{printf \"cgroup_peak_gib=%.2f\\n\", \$1/1073741824}' /sys/fs/cgroup/ledger_bench/m3/memory.peak >> '$OUT/$m/$w/llama_$frac.txt'; \
    grep -q '^RESULT' '$OUT/$m/$w/llama_$frac.txt'"
}

declare -A NL=( [qwen30b]=48 [mixtral8x7b]=32 [qwen235b]=94 )
declare -A NE=( [qwen30b]=128 [mixtral8x7b]=8 [qwen235b]=128 )
W8=$R/results/FLASHMOE; mkdir -p "$W8"
PO=0.59; PO_PREAD=0.59; PO_GTIER=0.085
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
ds(){
  local m=$1 w=$2 frac=$3 bt=${4:-1}
  local b=$(awk -v t="$(gib $m)" -v f="$frac" 'BEGIN{printf "%.2f", t*f}')
  local prof=""; for ow in longbench sharegpt mmlu; do [ $ow = $w ] || prof="$prof --profile-trace $R/results/SCOPE/rt_${m}_$ow.bin"; done
  local name=duoserve_$frac; [ "$bt" != 1 ] && name=duoserve_b${bt}_$frac
  local shards=""; for f in $(ggufs $m); do shards="$shards --shard $f"; done
  local need=$(awk -v b="$b" 'BEGIN{printf "%.0f", b+6}')
  mkdir -p "$OUT/$m/$w"
  step "m3_${m}_${w}_$name" "$need" -- bash -c "
    timeout 14400 '$R/scripts/in_cgroup.sh' m3 $(capb $b) '$BIN' $shards --trace '$R/results/SCOPE/rt_${m}_${w}.bin' $prof \
      --budget $b --window 0.5 --repeats 2 --compute-ms $(cal $m 2) --prompt-compute-ms $(cal $m 3) \
      --policy 14 --backend 3 --path-overhead $PO --overlap --batch $bt \
      > '$OUT/$m/$w/$name.txt' 2>&1; \
    awk '{printf \"cgroup_peak_gib=%.2f\\n\", \$1/1073741824}' /sys/fs/cgroup/ledger_bench/m3/memory.peak >> '$OUT/$m/$w/$name.txt'; \
    grep -q '^RESULT' '$OUT/$m/$w/$name.txt'"
}
# A budget a system cannot run at is a result: record why instead of failing.
cant(){  # cant <model> <workload> <name> <frac> <reason>
  mkdir -p "$OUT/$1/$2"; echo "NORUN budget=$4 reason=$5" > "$OUT/$1/$2/$3.txt"
}
tight(){
  local m=$1 w=$2 f=$3
  sb $m $w "lru_$f"     $f --policy 1  --backend 3 --path-overhead $PO_PREAD
  sb $m $w "moeinf_$f"  $f --policy 10 --backend 3 --path-overhead $PO_PREAD --overlap
  sb $m $w "mixtral_$f" $f --policy 11 --backend 3 --path-overhead $PO_PREAD --overlap
  sb $m $w "ledger_$f"  $f --policy 12 --backend 0 --path-overhead $PO_GTIER --overlap
  fm $m $w $f
  ds $m $w $f
  for k in lru moeinf mixtral flashmoe duoserve; do
    local t="$OUT/$m/$w/${k}_$f.txt"
    if [ -f "$t" ] && grep -q "budget too small" "$t"; then cant $m $w ${k}_$f $f "budget-below-dense+staging"; fi
  done
}
for w in longbench sharegpt mmlu; do for f in 0.10 0.15; do tight qwen30b $w $f; done; done
for f in 0.10 0.15; do tight mixtral8x7b mmlu $f; done
python3 "$R/scripts/summarize_matrix3.py" > "$OUT/SUMMARY.md"
git add -A results/MATRIX3 results/FLASHMOE scripts >/dev/null 2>&1
git diff --cached --quiet || { git commit -q -m "Tight budgets (0.10, 0.15): where the copy paths stop fitting

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q origin HEAD; }
say "=== stage 3e done ==="
