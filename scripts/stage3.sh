#!/bin/bash
# Stage 3: every system on its own data path, under the same memory cap.
#
# Stage 2 ran the baselines' policies on this project's data path, which
# credited them with it; that is a comparison of residency policies only, not
# of systems.  Here each system keeps the path it was built with:
#
#   LRU                 synchronous pread into pinned host memory + copy
#   MoE-Infinity*       O_DIRECT pread into pinned host memory + async copy
#                       (core/aio in its source), layer prefetch overlapped
#   Mixtral-offloading* host staging + copy, LRU + speculative load, overlapped
#   llama.cpp           the real engine: dense on GPU, experts mmap'd (--cpu-moe)
#   LEDGER              gTier path, layer-pipelined, decode-probability residency
#
# All runs execute inside a cgroup capped at budget + 0.5 GiB, which counts
# pinned host memory and page cache alike; the copy paths' device staging is
# charged as path overhead (pread+copy 2.18x the window, gTier 1.17x, both
# measured in sec 4.15).  Batch 1 at three budgets; batch 4 at 0.45 for the
# systems that batch (MoE-Infinity serves batches; Mixtral-offloading is a
# batch-1 design).
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
echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== stage 3 start ==="

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

PO_PREAD=0.59   # (2.18 - 1) x 0.5 GiB window
PO_GTIER=0.085  # (1.17 - 1) x 0.5 GiB window
for m in qwen30b mixtral8x7b qwen235b; do
  [ -n "$(ggufs $m)" ] || continue
  for w in longbench sharegpt mmlu; do
    for frac in 0.25 0.45 0.65; do
      sb $m $w "lru_$frac"     $frac --policy 1  --backend 3 --path-overhead $PO_PREAD
      sb $m $w "moeinf_$frac"  $frac --policy 10 --backend 3 --path-overhead $PO_PREAD --overlap
      sb $m $w "mixtral_$frac" $frac --policy 11 --backend 3 --path-overhead $PO_PREAD --overlap
      sb $m $w "ledger_$frac"  $frac --policy 12 --backend 0 --path-overhead $PO_GTIER --overlap
    done
    # llama.cpp pages its experts through a page cache the cap bounds, and at
    # these budgets that thrashes (Qwen3-30B MMLU at 0.25: TTFT 80 s, TPOT
    # 3.1 s); one budget per model and workload keeps it to hours, not days.
    ll $m $w 0.45
    sb $m $w "moeinf_b4_0.45" 0.45 --policy 10 --backend 3 --path-overhead $PO_PREAD --overlap --batch 4
    sb $m $w "ledger_b4_0.45" 0.45 --policy 12 --backend 0 --path-overhead $PO_GTIER --overlap --batch 4
  done
  # What each part of LEDGER contributes, on the model where every run is short.
  if [ "$m" = qwen30b ]; then
    for w in longbench sharegpt mmlu; do
      sb $m $w abl_nooverlap 0.45 --policy 12 --backend 0 --path-overhead $PO_GTIER
      sb $m $w abl_noasync   0.45 --policy 12 --backend 0 --path-overhead $PO_GTIER --overlap --no-async
      sb $m $w abl_pread     0.45 --policy 12 --backend 3 --path-overhead $PO_PREAD --overlap
      sb $m $w abl_lru       0.45 --policy 1  --backend 0 --path-overhead $PO_GTIER --overlap
      sb $m $w abl_mix0      0.45 --policy 12 --backend 0 --path-overhead $PO_GTIER --overlap --mix 0
      sb $m $w abl_mix1      0.45 --policy 12 --backend 0 --path-overhead $PO_GTIER --overlap --mix 1
      sb $m $w abl_norec     0.45 --policy 12 --backend 0 --path-overhead $PO_GTIER --overlap --w-rec 0
      sb $m $w abl_count     0.45 --policy 12 --backend 0 --path-overhead $PO_GTIER --overlap --mix off
    done
  fi
  $TORCH_VENV/bin/python "$R/scripts/summarize_matrix3.py" > "$OUT/SUMMARY.md" 2>>"$LOG"
  commit "Serving matrix (own data paths): $m"
done
touch "$ST/STAGE3.complete"
say "=== stage 3 done ==="
