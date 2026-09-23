#!/bin/bash
# Fourth queue: a real engine on the real target, so the serving numbers have
# something measured to stand against.
#
# Qwen3-235B-A22B Q4_K_M is 132.4 GiB against 122.8 GiB of memory, so
# llama.cpp cannot hold it and has to fault or stream.  llama-bench is used
# rather than llama-cli because it is non-interactive and reports prompt
# processing and token generation separately, which is TTFT and TPOT:
#
#   TTFT = n_prompt / pp_throughput      TPOT = 1000 / tg_throughput
#
# /proc/<pid>/io alongside gives what it pulled off the device, the same
# number serve_bench reports, so the two can be put side by side.
set -u
ROOT=/home/thor/kcj/thor_gtier
cd "$ROOT"
OUT=$ROOT/results/ENGINE; LOG=$OUT/progress.log
BENCH=${BENCH:-/home/thor/skim/llama.cpp/build/bin/llama-bench}
BIG=${BIG:-/home/thor/kcj/models/moe235b_q4km}
NP=${NP:-512}; NG=${NG:-32}
mkdir -p "$OUT"
MODEL=$(ls "$BIG"/*-00001-of-*.gguf 2>/dev/null | head -1)
[ -z "$MODEL" ] && MODEL=$(ls -S "$BIG"/*.gguf | head -1)

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
run(){ local name="$1"; shift
  [ -s "$OUT/$name.txt" ] && { say "skip $name"; return; }
  say "run  $name"; drop
  timeout 10800 "$ROOT/scripts/io_meter.py" --label "$name" -- \
      "$BENCH" -m "$MODEL" -p $NP -n $NG -r 1 -o json "$@" \
      < /dev/null > "$OUT/$name.txt" 2>&1 || say "  rc=$?"
  grep -E '"avg_ts"|IOMETER' "$OUT/$name.txt" | tail -3 | tee -a "$LOG"
}
commit(){ cd "$ROOT"; git add -A results/ENGINE >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1|tail -1|tee -a "$LOG"
  timeout 300 git push -q 2>&1|tail -1|tee -a "$LOG"; }

echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== engine queue: $(basename "$MODEL")  pp=$NP tg=$NG ==="
ls -la "$BIG"/*.gguf | awk '{s+=$5} END {printf "  model %.1f GiB vs 122.8 GiB of memory\n", s/1073741824}' | tee -a "$LOG"

# ngl is how many layers go to the GPU, which on this SoC is the same memory,
# so a high ngl is the case that cannot fit.  -ncmoe keeps that many layers'
# experts on the CPU, which is llama.cpp's own answer to a MoE too big to hold.
run "e9_ngl0_mmap"    -ngl 0
run "e9_ngl99_mmap"   -ngl 99
run "e9_ngl40_mmap"   -ngl 40
run "e9_ngl99_dio"    -ngl 99 -dio 1
run "e9_ngl99_nommap" -ngl 99 -mmp 0
run "e9_ncmoe40"      -ngl 99 -ncmoe 40
commit "Engine queue: llama.cpp on the DRAM-exceeding MoE model"
say "=== engine queue done ==="
