#!/bin/bash
# Fourth queue: a real engine on the real target, so the serving numbers have
# something measured to stand against.
#
# Qwen3-235B-A22B Q4_K_M is 132.4 GiB against 122.8 GiB of memory, so
# llama.cpp cannot hold it and has to fault or stream.  It reports prompt eval
# and eval separately, which is TTFT and TPOT, and /proc/<pid>/io gives what
# it actually pulled off the device -- the same number serve_bench reports, so
# the two can be put side by side.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd); OUT=$ROOT/results/ENGINE; LOG=$OUT/progress.log
LLAMA=${LLAMA:-/home/thor/skim/llama.cpp/build/bin/llama-cli}
BIG=${BIG:-/home/thor/kcj/models/moe235b_q4km}
NPRED=${NPRED:-32}
mkdir -p "$OUT"
MODEL=$(ls "$BIG"/*-00001-of-*.gguf 2>/dev/null | head -1)
[ -z "$MODEL" ] && MODEL=$(ls -S "$BIG"/*.gguf | head -1)

PROMPT="You are a careful systems engineer. Answer precisely and briefly. \
Consider the following context about storage hardware: NVMe drives deliver high \
throughput at large block sizes and collapse at small ones, and the page cache \
hides this from most applications. Question: why does random I/O improve with \
queue depth on an NVMe device, and what limits that improvement?"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
run(){ local name="$1"; shift
  [ -s "$OUT/$name.txt" ] && { say "skip $name"; return; }
  say "run  $name"; drop
  timeout 9000 ./scripts/io_meter.py --label "$name" -- \
      "$LLAMA" -m "$MODEL" -p "$PROMPT" -n $NPRED --no-warmup -no-cnv "$@" \
      > "$OUT/$name.txt" 2>&1 || say "  rc=$? (may still have partial timings)"
  grep -E "prompt eval time|^ *eval time|IOMETER" "$OUT/$name.txt" | tee -a "$LOG"
}
commit(){ cd "$ROOT"; git add -A results/ENGINE >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1|tail -1|tee -a "$LOG"
  timeout 300 git push -q 2>&1|tail -1|tee -a "$LOG"; }

echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== engine queue: $(basename $MODEL) ==="
ls -la "$BIG"/*.gguf | awk '{s+=$5} END {printf "  model %.1f GiB vs 122.8 GiB of memory\n", s/1073741824}' | tee -a "$LOG"

# Loading modes.  ngl 99 asks for the whole model on the GPU, which on this
# SoC is the same memory, so it is the case that cannot fit.
run "e9_ngl0_mmap"     -ngl 0
run "e9_ngl99_mmap"    -ngl 99
run "e9_ngl40_mmap"    -ngl 40
run "e9_ngl0_direct"   -ngl 0  --direct-io
run "e9_ngl99_direct"  -ngl 99 --direct-io
run "e9_ngl0_nommap"   -ngl 0  --no-mmap
commit "Engine queue: llama.cpp on the DRAM-exceeding MoE model"

say "=== engine queue done ==="
