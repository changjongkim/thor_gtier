#!/bin/bash
# The same file llama.cpp is reading, read by gTier.
#
# The engine queue measures what llama.cpp gets out of Qwen3-235B-A22B
# Q4_K_M; this measures what the same bytes can be delivered at, so the two
# sit on one axis instead of being compared across models.
set -u
ROOT=/home/thor/kcj/thor_gtier
cd "$ROOT"
OUT=$ROOT/results/ENGINE; LOG=$OUT/progress.log
BIN=$ROOT/lib/gguf_bench
BIG=${BIG:-/home/thor/kcj/models/moe235b_q4km}
mkdir -p "$OUT"
SH_ARGS=""; for f in "$BIG"/*.gguf; do SH_ARGS="$SH_ARGS --shard $f"; done
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
run(){ local name="$1"; shift
  [ -s "$OUT/$name.txt" ] && { say "skip $name"; return; }
  say "run  $name"; drop
  timeout 10800 "$BIN" $SH_ARGS "$@" > "$OUT/$name.txt" 2>&1 \
    && tail -1 "$OUT/$name.txt" | tee -a "$LOG" || say "  FAILED $name"
}
commit(){ cd "$ROOT"; git add -A results/ENGINE >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1|tail -1|tee -a "$LOG"
  timeout 300 git push -q 2>&1|tail -1|tee -a "$LOG"; }

echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== same-file queue: gTier on the model llama.cpp just read ==="

# Decode: 8 of 128 experts per token, which is what the engine's decode does.
for b in 0 1 2 3 4 5; do
  run "e10_backend$b" --backend $b --policy 0 --tokens 2 --batch 1 \
      --experts 128 --active 8 --skew 1.185 --slot 1048576 --slots 512
done
# Prefill: the union over a long prompt reaches most experts, so it is read as
# a near-full pass rather than a routed one.
run "e10_prefill_union" --backend 0 --async 1 --policy 0 --tokens 1 --batch 32 \
    --experts 128 --active 8 --skew 1.185 --slot 1048576 --slots 512
commit "Same-file queue: gTier on Qwen3-235B-A22B Q4_K_M"
say "=== same-file queue done ==="
python3 "$ROOT/scripts/summarize_engine.py" > "$OUT/SUMMARY.md" 2>/dev/null
commit "Engine summary"
