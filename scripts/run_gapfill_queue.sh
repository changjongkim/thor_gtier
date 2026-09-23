#!/bin/bash
# Fills the baselines the earlier queues left out, and re-runs anything whose
# policy numbering shifted when SERVE_LRU_PHASE was inserted into the enum.
set -u
ROOT=/home/thor/kcj/thor_gtier
cd "$ROOT"
OUT=$ROOT/results/SERVE; LOG=$OUT/progress.log
BIN=$ROOT/lib/serve_bench
MODEL=${MODEL:-/home/thor/kcj/models/qwen3_30b_a3b}
TRACE=$ROOT/results/SCOPE/routing_multi.bin
MAXDEC=${MAXDEC:-8}
SH_ARGS=""; for f in "$MODEL"/model-*.safetensors; do SH_ARGS="$SH_ARGS --shard $f"; done
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
run(){ local name="$1"; shift
  [ -s "$OUT/$name.txt" ] && { say "skip $name"; return; }
  say "run  $name"; drop
  timeout 7200 "$BIN" $SH_ARGS "$@" > "$OUT/$name.txt" 2>&1 \
    && tail -1 "$OUT/$name.txt" | tee -a "$LOG" || say "  FAILED $name"
}
commit(){ cd "$ROOT"; git add -A results >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1|tail -1|tee -a "$LOG"
  timeout 300 git push -q 2>&1|tail -1|tee -a "$LOG"; }

echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== gapfill queue ==="

# E6 asked for a no-prefix baseline and got lru+phase, because the policy
# numbering shifted.  per-layer is the baseline the prefix policies should be
# read against, so it is filled in here.
for B in 24 40; do
  run "e6_b${B}_perlayer" --trace "$TRACE" --policy 3 --budget $B --window 0.5 --max-decode $MAXDEC
done
# The online policies on the richer trace, so E6's prefix competition has an
# oracle-free point to stand against.
for B in 24 40; do
  run "e6_b${B}_online"  --trace "$TRACE" --policy 5 --budget $B --window 0.5 --repeats 3 --max-decode $MAXDEC
  run "e6_b${B}_onlinep" --trace "$TRACE" --policy 6 --budget $B --window 0.5 --repeats 3 --max-decode $MAXDEC
done
commit "Gapfill: per-layer baseline and online policies on the multi-prefix trace"
say "=== gapfill done ==="
python3 "$ROOT/scripts/summarize_serving.py" > "$OUT/SUMMARY.md" 2>/dev/null
commit "Serving queues: summary refresh"
