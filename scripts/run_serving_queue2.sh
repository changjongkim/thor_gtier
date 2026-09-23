#!/bin/bash
# Second queue: the online policies, which learn without an oracle.
#
# E1's per-layer and prefix orderings are ranked from counts taken over the
# whole trace, which a running system does not have.  These two learn the same
# thing from what they observe, so the gap between them and the oracle is the
# cost of not knowing the distribution in advance.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd); OUT=$ROOT/results/SERVE; LOG=$OUT/progress.log
BIN=$ROOT/lib/serve_bench
MODEL=${MODEL:-/home/thor/kcj/models/qwen3_30b_a3b}
MAXDEC=${MAXDEC:-8}
REPEATS=${REPEATS:-3}
mkdir -p "$OUT"
SH=""; for f in "$MODEL"/model-*.safetensors; do SH="$SH --shard $f"; done
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
run(){ local name="$1"; shift
  [ -s "$OUT/$name.txt" ] && { say "skip $name"; return; }
  say "run  $name"; drop
  if timeout 7200 "$BIN" $SH "$@" > "$OUT/$name.txt" 2>&1; then
    tail -1 "$OUT/$name.txt" | tee -a "$LOG"
  else say "  FAILED $name (rc=$?)"; tail -2 "$OUT/$name.txt" | tee -a "$LOG"; fi
}
commit(){ cd "$ROOT"; git add -A results/SERVE >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1|tail -1|tee -a "$LOG"
  timeout 300 git push -q 2>&1|tail -1|tee -a "$LOG"; }

echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== queue 2: online policies, repeats=$REPEATS ==="

# --- E5: online (no oracle) against the oracle orderings -----------------
# Repeated passes matter here: an online policy has to see traffic before it
# can hold the right thing, so one pass understates it.
for B in 16 24 32 40; do
  run "e5_b${B}_online"  --policy 5 --budget $B --window 0.5 --repeats $REPEATS --max-decode $MAXDEC
  run "e5_b${B}_onlinep" --policy 6 --budget $B --window 0.5 --repeats $REPEATS --max-decode $MAXDEC
  run "e5_b${B}_oracle"  --policy 4 --budget $B --window 0.5 --repeats $REPEATS --max-decode $MAXDEC
  run "e5_b${B}_lru"     --policy 1 --budget $B --window 0.5 --repeats $REPEATS --max-decode $MAXDEC
done
commit "Serving queue: online per-layer LFU against the oracle orderings"

say "=== queue 2 done ==="
python3 scripts/summarize_serving.py > "$OUT/SUMMARY.md" 2>>"$LOG" || true
commit "Serving queue: summary refresh"
