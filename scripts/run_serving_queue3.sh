#!/bin/bash
# Third queue: the questions the first two cannot answer with one prefix
# family and one request at a time.
#
#   - several system prompts competing for the same pinned residency, with a
#     cap that forces them to evict each other;
#   - prefills and decodes interleaved, which is what continuous batching does
#     and where a global phase switch stops being available;
#   - what a token actually waits for once the arithmetic is allowed to hide
#     the I/O behind it.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd); OUT=$ROOT/results/SERVE; LOG=$OUT/progress.log
BIN=$ROOT/lib/serve_bench
MODEL=${MODEL:-/home/thor/kcj/models/qwen3_30b_a3b}
TRACE=${TRACE:-$ROOT/results/SCOPE/routing_multi.bin}
MAXDEC=${MAXDEC:-8}
mkdir -p "$OUT"
SH=""; for f in "$MODEL"/model-*.safetensors; do SH="$SH --shard $f"; done
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
run(){ local name="$1"; shift
  [ -s "$OUT/$name.txt" ] && { say "skip $name"; return; }
  say "run  $name"; drop
  if timeout 7200 "$BIN" $SH --trace "$TRACE" "$@" > "$OUT/$name.txt" 2>&1; then
    tail -1 "$OUT/$name.txt" | tee -a "$LOG"
  else say "  FAILED $name (rc=$?)"; tail -2 "$OUT/$name.txt" | tee -a "$LOG"; fi
}
commit(){ cd "$ROOT"; git add -A results >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1|tail -1|tee -a "$LOG"
  timeout 300 git push -q 2>&1|tail -1|tee -a "$LOG"; }

echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
[ -s "$TRACE" ] || { say "trace $TRACE missing"; exit 1; }
say "=== queue 3: multi-prefix, interleaving, stall  trace=$(basename $TRACE) ==="

# --- E6: several prefixes under a cap ------------------------------------
# No cap means the prefixes take whatever they need and popularity-ordered
# residency gets the rest; a cap forces them to evict each other.
for B in 24 40; do
  run "e6_b${B}_noprefix"  --policy 2 --budget $B --window 0.5 --max-decode $MAXDEC
  run "e6_b${B}_single"    --policy 4 --budget $B --window 0.5 --max-decode $MAXDEC
  for CAP in 0 4 8 16; do
    run "e6_b${B}_multi_cap${CAP}" --policy 7 --budget $B --window 0.5 \
        --prefix-budget $CAP --max-decode $MAXDEC
  done
done
commit "Serving queue: several prefix families competing for pinned residency"

# --- E7: sequential against interleaved ----------------------------------
# The phase-aware policy assumes the system is in one phase at a time.  A
# server that batches continuously is not, so the same policies are run both
# ways and the difference is what that assumption is worth.
for B in 24 40; do
  for P in 1 2 4 7; do
    run "e7_b${B}_p${P}_seq"   --policy $P --budget $B --window 0.5 --max-decode $MAXDEC
    run "e7_b${B}_p${P}_intl"  --policy $P --budget $B --window 0.5 --max-decode $MAXDEC --interleave
  done
done
commit "Serving queue: sequential against interleaved request arrival"

# --- E8: what the arithmetic can hide ------------------------------------
# Decode is compute-bound once residency is good, so the I/O that fits inside
# the arithmetic costs nothing.  Sweeping the assumed compute time shows how
# much of the remaining I/O is really a stall.
for CMS in 0.0 0.89 5 20; do
  run "e8_cms${CMS}" --policy 7 --budget 40 --window 0.5 --compute-ms $CMS \
      --prompt-compute-ms $CMS --max-decode $MAXDEC
done
commit "Serving queue: how much of the remaining I/O the arithmetic hides"

say "=== queue 3 done ==="
python3 scripts/summarize_serving.py > "$OUT/SUMMARY.md" 2>>"$LOG" || true
commit "Serving queue: summary refresh"
