#!/bin/bash
# The definitive comparison: every policy against every budget, on one trace.
#
# Earlier queues were run as the policies were written, so they are spread
# across two traces and several binaries.  This runs all of them on the
# multi-prefix trace with one binary, which is what the tables should be
# built from.
set -u
ROOT=/home/thor/kcj/thor_gtier
cd "$ROOT"
OUT=$ROOT/results/FINAL; LOG=$OUT/progress.log
BIN=$ROOT/lib/serve_bench
MODEL=${MODEL:-/home/thor/kcj/models/qwen3_30b_a3b}
TRACE=$ROOT/results/SCOPE/routing_multi.bin
MAXDEC=${MAXDEC:-8}
mkdir -p "$OUT"
SH_ARGS=""; for f in "$MODEL"/model-*.safetensors; do SH_ARGS="$SH_ARGS --shard $f"; done
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
run(){ local name="$1"; shift
  grep -q "TTFT(io)" "$OUT/$name.txt" 2>/dev/null && { say "skip $name"; return; }
  say "run  $name"; drop
  timeout 7200 "$BIN" $SH_ARGS --trace "$TRACE" "$@" > "$OUT/$name.txt" 2>&1 \
    && tail -1 "$OUT/$name.txt" | tee -a "$LOG" || say "  FAILED $name"
}
commit(){ cd "$ROOT"; git add -A results >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1|tail -1|tee -a "$LOG"
  timeout 300 git push -q 2>&1|tail -1|tee -a "$LOG"; }

echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== final queue: all policies x budgets, one trace, one binary ==="

# 0 none · 1 lru · 2 lru+phase · 3 per-layer · 4 prefix · 5 online
# 6 online+prefix · 7 multi-prefix · 8 unified · 9 unified-online
# 10 moe-inf* · 11 mixtral*
run "f_none" --policy 0 --budget 40 --window 0.5 --max-decode $MAXDEC
for B in 16 24 32 40 48; do
  for P in 1 2 3 4 5 6 7 8 9 10 11; do
    run "f_b${B}_p${P}" --policy $P --budget $B --window 0.5 --max-decode $MAXDEC
  done
  commit "Final queue: budget $B"
done

# Lookahead: what a plain MoE knows, what pre-gating buys, and what the
# driver had been assuming.
for LA in 1 2 4 8 16; do
  run "f_look${LA}_p8" --policy 8 --budget 40 --window 0.5 --lookahead $LA --max-decode $MAXDEC
  run "f_look${LA}_p3" --policy 3 --budget 40 --window 0.5 --lookahead $LA --max-decode $MAXDEC
done
commit "Final queue: lookahead, which is what pre-gating buys"

# Interleaved arrival for the policies that matter, so the premise in 3.6 is
# stated with a number rather than assumed.
for P in 2 3 4 8 9; do
  run "f_intl_p${P}" --policy $P --budget 40 --window 0.5 --interleave --max-decode $MAXDEC
done
commit "Final queue: interleaved arrival"

say "=== final queue done ==="
python3 "$ROOT/scripts/summarize_final.py" > "$OUT/SUMMARY.md" 2>/dev/null
commit "Final queue: summary"
