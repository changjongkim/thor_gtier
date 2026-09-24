#!/bin/bash
# The policy comparison on the axis an engine is judged by.
#
# Every earlier table reports transfer seconds with the arithmetic modelled.
# --compute runs the routed experts' feed-forward for real, so these are token
# rates.  The kernels are the same under every policy, so what separates the
# rows is the I/O each one causes -- which is the point.
set -u
R=/home/thor/kcj/thor_gtier
cd "$R"
OUT=$R/results/TOKRATE; LOG=$OUT/progress.log
BIN=$R/lib/serve_bench
MODEL=${MODEL:-/home/thor/kcj/models/qwen3_30b_a3b}
TRACE=$R/results/SCOPE/routing_multi.bin
MAXDEC=${MAXDEC:-4}
mkdir -p "$OUT"
SH_ARGS=""; for f in "$MODEL"/model-*.safetensors; do SH_ARGS="$SH_ARGS --shard $f"; done
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
run(){ local name="$1"; shift
  grep -q "tok/s" "$OUT/$name.txt" 2>/dev/null && { say "skip $name"; return; }
  say "run  $name"; drop
  timeout 7200 "$BIN" $SH_ARGS --trace "$TRACE" --compute "$@" > "$OUT/$name.txt" 2>&1 \
    && tail -1 "$OUT/$name.txt" | tee -a "$LOG" || say "  FAILED $name"
}
commit(){ cd "$R"; git add -A results >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1|tail -1|tee -a "$LOG"
  timeout 300 git push -q 2>&1|tail -1|tee -a "$LOG"; }

echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== token rate queue: max-decode=$MAXDEC ==="
# 1 lru · 2 lru+phase · 3 per-layer · 4 prefix · 5 online · 6 online+prefix
# 7 multi-prefix · 8 unified · 9 unified-online · 10 moe-inf* · 11 mixtral*
for B in 24 40; do
  for P in 1 10 11 2 3 4 7 5 6 8 9; do
    run "t_b${B}_p${P}" --policy $P --budget $B --window 0.5 --max-decode $MAXDEC
  done
  commit "Token rate queue: budget $B"
done
say "=== token rate queue done ==="
