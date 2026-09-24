#!/bin/bash
# The scheme against the baselines, and against itself with pieces removed.
#
# Cold is one pass over the trace: nothing has been learned yet.  Steady is
# six, which is the same requests returning -- the regime a recency or
# frequency scheme is built for, and the one where it is hardest to beat.
set -u
R=/home/thor/kcj/thor_gtier
cd "$R"
OUT=$R/results/SCHEME; LOG=$OUT/progress.log
BIN=$R/lib/serve_bench
MODEL=${MODEL:-/home/thor/kcj/models/qwen3_30b_a3b}
TRACE=$R/results/SCOPE/routing_multi.bin
mkdir -p "$OUT"
SH_ARGS=""; for f in "$MODEL"/model-*.safetensors; do SH_ARGS="$SH_ARGS --shard $f"; done
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
run(){ local name="$1"; shift
  grep -q "request seconds" "$OUT/$name.txt" 2>/dev/null && { say "skip $name"; return; }
  say "run  $name"; drop
  timeout 7200 "$BIN" $SH_ARGS --trace "$TRACE" --compute "$@" > "$OUT/$name.txt" 2>&1 \
    && tail -2 "$OUT/$name.txt" | tee -a "$LOG" || say "  FAILED $name"
}
commit(){ cd "$R"; git add -A results >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1|tail -1|tee -a "$LOG"
  timeout 300 git push -q 2>&1|tail -1|tee -a "$LOG"; }

echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== scheme queue ==="
# Baselines:  1 lru · 10 moe-inf* · 11 mixtral*
# Ablations:  3 per-layer (the decode term alone)
#             4 prefix    (the prefill term as a hard pin)
# The scheme: 12, at three levels of trust in a profile.  A profile is not a
# different policy -- it is where the counts start, and the run keeps adding
# to them either way, so one knob runs from knowing nothing to knowing the
# workload.
for B in 24 40; do
  for RP in 1 6; do
    for P in 1 10 11 3 4; do
      run "sc_b${B}_r${RP}_p${P}" --policy $P --budget $B --window 0.5 \
          --max-decode 4 --repeats $RP
    done
    for PW in 0 0.25 1; do
      run "sc_b${B}_r${RP}_ours${PW}" --policy 12 --budget $B --window 0.5 \
          --max-decode 4 --repeats $RP --profile-weight $PW
    done
  done
  commit "Scheme queue: budget $B"
done
say "=== scheme queue done ==="
