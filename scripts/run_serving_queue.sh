#!/bin/bash
# The serving experiments, detached and resumable.
#
# Every step writes its own file under results/SERVE and is skipped if that
# file already exists, so the batch survives a disconnection and a re-run
# costs only what did not finish.  Each group is committed as it completes.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT=$ROOT/results/SERVE
LOG=$OUT/progress.log
BIN=$ROOT/lib/serve_bench
MODEL=${MODEL:-/home/thor/kcj/models/qwen3_30b_a3b}
MAXDEC=${MAXDEC:-8}
mkdir -p "$OUT"

SH=""
for f in "$MODEL"/model-*.safetensors; do SH="$SH --shard $f"; done

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
run(){ # run <name> <args...>
  local name="$1"; shift
  if [ -s "$OUT/$name.txt" ]; then say "skip $name"; return; fi
  say "run  $name"
  drop
  if timeout 5400 setsid --wait bash -c 'echo -700 > /proc/self/oom_score_adj 2>/dev/null; exec "$@"' _ \
        "$BIN" $SH "$@" > "$OUT/$name.txt" 2>&1; then
    tail -1 "$OUT/$name.txt" | tee -a "$LOG"
  else
    say "  FAILED $name (rc=$?)"; tail -2 "$OUT/$name.txt" | tee -a "$LOG"
  fi
}
commit(){ # commit <message>
  cd "$ROOT"
  git add -A results/SERVE >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1 | tail -1 | tee -a "$LOG"
  timeout 300 git push -q 2>&1 | tail -1 | tee -a "$LOG"
}

# Earlier long runs were killed as collateral when an unrelated process on
# this machine ballooned to 81 GiB and triggered a global OOM.  Lowering this
# queue's oom_score_adj does not touch that process; it only makes the kernel
# prefer the runaway over the measurement when it has to choose.
echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true

say "=== serving queue start  model=$MODEL  max-decode=$MAXDEC ==="

# --- E1: policy x budget --------------------------------------------------
# none is budget-independent (it holds nothing), so it runs once.
run "e1_none" --policy 0 --budget 40 --window 0.5 --max-decode $MAXDEC
for B in 12 16 24 32 40 48; do
  for P in 1 2 3 4; do
    run "e1_b${B}_p${P}" --policy $P --budget $B --window 0.5 --max-decode $MAXDEC
  done
done
commit "Serving queue: residency policy against memory budget"

# --- E2: the data path's footprint, charged to the same budget ------------
# Measured non-reclaimable footprints above a 2.00 GiB declared window
# (results/HF_MOE/footprint.txt): gtier 2.33, pread+copy 4.35, cuFile 10.62.
# Here each path keeps a 0.5 GiB window and is charged what it really holds.
for B in 24 40; do
  run "e2_b${B}_gtier"  --policy 4 --budget $B --window 0.5 --backend 0 --path-overhead 0.00 --max-decode $MAXDEC
  run "e2_b${B}_pread"  --policy 4 --budget $B --window 0.5 --backend 3 --path-overhead 2.02 --max-decode $MAXDEC
  run "e2_b${B}_cufile" --policy 4 --budget $B --window 0.5 --backend 4 --path-overhead 8.29 --max-decode $MAXDEC
done
commit "Serving queue: data path footprint charged against the serving budget"

# --- E3: how much of residency should the prefix take --------------------
# The prefix union grows with how much of the prompt is treated as shared, so
# sweeping the token count sweeps the split between prefix-pinned and
# popularity-ordered residency.
for T in 0 10 20 30 50 65; do
  run "e3_prefix${T}" --policy 4 --budget 40 --window 0.5 --prefix-tokens $T --max-decode $MAXDEC
done
commit "Serving queue: splitting residency between prefix and popularity"

# --- E4: window size under a serving budget ------------------------------
for W in 0.25 0.5 1 2 4; do
  run "e4_w${W}" --policy 4 --budget 40 --window $W --max-decode $MAXDEC
done
commit "Serving queue: staging window under a fixed serving budget"

say "=== queue done, summarising ==="
python3 scripts/summarize_serving.py > "$OUT/SUMMARY.md" 2>>"$LOG" || say "summary failed"
commit "Serving queue: summary"
say "=== all done ==="
