#!/bin/bash
# Chain everything so the batch runs unattended: the first queue, then a
# richer routing capture (several system prompts, needed for the multi-prefix
# questions), then the online and multi-prefix queues, then the summary.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd); LOG=$ROOT/results/SERVE/progress.log
mkdir -p "$ROOT/results/SERVE"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true

while pgrep -f run_serving_queue.sh >/dev/null; do sleep 30; done
./scripts/run_serving_queue.sh  >> "$ROOT/results/SERVE_queue.out"  2>&1

# The richer trace: three system prompts, three requests each, so prefix
# families actually compete.  Kept separate from routing.bin so the earlier
# runs stay comparable.
if [ ! -s "$ROOT/results/SCOPE/routing_multi.npz" ]; then
  say "capture: routing with three prefix families"
  . ./scripts/torch_env.sh
  $TORCH_VENV/bin/python scripts/capture_routing.py --decode 64 \
      --out results/SCOPE/routing_multi.npz >> "$ROOT/results/capture_multi.out" 2>&1 \
      && $TORCH_VENV/bin/python scripts/export_routing.py \
           --npz results/SCOPE/routing_multi.npz \
           --out results/SCOPE/routing_multi.bin >> "$ROOT/results/capture_multi.out" 2>&1
  say "capture done"
fi

./scripts/run_serving_queue2.sh >> "$ROOT/results/SERVE_queue2.out" 2>&1
./scripts/run_serving_queue3.sh >> "$ROOT/results/SERVE_queue3.out" 2>&1

python3 scripts/summarize_serving.py > "$ROOT/results/SERVE/SUMMARY.md" 2>/dev/null
cd "$ROOT"
git add -A results >/dev/null 2>&1
git diff --cached --quiet || {
  git commit -q -m "Serving queues: final summary

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
  timeout 300 git push -q 2>&1 | tail -1
}
say "=== run_all done ==="
