#!/bin/bash
# Resume the experiment queues, including after a reboot.
#
# This machine restarted three times during the run -- 19:25, 23:12 and 01:58
# -- taking whatever was in flight with it.  Every queue skips work whose
# output already carries a result line, so resuming costs only what did not
# finish; what was missing was something to do the resuming.  Registered with
# cron @reboot, and safe to run by hand: a lock keeps one copy at a time.
set -u
R=/home/thor/kcj/thor_gtier
cd "$R"
LOCK=/tmp/gtier_supervisor.lock
exec 9>"$LOCK"
flock -n 9 || { echo "supervisor already running"; exit 0; }

LOG=$R/results/supervisor.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
say "=== supervisor start (uptime $(cut -d. -f1 /proc/uptime)s) ==="

# Let the system settle before competing with it for the device.
sleep 60

for q in run_final_queue run_engine_queue run_samefile_queue run_placement_queue; do
  [ -x "$R/scripts/$q.sh" ] || continue
  cp "$R/scripts/$q.sh" "/tmp/sv_$q.sh"; chmod +x "/tmp/sv_$q.sh"
  say "-> $q"
  "/tmp/sv_$q.sh" >> "$R/results/${q}.out" 2>&1
  say "<- $q (rc=$?)"
done

for s in summarize_final summarize_engine summarize_serving; do
  case $s in
    summarize_final)   o=$R/results/FINAL/SUMMARY.md ;;
    summarize_engine)  o=$R/results/ENGINE/SUMMARY.md ;;
    *)                 o=$R/results/SERVE/SUMMARY.md ;;
  esac
  python3 "$R/scripts/$s.py" > "$o" 2>/dev/null
done

cd "$R"
git add -A results >/dev/null 2>&1
git diff --cached --quiet || {
  git commit -q -m "Queue results, resumed after reboot

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
  timeout 300 git push -q 2>&1 | tail -1 >> "$LOG"
}
say "=== supervisor done ==="
