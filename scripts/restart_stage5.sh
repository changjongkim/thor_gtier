#!/bin/bash
# Restart stage 5 between two runs without losing the run that just ended:
# pause the driver, wait for its current run to finish, record that run as
# stage 5 would (done / norun from its output), then restart the driver from
# scripts/stage5.sh (finished runs are skipped).  usage: restart_stage5.sh "<reason>"
cd /home/thor/kcj/thor_gtier; ST=results/PIPELINE; L=$ST/pipeline.log
RUNNER="^[^ ]*python[^ ]* (phasor_hf/phasor_serve|scripts/zipmoe_serve|scripts/check_tokens|baselines_hf/|phasor_hf/capture|scripts/memcal)"
P=$(ps -eo pid,args | awk '$2=="bash" && $3 ~ /stage5_run/ {print $1}'); [ -n "$P" ] && kill -STOP $P
while ps -eo args | grep -qE "$RUNNER"; do sleep 5; done
last=$(grep -E "\] run  s5" $L | tail -1 | awk '{print $4}')
if [ -n "$last" ] && [ ! -f $ST/$last.done ] && [ ! -f $ST/$last.norun ]; then
  out=$(ls -t results/MATRIX5/*/*/*.txt results/MATRIX5/*/extras/*/*.txt 2>/dev/null | head -1)
  if grep -q '^RESULT' "$out" 2>/dev/null; then touch $ST/$last.done; echo "[$(date '+%m-%d %H:%M:%S')]   ok $last (recorded at restart)" >> $L
  elif grep -q '^NORUN' "$out" 2>/dev/null; then touch $ST/$last.norun; echo "[$(date '+%m-%d %H:%M:%S')]   NORUN $last (recorded at restart)" >> $L; fi
fi
[ -n "$P" ] && { kill -CONT $P; kill $P; }; sleep 1
ps -eo pid,args | awk '$0 ~ /(memcal\.py|phasor_serve|zipmoe_serve|baseline_serve|fiddler_serve|mixoff_serve|check_tokens)/ && $0 !~ /awk/ {print $1}' | xargs -r kill
for c in /sys/fs/cgroup/ledger_bench/*/; do echo 1 | sudo -n tee ${c}cgroup.kill >/dev/null 2>&1; done
echo "[$(date '+%m-%d %H:%M:%S')] stage 5 restarted ($1)" >> $L
cp scripts/stage5.sh /tmp/claude-1000/stage5_run.sh
exec setsid bash /tmp/claude-1000/stage5_run.sh >> $ST/stage5_driver.out 2>&1
