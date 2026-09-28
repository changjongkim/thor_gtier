#!/bin/bash
# The rest of the pipeline, one stage after another, once the host has its
# memory back (09-28: an IDE language server outside the experiments grew to
# 60-110 GB; stage 6b's store rebuild and stage 7's self-test met the host
# guard, and stage 8 was stopped before its load test):
#   6b  ZipMoE Qwen3 store rebuild (4 GiB GPU pool), then its slower-SSD runs
#   7   MoE-APEX*, 8 FineMoE, 9 their E2 / own-setting references / recorded causes
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"; . scripts/torch_env.sh
L=results/PIPELINE/pipeline.log; ZPY=/home/thor/kcj/envs/zipmoe/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
gib(){ awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo; }
say "resume: waiting for >= 100 GiB available ($(gib) GiB now), then stage 6b, 7, 8, 9 in order"
until [ $(gib) -ge 100 ]; do sleep 60; done
( exec 9>/tmp/gtier_pipeline.lock; flock 9
  say "stage 6b: rebuilding the ZipMoE Qwen3 store with a 4 GiB GPU pool ($(gib) GiB available)"
  rm -rf "/home/thor/kcj/ZipMoE-ICML26/offload/qwen3"
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  timeout 10800 scripts/in_cgroup.sh prep max $ZPY scripts/zipmoe_serve.py --model-type qwen3 --workload results/WORKLOADS/mmlu.json \
    --budget-gib 4 --trace /home/thor/kcj/ZipMoE/trace/qwen3_mmlu_heldout.pt --limit 1 --out results/PREP/zip_qwen3_rebuild.json > results/PREP/zip_qwen3_rebuild.log 2>&1
  say "stage 6b: ZipMoE Qwen3 store: $(grep -c '^RESULT' results/PREP/zip_qwen3_rebuild.log) result(s), $(du -sh /home/thor/kcj/ZipMoE-ICML26/offload/qwen3 2>/dev/null | cut -f1)" )
if grep -q '^RESULT' results/PREP/zip_qwen3_rebuild.log; then
  F6=0.25 S6=zipmoe SKIP_REPEATS=1 bash scripts/stage6_ssd_only.sh >> results/PIPELINE/stage6_driver.out 2>&1
else
  say "stage 6b: rebuild FAILED ($(grep -hE 'HOSTGUARD|Error' results/PREP/zip_qwen3_rebuild.log | tail -1 | cut -c1-120)); ZipMoE slower-SSD left out"
fi
bash scripts/stage7_apex.sh >> results/PIPELINE/stage7_driver.out 2>&1
bash scripts/stage8_finemoe.sh >> results/PIPELINE/stage8_driver.out 2>&1
NOWAIT=1 bash scripts/stage9_apex_finemoe.sh >> results/PIPELINE/stage9_driver.out 2>&1
say "resume: stages 6b-9 finished"
