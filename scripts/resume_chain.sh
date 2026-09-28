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
say "resume: waiting for >= 100 GiB available ($(gib) GiB now), then stage 7, 8, 9 in order (6b after them)"
until [ $(gib) -ge 100 ]; do sleep 60; done
# stage 6b runs after stage 9 (scripts/stage6b_after.sh, waiting for the line below)
bash scripts/stage7_apex.sh >> results/PIPELINE/stage7_driver.out 2>&1
bash scripts/stage8_finemoe.sh >> results/PIPELINE/stage8_driver.out 2>&1
NOWAIT=1 bash scripts/stage9_apex_finemoe.sh >> results/PIPELINE/stage9_driver.out 2>&1
say "resume: stages 6b-9 finished"
