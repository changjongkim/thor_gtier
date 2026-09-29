#!/bin/bash
# After the hand-run MoE-Infinity Mixtral smoke (09-29 12:07), stage 12b (Mixtral MMLU E1), then
# stage 13 (Qwen3: tokens, memcal, E1, E2, E3), one after the other.
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
while pgrep -f "scratchpad/mi_smoke.sh" >/dev/null; do sleep 30; done
echo "[$(date '+%m-%d %H:%M:%S')] chain_moeinf: hand smoke finished ($(grep -h '^RESULT' results/PREP/moeinf2408/smoke.log | cut -c1-100)); stage 12b, then stage 13" >> results/PIPELINE/pipeline.log
NOWAIT=1 SETTLE_GIB=${SETTLE_GIB:-75} bash scripts/stage12b_moeinf_legacy.sh >> results/PIPELINE/stage12b2_driver.out 2>&1
NOWAIT=1 SETTLE_GIB=${SETTLE_GIB:-75} bash scripts/stage13_moeinf_qwen3.sh >> results/PIPELINE/stage13_driver.out 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] chain_moeinf: finished" >> results/PIPELINE/pipeline.log
