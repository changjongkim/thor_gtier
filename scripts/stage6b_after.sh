#!/bin/bash
# Stage 6b once more, after stage 9 (resume_chain.sh): the ZipMoE Qwen3 store
# rebuild, then ZipMoE's slower-SSD runs.  On 09-28 14:08 the rebuild's restart
# (after outside memory pressure stopped its first attempt) met the store that
# attempt had left half-written ("Tensor is offloaded twice"); every attempt now
# starts from an empty store (BEFORE_EACH).
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"; . scripts/torch_env.sh
L=results/PIPELINE/pipeline.log; ZPY=/home/thor/kcj/envs/zipmoe/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
gib(){ awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo; }
until grep -q "resume: stages 6b-9 finished" "$L"; do sleep 120; done
until [ $(gib) -ge 100 ]; do sleep 60; done
( exec 9>/tmp/gtier_pipeline.lock; flock 9
  say "stage 6b (after stage 9): rebuilding the ZipMoE Qwen3 store with a 4 GiB GPU pool ($(gib) GiB available)"
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  BEFORE_EACH='rm -rf /home/thor/kcj/ZipMoE-ICML26/offload/qwen3' timeout 14400 scripts/in_cgroup.sh prep max $ZPY scripts/zipmoe_serve.py \
    --model-type qwen3 --workload results/WORKLOADS/mmlu.json --budget-gib 4 --trace /home/thor/kcj/ZipMoE/trace/qwen3_mmlu_heldout.pt \
    --limit 1 --out results/PREP/zip_qwen3_rebuild.json > results/PREP/zip_qwen3_rebuild.log 2>&1
  say "stage 6b: ZipMoE Qwen3 store: $(grep -c '^RESULT' results/PREP/zip_qwen3_rebuild.log) result(s), $(du -sh /home/thor/kcj/ZipMoE-ICML26/offload/qwen3 2>/dev/null | cut -f1)" )
if grep -q '^RESULT' results/PREP/zip_qwen3_rebuild.log; then
  F6=0.25 S6=zipmoe SKIP_REPEATS=1 bash scripts/stage6_ssd_only.sh >> results/PIPELINE/stage6_driver.out 2>&1
  python3 scripts/summarize_matrix5.py > results/MATRIX5/SUMMARY.md 2>>"$L"
  git add -f results/MATRIX5/qwen30b/ssd results/MATRIX5/SUMMARY.md scripts/stage6b_after.sh scripts/in_cgroup.sh 2>/dev/null
  git commit -q -m "ZipMoE on the slower SSD (stage 6b)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
else
  say "stage 6b: rebuild FAILED ($(grep -hE 'HOSTGUARD|FATAL|Error' results/PREP/zip_qwen3_rebuild.log | tail -1 | cut -c1-120)); ZipMoE slower-SSD left out"
fi
say "=== stage 6b (after stage 9) done ==="
