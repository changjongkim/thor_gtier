#!/bin/bash
# Stage 12b: MoE-Infinity at 350f0dd (2024-08-15), after stage 11q.  Stage 12's
# 48bb3bc (2025-02-13, a dev merge) builds but its Mixtral block passes the routed
# expert tensor itself as a dict key (models/mixtral.py:71 -> KeyError, 09-28 23:01);
# 350f0dd, the release before it, has the paper's predictor + prefetch path there
# and the same SSD tier (no preloading).  Copy of stage12_moeinf_legacy.sh.
# Stage 12: MoE-Infinity at its last release with an SSD tier (48bb3bc,
# 2025-02-13).  From c098c15 (2026-02-16) on, it copies every expert into its
# host pool at load (the version in results/PREP/MOE_INFINITY.md, which cannot
# run here); before, experts stayed in the offload store on the SSD and moved
# SSD -> host pool -> GPU on demand -- the "SSD offloading mode" ZipMoE (ICML'26)
# compared against.  That version supports Mixtral (not Qwen3: Qwen3 came with
# the preloading versions), so it joins the Mixtral generality check, MMLU
# 25/45/65%.  Unmodified; its host pool size is its build-time macro
# HOST_MEMORY_RATIO (default 0.8 of system memory, which on the unified pool is
# the same memory as the GPU cache), set to 0.04 (4.9 GiB of staging); memcal
# calibrates its runtime knob device_memory_ratio (the GPU expert cache).
# After stage 11, alone (pipeline lock).
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/torch_env.sh; . scripts/memguard.sh
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log; P=$R/results/PREP
SRC=/home/thor/kcj/MoE-Infinity-2408; V=/home/thor/kcj/envs/moeinf2408; MPY=$V/bin/python
OFF=/home/thor/kcj/offload_tmp/mixtral8x7b_2408
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
settle(){ local i=0; until [ $(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo) -ge ${SETTLE_GIB:-100} ]; do   # no time limit: a run under
  i=$((i+1)); [ $i = 120 ] && say "waiting for >= 100 GiB available"; sleep 5; done; }   # outside memory pressure would say nothing
eval "$(sed -n '/^run(){/,/^}/p; /^capfor(){/,/^}/p' scripts/stage5.sh)"
eval "$(sed -n '/^cal(){/,/^}/p; /^flag14(){/,/^}/p; /^over14(){/,/^}/p; /^because(){/,/^}/p' scripts/stage_helpers.sh)"
[ -n "${NOWAIT:-}" ] || until grep -q "=== stage 11q done" "$LOG"; do sleep 120; done
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== stage 12b (MoE-Infinity 2024-08 release 350f0dd, SSD tier, Mixtral MMLU) start ==="
m=mixtral8x7b; ck=/home/thor/kcj/models/mixtral8x7b_bf16; gb=87.0; O=results/MATRIX5/$m; w=mmlu
mkdir -p $P/moeinf2408
rec(){ echo "$*" >> $P/MOE_INFINITY.md; }
fail(){  # fail <why>: every cell NORUN with the cause, evidence in MOE_INFINITY.md
  say "stage 12b: $1"
  rec ""; rec "## Version 350f0dd (2024-08, SSD tier), Mixtral-8x7B: $1"
  for f in 0.25 0.45 0.65; do echo "NORUN budget=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}') reason=$(echo "$1" | tr ' ' '-' | cut -c1-80) (results/PREP/MOE_INFINITY.md)" > $O/$w/moeinf2408_$f.txt; done
  git add -f $P/MOE_INFINITY.md $P/moeinf2408 $O/$w/moeinf2408_* scripts/stage12b_moeinf_legacy.sh 2>/dev/null
  git commit -q -m "MoE-Infinity (2024-08 release, SSD tier) on Mixtral: $1

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
  say "=== stage 12b done ($1) ==="; exit 0; }
# disk: its offload store is a full copy of the checkpoint (~90 GB)
free_gb(){ df -BG --output=avail /home/thor/kcj | tail -1 | tr -dc 0-9; }
if [ $(free_gb) -lt 100 ]; then
  for gg in /home/thor/kcj/models/gguf_bf16/mixtral8x7b_bf16.gguf /home/thor/kcj/models/gguf_bf16/qwen30b_bf16.gguf; do
    [ $(free_gb) -ge 100 ] && break
    if [ -f $ST/ok_delete_gguf ] && [ -f "$gg" ]; then rm -f "$gg"; say "stage 12b: removed $gg (bf16 GGUF of stage 11, regenerable; approved) for disk"; fi
  done
  [ $(free_gb) -ge 100 ] || fail "not enough disk for its offload store ($(free_gb) GB free, needs ~100)"
fi
# build: the release at 48bb3bc, its own ops, in its own venv on the t26 torch
if ! $MPY -c "import moe_infinity,sys; sys.exit(0 if '/envs/moeinf2408/' in moe_infinity.__file__ else 1)" 2>/dev/null; then
  [ -d $SRC/.git ] || timeout 1800 git clone -q https://github.com/EfficientMoE/MoE-Infinity $SRC
  ( cd $SRC && git checkout -q 350f0dd ) || fail "checkout of 350f0dd failed"
  [ -x $MPY ] || { $TORCH_VENV/bin/python -m venv $V && echo "$TORCH_VENV/lib/python3.12/site-packages" > $V/lib/python3.12/site-packages/t26.pth; }
  # its requirements that the t26 venv does not satisfy (transformers >= 4.37.1, < 4.47; pydantic 1)
  timeout 3600 $MPY -m pip install -q "transformers>=4.37.1,<4.47" "pydantic==1.10.12" hjson py-cpuinfo ninja "accelerate<1.3" "optimum<1.24" "setuptools<75" > $P/moeinf2408/pip.log 2>&1
  timeout 3600 $MPY -m pip install -q --no-deps "peft==0.13.2" gekko >> $P/moeinf2408/pip.log 2>&1
  # build compatibility only (third_party/moeinf2408_build.patch): its log buffer constant assumes x86's
  # 80-bit long double (aarch64: 33 digits), and GCC 13's headers no longer pull <string> in transitively
  ( cd $SRC && git apply --check $R/third_party/moeinf2502_build.patch 2>/dev/null && git apply $R/third_party/moeinf2502_build.patch ) || true
  BUILD_CUDA_EXT=0 timeout 3600 $MPY -m pip install -q --no-deps --no-build-isolation "auto-gptq==0.7.1" >> $P/moeinf2408/pip.log 2>&1
  ( cd $SRC && BUILD_OPS=1 TORCH_CUDA_ARCH_LIST="11.0" MAX_JOBS=8 \
      CFLAGS="-include string -DHOST_MEMORY_RATIO=0.04" CXXFLAGS="-include string -DHOST_MEMORY_RATIO=0.04" NVCC_APPEND_FLAGS="-include string -DHOST_MEMORY_RATIO=0.04" \
      timeout 7200 $MPY -m pip install --no-deps --no-build-isolation . > $P/moeinf2408/build.log 2>&1 )
  $MPY -c "import moe_infinity,sys; from moe_infinity import MoE; sys.exit(0 if '/envs/moeinf2408/' in moe_infinity.__file__ else 1)" > $P/moeinf2408/import.log 2>&1 || fail "does not build or import on CUDA 13 / sm_110: $(grep -hE 'error|Error' $P/moeinf2408/build.log $P/moeinf2408/import.log | tail -1 | cut -c1-120)"
fi
rec ""; rec "## Version 350f0dd (2024-08-15): release with an SSD tier (48bb3bc, 2025-02, fails at its first Mixtral request)"
rec "- Experts stay in its offload store on the SSD and move SSD -> host pool -> GPU on demand; the preloading"
rec "  (\"Moving sparse parameters to CPU\") arrived in c098c15 (2026-02-16). Supports Mixtral, not Qwen3."
rec "- Built in its own venv (transformers 4.46.3, < 4.47 as it requires) with build-compatibility fixes only:"
rec "  \`-include string\` (GCC 13) and its log buffer constant kMaxNumericSize 32 -> 48 (aarch64 long double;"
rec "  third_party/moeinf2408_build.patch). Caching, prefetching and the data path are unchanged. HOST_MEMORY_RATIO (its build-time"
rec "  host pool size, default 0.8 of system memory) = 0.04; memcal calibrates device_memory_ratio."
export SCRUB_GLOB="$ck/*.safetensors $OFF/*"
MI(){ echo "$MPY scripts/sota_serve.py --system moe-infinity --checkpoint $ck --workload results/WORKLOADS/$w.json --offload-dir $OFF"; }
# smoke (also builds its offload store on the first load)
settle; drop
timeout 14400 scripts/in_cgroup.sh prep max $(MI) --budget-gib 8 --limit 2 --out $P/moeinf2408/smoke.json > $P/moeinf2408/smoke.log 2>&1
grep -q '^RESULT' $P/moeinf2408/smoke.log || fail "does not serve: $(grep -hE 'HOSTGUARD|Error|error' $P/moeinf2408/smoke.log | tail -1 | cut -c1-120)"
say "stage 12b: smoke $(grep -h '^RESULT' $P/moeinf2408/smoke.log | cut -c1-140)"
rec "- Smoke (2 MMLU prompts, 8 GiB GPU cache): $(grep -h '^RESULT' $P/moeinf2408/smoke.log | cut -c1-160)"
for f in 0.25 0.45 0.65; do
  b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); c=$O/memcal/moeinf2408_$b.json
  T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$b.json'))['peak_gib'],2))")
  cal $c $T $b $(MI) --budget-gib {B} --out {OUT} --limit 2
  say "memcal $m $f moeinf2408: $(tail -1 $c.log)"
done
for f in 0.25 0.45 0.65; do
  nb=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); o=$O/$w/moeinf2408_$f
  T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$nb.json'))['peak_gib'],2))")
  kb=$(python3 -c "import json;v=json.load(open('$O/memcal/moeinf2408_$nb.json'))['budget_gib'];print(v if v else 'none')" 2>/dev/null || echo none)
  if [ "$kb" = none ]; then
    CAP_GIB=$(awk -v t=$T 'BEGIN{printf "%.2f", 1.4*t}') run "s12b_${m}_${w}_moeinf2408_${nb}_min" 1 $o $(MI) --budget-gib 1 --out $o.json
    over14 $o $T $nb
  else
    CAP_GIB=$(capfor $nb) run "s12b_${m}_${w}_moeinf2408_$nb" $kb $o $(MI) --budget-gib $kb --out $o.json
    if grep -q "^NORUN.*oom-under-cap" $o.txt 2>/dev/null; then
      for k in 0.85 0.7 0.55; do
        kk=$(awk -v a=$kb -v k=$k 'BEGIN{printf "%.2f", a*k}'); ok=$O/$w/moeinf2408_${f}_k$k
        CAP_GIB=$(capfor $nb) run "s12b_${m}_${w}_moeinf2408_${nb}_k$k" $kk $ok $(MI) --budget-gib $kk --out $ok.json && break
      done
    fi
  fi
  because $o $nb
  say "E1 $m moeinf2408 $f: $(grep -hE '^(RESULT|NORUN)' $o.txt | tail -1 | cut -c1-150)"
done
cp -f $O/$w/phasor_1.08.txt $O/$w/moeinf2408_1.08.txt 2>/dev/null && sed -i -n '/^NORUN/p' $O/$w/moeinf2408_1.08.txt
python3 scripts/summarize_matrix5.py > results/MATRIX5/SUMMARY.md 2>>"$LOG"
git add -f $P/MOE_INFINITY.md $P/moeinf2408 $O/$w/moeinf2408_* $O/memcal/moeinf2408_* results/MATRIX5/SUMMARY.md scripts/stage12b_moeinf_legacy.sh 2>/dev/null
git commit -q -m "MoE-Infinity (2024-08 release 350f0dd, SSD tier) in the Mixtral MMLU generality check

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
rm -rf "${OFF:?}"; say "stage 12b: removed its offload store (ours, regenerable) for disk"
say "=== stage 12b done ==="
