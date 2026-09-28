#!/bin/bash
# Stage 7: MoE-APEX* (ASPLOS'26; reimplemented from the paper, bf16 mode) on
# Qwen3-30B: self-test, memcal at the four budgets, E1 (three workloads x four
# budgets), E3 (batch 4/8 at 45%).  Same caps,
# scrubbing and run() as stage 5.  Starts after stage 6 and 6b.
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/torch_env.sh; . scripts/memguard.sh
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log
ZPY=/home/thor/kcj/envs/zipmoe/bin/python; TPY=$TORCH_VENV/bin/python; OPY=/home/thor/kcj/envs/oldhf/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
settle(){ local i=0; until [ $(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo) -ge 100 ]; do   # no time limit: a run under
  i=$((i+1)); [ $i = 120 ] && say "waiting for >= 100 GiB available"; sleep 5; done; }   # outside memory pressure would say nothing
eval "$(sed -n '/^run(){/,/^}/p; /^capfor(){/,/^}/p' scripts/stage5.sh)"
eval "$(sed -n '/^cal(){/,/^}/p; /^over14(){/,/^}/p' scripts/stage_helpers.sh)"
until [ $(grep -c "=== stage 6 done ===" "$LOG") -ge 2 ]; do sleep 120; done     # stage 6, then 6b
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== stage 7 (MoE-APEX* on Qwen3-30B) start ==="
m=qwen30b; ck=/home/thor/kcj/models/qwen3_30b_a3b; gb=57.0; O=results/MATRIX5/$m
# the evaluation runs MoE-APEX* in bf16 mode: precision adaptation off (outputs
# equal the original model, as for every system here); LCU caching and the
# adaptive prefetcher are the paper's.  The mixed (int2) mode is not run here.
export SCRUB_GLOB="$ck/*.safetensors"
APEX(){ echo "$ZPY baselines_hf/baseline_serve.py --system apex --weights bf16 --checkpoint $ck --workload results/WORKLOADS/$1.json --budget-gib $2 --out $3.json"; }
settle; drop; timeout 3600 scripts/in_cgroup.sh prep max $(APEX mmlu 25.65 results/PREP/selftest/apex) --limit 2 > results/PREP/selftest/apex.log 2>&1
say "stage 7: selftest $(grep -h '^RESULT' results/PREP/selftest/apex.log | cut -c1-140)"
grep -q '^RESULT' results/PREP/selftest/apex.log || { say "stage 7: MoE-APEX* self-test FAILED"; exit 1; }
# memcal: the knob whose peak matches PHASOR's at each budget
for f in 0.25 0.45 0.65 1.08; do
  b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); c=$O/memcal/apex_$b.json
  T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$b.json'))['peak_gib'],2))")
  cal $c $T $b $(APEX mmlu {B} {OUT}) --limit 2
  say "memcal $m $f apex: $(tail -1 $c.log)"
done
# E1
for w in mmlu sharegpt longbench; do for f in 0.25 0.45 0.65 1.08; do
  nb=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); cal=$O/memcal/apex_$nb.json
  kb=$(python3 -c "import json;v=json.load(open('$cal'))['budget_gib'];print(v if v else 'none')" 2>/dev/null || echo none)
  T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$nb.json'))['peak_gib'],2))")
  if [ "$kb" = none ]; then   # no knob within PHASOR's peak: the smallest cache (1 GiB) under 1.4 x that peak (as FineMoE, stage 8)
    CAP_GIB=$(awk -v t=$T 'BEGIN{printf "%.2f", 1.4*t}') run "s7_${m}_${w}_apex_${nb}_min" 1 $O/$w/apex_$f $(APEX $w 1 $O/$w/apex_$f)
    over14 $O/$w/apex_$f $T $nb; continue
  fi
  CAP_GIB=$(capfor $nb) run "s5_${m}_${w}_apex_$nb" $kb $O/$w/apex_$f $(APEX $w $kb $O/$w/apex_$f)
done; done
# E1 retries, as for the other baselines (stage 5 extras): a calibrated run that hit
# the cap is retried with its knob at x0.85, x0.7, x0.55, stopping at the first that runs
for w in mmlu sharegpt longbench; do for f in 0.25 0.45 0.65 1.08; do
  nb=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}')
  grep -q "^NORUN.*oom-under-cap" $O/$w/apex_$f.txt 2>/dev/null || continue
  k0=$(python3 -c "import json;v=json.load(open('$O/memcal/apex_$nb.json'))['budget_gib'];print(v if v else 'none')" 2>/dev/null || echo none)
  [ "$k0" = none ] && continue
  for k in 0.85 0.7 0.55; do
    kb=$(awk -v a=$k0 -v k=$k 'BEGIN{printf "%.2f", a*k}'); o=$O/$w/apex_${f}_k$k
    CAP_GIB=$(capfor $nb) run "s5_${m}_${w}_apex_${nb}_k$k" $kb $o $(APEX $w $kb $o) && break
  done
done; done
# E3: batch 4 and 8 at 45% (MMLU, ShareGPT), E1's equal-memory knob, cap + 8 GiB for batch KV
b45=$(awk -v g=$gb 'BEGIN{printf "%.2f", g*0.45}'); mkdir -p $O/extras/e3
kb45=$(python3 -c "import json;v=json.load(open('$O/memcal/apex_$b45.json'))['budget_gib'];print(v if v else 'none')" 2>/dev/null || echo none)
if [ "$kb45" != none ]; then
  for B in 4 8; do for w in mmlu sharegpt; do
    o=$O/extras/e3/${w}_apex_b$B
    CAP_GIB=$(awk -v c=$(capfor $b45) 'BEGIN{printf "%.2f", c+8}') run "s5x_${m}_${w}_apex_b$B" $kb45 $o $(APEX $w $kb45 $o) --batch $B
  done; done
else
  for B in 4 8; do for w in mmlu sharegpt; do echo "NORUN budget=$b45 reason=no-setting-within-phasor-memory-at-45%" > $O/extras/e3/${w}_apex_b$B.txt; done; done
fi
python3 scripts/summarize_matrix5.py > results/MATRIX5/SUMMARY.md 2>>"$LOG"
git add -f baselines_hf/apex_hf.py baselines_hf/offload_hf.py baselines_hf/baseline_serve.py scripts/apex_build_store.py scripts/stage7_apex.sh \
  $O/*/apex_* $O/memcal/apex_* $O/extras/e3/*apex* results/MATRIX5/SUMMARY.md 2>/dev/null
git commit -q -m "MoE-APEX* (ASPLOS'26, reimplemented from HOBBIT) on Qwen3-30B: E1

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
say "=== stage 7 done ==="
