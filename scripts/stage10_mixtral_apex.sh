#!/bin/bash
# Stage 10: MoE-APEX* (bf16 mode) in the Mixtral-8x7B generality check -- MMLU
# at 25/45/65%, as the four other bf16 systems (stage 5): memcal to PHASOR's
# measured peak, E1, E1 retries, the 1.4 x rule for a budget without a setting.
# 108%: not run for any system (host safety ceiling), recorded the same way.
# Runs after stage 6b (after stage 9), alone (pipeline lock).
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/torch_env.sh; . scripts/memguard.sh
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log
ZPY=/home/thor/kcj/envs/zipmoe/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
settle(){ local i=0; until [ $(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo) -ge 100 ]; do   # no time limit: a run under
  i=$((i+1)); [ $i = 120 ] && say "waiting for >= 100 GiB available"; sleep 5; done; }   # outside memory pressure would say nothing
eval "$(sed -n '/^run(){/,/^}/p; /^capfor(){/,/^}/p' scripts/stage5.sh)"
eval "$(sed -n '/^cal(){/,/^}/p; /^flag14(){/,/^}/p; /^over14(){/,/^}/p; /^because(){/,/^}/p' scripts/stage_helpers.sh)"
until grep -q "=== stage 6b (after stage 9) done ===" "$LOG"; do sleep 120; done
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== stage 10 (MoE-APEX* on Mixtral-8x7B, MMLU) start ==="
m=mixtral8x7b; ck=/home/thor/kcj/models/mixtral8x7b_bf16; gb=87.0; O=results/MATRIX5/$m; w=mmlu
export SCRUB_GLOB="$ck/*.safetensors"
APEX(){ echo "$ZPY baselines_hf/baseline_serve.py --system apex --weights bf16 --checkpoint $ck --workload results/WORKLOADS/$1.json --budget-gib $2 --out $3.json"; }
mkdir -p results/PREP/selftest
settle; drop; timeout 7200 scripts/in_cgroup.sh prep max $(APEX mmlu 21.75 results/PREP/selftest/apex_mixtral) --limit 2 > results/PREP/selftest/apex_mixtral.log 2>&1
say "stage 10: selftest $(grep -h '^RESULT' results/PREP/selftest/apex_mixtral.log | cut -c1-140)"
if ! grep -q '^RESULT' results/PREP/selftest/apex_mixtral.log; then
  for f in 0.25 0.45 0.65; do nb=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}')
    echo "NORUN budget=$nb reason=self-test-failed (results/PREP/selftest/apex_mixtral.log)" > $O/$w/apex_$f.txt; done
else
  for f in 0.25 0.45 0.65; do
    b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); c=$O/memcal/apex_$b.json
    T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$b.json'))['peak_gib'],2))")
    cal $c $T $b $ZPY baselines_hf/baseline_serve.py --system apex --weights bf16 --checkpoint $ck --workload results/WORKLOADS/mmlu.json --budget-gib {B} --out {OUT} --limit 2
    say "memcal $m $f apex: $(tail -1 $c.log)"
  done
  for f in 0.25 0.45 0.65; do
    nb=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); o=$O/$w/apex_$f
    kb=$(python3 -c "import json;v=json.load(open('$O/memcal/apex_$nb.json'))['budget_gib'];print(v if v else 'none')" 2>/dev/null || echo none)
    T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$nb.json'))['peak_gib'],2))")
    if [ "$kb" = none ]; then   # no knob within PHASOR's peak: the smallest cache (1 GiB) under 1.4 x that peak
      CAP_GIB=$(awk -v t=$T 'BEGIN{printf "%.2f", 1.4*t}') run "s10_${m}_${w}_apex_${nb}_min" 1 $o $(APEX $w 1 $o)
      over14 $o $T $nb
    else
      CAP_GIB=$(capfor $nb) run "s10_${m}_${w}_apex_$nb" $kb $o $(APEX $w $kb $o)
      if grep -q "^NORUN.*oom-under-cap" $o.txt 2>/dev/null; then   # E1 retries, as for the other baselines
        for k in 0.85 0.7 0.55; do
          kk=$(awk -v a=$kb -v k=$k 'BEGIN{printf "%.2f", a*k}'); ok=$O/$w/apex_${f}_k$k
          CAP_GIB=$(capfor $nb) run "s10_${m}_${w}_apex_${nb}_k$k" $kk $ok $(APEX $w $kk $ok) && break
        done
      fi
    fi
    because $o $nb
    say "E1 $m apex $f: $(grep -hE '^(RESULT|NORUN)' $o.txt | tail -1 | cut -c1-150)"
  done
fi
nb=$(awk -v g=$gb 'BEGIN{printf "%.2f", g*1.08}')
cp -f $O/$w/phasor_1.08.txt $O/$w/apex_1.08.txt 2>/dev/null && sed -i -n '/^NORUN/p' $O/$w/apex_1.08.txt
[ -s $O/$w/apex_1.08.txt ] || echo "NORUN budget=$nb reason=host-safety-ceiling (whole bf16 model resident; not run for any system)" > $O/$w/apex_1.08.txt
python3 scripts/summarize_matrix5.py > results/MATRIX5/SUMMARY.md 2>>"$LOG"
git add -f scripts/stage10_mixtral_apex.sh $O/$w/apex_* $O/memcal/apex_* results/PREP/selftest/apex_mixtral* results/MATRIX5/SUMMARY.md 2>/dev/null
git commit -q -m "MoE-APEX* in the Mixtral-8x7B generality check (MMLU)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
say "=== stage 10 done ==="
