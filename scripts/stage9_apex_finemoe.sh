#!/bin/bash
# Stage 9: MoE-APEX* (bf16 mode) and FineMoE in the paper's other experiments on
# Qwen3-30B, after stage 8, alone (pipeline lock).  Same run(), caps, scrubbing
# and host guard as stage 5.
#  E2       MMLU at 20/15/10/5%, each system at its own setting (cache = budget),
#           as the other systems' E2 in stage 5 (no memcal there); a system stops
#           at the first budget it cannot run at.  A run whose peak exceeds 1.4 x
#           PHASOR's E2 peak at that budget is recorded as not fitting.
#  nominal  MoE-APEX* at its own setting for 25/45/65/108% (the other baselines'
#           reference rows); at 5-20% its E2 runs are that setting (same command
#           and cap, as in stage 5).  FineMoE sizes its cache from free memory
#           itself: stage 8's device_memory_ratio run is its reference.
#  gaps     every E1/E3 cell of the two systems that ended without a result gets
#           its cause recorded.
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/torch_env.sh; . scripts/memguard.sh
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log
ZPY=/home/thor/kcj/envs/zipmoe/bin/python; FPY=/home/thor/kcj/envs/finemoe/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
settle(){ local i=0; until [ $(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo) -ge 100 ]; do   # no time limit: a run under
  i=$((i+1)); [ $i = 120 ] && say "waiting for >= 100 GiB available"; sleep 5; done; }   # outside memory pressure would say nothing
eval "$(sed -n '/^run(){/,/^}/p; /^capfor(){/,/^}/p' scripts/stage5.sh)"
eval "$(sed -n '/^flag14(){/,/^}/p; /^over14(){/,/^}/p; /^because(){/,/^}/p' scripts/stage_helpers.sh)"
[ -n "${NOWAIT:-}" ] || until grep -q "=== stage 8 done" "$LOG"; do sleep 120; done   # NOWAIT: run by resume_chain.sh after stage 8
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== stage 9 (MoE-APEX* and FineMoE: E2, nominal, gaps) start ==="
m=qwen30b; ck=/home/thor/kcj/models/qwen3_30b_a3b; gb=57.0; O=results/MATRIX5/$m; F=results/FINEMOE
export SCRUB_GLOB="$ck/*.safetensors"
APEX(){ echo "$ZPY baselines_hf/baseline_serve.py --system apex --weights bf16 --checkpoint $ck --workload results/WORKLOADS/$1.json --budget-gib $2 --out $3.json"; }
FM(){ echo "$FPY scripts/finemoe_serve.py --checkpoint $ck --workload results/WORKLOADS/$1.json --maps $F/maps/${m}_$1 --budget-gib $2 --out $3.json"; }
ok_apex=0; grep -q '^RESULT' results/PREP/selftest/apex.log 2>/dev/null && ok_apex=1
ok_fm=0; grep -q '^RESULT' results/PREP/finemoe/smoke.log 2>/dev/null && ok_fm=1
# E2
for s in apex finemoe; do
  stop=""
  for f in 0.20 0.15 0.10 0.05; do
    b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); o=$O/mmlu/${s}_$f
    T=$(python3 -c "import json;print(round(json.load(open('$O/mmlu/phasor_$f.json'))['peak_gib'],2))")
    if [ -n "$stop" ]; then echo "NORUN budget=$b reason=not-tried (did not run at $stop)" > $o.txt; continue; fi
    if [ $s = apex ] && [ $ok_apex = 0 ]; then echo "NORUN budget=$b reason=self-test-failed (results/PREP/selftest/apex.log)" > $o.txt; stop=$f; continue; fi
    if [ $s = finemoe ] && [ $ok_fm = 0 ]; then echo "NORUN budget=$b reason=does-not-serve-on-this-device (results/PREP/FINEMOE.md)" > $o.txt; stop=$f; continue; fi
    # every system at an E2 budget has the same cap (stage 5: capfor, 1.05 x (budget + 6 GiB));
    # FineMoE, which pins every expert in host memory, is held to 1.4 x PHASOR's peak instead
    if [ $s = apex ]; then cap=$(capfor $b); else cap=$(awk -v t=$T 'BEGIN{printf "%.2f", 1.4*t}'); fi
    if [ $s = apex ]; then cmd=$(APEX mmlu $b $o); else cmd=$(FM mmlu $b $o); fi
    CAP_GIB=$cap run "s9_${m}_mmlu_${s}_$b" $b $o $cmd
    because $o $b; flag14 $o $T $b
    grep -q '^RESULT' $o.txt 2>/dev/null || stop=$f
    say "E2 $s $f: $(grep -hE '^(RESULT|NORUN)' $o.txt | tail -1 | cut -c1-150)"
  done
done
# nominal: MoE-APEX* with its cache at the budget
if [ $ok_apex = 1 ]; then
  for f in 0.20 0.15 0.10 0.05; do   # = its E2 run (same command and cap)
    [ -f $O/mmlu/apex_$f.txt ] || continue
    cp -f $O/mmlu/apex_$f.txt $O/mmlu/apex_${f}_nominal.txt; echo "NOTE same-run-as apex_$f (E2 runs each system at its own setting)" >> $O/mmlu/apex_${f}_nominal.txt
    [ -f $O/mmlu/apex_$f.json ] && cp -f $O/mmlu/apex_$f.json $O/mmlu/apex_${f}_nominal.json
  done
  for f in 0.25 0.45 0.65 1.08; do
    nb=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); o=$O/mmlu/apex_${f}_nominal
    CAP_GIB=$(capfor $nb) run "s9_${m}_mmlu_apex_nominal_$nb" $nb $o $(APEX mmlu $nb $o)
    because $o $nb
    say "nominal apex $f: $(grep -hE '^(RESULT|NORUN)' $o.txt | tail -1 | cut -c1-150)"
  done
fi
# gaps: E1 and E3 cells of both systems that ended without a result or a NORUN line
# (a failed run is run once more, as run() allows two attempts; then its cause is
# recorded), and the cells of a system that could not start at all
knob(){ python3 -c "import json;v=json.load(open('$O/memcal/$1_$2.json'))['budget_gib'];print(v if v else 'none')" 2>/dev/null || echo none; }
b45=$(awk -v g=$gb 'BEGIN{printf "%.2f", g*0.45}')
for s in apex finemoe; do
  ok=$ok_apex; why="self-test-failed (results/PREP/selftest/apex.log)"; [ $s = finemoe ] && { ok=$ok_fm; why="does-not-serve-on-this-device (results/PREP/FINEMOE.md)"; }
  for w in mmlu sharegpt longbench; do for f in 0.25 0.45 0.65 1.08; do
    nb=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); o=$O/$w/${s}_$f
    if [ $ok = 0 ]; then [ -f $o.txt ] || echo "NORUN budget=$nb reason=$why" > $o.txt; continue; fi
    [ -f $o.txt ] && ! grep -qE '^(RESULT|NORUN)' $o.txt || continue
    T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$nb.json'))['peak_gib'],2))"); kb=$(knob $s $nb)
    if [ $s = apex ]; then nm=s5_${m}_${w}_apex_$nb; [ $kb = none ] && nm=s7_${m}_${w}_apex_${nb}_min; else nm=s8_${m}_${w}_finemoe_$nb; fi
    if [ $kb = none ]; then cap=$(awk -v t=$T 'BEGIN{printf "%.2f", 1.4*t}'); k=1; else cap=$(capfor $nb); k=$kb; fi
    if [ $s = apex ]; then cmd=$(APEX $w $k $o); else cmd=$(FM $w $k $o); fi
    say "E1 $s $w $f ended without a result: run once more"
    CAP_GIB=$cap run $nm $k $o $cmd; [ $kb = none ] && over14 $o $T $nb; because $o $nb
  done; done
  for B in 4 8; do for w in mmlu sharegpt; do
    o=$O/extras/e3/${w}_${s}_b$B
    if [ $ok = 0 ]; then [ -f $o.txt ] || echo "NORUN budget=$b45 reason=$why" > $o.txt; continue; fi
    [ -f $o.txt ] && ! grep -qE '^(RESULT|NORUN)' $o.txt || continue
    kb=$(knob $s $b45); [ $kb = none ] && continue
    if [ $s = apex ]; then nm=s5x_${m}_${w}_apex_b$B; cmd=$(APEX $w $kb $o); else nm=s8x_${m}_${w}_finemoe_b$B; cmd=$(FM $w $kb $o); fi
    say "E3 $s $w b$B ended without a result: run once more"
    CAP_GIB=$(awk -v c=$(capfor $b45) 'BEGIN{printf "%.2f", c+8}') run $nm $kb $o $cmd --batch $B; because $o $b45
  done; done
done
python3 scripts/summarize_matrix5.py > results/MATRIX5/SUMMARY.md 2>>"$LOG"
git add -f scripts/stage9_apex_finemoe.sh scripts/stage_helpers.sh $O/mmlu/apex_* $O/mmlu/finemoe_* $O/*/apex_* $O/*/finemoe_* \
  $O/extras/e3/*apex* $O/extras/e3/*finemoe* results/MATRIX5/SUMMARY.md 2>/dev/null
git commit -q -m "MoE-APEX* and FineMoE: E2 small budgets, MoE-APEX* own-setting references, recorded causes

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
say "=== stage 9 done ==="
