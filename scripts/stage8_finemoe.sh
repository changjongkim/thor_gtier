#!/bin/bash
# Stage 8: FineMoE (EuroSys'26), its released code ported to Qwen3-30B-A3B.
#  1 load/serve test (2 MMLU prompts, host guard on): FineMoE pins every expert
#    in host memory and loads the whole checkpoint to CPU first; on the unified
#    pool that may not fit -- then results/PREP/FINEMOE.md records why and stop
#  2 token check against stock transformers
#  3 expert maps from the workloads other than the evaluated one
#  4 fidelity: FineMoE's own measure() vs our runner, same prompts and cache
#  5 memcal, E1 (3 workloads x 4 budgets); a budget whose memcal finds no knob is
#    tried at the smallest cache under 1.4 x PHASOR's peak, else NORUN (> 1.4x)
#  6 nominal reference (FineMoE's own device_memory_ratio=0.8), MMLU
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/torch_env.sh; . scripts/memguard.sh
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log; P=$R/results/PREP
ZPY=/home/thor/kcj/envs/zipmoe/bin/python; TPY=$TORCH_VENV/bin/python; OPY=/home/thor/kcj/envs/oldhf/bin/python
FPY=/home/thor/kcj/envs/finemoe/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
settle(){ local i; for i in $(seq 1 36); do
  [ $(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo) -ge 100 ] && return 0; sleep 5; done; }
eval "$(sed -n '/^run(){/,/^}/p; /^capfor(){/,/^}/p' scripts/stage5.sh)"
until grep -qE "=== stage 7 done ===|stage 7: MoE-APEX\* self-test FAILED" "$LOG"; do sleep 120; done
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== stage 8 (FineMoE on Qwen3-30B) start ==="
m=qwen30b; ck=/home/thor/kcj/models/qwen3_30b_a3b; gb=57.0; O=results/MATRIX5/$m; F=results/FINEMOE; mkdir -p $F/maps $P/finemoe
export SCRUB_GLOB="$ck/*.safetensors"
FM(){ echo "$FPY scripts/finemoe_serve.py --checkpoint $ck"; }
# 1 load/serve test
settle; drop
timeout 7200 scripts/in_cgroup.sh prep max $(FM) --workload results/WORKLOADS/mmlu.json --budget-gib 4 --limit 2 --out $P/finemoe/smoke.json \
  > $P/finemoe/smoke.log 2>&1
if ! grep -q '^RESULT' $P/finemoe/smoke.log; then
  say "stage 8: FineMoE could not serve on this device: $(grep -hE 'HOSTGUARD|Error|error' $P/finemoe/smoke.log | tail -1 | cut -c1-160)"
  python3 scripts/finemoe_report.py > $P/FINEMOE.md 2>>"$LOG"
  git add -f $P/FINEMOE.md $P/finemoe scripts/finemoe_serve.py scripts/finemoe_report.py scripts/stage8_finemoe.sh third_party/finemoe_qwen3_sm110.patch 2>/dev/null
  git commit -q -m "FineMoE (EuroSys'26) on the unified pool: does not serve (evidence)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
  say "=== stage 8 done (FineMoE cannot run here) ==="; exit 0
fi
say "stage 8: smoke $(grep -h '^RESULT' $P/finemoe/smoke.log | cut -c1-140)"
# 2 token check: the two prompts of check_tokens.py, 24 new tokens, against stock
python3 - <<'PY'
import json
ps = ["Explain why the sky is blue in two sentences.", "Write a Python function that returns the n-th Fibonacci number."]
json.dump([{"name": f"tok{i}", "prompt": p, "max_new": 24} for i, p in enumerate(ps)], open("results/PREP/finemoe/tok_prompts.json", "w"))
PY
settle; drop
timeout 3600 scripts/in_cgroup.sh prep max $(FM) --workload $P/finemoe/tok_prompts.json --budget-gib 8 --max-new 24 --out $P/finemoe/tok.json > $P/finemoe/tok.log 2>&1
TOK=$(python3 - <<'PY'
import json
try:
    a = json.load(open("results/PREP/finemoe/tok.json"))["rows"]; b = json.load(open("results/PHASOR_HF/correct_stock.json"))
    ps = ["Explain why the sky is blue in two sentences.", "Write a Python function that returns the n-th Fibonacci number."]
    same = [r["out_ids"] == b[p] for r, p in zip(a, ps)]
    print("identical" if all(same) else "DIFF " + str([sum(x == y for x, y in zip(r["out_ids"], b[p])) for r, p in zip(a, ps)]))
except Exception as e: print("n/a", e)
PY
)
say "stage 8: FineMoE vs stock tokens: $TOK"
# 3 expert maps from the other workloads (16 prompts each)
for w in mmlu sharegpt longbench; do
  d=$F/maps/${m}_$w; [ -s $d/.done ] && continue
  others=""; for o in mmlu sharegpt longbench; do [ $o = $w ] || others="$others results/WORKLOADS/$o.json"; done
  python3 - $others > $F/maps/${m}_$w.heldout.json <<'PY'
import json, sys
out = []
for p in sys.argv[1:]: out += json.load(open(p))[:16]
print(json.dumps(out))
PY
  settle; drop
  timeout 10800 scripts/in_cgroup.sh prep max $(FM) --collect $d --workload $F/maps/${m}_$w.heldout.json > $d.log 2>&1 && touch $d/.done
  say "stage 8: maps for $w: $(grep -h COLLECTED $d.log | cut -c1-80)"
done
# 4 fidelity: FineMoE's measure() (demo/eval.py) vs our runner, 8 MMLU prompts, same cache and maps
settle; drop
timeout 3600 scripts/in_cgroup.sh prep max $FPY scripts/finemoe_fidelity.py $ck results/WORKLOADS/mmlu.json $F/maps/${m}_mmlu 25.65 8 $F/fidelity_theirs.json > $F/fidelity_theirs.log 2>&1
settle; drop
timeout 3600 scripts/in_cgroup.sh prep max $(FM) --workload results/WORKLOADS/mmlu.json --budget-gib 25.65 --maps $F/maps/${m}_mmlu --limit 8 --out $F/fidelity_ours.json > $F/fidelity_ours.log 2>&1
say "stage 8: fidelity theirs $(python3 -c "import json;print(round(json.load(open('$F/fidelity_theirs.json'))['request_s'],3))" 2>/dev/null) s, ours $(python3 -c "import json;print(round(json.load(open('$F/fidelity_ours.json'))['request_s'],3))" 2>/dev/null) s"
# 5 memcal and E1
for f in 0.25 0.45 0.65 1.08; do
  b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); c=$O/memcal/finemoe_$b.json
  T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$b.json'))['peak_gib'],2))")
  [ -s $c ] || { settle; drop; timeout 14400 python3 scripts/memcal.py $T $b $c -- $(FM) --workload results/WORKLOADS/mmlu.json --maps $F/maps/${m}_mmlu --budget-gib {B} --limit 2 --out {OUT} > $c.log 2>&1; }
  say "memcal $m $f finemoe: $(tail -1 $c.log)"
done
for w in mmlu sharegpt longbench; do for f in 0.25 0.45 0.65 1.08; do
  nb=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); cal=$O/memcal/finemoe_$nb.json
  T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$nb.json'))['peak_gib'],2))")
  kb=$(python3 -c "import json;v=json.load(open('$cal'))['budget_gib'];print(v if v else 'none')" 2>/dev/null || echo none)
  o=$O/$w/finemoe_$f
  if [ "$kb" != none ]; then
    CAP_GIB=$(capfor $nb) run "s8_${m}_${w}_finemoe_$nb" $kb $o $(FM) --workload results/WORKLOADS/$w.json --maps $F/maps/${m}_$w --budget-gib $kb --out $o.json
  else   # no knob within PHASOR's peak: the smallest cache (1 GiB) under 1.4 x that peak
    CAP_GIB=$(awk -v t=$T 'BEGIN{printf "%.2f", 1.4*t}') run "s8_${m}_${w}_finemoe_$nb" 1 $o $(FM) --workload results/WORKLOADS/$w.json --maps $F/maps/${m}_$w --budget-gib 1 --out $o.json
    if grep -q '^NORUN' $o.txt 2>/dev/null; then sed -i 's/reason=oom-under-cap/reason=exceeds-1.4x-phasor-peak/' $o.txt; fi
    if grep -q '^RESULT' $o.txt 2>/dev/null; then pk=$(grep -o 'peak_gib=[0-9.]* compute' $o.txt | grep -o '[0-9.]*'); \
      awk -v p=$pk -v t=$T 'BEGIN{exit !(p>1.4*t)}' && echo "NORUN budget=$nb reason=exceeds-1.4x-phasor-peak (peak $pk vs $T)" >> $o.txt; fi
  fi
done; done
# 6 nominal reference (FineMoE sizes its GPU cache from free memory itself)
settle; drop
timeout 7200 scripts/in_cgroup.sh prep max $(FM) --workload results/WORKLOADS/mmlu.json --maps $F/maps/${m}_mmlu --device-memory-ratio 0.8 --out $O/mmlu/finemoe_nominal.json > $O/mmlu/finemoe_nominal.txt 2>&1
python3 scripts/finemoe_report.py > $P/FINEMOE.md 2>>"$LOG"
python3 scripts/summarize_matrix5.py > results/MATRIX5/SUMMARY.md 2>>"$LOG"
git add -f $P/FINEMOE.md $P/finemoe $F $O/*/finemoe_* $O/memcal/finemoe_* results/MATRIX5/SUMMARY.md scripts/finemoe_serve.py \
  scripts/finemoe_fidelity.py scripts/finemoe_report.py scripts/stage8_finemoe.sh third_party/finemoe_qwen3_sm110.patch 2>/dev/null
git commit -q -m "FineMoE (EuroSys'26) on Qwen3-30B: tokens, maps, fidelity, E1

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
say "=== stage 8 done ==="
