#!/bin/bash
# Stage 13: MoE-Infinity (2024-08 release 350f0dd, SSD tier) on Qwen3-30B-A3B, the same set as
# every baseline (user, 09-29): tokens against stock, memcal to PHASOR's peak, E1 (3 workloads x
# 4 budgets, retries, the 1.4 x rule), E2 (MMLU 20-5% at its own setting), E3 (batch 4/8 at 45%).
# That release has no Qwen3 model code; third_party/moeinf2408_qwen3.patch adds it (the Mixtral
# block generalized to top-k with Qwen3's renormalization; predictor and prefetcher calls
# unchanged), built with the same compatibility flags, in its own venv (transformers 4.51.3,
# which has Qwen3-MoE and satisfies its < 5.0 pin).  After stage 12b, alone (pipeline lock).
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/torch_env.sh; . scripts/memguard.sh
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log; P=$R/results/PREP
MPY=/home/thor/kcj/envs/moeinf2408q/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
settle(){ local i=0; until [ $(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo) -ge ${SETTLE_GIB:-100} ]; do
  i=$((i+1)); [ $i = 120 ] && say "waiting for >= ${SETTLE_GIB:-100} GiB available"; sleep 5; done; }
eval "$(sed -n '/^run(){/,/^}/p; /^capfor(){/,/^}/p' scripts/stage5.sh)"
eval "$(sed -n '/^cal(){/,/^}/p; /^flag14(){/,/^}/p; /^over14(){/,/^}/p; /^because(){/,/^}/p' scripts/stage_helpers.sh)"
[ -n "${NOWAIT:-}" ] || until grep -q "=== stage 12b done" "$LOG"; do sleep 120; done
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== stage 13 (MoE-Infinity 2024-08 + Qwen3 port, Qwen3-30B: tokens, memcal, E1, E2, E3) start ==="
m=qwen30b; ck=/home/thor/kcj/models/qwen3_30b_a3b; gb=57.0; O=results/MATRIX5/$m; K=moeinf2408
OFF=/home/thor/kcj/offload_tmp/qwen30b_2408; mkdir -p $P/moeinf2408q
export SCRUB_GLOB="$ck/*.safetensors $OFF/*"
MI(){ echo "$MPY scripts/sota_serve.py --system moe-infinity --checkpoint $ck --workload results/WORKLOADS/$1.json --offload-dir $OFF"; }
rec(){ echo "$*" >> $P/MOE_INFINITY.md; }
$MPY -c "import moe_infinity,sys; from moe_infinity.models import SyncQwen3MoeSparseMoeBlock; sys.exit(0 if '/envs/moeinf2408q/' in moe_infinity.__file__ else 1)" \
  || { say "stage 13: the Qwen3-ported build does not import"; say "=== stage 13 done (no build) ==="; exit 0; }
# tokens (also builds its Qwen3 offload store on the first load): the two prompts of check_tokens.py, 24 new tokens
python3 - <<'PY'
import json
ps = ["Explain why the sky is blue in two sentences.", "Write a Python function that returns the n-th Fibonacci number."]
json.dump([{"name": f"tok{i}", "prompt": p, "max_new": 24} for i, p in enumerate(ps)], open("results/PREP/moeinf2408q/tok_prompts.json", "w"))
PY
settle; drop
BEFORE_EACH="rm -rf $OFF" timeout 10800 scripts/in_cgroup.sh prep max $MPY scripts/sota_serve.py --system moe-infinity --checkpoint $ck \
  --workload $P/moeinf2408q/tok_prompts.json --offload-dir $OFF --budget-gib 8 --max-new 24 --out $P/moeinf2408q/tok.json > $P/moeinf2408q/tok.log 2>&1
if ! grep -q '^RESULT' $P/moeinf2408q/tok.log; then
  why=$(grep -hE 'HOSTGUARD|Error|error' $P/moeinf2408q/tok.log | tail -1 | cut -c1-150)
  say "stage 13: MoE-Infinity (Qwen3 port) does not serve: $why"
  rec ""; rec "## Qwen3-30B (350f0dd + Qwen3 port): does not serve: $why (results/PREP/moeinf2408q/tok.log)"
  for w in mmlu sharegpt longbench; do for f in 0.25 0.45 0.65 1.08; do echo "NORUN budget=- reason=does-not-serve (results/PREP/MOE_INFINITY.md)" > $O/$w/${K}_$f.txt; done; done
  git add -f $P/MOE_INFINITY.md $P/moeinf2408q $O/*/${K}_* scripts/stage13_moeinf_qwen3.sh 2>/dev/null
  git commit -q -m "MoE-Infinity (2024-08 + Qwen3 port) on Qwen3-30B: does not serve (evidence)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
  say "=== stage 13 done (does not serve) ==="; exit 0
fi
TOK=$(python3 - <<'PY'
import json
try:
    a = json.load(open("results/PREP/moeinf2408q/tok.json"))["rows"]; b = json.load(open("results/PHASOR_HF/correct_stock.json"))
    ps = ["Explain why the sky is blue in two sentences.", "Write a Python function that returns the n-th Fibonacci number."]
    same = [sum(x == y for x, y in zip(r["out_ids"], b[p])) for r, p in zip(a, ps)]
    print("identical" if all(r["out_ids"] == b[p] for r, p in zip(a, ps)) else f"{same} of 24 tokens equal")
except Exception as e: print("n/a", e)
PY
)
say "stage 13: MoE-Infinity (Qwen3 port) vs stock tokens: $TOK"
rec ""; rec "## Qwen3-30B: 350f0dd + Qwen3 port (third_party/moeinf2408_qwen3.patch)"
rec "- Model code added (models/qwen3.py: the Mixtral block with top-k routing and Qwen3's norm_topk_prob; config"
rec "  parsing and module wiring); predictor, prefetcher, caching and the SSD tier are the release's. Built with the"
rec "  same compatibility flags; venv transformers 4.51.3. Tokens vs stock transformers (2 prompts x 24): $TOK."
# memcal (knob: its GPU expert cache via device_memory_ratio), E1
for f in 0.25 0.45 0.65 1.08; do
  b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); c=$O/memcal/${K}_$b.json
  T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$b.json'))['peak_gib'],2))")
  cal $c $T $b $(MI mmlu) --budget-gib {B} --out {OUT} --limit 2
  say "memcal $m $f $K: $(tail -1 $c.log)"
done
for w in mmlu sharegpt longbench; do for f in 0.25 0.45 0.65 1.08; do
  nb=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); o=$O/$w/${K}_$f
  T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$nb.json'))['peak_gib'],2))")
  kb=$(python3 -c "import json;v=json.load(open('$O/memcal/${K}_$nb.json'))['budget_gib'];print(v if v else 'none')" 2>/dev/null || echo none)
  if [ "$kb" = none ]; then
    CAP_GIB=$(awk -v t=$T 'BEGIN{printf "%.2f", 1.4*t}') run "s13_${m}_${w}_${K}_${nb}_min" 1 $o $(MI $w) --budget-gib 1 --out $o.json
    over14 $o $T $nb
  else
    CAP_GIB=$(capfor $nb) run "s13_${m}_${w}_${K}_$nb" $kb $o $(MI $w) --budget-gib $kb --out $o.json
    if grep -q "^NORUN.*oom-under-cap" $o.txt 2>/dev/null; then
      for k in 0.85 0.7 0.55; do
        kk=$(awk -v a=$kb -v k=$k 'BEGIN{printf "%.2f", a*k}'); ok=$O/$w/${K}_${f}_k$k
        CAP_GIB=$(capfor $nb) run "s13_${m}_${w}_${K}_${nb}_k$k" $kk $ok $(MI $w) --budget-gib $kk --out $ok.json && break
      done
    fi
  fi
  because $o $nb
  say "E1 $m $w $K $f: $(grep -hE '^(RESULT|NORUN)' $o.txt | tail -1 | cut -c1-150)"
done; done
# E3: batch 4 and 8 at 45% (MMLU, ShareGPT), E1's knob, run cap + 8 GiB
b45=$(awk -v g=$gb 'BEGIN{printf "%.2f", g*0.45}'); mkdir -p $O/extras/e3
kb45=$(python3 -c "import json;v=json.load(open('$O/memcal/${K}_$b45.json'))['budget_gib'];print(v if v else 'none')" 2>/dev/null || echo none)
for B in 4 8; do for w in mmlu sharegpt; do
  o=$O/extras/e3/${w}_${K}_b$B
  if [ "$kb45" = none ]; then echo "NORUN budget=$b45 reason=no-setting-within-phasor-memory-at-45%" > $o.txt; continue; fi
  CAP_GIB=$(awk -v c=$(capfor $b45) 'BEGIN{printf "%.2f", c+8}') run "s13x_${m}_${w}_${K}_b$B" $kb45 $o $(MI $w) --budget-gib $kb45 --batch $B --out $o.json
  because $o $b45
  say "E3 $w b$B $K: $(grep -hE '^(RESULT|NORUN)' $o.txt | tail -1 | cut -c1-150)"
done; done
# E2: MMLU 20/15/10/5% at its own setting (cache = budget), stop at the first budget it cannot run at
stop=""
for f in 0.20 0.15 0.10 0.05; do
  b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); o=$O/mmlu/${K}_$f
  T=$(python3 -c "import json;print(round(json.load(open('$O/mmlu/phasor_$f.json'))['peak_gib'],2))")
  if [ -n "$stop" ]; then echo "NORUN budget=$b reason=not-tried (did not run at $stop)" > $o.txt; continue; fi
  CAP_GIB=$(capfor $b) run "s13_${m}_mmlu_${K}_$b" $b $o $(MI mmlu) --budget-gib $b --out $o.json
  because $o $b; flag14 $o $T $b
  grep -q '^RESULT' $o.txt 2>/dev/null || stop=$f
  say "E2 $K $f: $(grep -hE '^(RESULT|NORUN)' $o.txt | tail -1 | cut -c1-150)"
done
rm -rf "${OFF:?}"; say "stage 13: removed its Qwen3 offload store (ours, regenerable)"
python3 scripts/summarize_matrix5.py > results/MATRIX5/SUMMARY.md 2>>"$LOG"
git add -f $P/MOE_INFINITY.md $P/moeinf2408q $O/*/${K}_* $O/memcal/${K}_* $O/extras/e3/*${K}* results/MATRIX5/SUMMARY.md \
  scripts/stage13_moeinf_qwen3.sh scripts/sota_serve.py third_party/moeinf2408_qwen3.patch 2>/dev/null
git commit -q -m "MoE-Infinity (2024-08 release + Qwen3 port) on Qwen3-30B: tokens, memcal, E1, E2, E3

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
say "=== stage 13 done ==="
