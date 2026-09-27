#!/bin/bash
# Stage 6: repeats.  Every E1 cell at 45% of the model (the headline budget),
# all workloads and every system that ran there, is run twice more (r2, r3),
# so each has three measurements for a mean and spread.  Same commands, knobs,
# caps and page-cache scrubbing as stage 5 (its functions are reused as is).
# Mixtral first, while its conversion stores exist; then its ZipMoE store is
# removed for disk and Qwen3 runs (ZipMoE rebuilds its Qwen3 store on first use).
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/torch_env.sh; . scripts/memguard.sh
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log
ZPY=/home/thor/kcj/envs/zipmoe/bin/python; TPY=$TORCH_VENV/bin/python; OPY=/home/thor/kcj/envs/oldhf/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
# stage 5's run(), capfor(), sys_cmd(), fm_weights(), held()
eval "$(sed -n '/^run(){/,/^}/p; /^capfor(){/,/^}/p; /^sys_cmd(){/,/^}/p; /^fm_weights(){/,/^}/p; /^held(){/p' scripts/stage5.sh)"

until grep -q "=== stage 5 done ===" "$LOG"; do sleep 120; done
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== stage 6 (repeats at 45%) start ==="
for spec in "mixtral8x7b /home/thor/kcj/models/mixtral8x7b_bf16 mixtral 128 1.5 87.0 phasor,zipmoe,flashmoe,duoserve,fiddler,mixoff" \
            "qwen30b /home/thor/kcj/models/qwen3_30b_a3b qwen3 4 0.5 57.0 phasor,zipmoe,flashmoe,duoserve"; do
  set -- $spec; m=$1 ck=$2 zt=$3 slot=$4 win=$5 gb=$6 systems=${7//,/ }
  O=results/MATRIX5/$m; export SCRUB_GLOB="$ck/*.safetensors /home/thor/kcj/ZipMoE/offload/$zt/*"
  if [ $m = qwen30b ]; then
    rm -rf /home/thor/kcj/ZipMoE-ICML26/offload/mixtral; say "removed Mixtral ZipMoE store (disk) before the Qwen3 repeats"
  fi
  b=$(awk -v g=$gb 'BEGIN{printf "%.2f", g*0.45}'); CAP_GIB=$(capfor $b)
  for rep in 2 3; do for w in mmlu sharegpt longbench; do
    mkdir -p $O/repeats/$w
    for s in $systems; do
      t=$O/$w/${s}_0.45.txt; grep -q '^RESULT' $t 2>/dev/null || continue     # only cells E1 measured
      o=$O/repeats/$w/${s}_0.45_r$rep
      if [ $s = phasor ]; then
        run "s6_${m}_${w}_phasor_r$rep" $b $o $ZPY phasor_hf/phasor_serve.py --checkpoint $ck --workload results/WORKLOADS/$w.json \
          --budget-gib $b --slot-mib $slot --window-gib $win --out $o.json
      else
        kb=$b; cal=$O/memcal/${s}_$b.json
        [ -s $cal ] && kb=$(python3 -c "import json;v=json.load(open('$cal'))['budget_gib'];print(v if v else '$b')")
        run "s6_${m}_${w}_${s}_r$rep" $kb $o $(sys_cmd $s $m $ck $zt $slot $win $w $kb $o)
      fi
    done
  done; done
  unset CAP_GIB
  git add -A $O/repeats >/dev/null 2>&1
  git diff --cached --quiet || { git commit -q -m "Repeats at 45% ($m)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q origin HEAD; }
done
rm -rf /home/thor/kcj/ZipMoE-ICML26/offload/qwen3
python3 scripts/summarize_matrix5.py > results/MATRIX5/SUMMARY.md 2>>"$LOG"
git add results/MATRIX5/SUMMARY.md; git diff --cached --quiet || { git commit -q -m "Matrix summary with repeats

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q origin HEAD; }
say "=== stage 6 done ==="
