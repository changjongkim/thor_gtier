#!/bin/bash
# Stage 5: the architecture-level matrix (docs/EXPERIMENT_PLAN.md v3).
# Every system is its real code on the transformers stack; same prompts, same
# cgroup cap (budget + 0.5 GiB); each run also records the peak drop in
# MemAvailable, which sees cudaMalloc'd GPU caches that the cgroup does not.
#   E1  all systems x 2 models x 3 workloads x budgets 0.25/0.45/0.65/1.08
#   E7  PHASOR ablations at 0.45
#   E2  smallest budget each system runs at (0.20, 0.15, 0.10, 0.05)
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/torch_env.sh; . scripts/memguard.sh
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log
ZPY=/home/thor/kcj/envs/zipmoe/bin/python; TPY=$TORCH_VENV/bin/python
# Fiddler and Mixtral-offloading pin transformers 4.36 (and hqq at a fixed commit);
# this venv has those and borrows torch from t26
OPY=/home/thor/kcj/envs/oldhf/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
# run <name> <budget> <out-prefix> <cmd...>: capped run; RESULT line decides success
run(){
  local name=$1 b=$2 o=$3; shift 3
  [ -f "$ST/$name.done" ] && return 0
  [ -f "$ST/$name.norun" ] && return 1
  local fails=$(cat "$ST/$name.fail" 2>/dev/null || echo 0)
  if [ "$fails" -ge 2 ]; then say "give up $name"; return 1; fi
  local need=$(awk -v b=${CAP_GIB:-$b} 'BEGIN{printf "%.0f", b+12}')
  mg_check $need >>"$LOG" 2>&1 || { say "REFUSED $name"; return 1; }
  # cap: the run's memory bound in GiB (CAP_GIB, set per nominal budget by the
  # caller; see capfor), plus 0.5.  Without it, the budget argument itself.
  local cap=$(awk -v b=${CAP_GIB:-$b} 'BEGIN{printf "%.0f", (b+0.5)*1073741824}')
  say "run  $name"; drop
  timeout 21600 "$R/scripts/in_cgroup.sh" m5 $cap "$@" > "$o.txt" 2>&1
  local rc=$?
  awk '{printf "cgroup_peak_gib=%.2f\n", $1/1073741824}' /sys/fs/cgroup/ledger_bench/m5/memory.peak >> "$o.txt" 2>/dev/null
  if grep -q '^RESULT' "$o.txt"; then touch "$ST/$name.done"; say "  ok $name"; return 0; fi
  # an OOM kill under the cap is a result (the system does not run at this budget)
  if grep -q "oom_kill [1-9]" /sys/fs/cgroup/ledger_bench/m5/memory.events 2>/dev/null || [ $rc = 137 ]; then
    echo "NORUN budget=$b reason=oom-under-cap" >> "$o.txt"; touch "$ST/$name.norun"; say "  NORUN $name (oom under cap)"; return 1
  fi
  echo $((fails+1)) > "$ST/$name.fail"; say "  FAILED $name (rc=$rc)"; return 1
}
fm_weights(){  # fm_weights <model> <workload> <budget> -> weights file (trained on held-out workloads)
  local m=$1 w=$2 b=$3 unit L
  if [ $m = qwen30b ]; then unit=9437184; L=48; else unit=352321536; L=32; fi
  local s=$(awk -v b=$b -v u=$unit -v l=$L 'BEGIN{n=int(b*1073741824/u/l); if(n<1)n=1; print n}')
  local f=results/FLASHMOE/bf16/${m}_${w}_s$s.txt
  if [ ! -s $f ]; then
    local h=""; for o in longbench sharegpt mmlu; do [ $o = $w ] || h="$h results/SCOPE/rt_${m}_$o.npz"; done
    OMP_NUM_THREADS=4 $ZPY scripts/train_flashmoe.py $f $s $h > $f.log 2>&1
  fi
  echo $f
}
sys_cmd(){  # sys_cmd <sys> <model> <ckpt> <zt> <slot> <win> <w> <b> <out>: the runner command line
  local s=$1 m=$2 ck=$3 zt=$4 slot=$5 win=$6 w=$7 b=$8 o=$9 W=results/WORKLOADS/$7.json
  case $s in
    zipmoe)  echo "$ZPY scripts/zipmoe_serve.py --model-type $zt --workload $W --budget-gib $b --trace /home/thor/kcj/ZipMoE/trace/${zt}_${w}_heldout.pt --out $o.json" ;;
    moeinf)  echo "$TPY scripts/sota_serve.py --system moe-infinity --checkpoint $ck --workload $W --offload-dir /home/thor/kcj/offload_tmp/$m --budget-gib $b --out $o.json" ;;
    flashmoe) echo "$ZPY baselines_hf/baseline_serve.py --system flashmoe --checkpoint $ck --workload $W --budget-gib $b --weights $(fm_weights $m $w $b) --out $o.json" ;;
    duoserve) echo "$ZPY baselines_hf/baseline_serve.py --system duoserve --checkpoint $ck --workload $W --budget-gib $b --predictor results/DUOSERVE/${m}_$w.pt --trace $(held $m $w) --out $o.json" ;;
    fiddler) echo "$OPY baselines_hf/fiddler_serve.py --checkpoint $ck --workload $W --budget-gib $b --out $o.json" ;;
    mixoff)  echo "$OPY baselines_hf/mixoff_serve.py --state /home/thor/kcj/models/mixtral_offloading_demo --workload $W --budget-gib $b --out $o.json" ;;
  esac
}
held(){ local m=$1 w=$2 h=""; for o in longbench sharegpt mmlu; do [ $o = $w ] || h="$h results/SCOPE/rt_${m}_$o.npz"; done; echo $h; }

capfor(){  # capfor <nominal budget>: 1.05 x PHASOR's measured peak there (memcal's
  # probe cap), else 1.05 x (budget + 6 GiB of non-expert weights, KV and CUDA
  # context, PHASOR's measured overhead at the calibrated budgets: 5.3-5.8 GiB)
  local t=$O/memcal/phasor_$1.json
  python3 -c "import json,os;p='$t';print(round(1.05*(json.load(open(p))['peak_gib'] if os.path.exists(p) and os.path.getsize(p) else $1+6),2))"
}
system(){  # system <sys> <model> <ckpt> <zt> <slot> <win> <w> <b> <out>  -> runs one system
  local s=$1 m=$2 ck=$3 zt=$4 slot=$5 win=$6 w=$7 b=$8 o=$9 W=results/WORKLOADS/$7.json
  # equal memory: a baseline gets the knob value that makes its measured peak
  # equal PHASOR's at this budget (memcal); the run name keeps the nominal budget
  local nb=$b cal=results/MATRIX5/$m/memcal/${s}_$b.json
  local CAP_GIB=$(capfor $nb)   # every system at this nominal budget gets the same cap
  if [ -s $cal ]; then
    b=$(python3 -c "import json;v=json.load(open('$cal'))['budget_gib'];print(v if v else 'none')")
    if [ "$b" = none ]; then   # no setting fits within PHASOR's memory: a result, not a failure
      mkdir -p $(dirname $o); echo "NORUN budget=$nb reason=exceeds-phasor-memory-at-every-setting" > $o.txt; return 1
    fi
  fi
  # reference run at the system's own setting (it may exceed the budget; its peak shows by how much)
  if [ "$s" != phasor ] && [ "$w" = mmlu ]; then
    run "s5_${m}_${w}_${s}_nominal_$nb" $nb ${o}_nominal $(sys_cmd $s $m $ck $zt $slot $win $w $nb ${o}_nominal) || true
  fi
  case $s in
    phasor)  run "s5_${m}_${w}_phasor_$nb" $b $o $ZPY phasor_hf/phasor_serve.py --checkpoint $ck --workload $W --budget-gib $b --slot-mib $slot --window-gib $win --out $o.json ;;
    zipmoe)  run "s5_${m}_${w}_zipmoe_$nb" $b $o $ZPY scripts/zipmoe_serve.py --model-type $zt --workload $W --budget-gib $b --trace /home/thor/kcj/ZipMoE/trace/${zt}_${w}_heldout.pt --out $o.json ;;
    moeinf)  run "s5_${m}_${w}_moeinf_$nb" $b $o $TPY scripts/sota_serve.py --system moe-infinity --checkpoint $ck --workload $W --offload-dir /home/thor/kcj/offload_tmp/$m --budget-gib $b --out $o.json ;;
    flashmoe) run "s5_${m}_${w}_flashmoe_$nb" $b $o $ZPY baselines_hf/baseline_serve.py --system flashmoe --checkpoint $ck --workload $W --budget-gib $b --weights $(fm_weights $m $w $b) --out $o.json ;;
    duoserve) run "s5_${m}_${w}_duoserve_$nb" $b $o $ZPY baselines_hf/baseline_serve.py --system duoserve --checkpoint $ck --workload $W --budget-gib $b --predictor results/DUOSERVE/${m}_$w.pt --trace $(held $m $w) --out $o.json ;;
    fiddler) run "s5_${m}_${w}_fiddler_$nb" $b $o $OPY baselines_hf/fiddler_serve.py --checkpoint $ck --workload $W --budget-gib $b --out $o.json ;;
    mixoff)  run "s5_${m}_${w}_mixoff_$nb" $b $o $OPY baselines_hf/mixoff_serve.py --state /home/thor/kcj/models/mixtral_offloading_demo --workload $W --budget-gib $b --out $o.json ;;
  esac
}

# only after every system has passed its preparation
until grep -q "=== prep done ===" "$LOG"; do sleep 60; done
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== stage 5 (architecture-level matrix) start ==="
# CPU cost, real-model cuFile: OOM-killed under the 8 GiB cap of the other
# backends (its footprint is 10.6 GiB, HF_MOE/FOOTPRINT.md); measured again
# under 16 GiB so the table has its CPU cost, and both facts are reported.
if ! grep -q "cap=16G" results/CPU_COST/real.log 2>/dev/null; then
  say "cpu cost: cufile real-model rerun (16 GiB cap)"; drop
  { echo "NOTE cufile was OOM-killed under the 8 GiB cap (footprint 10.6 GiB, HF_MOE/FOOTPRINT.md); rerun under a 16 GiB cap"
    echo "RUN backend=4 cap=16G"
    a=$(awk '/^cpu /{print $2+$3+$4+$7+$8+$9}' /proc/stat); sleep 3; b=$(awk '/^cpu /{print $2+$3+$4+$7+$8+$9}' /proc/stat)
    echo "BASE machine_cores=$(awk -v a=$a -v b=$b -v h=$(getconf CLK_TCK) 'BEGIN{printf "%.3f", (b-a)/h/3}')"
    SH=""; for x in /home/thor/kcj/models/qwen3_30b_a3b/model-*.safetensors; do SH="$SH --shard $x"; done
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib timeout 3600 scripts/in_cgroup.sh cpucost $((16<<30)) lib/gguf_bench $SH \
      --experts 128 --active 8 --skew 0.8 --tokens 16 --batch 8 --slot 4194304 --slots 512 --backend 4 || echo "FAILED rc=$?"
  } >> results/CPU_COST/real.log 2>&1
  python3 scripts/cpu_cost_table.py > /dev/null && git add results/CPU_COST results/CPU_COST.md && git commit -q -m "CPU cost: cuFile real-model run under 16 GiB (OOM under 8 GiB)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
fi
# Fiddler and Mixtral-offloading failed their prep smoke on the newer
# transformers; smoke them in their own venv first, so a failure shows now and
# not after the Qwen3 matrix.  Host guard on; results in results/PREP.
for s_ in "fiddler baselines_hf/fiddler_serve.py --checkpoint /home/thor/kcj/models/mixtral8x7b_bf16 --budget-gib 94.0" \
          "mixoff baselines_hf/mixoff_serve.py --state /home/thor/kcj/models/mixtral_offloading_demo"; do
  set -- $s_; n=$1; shift
  [ -f $ST/s5_smoke_$n.done ] && continue
  say "run  s5_smoke_$n"; drop
  if timeout 7200 scripts/in_cgroup.sh prep max $OPY "$@" --workload results/WORKLOADS/mmlu.json --limit 2 \
       --out results/PREP/smoke_${n}_oldhf.json > results/PREP/smoke_${n}_oldhf.log 2>&1 && grep -q '^RESULT' results/PREP/smoke_${n}_oldhf.log; then
    touch $ST/s5_smoke_$n.done; say "  ok s5_smoke_$n ($(grep '^RESULT' results/PREP/smoke_${n}_oldhf.log | grep -o 'request_s=[0-9.]* peak_gib=[0-9.]*'))"
  else
    say "  FAILED s5_smoke_$n: $(grep -E 'Error|HOSTGUARD' results/PREP/smoke_${n}_oldhf.log | tail -1)"
  fi
done
# Self-test of the E3/E5 runner modes (batch, profile) on two MMLU prompts, so a
# broken mode shows now rather than when the extras are reached; not a measurement.
if [ ! -f $ST/s5_selftest_extras.done ]; then
  Q=/home/thor/kcj/models/qwen3_30b_a3b; W=results/WORKLOADS/mmlu.json; T=results/PREP/selftest; mkdir -p $T; okall=1
  for t in "phasor_b2 $ZPY phasor_hf/phasor_serve.py --checkpoint $Q --workload $W --budget-gib 25.65 --slot-mib 4 --window-gib 0.5 --batch 2" \
           "phasor_prof $ZPY phasor_hf/phasor_serve.py --checkpoint $Q --workload $W --budget-gib 25.65 --slot-mib 4 --window-gib 0.5 --profile" \
           "flashmoe_b2 $ZPY baselines_hf/baseline_serve.py --system flashmoe --checkpoint $Q --workload $W --budget-gib 25.65 --weights $(ls results/FLASHMOE/bf16/qwen30b_mmlu_s*.txt | head -1) --batch 2" \
           "duoserve_b2 $ZPY baselines_hf/baseline_serve.py --system duoserve --checkpoint $Q --workload $W --budget-gib 25.65 --predictor results/DUOSERVE/qwen30b_mmlu.pt --trace $(held qwen30b mmlu) --batch 2" \
           "zipmoe_b2 $ZPY scripts/zipmoe_serve.py --model-type qwen3 --workload $W --budget-gib 25.65 --trace /home/thor/kcj/ZipMoE/trace/qwen3_mmlu_heldout.pt --batch 2"; do
    set -- $t; n=$1; shift
    grep -q '^RESULT' $T/$n.log 2>/dev/null && continue       # passed on an earlier start
    drop
    if timeout 3600 scripts/in_cgroup.sh prep max "$@" --limit 4 --out $T/$n.json > $T/$n.log 2>&1 && grep -q '^RESULT' $T/$n.log; then
      say "  selftest ok $n ($(python3 -c "import json;d=json.load(open('$T/$n.json'));print({k:d[k] for k in ('tok_per_s','profile_s') if k in d})" 2>&1 | cut -c1-200))"
    else okall=0; say "  selftest FAILED $n: $(grep -E 'Error|error|HOSTGUARD' $T/$n.log | tail -1 | cut -c1-200)"; fi
  done
  [ $okall = 1 ] && touch $ST/s5_selftest_extras.done
fi
for spec in "qwen30b /home/thor/kcj/models/qwen3_30b_a3b qwen3 4 0.5 57.0 phasor,zipmoe,moeinf,flashmoe,duoserve" \
            "mixtral8x7b /home/thor/kcj/models/mixtral8x7b_bf16 mixtral 128 1.5 87.0 phasor,zipmoe,moeinf,flashmoe,duoserve,fiddler,mixoff"; do
  # Queued SSD-idle work (e.g. CPU-cost measurements) runs here, between
  # models, under this script's lock; each hook runs once.
  for h in $ST/hooks/*.sh; do
    [ -f "$h" ] && [ ! -f "$h.done" ] || continue
    say "hook start $(basename $h)"; bash "$h" >> "$h.log" 2>&1; touch "$h.done"; say "hook done $(basename $h)"
  done
  set -- $spec; m=$1 ck=$2 zt=$3 slot=$4 win=$5 gb=$6 systems=${7//,/ }
  for n in fiddler mixoff; do
    [[ $systems == *$n* ]] && [ ! -f $ST/s5_smoke_$n.done ] && { systems=${systems/$n/}; say "$n smoke failed: left out of $m"; }
  done
  O=results/MATRIX5/$m
  if [ $m = mixtral8x7b ] && [ ! -f $ST/mixtral_bf16_traces.done ]; then
    # The Mixtral traces were captured from the Q4 GGUF with llama.cpp; the
    # matrix serves bf16, so its routing is recaptured from the bf16 model
    # through PHASOR-HF (tokens equal stock transformers), and everything
    # trained or planned from traces is rebuilt from the new ones.
    mkdir -p results/SCOPE/gguf_invalid
    for w in longbench sharegpt mmlu; do
      say "capture bf16 Mixtral routing: $w"; drop
      timeout 21600 $ZPY phasor_hf/capture_routing.py --checkpoint $ck --workload results/WORKLOADS/$w.json \
        --out results/SCOPE/rt_${m}_${w}.bf16.npz --budget-gib 60 > results/SCOPE/capture_bf16_$w.log 2>&1 || { say "capture failed: $w"; exit 1; }
    done
    for w in longbench sharegpt mmlu; do
      mv results/SCOPE/rt_${m}_$w.npz results/SCOPE/rt_${m}_$w.bin results/SCOPE/gguf_invalid/ 2>/dev/null
      mv results/SCOPE/rt_${m}_$w.bf16.npz results/SCOPE/rt_${m}_$w.npz
      $TPY scripts/export_routing.py --npz results/SCOPE/rt_${m}_$w.npz --out results/SCOPE/rt_${m}_$w.bin >> $LOG 2>&1
    done
    rm -f results/FLASHMOE/bf16/${m}_*.txt
    for w in longbench sharegpt mmlu; do
      h=$(held $m $w)
      OMP_NUM_THREADS=4 $ZPY baselines_hf/train_duoserve.py results/DUOSERVE/${m}_$w.pt $h >> $LOG 2>&1
      $ZPY scripts/zipmoe_trace.py /home/thor/kcj/ZipMoE-ICML26/trace/mixtral_${w}_heldout.pt $h >> $LOG 2>&1
    done
    touch $ST/mixtral_bf16_traces.done; say "Mixtral bf16 traces captured; predictors and plans rebuilt"
  fi
  if [ $m = mixtral8x7b ]; then
    # conversions that could not coexist with the Qwen3-30B stores: token check and
    # smoke for ZipMoE and MoE-Infinity here; a system that fails is left out
    P=results/PREP; mkdir -p /home/thor/kcj/offload_tmp/$m
    drop; timeout 14400 scripts/in_cgroup.sh prep max $ZPY scripts/check_tokens.py zipmoe $ck mixtral 39.15 $P/tok_zipmoe_mixtral.json > $P/tok_zipmoe_mixtral.log 2>&1 \
      || { systems=${systems/zipmoe/}; say "ZipMoE/Mixtral token check failed: left out"; }
  fi
  # MoE-Infinity keeps the whole model pinned in host memory and adds its GPU
  # cache on top (device_memory_ratio), so on a unified pool it needs more
  # than the model.  The smoke only checks that it serves at all (4 GiB GPU
  # cache, host guard on); whether it fits a budget is memcal's question.
  if [[ " $systems " == *" moeinf "* ]] || [[ $systems == *moeinf* ]]; then
    mkdir -p /home/thor/kcj/offload_tmp/$m
    if [ ! -f $ST/s5_smoke_mi_$m.done ]; then
      say "run  s5_smoke_mi_$m"; drop
      if timeout 7200 scripts/in_cgroup.sh prep max $TPY scripts/sota_serve.py --system moe-infinity --checkpoint $ck \
           --workload results/WORKLOADS/mmlu.json --offload-dir /home/thor/kcj/offload_tmp/$m --budget-gib 4 --limit 2 \
           --out results/PREP/smoke_mi_$m.json > results/PREP/smoke_mi_$m.log 2>&1 && grep -q '^RESULT' results/PREP/smoke_mi_$m.log; then
        touch $ST/s5_smoke_mi_$m.done; say "  ok s5_smoke_mi_$m ($(grep '^RESULT' results/PREP/smoke_mi_$m.log | grep -o 'peak_gib=[0-9.]*'))"
      else
        systems=${systems/moeinf/}; say "MoE-Infinity/$m smoke failed: left out ($(grep -o 'HOSTGUARD.*' results/PREP/smoke_mi_$m.log | head -1))"
      fi
    fi
  fi
  # Equal memory.  PHASOR's measured peak at each budget is the target; each
  # calibratable baseline gets the knob value that reaches it (2-prompt MMLU).
  mkdir -p $O/memcal
  for f in 0.25 0.45 0.65 1.08; do
    b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}')
    t=$O/memcal/phasor_$b.json
    if [ ! -s $t ]; then drop; timeout 3600 $ZPY phasor_hf/phasor_serve.py --checkpoint $ck --workload results/WORKLOADS/mmlu.json \
        --budget-gib $b --slot-mib $slot --window-gib $win --limit 2 --out $t > $t.log 2>&1; fi
    T=$(python3 -c "import json;print(round(json.load(open('$t'))['peak_gib'],2))" 2>/dev/null) || continue
    say "memcal $m $f: PHASOR peak $T GiB"
    for s in $systems; do
      case $s in phasor|fiddler|mixoff) continue ;; esac
      c=$O/memcal/${s}_$b.json; [ -s $c ] && continue
      case $s in
        zipmoe)   cmd="$ZPY scripts/zipmoe_serve.py --model-type $zt --workload results/WORKLOADS/mmlu.json --budget-gib {B} --trace /home/thor/kcj/ZipMoE/trace/${zt}_mmlu_heldout.pt --limit 2 --out {OUT}" ;;
        moeinf)   cmd="$TPY scripts/sota_serve.py --system moe-infinity --checkpoint $ck --workload results/WORKLOADS/mmlu.json --offload-dir /home/thor/kcj/offload_tmp/$m --budget-gib {B} --limit 2 --out {OUT}" ;;
        flashmoe) cmd="$ZPY baselines_hf/baseline_serve.py --system flashmoe --checkpoint $ck --workload results/WORKLOADS/mmlu.json --budget-gib {B} --weights $(fm_weights $m mmlu $b) --limit 2 --out {OUT}" ;;
        duoserve) cmd="$ZPY baselines_hf/baseline_serve.py --system duoserve --checkpoint $ck --workload results/WORKLOADS/mmlu.json --budget-gib {B} --predictor results/DUOSERVE/${m}_mmlu.pt --trace $(held $m mmlu) --limit 2 --out {OUT}" ;;
      esac
      drop; timeout 14400 python3 scripts/memcal.py $T $b $c -- $cmd > $c.log 2>&1
      say "memcal $m $f $s: $(tail -1 $c.log)"
    done
  done
  # E1
  for w in mmlu sharegpt longbench; do mkdir -p $O/$w; for f in 0.25 0.45 0.65 1.08; do
    b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}')
    for s in $systems; do system $s $m $ck $zt $slot $win $w $b $O/$w/${s}_$f; done
  done; done
  # E7: PHASOR ablations at 0.45
  b=$(awk -v g=$gb 'BEGIN{printf "%.2f", g*0.45}'); CAP_GIB=$(capfor $b)
  for w in mmlu sharegpt longbench; do
    for ab in "lru --policy lru" "count --policy count" "copy --policy phasor+copy" "nopipe --no-pipeline" "noprompt --mix 0" "admitall --policy phasor+all" "pfall --policy phasor+pfall"; do
      set -- $ab; n=$1; shift
      run "s5_${m}_${w}_abl_$n" $b $O/$w/abl_$n $ZPY phasor_hf/phasor_serve.py --checkpoint $ck --workload results/WORKLOADS/$w.json \
        --budget-gib $b --slot-mib $slot --window-gib $win "$@" --out $O/$w/abl_$n.json
    done
  done
  unset CAP_GIB
  # E2: smallest budget, MMLU (shortest requests); a system stops at its first budget it cannot run at
  for s in $systems; do
    for f in 0.20 0.15 0.10 0.05; do
      b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}')
      system $s $m $ck $zt $slot $win mmlu $b $O/mmlu/${s}_$f || break
    done
  done
  # E3 (batch), E5 (latency breakdown), E9 (window, SSD bandwidth): in their
  # own script, read when reached, while this model's conversion stores exist
  [ -f scripts/stage5_extras.sh ] && . scripts/stage5_extras.sh   # sourced: uses run/system/sys_cmd and this loop's variables
  python3 scripts/summarize_matrix5.py > results/MATRIX5/SUMMARY.md 2>>"$LOG"
  git add -A results/MATRIX5 results/FLASHMOE >/dev/null 2>&1
  git diff --cached --quiet || { git commit -q -m "Architecture-level matrix: $m

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q origin HEAD; }
  # the next model needs the disk: remove this model's conversion stores (ours, regenerable)
  if [ $m = qwen30b ]; then rm -rf /home/thor/kcj/offload_tmp/qwen30b /home/thor/kcj/ZipMoE-ICML26/offload/qwen3; say "removed Qwen3-30B conversion stores"; fi
done
say "=== stage 5 done ==="
