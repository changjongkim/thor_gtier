# Sourced by stage5.sh inside its per-model loop, after E1/E7/E2 and before
# the model's conversion stores are removed.  Uses stage5's run(), sys_cmd(),
# fm_weights(), held() and the loop variables m ck zt slot win gb systems O.
#
#   E5  PHASOR latency breakdown (wait for expert bytes / expert matmuls / MoE
#       block, per phase), a separate profiled run at 0.45: it synchronizes
#   E9  staging-window sensitivity: PHASOR at 0.45 with the window x0.5/x2/x4
#       (x1 is E1).  The SSD-bandwidth half of E9 is not run: this kernel has
#       no block-I/O throttling (CONFIG_BLK_DEV_THROTTLING unset).
#   E3  batching at 0.45: batch 4 and 8 (batch 1 is E1) for the systems whose
#       code serves a batch (PHASOR, ZipMoE, FlashMoE*, DuoServe*; Fiddler and
#       Mixtral-offloading generate one sequence).  Each baseline keeps its
#       equal-memory setting from E1's memcal; a batch adds KV cache and
#       activations to every system, so the cap is E1's cap + 8 GiB and the
#       measured peaks are reported.
say "=== extras ($m): E1 retries, E5, E9 window, E3 ==="
# E1 retries.  memcal calibrates on two prompts; a baseline whose memory keeps
# growing over a workload (e.g. DuoServe*'s host LRU) can then exceed the cap
# on the full run.  Such a cell is retried with its knob lowered to x0.85,
# x0.7, x0.55 of the calibrated value, stopping at the first that runs; the
# cell reports that run and its knob.  PHASOR is never retried.
for w in mmlu sharegpt longbench; do for f in 0.25 0.45 0.65 1.08; do
  nb=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}')
  for s in $systems; do
    [ $s = phasor ] && continue
    t=$O/$w/${s}_$f.txt
    grep -q "^NORUN.*oom-under-cap" $t 2>/dev/null || continue
    cal=$O/memcal/${s}_$nb.json; [ -s $cal ] || continue
    k0=$(python3 -c "import json;v=json.load(open('$cal'))['budget_gib'];print(v if v else 'none')"); [ "$k0" = none ] && continue
    CAP_GIB=$(capfor $nb)
    for k in 0.85 0.7 0.55; do
      kb=$(awk -v a=$k0 -v k=$k 'BEGIN{printf "%.2f", a*k}'); o=$O/$w/${s}_${f}_k$k
      if [ $s = zipmoe ] || [ $s = flashmoe ] || [ $s = duoserve ] || [ $s = fiddler ] || [ $s = mixoff ]; then
        run "s5_${m}_${w}_${s}_${nb}_k$k" $kb $o $(sys_cmd $s $m $ck $zt $slot $win $w $kb $o) && break
      fi
    done
    unset CAP_GIB
  done
done; done
X=$O/extras; mkdir -p $X/e5 $X/e9 $X/e3
b45=$(awk -v g=$gb 'BEGIN{printf "%.2f", g*0.45}')
calb(){  # calb <sys> -> equal-memory budget at 0.45, or "none"
  local c=$O/memcal/${1}_$b45.json
  [ "$1" = phasor ] && { echo $b45; return; }
  [ -s $c ] || { echo $b45; return; }
  python3 -c "import json;v=json.load(open('$c'))['budget_gib'];print(v if v else 'none')"
}
CAP_GIB=$(capfor $b45)          # E5, E9: PHASOR at 0.45, the E1 cap
# E5
for w in mmlu sharegpt longbench; do
  run "s5x_${m}_${w}_phasor_profile" $b45 $X/e5/${w}_phasor $ZPY phasor_hf/phasor_serve.py --checkpoint $ck \
    --workload results/WORKLOADS/$w.json --budget-gib $b45 --slot-mib $slot --window-gib $win --profile --out $X/e5/${w}_phasor.json
done
# E9: window
for x in 0.5 2 4; do
  wn=$(awk -v a=$win -v x=$x 'BEGIN{printf "%.3g", a*x}')
  for w in mmlu sharegpt; do
    run "s5x_${m}_${w}_phasor_win$wn" $b45 $X/e9/${w}_phasor_win$wn $ZPY phasor_hf/phasor_serve.py --checkpoint $ck \
      --workload results/WORKLOADS/$w.json --budget-gib $b45 --slot-mib $slot --window-gib $wn --out $X/e9/${w}_phasor_win$wn.json
  done
done
# E3: batch
for B in 4 8; do
  for w in mmlu sharegpt; do
    for s in phasor zipmoe flashmoe duoserve; do
      [[ " $systems " == *" $s "* ]] || continue
      b=$(calb $s)
      if [ "$b" = none ]; then echo "NORUN budget=$b45 reason=exceeds-phasor-memory-at-every-setting" > $X/e3/${w}_${s}_b$B.txt; continue; fi
      capb=$(awk -v c=$(capfor $b45) 'BEGIN{printf "%.2f", c+8}')   # the E1 cap + 8 GiB for batch KV/activations
      o=$X/e3/${w}_${s}_b$B
      if [ $s = phasor ]; then
        cmd="$ZPY phasor_hf/phasor_serve.py --checkpoint $ck --workload results/WORKLOADS/$w.json --budget-gib $b --slot-mib $slot --window-gib $win --out $o.json"
      else
        cmd=$(sys_cmd $s $m $ck $zt $slot $win $w $b $o)
      fi
      CAP_GIB=$capb run "s5x_${m}_${w}_${s}_b$B" $b $o $cmd --batch $B
    done
  done
done
git add -A $X >/dev/null 2>&1
git diff --cached --quiet || { git commit -q -m "Extras ($m): E5 breakdown, E9 window, E3 batch

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q origin HEAD; }
unset CAP_GIB
say "=== extras ($m) done ==="
