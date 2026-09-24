#!/bin/bash
# Why LEDGER lost to LRU once its initial counts were held out: sweep the
# utility's half-life (0 = pure frequency, the version that lost) on
# Qwen3-30B, where one run takes about three minutes.
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/memguard.sh
O=$R/results/DIAG_decay
exec 9>/tmp/gtier_pipeline.lock; flock 9
G=/home/thor/kcj/models/qwen3_30b_q4km/qwen3-30b-a3b-Q4_K_M.gguf
TOT=$(du -b $G | awk '{print $1/1073741824}')
cal(){ awk -v c="$1" '$1=="qwen30b"{print $c}' results/MATRIX/calib.tsv; }
run(){ local w=$1 frac=$2 name=$3; shift 3
  local f=$O/${w}_${frac}_$name.txt; [ -s "$f" ] && grep -q RESULT "$f" && return
  local b=$(awk -v t=$TOT -v f=$frac 'BEGIN{printf "%.2f",t*f}')
  mg_check $(awk -v b=$b 'BEGIN{printf "%.0f",b+6}') >/dev/null || return
  local prof=""; for ow in longbench sharegpt mmlu; do [ $ow = $w ] || prof="$prof --profile-trace results/SCOPE/rt_qwen30b_$ow.bin"; done
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  timeout 3600 lib/serve_bench --shard $G --trace results/SCOPE/rt_qwen30b_$w.bin $prof \
    --budget $b --window 0.5 --repeats 2 --compute-ms $(cal 2) --prompt-compute-ms $(cal 3) "$@" > "$f" 2>&1
  echo "$w $frac $name $(grep RESULT $f | grep -oE '(request_s|tpot_ms|prefill_io_s|decode_gib_tok)=[0-9.]+' | tr '\n' ' ')"
}
# hl0 / hl16 = the count utility (--mix off); pred_* = decode-probability
# estimate mixing this request's prefill routing with decode history.
for wf in "longbench 0.25" "sharegpt 0.25" "mmlu 0.25" "longbench 0.45" "sharegpt 0.45" "mmlu 0.45"; do
  set -- $wf; w=$1; fr=$2
  run $w $fr lru        --policy 1
  run $w $fr hl0        --policy 12 --mix off
  run $w $fr pred_m05   --policy 12 --mix 0.5
  run $w $fr pred_m05_nosel --policy 12 --mix 0.5 --selective 0
  run $w $fr pred_m0    --policy 12 --mix 0
  run $w $fr pred_m1    --policy 12 --mix 1
  run $w $fr pred_m05_noprof --policy 12 --mix 0.5 --profile-weight 0
done
echo DIAG_DONE
