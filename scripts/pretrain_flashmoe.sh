#!/bin/bash
# Train every FlashMoE* network the stages will need, ahead of time, on two
# cores at the lowest priority so the serving runs are not disturbed.  The
# weight files and .done markers match stage3b/3d/3e, which then skip training.
R=/home/thor/kcj/thor_gtier; cd "$R"
. "$R/scripts/torch_env.sh"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ST=$R/results/PIPELINE; W8=$R/results/FLASHMOE; mkdir -p "$W8"
MD=/home/thor/kcj/models
declare -A MDIR=( [qwen30b]=$MD/qwen3_30b_q4km [mixtral8x7b]=$MD/mixtral8x7b_q4km [qwen235b]=$MD/moe235b_q4km )
declare -A NL=( [qwen30b]=48 [mixtral8x7b]=32 [qwen235b]=94 )
declare -A NE=( [qwen30b]=128 [mixtral8x7b]=8 [qwen235b]=128 )
ggufs(){ ls "${MDIR[$1]}"/*.gguf 2>/dev/null | grep -v q8_0 | sort; }
gib(){ du -cb $(ggufs "$1") | tail -1 | awk '{printf "%.2f", $1/1073741824}'; }
one(){
  local m=$1 w=$2 f=$3
  local b=$(awk -v t="$(gib $m)" -v f="$f" 'BEGIN{printf "%.2f", t*f}')
  local s=$(python3 "$R/scripts/fm_slots.py" $b 0.5 0.59 ${NL[$m]} ${NE[$m]} $(ggufs $m))
  local name=fmtrain_${m}_${w}_s$s
  [ -f "$ST/$name.done" ] && return
  local train=""; for ow in longbench sharegpt mmlu; do [ $ow = $w ] || train="$train $R/results/SCOPE/rt_${m}_$ow.npz"; done
  echo "[$(date +%T)] $name"
  if nice -n 19 taskset -c 12,13 $TORCH_VENV/bin/python "$R/scripts/train_flashmoe.py" "$W8/${m}_${w}_s$s.txt" $s $train >> "$ST/$name.out" 2>&1; then
    touch "$ST/$name.done"
  fi
}
for m in qwen30b mixtral8x7b qwen235b; do for w in longbench sharegpt mmlu; do
  for f in 0.10 0.15 0.25 0.45 0.65 1.08; do
    [ "$m" = qwen235b ] && [ "$f" = 1.08 ] && continue
    one $m $w $f
  done
done; done
echo PRETRAIN_DONE
