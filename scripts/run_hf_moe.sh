#!/bin/bash
# Same model, same files, same cold cache, same memory budget.
#
# The budget is the part that is easy to get wrong.  An mmap or pread backend
# has the whole page cache behind it, so on a 122 GiB machine it gets a cache
# two orders of magnitude larger than the window gTier is configured with, and
# any comparison against it is really a comparison of cache sizes.  Every run
# here is therefore placed in a cgroup with the same memory.max, which bounds
# the page cache to the same budget the explicit window costs.
set -u
ROOT=/home/thor/kcj/thor_gtier
MODEL=${MODEL:-/home/thor/kcj/models/qwen3_30b_a3b}
OUT=$ROOT/results/HF_MOE
TOKENS=${TOKENS:-8}
EXPERTS=${EXPERTS:-128}; ACTIVE=${ACTIVE:-8}; SKEW=${SKEW:-0.8}
SLOT=${SLOT:-$((4*1024*1024))}; SLOTS=${SLOTS:-512}
BUDGET=${BUDGET:-8G}
mkdir -p "$OUT"

SHARDS=""
for f in "$MODEL"/model-*.safetensors; do SHARDS="$SHARDS --shard $f"; done
N=$(ls "$MODEL"/model-*.safetensors 2>/dev/null | wc -l)
[ "$N" -eq 0 ] && { echo "no safetensors in $MODEL"; exit 1; }

drop(){ sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
capped(){ sudo systemd-run --scope -q -p MemoryMax=$BUDGET --slice=gtierbench.slice -- "$@"; }

echo "== model $MODEL ($N shards, $(du -sh "$MODEL"|cut -f1)) =="
echo "== trace: $TOKENS tokens, $ACTIVE-of-$EXPERTS experts, skew $SKEW, budget $BUDGET =="

run(){  # run <label> <extra args...>
  local label="$1"; shift
  drop
  local line
  line=$(capped timeout 3600 "$ROOT/lib/gguf_bench" $SHARDS \
      --experts $EXPERTS --active $ACTIVE --skew $SKEW --tokens $TOKENS \
      --slot $SLOT --slots $SLOTS "$@" 2>&1 | tail -1)
  printf '%-26s %s\n' "$label" "$line" | tee -a "$OUT/backends.txt"
}

: > "$OUT/backends.txt"
# Backend ids follow gtier.h: 0 gtier, 1 mmap_gpu, 2 mmap_cpu, 3 pread+copy,
# 4 cuFile, 5 UVM.  3/4/5 are the published patterns -- FlexGen and
# ZeRO-Infinity stage through host memory, GDS is cuFile, DeepUM is UVM.
for b in 0 3 4 5 1 2; do run "backend$b" --backend $b; done
# The levers gTier actually has on top of the plain path.
run "gtier+async"      --backend 0 --async 1
run "gtier+pin"        --backend 0 --policy 4
run "gtier+block"      --backend 0 --policy 1
run "gtier+adaptive"   --backend 0 --policy 3
echo "backends -> $OUT/backends.txt"
