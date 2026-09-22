#!/bin/bash
# E2: what happens when the model does not fit?
#
# cgroup v2 MemoryMax bounds the page cache as well as anonymous memory, so it
# emulates a smaller DRAM faithfully: the model, its cache pages and the CUDA
# allocations all have to live inside the limit.  This reaches the
# oversubscription regime without waiting on a 132 GiB download.
#
# Paths compared -- these are the real baselines:
#   ngl99-cudaMalloc   llama.cpp default: device buffer, cannot exceed memory
#   ngl99-UVM          GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 -> cudaMallocManaged
#                      (the DeepUM / UVM-oversubscription baseline)
#   ngl0-mmap          CPU paging through the kernel
#   ngl0-nommap        explicit read into anonymous memory
set -u
LLAMA=/home/thor/skim/llama.cpp/build/bin/llama-bench
MODEL="$1"; shift
SIZE_GIB=$(python3 -c "
import os,glob,re,sys
p='$MODEL'; d=os.path.dirname(p)
m=re.match(r'^(.*)-\d{5}-of-(\d{5})\.gguf$', os.path.basename(p))
print(round(sum(os.path.getsize(f) for f in glob.glob(os.path.join(d,'*.gguf')))/2**30,1) if m else round(os.path.getsize(p)/2**30,1))")
echo "model $(basename $(dirname $MODEL))  ${SIZE_GIB} GiB"
run() { # label, memmax, extra-env, args...
  local label="$1" mm="$2" env="$3"; shift 3
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
  local ratio=$(python3 -c "print(f'{$SIZE_GIB/${mm%G}:.2f}')")
  printf "  %-18s cap=%-5s (%sx model)  " "$label" "$mm" "$ratio"
  local out
  out=$(timeout 900 systemd-run --user --scope -q -p MemoryMax=$mm -p MemorySwapMax=0 \
        env $env $LLAMA -m "$MODEL" "$@" -o json 2>&1)
  if echo "$out" | grep -q '"avg_ts"'; then
    echo "$out" | python3 -c "
import sys,json,re
t=sys.stdin.read(); j=json.loads(t[t.index('['):t.rindex(']')+1])
d={('pp' if r['n_prompt'] else 'tg'):round(r['avg_ts'],2) for r in j}
print(f\"OK   pp={d.get('pp','-')} tg={d.get('tg','-')} t/s\")"
  else
    echo "FAIL $(echo "$out" | grep -oiE 'out of memory|oom|killed|cuda error[^\"]*|error[^\"]*' | head -1)"
  fi
}
for mm in "$@"; do
  echo " --- cap $mm ---"
  run ngl99-cudaMalloc "$mm" "X=1"                                   -ngl 99 -p 128 -n 32 -r 1
  run ngl99-UVM        "$mm" "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1"     -ngl 99 -p 128 -n 32 -r 1
  run ngl0-mmap        "$mm" "X=1"                                   -ngl 0  -p 128 -n 32 -r 1
  run ngl0-nommap      "$mm" "X=1"                                   -ngl 0  -p 128 -n 32 -r 1 --mmap 0
done
