#!/bin/bash
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib
printf "%-12s %13s %13s %8s\n" backend serial pipelined gain
for i in 0 1 2 3 4 5; do
  a=""; b=""
  for p in 0 1; do
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; sleep 1
    v=$(./gtier_bench --only $i --pipeline $p "$@" 2>/dev/null | tail -1 | awk '{print $2}')
    [ $p -eq 0 ] && a=$v || b=$v
  done
  name=$(./gtier_bench --only $i --iters 1 --pipeline 0 "$@" 2>/dev/null | tail -1 | awk '{print $1}')
  g=$(python3 -c "
a='$a'; b='$b'
try: print(f'{float(b)/float(a):.2f}x')
except: print('-')")
  printf "%-12s %10s GiB/s %10s GiB/s %8s\n" "${name:-$i}" "${a:-n/a}" "${b:-n/a}" "$g"
done
