#!/bin/bash
# Drives each backend in its own process with the page cache dropped between
# them, so backends that bypass the cache are not penalised and backends that
# use it are not flattered.
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib
B=./gtier_bench
echo "$($B --only 99 "$@" 2>/dev/null)"
printf "%-12s %12s %8s %8s %10s\n" backend useful amp reads hitrate
for i in 0 1 2 3 4 5; do
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; sleep 1
  $B --only $i "$@" 2>/dev/null | tail -1
done
