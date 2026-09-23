#!/bin/bash
# True footprint on a unified-memory SoC.
#
# RSS misses cudaMalloc, cudaMemGetInfo double-counts cudaHostAlloc'd mapped
# memory (it is one physical pool), and the memory cgroup charges neither
# device allocations nor, usefully, reclaimable page cache.  What is
# unambiguous is how much memory the rest of the system can no longer have:
# MemAvailable already discounts reclaimable page cache, so its drop is the
# non-reclaimable footprint and nothing else.
set -u
avail(){ awk '/^MemAvailable:/{print $2}' /proc/meminfo; }
base=$(avail)
"$@" & pid=$!
low=$base
while kill -0 $pid 2>/dev/null; do
    a=$(avail); [ "$a" -lt "$low" ] && low=$a
done
wait $pid; rc=$?
echo "MEMFOOT_GIB=$(echo "scale=2;($base-$low)/1048576"|bc)"
exit $rc
