#!/bin/bash
# CPU cost microbenchmark alone, with the run length of BACKENDS.md /
# ASYNC_FIX.md (iters 128 = 512 MiB per run).  cpu_cost.sh's first pass read
# 4 GiB per run over the same 16 GiB span, which lets the page-cache backends
# reuse their readahead and so differs from the earlier table.  Called as a
# stage 5 hook, which already holds the pipeline lock (so no flock here).
# usage: cpu_cost_micro.sh <iters> <repeats> <log>
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
IT=$1; REP=$2; LOGF=$3; F=/home/thor/kcj/mmap_gpu/real32.bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; sleep 1; }
busy(){ awk '/^cpu /{print $2+$3+$4+$7+$8+$9}' /proc/stat; }
base(){ local a=$(busy); sleep 3; local b=$(busy)
  echo "BASE machine_cores=$(awk -v a=$a -v b=$b -v h=$(getconf CLK_TCK) 'BEGIN{printf "%.3f", (b-a)/h/3}')"; }
if [ ! -s "$F" ]; then head -c $((32<<30)) /dev/urandom > "$F" || exit 1; fi
sync; sleep 300      # let the SSD finish absorbing the 32 GiB write before measuring
: > "$LOGF"
for rep in $(seq 1 $REP); do
  for item in 16384 65536 262144 1048576 4194304; do
    n=$((4194304 / item))
    for b in 0 4 3 2 5 1; do
      drop; echo "RUN rep=$rep item=$item backend=$b" >> "$LOGF"; base >> "$LOGF"
      timeout 1800 lib/gtier_bench --file "$F" --only $b --item $item --n $n --slot $item \
        --iters $IT --span 16 --async 1 >> "$LOGF" 2>&1 || echo "FAILED rc=$?" >> "$LOGF"
    done
  done
done
rm -f "${F:?}"
