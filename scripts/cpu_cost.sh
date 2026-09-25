#!/bin/bash
# CPU cost of each data path (paper §4.6).  Runs only while the SSD is idle:
# it waits for stage 4 to stop and then takes the pipeline lock, so it sits
# between stages and never runs next to a measurement.
#
# Same conditions as ASYNC_FIX.md / BACKENDS.md: a 32 GiB random file, each
# backend in its own process after the page cache is dropped, 4 MiB per fetch
# (n = 4 MiB / item, slot = item), scattered offsets over 16 GiB.  Per run:
# getrusage(RUSAGE_SELF) user and sys over the timed loop (all threads), and
# the whole machine's busy time from /proc/stat, since GPU fault service and
# completion interrupts run outside the process.  An idle baseline of the
# machine is taken before each run and subtracted in the table.
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/memguard.sh
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log; O=$R/results/CPU_COST; mkdir -p "$O"
F=/home/thor/kcj/mmap_gpu/real32.bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; sleep 1; }
busy(){ awk '/^cpu /{print $2+$3+$4+$7+$8+$9}' /proc/stat; }
base(){  # machine busy cores over 3 s of idle
  local a=$(busy); sleep 3; local b=$(busy)
  echo "BASE machine_cores=$(awk -v a=$a -v b=$b -v h=$(getconf CLK_TCK) 'BEGIN{printf "%.3f", (b-a)/h/3}')"
}

until grep -q "stage 4 stopped" "$LOG" || ! pgrep -f stage4_run >/dev/null; do sleep 30; done
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== cpu cost (data paths) start ==="
mg_check 20 >>"$LOG" 2>&1 || { say "REFUSED cpu cost"; exit 1; }

if [ ! -s "$F" ] || [ $(stat -c %s "$F") -lt $((32<<30)) ]; then
  say "cpu cost: writing 32 GiB random file"
  mkdir -p "$(dirname "$F")"
  head -c $((32<<30)) /dev/urandom > "$F" || { say "cpu cost: file write FAILED"; exit 1; }
fi

# Microbenchmark: 5 item sizes x 6 backends, 4 GiB read per run, 2 repeats.
MICRO=$O/micro.log; : > "$MICRO"
for rep in 1 2; do
  for item in 16384 65536 262144 1048576 4194304; do
    n=$((4194304 / item)); iters=$(( (4<<30) / 4194304 ))
    for b in 0 4 3 2 5 1; do   # gtier(async) cufile pread+copy mmap-cpu uvm mmap-gpu
      drop
      echo "RUN rep=$rep item=$item backend=$b" >> "$MICRO"
      base >> "$MICRO"
      timeout 1800 lib/gtier_bench --file "$F" --only $b --item $item --n $n --slot $item \
        --iters $iters --span 16 --async 1 >> "$MICRO" 2>&1 || echo "FAILED rc=$?" >> "$MICRO"
    done
  done
done
say "cpu cost: micro done"

# Real model: Qwen3-30B-A3B bf16 shards, 8 of 128 experts (skew 0.8), batch 8,
# window 2 GiB (512 x 4 MiB), under the 8 GiB cap of HF_MOE/footprint.txt.
REAL=$O/real.log; : > "$REAL"
SH=""; for s in /home/thor/kcj/models/qwen3_30b_a3b/model-*.safetensors; do SH="$SH --shard $s"; done
for b in 0 4 3 2 5 1; do
  extra=""; [ $b = 0 ] && extra="--async 1"
  drop
  echo "RUN backend=$b" >> "$REAL"
  base >> "$REAL"
  timeout 3600 scripts/in_cgroup.sh cpucost $((8<<30)) lib/gguf_bench $SH --experts 128 --active 8 \
    --skew 0.8 --tokens 16 --batch 8 --slot 4194304 --slots 512 --backend $b $extra >> "$REAL" 2>&1 \
    || echo "FAILED rc=$?" >> "$REAL"
done
say "cpu cost: real-model done"

python3 scripts/cpu_cost_table.py > /dev/null 2>>"$LOG"
rm -f "${F:?}"
git add scripts/cpu_cost.sh scripts/cpu_cost_table.py lib/cpucost.h lib/bench.cu lib/gguf_bench.cu \
  results/CPU_COST results/CPU_COST.md
git commit -q -m "CPU cost per data path (getrusage user/sys, machine busy)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
say "=== cpu cost done ==="
