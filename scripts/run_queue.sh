#!/bin/bash
# Detached experiment queue.  Launched with setsid so it outlives the shell that
# started it; every step writes its own file under results/auto and the whole
# batch is committed and pushed at the end, so a dropped connection loses
# nothing.  Re-running skips steps whose output already exists.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT=$ROOT/results/auto
LOG=$OUT/progress.log
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib
BIN=$ROOT/lib/gguf_bench
MOE=/home/thor/kcj/models/moe235b_q4km
DENSE=/home/thor/kcj/models/qwen32b_q8

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; sleep 2; }
shards(){ local s=""; for f in "$1"/*.gguf; do s="$s --shard $f"; done; echo "$s"; }
save(){ # name, command...
  local name="$1"; shift
  if [ -s "$OUT/$name.txt" ]; then say "skip $name (already done)"; return; fi
  say "run  $name"
  drop
  timeout 7200 "$@" > "$OUT/$name.txt" 2>&1 || say "  FAILED $name (rc=$?)"
  tail -1 "$OUT/$name.txt" | tee -a "$LOG"
}

say "=== queue start ==="
say "MoE  $(du -sh $MOE 2>/dev/null | cut -f1)   DRAM $(free -g | awk '/Mem:/{print $2}') GiB"
S_MOE=$(shards $MOE)
S_DEN=$(shards $DENSE)

# --- E2: a model that does not fit.  132.4 GiB against 122.8 GiB of DRAM. -----
for b in 0 1 2 3 4 5; do
  save "e2_moe_backend$b" $BIN $S_MOE --backend $b --policy 0 --tokens 1 \
       --slot 1048576 --slots 192
done

# --- E2b: PIN vs LRU where most but not all of the model is resident ----------
# Window is capped below DRAM so the OS keeps room; at 1 MiB slots this is the
# regime the real target sits in.
for slots in 24576 49152 73728 98304; do
  for pol in 1 4; do
    save "e2_res_${slots}_pol${pol}" $BIN $S_MOE --backend 0 --policy $pol \
         --tokens 3 --slot 1048576 --slots $slots
  done
done

# --- E3: dense model, continuous-submission check at depth --------------------
for n in 64 128 256; do
  save "e3_depth_$n" $BIN $S_DEN --backend 0 --policy 0 --tokens 1 \
       --slot 1048576 --slots $n
done

say "=== queue done, committing ==="
cd "$ROOT"
git add -A results/auto
git commit -q -m "Automated queue results: DRAM-exceeding MoE model and residency sweep

Ran detached so the batch survives a dropped connection. Covers the 132.4 GiB
Qwen3-235B-A22B Q4_K_M against 122.8 GiB of DRAM across all six backends, the
PIN versus LRU comparison at several residency fractions, and a queue-depth
check on the dense model.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1 | tail -2 | tee -a "$LOG"
timeout 300 git push 2>&1 | tail -2 | tee -a "$LOG"
say "=== all done ==="
