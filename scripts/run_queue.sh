#!/bin/bash
# Overnight experiment queue.  Launched detached so it outlives the shell that
# starts it.  Every step writes its own file under results/auto and is committed
# as it completes, so a dropped connection or an interrupted run loses nothing;
# re-running skips whatever is already there.  A summary is generated at the end.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT=$ROOT/results/auto
LOG=$OUT/progress.log
mkdir -p "$OUT"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib
BIN=$ROOT/lib/gguf_bench
MOE=/home/thor/kcj/models/moe235b_q4km      # 132.4 GiB, exceeds 122.8 GiB DRAM
DENSE=/home/thor/kcj/models/qwen32b_q8      # 32.42 GiB, fits

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; sleep 2; }
shards(){ local s=""; for f in "$1"/*.gguf; do s="$s --shard $f"; done; echo "$s"; }
commit(){
  cd "$ROOT"
  git add -A results/auto >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "Automated queue: $1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" >/dev/null 2>&1
  timeout 300 git push -q >/dev/null 2>&1 && say "  pushed ($1)" || say "  push failed ($1), will retry later"
}
save(){ local name="$1"; shift
  if [ -s "$OUT/$name.txt" ]; then say "skip $name"; return; fi
  say "run  $name"
  drop
  if timeout 10800 "$@" > "$OUT/$name.tmp" 2>&1; then
    mv "$OUT/$name.tmp" "$OUT/$name.txt"
    tail -1 "$OUT/$name.txt" | sed 's/^/       /' | tee -a "$LOG"
  else
    say "     FAILED (rc=$?)"; mv "$OUT/$name.tmp" "$OUT/$name.failed" 2>/dev/null
  fi
}

say "================ queue start ================"
say "MoE $(du -sh $MOE 2>/dev/null|cut -f1)  dense $(du -sh $DENSE 2>/dev/null|cut -f1)  DRAM $(free -g|awk '/Mem:/{print $2}') GiB"
sudo -n jetson_clocks >/dev/null 2>&1
S_MOE=$(shards $MOE); S_DEN=$(shards $DENSE)

# --- A. dense model, all backends, gtier with continuous submission ----------
save "a_dense_gtier_async" $BIN $S_DEN --backend 0 --policy 0 --tokens 1 --slot 1048576 --slots 576 --async 1
for b in 0 1 2 3 4 5; do
  save "a_dense_backend$b" $BIN $S_DEN --backend $b --policy 0 --tokens 1 --slot 1048576 --slots 576
done
commit "dense model backend comparison"

# --- B. a model that does not fit: 132.4 GiB against 122.8 GiB of DRAM -------
save "b_moe_gtier_async" $BIN $S_MOE --backend 0 --policy 0 --tokens 1 --slot 1048576 --slots 576 --async 1
for b in 0 1 2 3 4 5; do
  save "b_moe_backend$b" $BIN $S_MOE --backend $b --policy 0 --tokens 1 --slot 1048576 --slots 576
done
commit "DRAM-exceeding MoE backend comparison"

# --- C. residency on the MoE model: PIN versus LRU ---------------------------
# 16 MiB slots keep the allocation count sane at these window sizes.
for slots in 768 1536 3072 4608 6144; do
  for pol in 1 4; do
    save "c_moe_res_${slots}_pol${pol}" $BIN $S_MOE --backend 0 --policy $pol \
         --tokens 3 --slot 16777216 --slots $slots
  done
  commit "MoE residency at $slots slots"
done

# --- D. granularity on the real MoE trace ------------------------------------
for sp in 65536 262144 1048576; do
  for b in 0 4 1; do
    save "d_moe_gran_${sp}_b${b}" $BIN $S_MOE --backend $b --policy 0 --tokens 1 \
         --slot 2097152 --slots 576 --split $sp
  done
done
commit "MoE granularity sweep"

# --- summary -----------------------------------------------------------------
say "building summary"
python3 "$ROOT/scripts/summarize.py" > "$OUT/SUMMARY.md" 2>&1 || say "summary failed"
commit "summary"
say "================ all done ================"
