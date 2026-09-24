#!/bin/bash
# Does which layers are placed where change what the engine delivers?
#
# llama.cpp's -ncmoe N sends layers 0..N-1's experts to the CPU: a count, not
# a choice.  The routing measurement says residency should be ordered by
# value, so the first question is whether the engine's placement is even
# sensitive to the ordering -- if moving the same number of layers gives the
# same throughput whichever layers they are, then a value-based order has
# nowhere to act through this interface and that is worth knowing.
#
# -ot takes a regex per tensor, and GGUF stacks a layer's experts into one
# tensor, so the finest placement the engine admits is per layer.  Per-expert
# residency, which is what the measurement actually calls for, cannot be
# expressed here at all.
set -u
R=/home/thor/kcj/thor_gtier
cd "$R"
OUT=$R/results/PLACE; LOG=$OUT/progress.log
BENCH=${BENCH:-/home/thor/skim/llama.cpp/build/bin/llama-bench}
BIG=${BIG:-/home/thor/kcj/models/moe235b_q4km}
NP=${NP:-256}; NG=${NG:-16}
mkdir -p "$OUT"
MODEL=$(ls "$BIG"/*-00001-of-*.gguf 2>/dev/null | head -1)
NL=94                      # Qwen3-235B-A22B layers

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
run(){ local name="$1"; shift
  grep -qE "IOMETER|OUTCOME" "$OUT/$name.txt" 2>/dev/null && { say "skip $name"; return; }
  say "run  $name"; drop
  local rc=0
  timeout 10800 "$R/scripts/io_meter.py" --label "$name" -- \
      "$BENCH" -m "$MODEL" -p $NP -n $NG -r 1 -o json "$@" \
      < /dev/null >> "$OUT/$name.txt" 2>&1 || rc=$?
  grep -q IOMETER "$OUT/$name.txt" || echo "OUTCOME failed rc=$rc" >> "$OUT/$name.txt"
  grep -E '"avg_ts"|IOMETER|OUTCOME' "$OUT/$name.txt" | tail -3 | tee -a "$LOG"
}
commit(){ cd "$R"; git add -A results >/dev/null 2>&1
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>&1|tail -1|tee -a "$LOG"
  timeout 300 git push -q 2>&1|tail -1|tee -a "$LOG"; }

# blk.<i>.ffn_(gate|up|down)_exps on CPU, for a chosen set of layers.
pat(){ local p=""; for i in "$@"; do p="${p}blk\\.${i}\\.ffn_(gate|up|down)_exps=CPU;"; done; echo "$p"; }

echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== placement queue: $(basename "$MODEL") pp=$NP tg=$NG ==="

# How much the engine gets as more of the model is pushed off the GPU.
#
# N is not swept from zero.  The model is 132.4 GiB and the device has 122.8,
# so leaving every layer's experts on the GPU is an overcommit, and measured,
# that configuration does not fail the process -- it restarts the host, five
# times out of five, eighteen to thirty-nine minutes in.  The experts are
# 96.6% of the model across 94 layers, about 1.36 GiB each, so roughly a dozen
# layers have to come off before it fits at all; the sweep starts at 24 to
# leave room for the KV cache and activations on top.
for N in 24 47 70 94; do
  run "p_ncmoe$N" -ngl 99 -ncmoe $N
done
commit "Placement queue: llama.cpp's own residency control swept"

# Same count, three different choices of which layers.  If these differ, the
# ordering matters and a value-based one has room to act.
HALF=47
FIRST=$(seq 0 $((HALF-1)));                 LAST=$(seq $((NL-HALF)) $((NL-1)))
ALT=$(seq 0 2 $((NL-1)) | head -$HALF)
run "p_half_first" -ngl 99 -ot "$(pat $FIRST)"
run "p_half_last"  -ngl 99 -ot "$(pat $LAST)"
run "p_half_alt"   -ngl 99 -ot "$(pat $ALT)"
commit "Placement queue: the same count of layers, three different choices"
say "=== placement queue done ==="
