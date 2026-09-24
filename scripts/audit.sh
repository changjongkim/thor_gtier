#!/bin/bash
# Turn each claimed component off and see whether the numbers move.
#
# Three components were inert before anyone checked -- two patches that never
# applied, and the continuous-submission path that was simply not wired in --
# and each time the symptom was a knob that did nothing, which was read as
# "the component does not help" rather than "the component is not running".
# This exists so that reading is never made again.
set -u
R=/home/thor/kcj/thor_gtier
cd "$R"
OUT=$R/results/AUDIT; mkdir -p "$OUT"
M=${MODEL:-/home/thor/kcj/models/qwen3_30b_a3b}
SH_ARGS=""; for f in "$M"/model-*.safetensors; do SH_ARGS="$SH_ARGS --shard $f"; done
BASE="--trace $R/results/SCOPE/routing_multi.bin --budget 24 --window 0.5 --max-decode 4 --repeats 6 --compute"
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
one(){ local name="$1"; shift
  [ -s "$OUT/$name.txt" ] && { echo "skip $name"; return; }
  drop
  timeout 3600 "$R/lib/serve_bench" $SH_ARGS $BASE "$@" > "$OUT/$name.txt" 2>&1
  printf '%-22s %s\n' "$name" "$(grep -oE 'prefill I/O +[0-9.]+|io +[0-9.]+ ms/tok|n=32 +[0-9.]+' "$OUT/$name.txt" | tr '\n' ' ')"
}
echo "=== audit: budget 24, steady, policy 12 unless noted ==="
one base            --policy 12 --profile-weight 0.25
one off_async       --policy 12 --profile-weight 0.25 --no-async
one off_liveset     --policy 12 --profile-weight 0.25 --no-live-set
one off_prefixpin   --policy 12 --profile-weight 0.25 --no-prefix-pin
one off_profile     --policy 12 --profile-weight 0
one W1              --policy 12 --profile-weight 0.25 --decode-weight 1
one W16             --policy 12 --profile-weight 0.25 --decode-weight 16
one path_pread      --policy 12 --profile-weight 0.25 --backend 3
one policy_lru      --policy 1
