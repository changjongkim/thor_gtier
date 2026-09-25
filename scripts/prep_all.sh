#!/bin/bash
# Before the matrix: every system proves it runs (two prompts each, one-time
# conversions done here) and the ported/integrated ones prove their tokens
# equal stock transformers.  Nothing here is a measurement.
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/torch_env.sh; . scripts/memguard.sh
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log; P=$R/results/PREP; mkdir -p "$P"
ZPY=/home/thor/kcj/envs/zipmoe/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
chk(){ local name=$1 need=$2; shift 2
  [ -f "$P/$name.ok" ] && { say "prep skip $name"; return 0; }
  mg_check $need >>"$LOG" 2>&1 || { say "prep REFUSED $name"; return 1; }
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  say "prep run  $name"
  if timeout 14400 "$@" > "$P/$name.log" 2>&1; then touch "$P/$name.ok"; say "prep ok   $name"; else say "prep FAIL $name (rc=$?)"; fi
}
# inputs that are still being prepared off the critical path
until [ -s results/FLASHMOE/bf16/mixtral8x7b_mmlu_s5.txt ] && [ -s results/DUOSERVE/mixtral8x7b_mmlu.pt ] \
      && [ -f /home/thor/kcj/models/mixtral_offloading_demo/config.json ] \
      && ! pgrep -f "snapshot_download|train_duoserve|train_flashmoe" >/dev/null; do sleep 30; done
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== prep (all systems) start ==="
Q=/home/thor/kcj/models/qwen3_30b_a3b; M=/home/thor/kcj/models/mixtral8x7b_bf16; W=$R/results/WORKLOADS/mmlu.json
# correctness
chk tok_zipmoe_qwen 70 $ZPY scripts/check_tokens.py zipmoe $Q qwen3 25.65 $P/tok_zipmoe_qwen.json
chk tok_stock_mixtral 100 $ZPY scripts/check_tokens.py stock $M mixtral 0 $P/tok_stock_mixtral.json
chk tok_phasor_mixtral 50 $ZPY scripts/check_tokens.py phasor $M mixtral 39.15 $P/tok_phasor_mixtral.json
# smoke: reimplemented baselines (Qwen3-30B and Mixtral)
chk smoke_flash_qwen 40 $ZPY baselines_hf/baseline_serve.py --system flashmoe --checkpoint $Q --workload $W \
  --budget-gib 25.65 --weights results/FLASHMOE/bf16/qwen30b_mmlu_s60.txt --limit 2 --out $P/smoke_flash_qwen.json
chk smoke_duo_qwen 40 $ZPY baselines_hf/baseline_serve.py --system duoserve --checkpoint $Q --workload $W \
  --budget-gib 25.65 --predictor results/DUOSERVE/qwen30b_mmlu.pt \
  --trace results/SCOPE/rt_qwen30b_longbench.npz results/SCOPE/rt_qwen30b_sharegpt.npz --limit 2 --out $P/smoke_duo_qwen.json
chk smoke_flash_mix 50 $ZPY baselines_hf/baseline_serve.py --system flashmoe --checkpoint $M --workload $W \
  --budget-gib 39.15 --weights results/FLASHMOE/bf16/mixtral8x7b_mmlu_s3.txt --limit 2 --out $P/smoke_flash_mix.json
chk smoke_duo_mix 50 $ZPY baselines_hf/baseline_serve.py --system duoserve --checkpoint $M --workload $W \
  --budget-gib 39.15 --predictor results/DUOSERVE/mixtral8x7b_mmlu.pt \
  --trace results/SCOPE/rt_mixtral8x7b_longbench.npz results/SCOPE/rt_mixtral8x7b_sharegpt.npz --limit 2 --out $P/smoke_duo_mix.json
# released systems on Mixtral (ZipMoE and MoE-Infinity convert the checkpoint first;
# the disk holds one model's conversions at a time, so theirs run in stage 5
# after the Qwen3-30B stores are removed)
chk smoke_mixoff 40 $ZPY baselines_hf/mixoff_serve.py --state /home/thor/kcj/models/mixtral_offloading_demo \
  --workload $W --limit 2 --out $P/smoke_mixoff.json
chk smoke_fiddler_mix 100 $ZPY baselines_hf/fiddler_serve.py --checkpoint $M --workload $W --budget-gib 94.0 \
  --limit 2 --out $P/smoke_fiddler_mix.json
chk smoke_phasor_mix 50 $ZPY phasor_hf/phasor_serve.py --checkpoint $M --workload $W --budget-gib 39.15 \
  --slot-mib 128 --window-gib 1.5 --limit 2 --out $P/smoke_phasor_mix.json
python3 - <<'PY'
import json, os
P = "/home/thor/kcj/thor_gtier/results/PREP"
def cmp(a, b):
    try:
        A = json.load(open(f"{P}/{a}")); B = json.load(open(b))
        return "identical" if A == B else "DIFF " + str([sum(x == y for x, y in zip(A[k], B[k])) for k in A])
    except Exception as e: return f"n/a ({e})"
print("zipmoe qwen   vs stock:", cmp("tok_zipmoe_qwen.json", "/home/thor/kcj/thor_gtier/results/PHASOR_HF/correct_stock.json"))
print("phasor mixtral vs stock:", cmp("tok_phasor_mixtral.json", f"{P}/tok_stock_mixtral.json"))
print("zipmoe mixtral vs stock:", cmp("tok_zipmoe_mixtral.json", f"{P}/tok_stock_mixtral.json"))
PY
say "=== prep done ==="
