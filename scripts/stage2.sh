#!/bin/bash
# Stage 2: the serving matrix.  Runs after the capture pipeline (it waits for
# the same lock), one step at a time, every step through the memory guard.
#
#   models     qwen30b, mixtral8x7b, qwen235b -- all Q4_K_M GGUF, one I/O path
#   workloads  longbench, sharegpt, mmlu     -- the same prompts on every model
#   budgets    0.25 / 0.45 / 0.65 of the model's bytes
#   policies   lru, moe-inf*, mixtral*, ledger
#   ablations  at 0.45: each LEDGER component off, decode weight 1/16/auto
#
# The per-token arithmetic is calibrated, not modelled from FLOPs: llama-bench
# on the same Q4_K_M file with every layer on the GPU (qwen30b, mixtral), and
# for 235B -- which does not fit -- qwen30b's cost scaled by the ratio of bytes
# a token touches, the two being the same architecture at the same precision.
# LEDGER's initial counts come from the other two workloads of the same model,
# never from the trace being served.
set -u
R=/home/thor/kcj/thor_gtier
cd "$R"
. "$R/scripts/torch_env.sh"
. "$R/scripts/memguard.sh"
ST=$R/results/PIPELINE; mkdir -p "$ST"
OUT=$R/results/MATRIX; mkdir -p "$OUT"
LOG=$ST/pipeline.log
LL=/home/thor/skim/llama.cpp
MD=/home/thor/kcj/models
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
step(){
  local name=$1 need=$2; shift 2; [ "${1:-}" = "--" ] && shift
  if [ -f "$ST/$name.done" ]; then return 0; fi
  # A step that failed twice is left for a person: retrying it on every
  # restart would turn one bad configuration into an endless loop.
  local fails=$(cat "$ST/$name.fail" 2>/dev/null || echo 0)
  if [ "$fails" -ge 2 ]; then say "give up $name (failed $fails times)"; return 0; fi
  if ! mg_check "$need" >>"$LOG" 2>&1; then
    say "REFUSED $name (needs $need GiB)"; echo refused > "$ST/$name.refused"; return 0
  fi
  say "run  $name"
  drop
  if "$@" >> "$ST/$name.out" 2>&1; then
    touch "$ST/$name.done"; say "  ok $name"
  else
    local rc=$?
    echo $((fails+1)) > "$ST/$name.fail"
    say "  FAILED $name (rc=$rc)"
  fi
}
commit(){
  git add -A results scripts lib/*.cu lib/*.h >/dev/null 2>&1
  git diff --cached --quiet || { git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"; timeout 300 git push -q origin HEAD; }
}

exec 9>/tmp/gtier_pipeline.lock
flock 9                                   # wait for the capture pipeline
echo -500 | sudo -n tee /proc/self/oom_score_adj >/dev/null 2>&1 || true
say "=== stage 2 start ==="

# ---- 0. Qwen3-30B as Q4_K_M, so all three models share precision and path --
Q30=$MD/qwen3_30b_q4km
step conv_qwen30b_q4km 40 -- bash -c "
  set -o pipefail; mkdir -p '$Q30' && \
  $TORCH_VENV/bin/python '$LL/convert_hf_to_gguf.py' '$MD/qwen3_30b_a3b' --outtype q8_0 \
      --outfile '$Q30/qwen3-30b-a3b-q8_0.gguf' 2>&1 | tail -2 && \
  '$LL/build/bin/llama-quantize' --allow-requantize '$Q30/qwen3-30b-a3b-q8_0.gguf' \
      '$Q30/qwen3-30b-a3b-Q4_K_M.gguf' Q4_K_M 2>&1 | tail -2 && \
  rm -f '$Q30/qwen3-30b-a3b-q8_0.gguf' && test -s '$Q30/qwen3-30b-a3b-Q4_K_M.gguf'"

declare -A MDIR=( [qwen30b]=$Q30 [mixtral8x7b]=$MD/mixtral8x7b_q4km [qwen235b]=$MD/moe235b_q4km )
declare -A NLAY=( [qwen30b]=48 [mixtral8x7b]=32 [qwen235b]=94 )
declare -A NEXP=( [qwen30b]=128 [mixtral8x7b]=8 [qwen235b]=128 )
declare -A TOPK=( [qwen30b]=8 [mixtral8x7b]=2 [qwen235b]=8 )
ggufs(){ ls "${MDIR[$1]}"/*.gguf 2>/dev/null | grep -v q8_0 | sort; }
gib(){ du -cb $(ggufs "$1") | tail -1 | awk '{printf "%.2f", $1/1073741824}'; }

# ---- 1. per-token compute, measured on the GPU with every layer resident --
CAL=$OUT/calib.tsv
for m in qwen30b mixtral8x7b; do
  g=$(ggufs $m | head -1); [ -n "$g" ] || continue
  need=$(mg_gguf_need "${MDIR[$m]}" 99 "${NLAY[$m]}")
  step "calib_$m" "$need" -- bash -c "
    '$LL/build/bin/llama-bench' -m '$g' -ngl 99 -fa 1 -p 4096 -n 32 -r 3 -o json \
      > '$OUT/calib_$m.json'"
done
# Rebuilt whenever a calibration is newer, so a missing model does not hold
# back the others.
if ls "$ST"/calib_*.done >/dev/null 2>&1; then
  $TORCH_VENV/bin/python - "$OUT" > "$CAL" <<'PY'
import json, sys, subprocess
out = sys.argv[1]
def ms(m):
    d = json.load(open(f"{out}/calib_{m}.json"))
    pp = [x for x in d if x["n_prompt"] > 0 and x["n_gen"] == 0][0]["avg_ts"]
    tg = [x for x in d if x["n_gen"] > 0 and x["n_prompt"] == 0][0]["avg_ts"]
    return 1000.0 / tg, 1000.0 / pp
def act(k, e, paths):
    r = subprocess.run(["python3", "/home/thor/kcj/thor_gtier/scripts/active_bytes.py",
                        str(k), str(e)] + paths, capture_output=True, text=True)
    return float(r.stdout.split()[0])
import glob
g = lambda d: sorted(p for p in glob.glob(d + "/*.gguf") if "q8_0" not in p)
M = "/home/thor/kcj/models"
print("model\tdecode_ms\tprompt_ms\tsource")
got = {}
for m in ("qwen30b", "mixtral8x7b"):
    try:
        got[m] = ms(m)
        print(f"{m}\t{got[m][0]:.4f}\t{got[m][1]:.5f}\tllama-bench ngl99")
    except Exception:
        pass
if "qwen30b" in got:
    d30, p30 = got["qwen30b"]
    a30 = act(8, 128, g(M + "/qwen3_30b_q4km"))
    a235 = act(8, 128, g(M + "/moe235b_q4km"))
    r = a235 / a30
    print(f"qwen235b\t{d30*r:.4f}\t{p30*r:.5f}\tqwen30b x {r:.3f} (bytes/token {a235:.3e}/{a30:.3e})")
PY
fi
cal(){ awk -v m="$1" -v c="$2" '$1==m{print $c}' "$CAL" 2>/dev/null; }

# ---- 2. the matrix ---------------------------------------------------------
# sb <model> <workload> <run-name> <budget-frac> <args...>
sb(){
  local m=$1 w=$2 name=$3 frac=$4; shift 4
  local tr=$R/results/SCOPE/rt_${m}_${w}.bin
  [ -s "$tr" ] || { say "skip $m/$w/$name (no trace)"; return; }
  local cd=$(cal $m 2) cp=$(cal $m 3)
  [ -n "$cd" ] || { say "skip $m/$w/$name (no calibration)"; return; }
  local tot=$(gib $m)
  local b=$(awk -v t="$tot" -v f="$frac" 'BEGIN{printf "%.2f", t*f}')
  local need=$(awk -v b="$b" 'BEGIN{printf "%.0f", b+6}')
  local prof=""
  for ow in longbench sharegpt mmlu; do
    [ "$ow" = "$w" ] && continue
    [ -s "$R/results/SCOPE/rt_${m}_${ow}.bin" ] && prof="$prof --profile-trace $R/results/SCOPE/rt_${m}_${ow}.bin"
  done
  local shards=""; for f in $(ggufs $m); do shards="$shards --shard $f"; done
  mkdir -p "$OUT/$m/$w"
  step "mx_${m}_${w}_${name}" "$need" -- bash -c "
    timeout 10800 '$R/lib/serve_bench' $shards --trace '$tr' $prof \
      --budget $b --window 0.5 --repeats 2 --compute-ms $cd --prompt-compute-ms $cp $* \
      > '$OUT/$m/$w/$name.txt' 2>&1 && grep -q '^RESULT' '$OUT/$m/$w/$name.txt'"
}

# A short run on the smallest configuration first: if the GGUF unit slicing or
# the RESULT line is wrong, it is better to stop here than after a night.
sb mixtral8x7b mmlu sanity_ledger 0.25 --policy 12 --max-decode 4
if ! grep -q '^RESULT' "$OUT/mixtral8x7b/mmlu/sanity_ledger.txt" 2>/dev/null; then
  say "sanity run failed -- stage 2 stops"; exit 1
fi

for m in qwen30b mixtral8x7b qwen235b; do
  [ -n "$(ggufs $m)" ] || { say "skip $m (no gguf)"; continue; }
  for w in longbench sharegpt mmlu; do
    for frac in 0.25 0.45 0.65; do
      sb $m $w "lru_$frac"     $frac --policy 1
      sb $m $w "moeinf_$frac"  $frac --policy 10
      sb $m $w "mixtral_$frac" $frac --policy 11
      sb $m $w "ledger_$frac"  $frac --policy 12
    done
  done
  for w in longbench sharegpt mmlu; do
    [ "$m" = qwen235b ] && [ "$w" != longbench ] && continue
    sb $m $w "abl_noasync"   0.45 --policy 12 --no-async
    sb $m $w "abl_mix0"      0.45 --policy 12 --mix 0
    sb $m $w "abl_mix1"      0.45 --policy 12 --mix 1
    sb $m $w "abl_norec"     0.45 --policy 12 --w-rec 0
    sb $m $w "abl_nosel"     0.45 --policy 12 --selective 0
    sb $m $w "abl_count"     0.45 --policy 12 --mix off
    sb $m $w "abl_profile"   0.45 --policy 12 --profile-weight 1
    sb $m $w "abl_noprefix"  0.45 --policy 12 --no-prefix-pin
    sb $m $w "abl_nolive"    0.45 --policy 12 --no-live-set
    sb $m $w "abl_pread"     0.45 --policy 12 --backend 3
    sb $m $w "ref_none"      0.45 --policy 0
  done
  $TORCH_VENV/bin/python "$R/scripts/summarize_matrix.py" > "$OUT/SUMMARY.md" 2>>"$LOG"
  commit "Serving matrix: $m"
done
# ---- 3. a real system on the same prompts: MoE-Infinity, Qwen3-30B bf16 ----
# MoE-Infinity reads HF checkpoints only, so this track is bf16, and the driver
# runs the same model file with its real kernels (--compute) at the same
# absolute budgets.  Fiddler and Mixtral-offloading keep every expert in host
# memory; on a unified pool that is the same memory, so they cannot run below
# the model's size and are not a budgeted comparison (results/SOTA/PORTING.md).
Q30BF=$MD/qwen3_30b_a3b
BFSH=""; for f in "$Q30BF"/model-*.safetensors; do BFSH="$BFSH --shard $f"; done
BFGIB=$(du -cb "$Q30BF"/model-*.safetensors | tail -1 | awk '{printf "%.2f", $1/1073741824}')
CP30=$(cal qwen30b 3)
for w in longbench sharegpt mmlu; do
  tr=$R/results/SCOPE/rt_qwen30b_${w}.bin
  prof=""; for ow in longbench sharegpt mmlu; do [ "$ow" = "$w" ] || prof="$prof --profile-trace $R/results/SCOPE/rt_qwen30b_${ow}.bin"; done
  mkdir -p "$OUT/qwen30b_bf16/$w"
  for frac in 0.25 0.45 0.65; do
    b=$(awk -v t="$BFGIB" -v f="$frac" 'BEGIN{printf "%.2f", t*f}')
    need=$(awk -v b="$b" 'BEGIN{printf "%.0f", b+8}')
    for pp in "lru 1" "moeinf 10" "mixtral 11" "ledger 12"; do
      set -- $pp
      step "bf_${w}_$1_$frac" "$need" -- bash -c "
        timeout 10800 '$R/lib/serve_bench' $BFSH --trace '$tr' $prof --budget $b --window 0.5 \
          --repeats 2 --compute --prompt-compute-ms $CP30 --policy $2 \
          > '$OUT/qwen30b_bf16/$w/$1_$frac.txt' 2>&1 && grep -q '^RESULT' '$OUT/qwen30b_bf16/$w/$1_$frac.txt'"
    done
    step "sota_moeinf_${w}_$frac" "$need" -- bash -c "
      mkdir -p /home/thor/kcj/offload_tmp/qwen30b; \
      timeout 14400 $TORCH_VENV/bin/python '$R/scripts/sota_serve.py' --system moe-infinity \
        --checkpoint '$Q30BF' --workload '$R/results/WORKLOADS/$w.json' \
        --offload-dir /home/thor/kcj/offload_tmp/qwen30b --budget-gib $b \
        --out '$OUT/qwen30b_bf16/$w/moeinf_real_$frac.json' \
        > '$OUT/qwen30b_bf16/$w/moeinf_real_$frac.txt' 2>&1"
  done
done
$TORCH_VENV/bin/python "$R/scripts/summarize_matrix.py" > "$OUT/SUMMARY.md" 2>>"$LOG"
commit "Serving matrix: bf16 track with MoE-Infinity"
touch "$ST/STAGE2.complete"
say "=== stage 2 done ==="
