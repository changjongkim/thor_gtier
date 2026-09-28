#!/bin/bash
# Stage 11q: stage 11 again (copy of stage11_llamacpp.sh), after stage 12, Mixtral first (its GGUF exists).
# Its 09-28 22:43 Mixtral smoke met the host guard: llama.cpp's load mode "auto" disables mmap on an iGPU
# and read the whole model into memory; the runner now passes --load-mode mmap.
# On 09-28 22:35 the converter's Qwen3 pass met the host guard (> 100 GiB at the start of writing; the
# Mixtral pass stayed low); here it writes through a temporary file (--use-temp-file), and the disk it
# needs (GGUF + temporary copy, ~125 GB) is freed first by removing stage 12's MoE-Infinity offload
# store (ours, regenerable) and any leftover GGUF.
# Stage 11: llama.cpp (ggml-org/llama.cpp, stock) as the practical baseline:
# bf16 GGUF, experts mmap'd from the SSD and computed on the CPU (--cpu-moe),
# the rest on the GPU; its page cache is its expert cache, bounded by the run's
# cgroup (scripts/llamacpp_serve.py).  After stage 10, alone (pipeline lock).
#  0 build llama-server (CUDA, sm_110), convert the checkpoints to bf16 GGUF
#  1 smoke (2 MMLU prompts); a failure is recorded in results/PREP/LLAMACPP.md
#  2 tokens against stock transformers
#  3 Qwen3-30B: memcal (cgroup cap) + E1 (3 workloads x 4 budgets, retries,
#    the 1.4 x rule), E2 (MMLU 20-5%, cap = budget), E3 (batch 4/8 at 45%)
#  4 Mixtral-8x7B: MMLU 25/45/65% (108%: host safety ceiling, as every system)
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"
. scripts/torch_env.sh; . scripts/memguard.sh
ST=$R/results/PIPELINE; LOG=$ST/pipeline.log; P=$R/results/PREP
ZPY=/home/thor/kcj/envs/zipmoe/bin/python
L=/home/thor/kcj/llama.cpp; BIN=$L/build/bin/llama-server; G=/home/thor/kcj/models/gguf_bf16
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
drop(){ sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
settle(){ local i=0; until [ $(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo) -ge ${SETTLE_GIB:-100} ]; do   # no time limit: a run under
  i=$((i+1)); [ $i = 120 ] && say "waiting for >= 100 GiB available"; sleep 5; done; }   # outside memory pressure would say nothing
eval "$(sed -n '/^run(){/,/^}/p; /^capfor(){/,/^}/p' scripts/stage5.sh)"
eval "$(sed -n '/^cal(){/,/^}/p; /^flag14(){/,/^}/p; /^over14(){/,/^}/p; /^because(){/,/^}/p' scripts/stage_helpers.sh)"
[ -n "${NOWAIT:-}" ] || until [ $(grep -c "=== stage 12 done" "$LOG") -ge 2 ]; do sleep 120; done
exec 9>/tmp/gtier_pipeline.lock; flock 9
say "=== stage 11q (llama.cpp, Qwen3-30B) start ==="
free_gb(){ df -BG --output=avail /home/thor/kcj | tail -1 | tr -dc 0-9; }
rm -rf /home/thor/kcj/offload_tmp/mixtral8x7b_2502; rm -f /home/thor/kcj/models/gguf_bf16/*.part
say "stage 11q: removed stage 12's MoE-Infinity offload store (ours, regenerable) for disk ($(free_gb) GB free)"
mkdir -p $P/llamacpp $G
export SCRUB_GLOB=""    # its page cache is its cache: bounded by the cgroup, not scrubbed
note(){ echo "$*" >> $P/LLAMACPP.md; }
[ -n "${FRESH:-}" ] && rm -f $P/LLAMACPP.md
[ -f $P/LLAMACPP.md ] || printf '# llama.cpp (stock) on the unified-memory Thor\n\n' > $P/LLAMACPP.md
# 0 build and convert
if [ ! -x $BIN ]; then
  [ -d $L/.git ] || timeout 1800 git clone -q https://github.com/ggml-org/llama.cpp $L
  # the CUDA 13.0 toolkit: /usr/local/cuda (13.2) here has no cuBLAS (09-28 22:26: "CUDA::cublas not found")
  CT=/usr/local/cuda-13.0; rm -rf "${L:?}/build"
  ( cd $L && PATH=$CT/bin:$PATH CUDACXX=$CT/bin/nvcc cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=110 \
      -DCUDAToolkit_ROOT=$CT -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > $P/llamacpp/cmake.log 2>&1 && \
    PATH=$CT/bin:$PATH cmake --build build -j 12 --target llama-server > $P/llamacpp/build.log 2>&1 )
fi
if [ ! -x $BIN ]; then
  say "stage 11: llama.cpp build FAILED: $(grep -hiE 'error' $P/llamacpp/cmake.log $P/llamacpp/build.log 2>/dev/null | head -1 | cut -c1-150)"
  note "- Build failed (commit $(cd $L 2>/dev/null && git log -1 --format=%h)): see results/PREP/llamacpp/build.log"
  git add -f $P/LLAMACPP.md $P/llamacpp scripts/stage11_llamacpp.sh scripts/llamacpp_serve.py 2>/dev/null
  git commit -q -m "llama.cpp baseline: build failed (evidence)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
  say "=== stage 11 done (llama.cpp build failed) ==="; exit 0
fi
note "- Code: llama.cpp commit $(cd $L && git log -1 --format='%h (%cd)' --date=short), llama-server built with CUDA for sm_110."
conv(){  # conv <ckpt> <out.gguf>
  [ -s $2 ] && return 0
  settle; drop
  TMPDIR=$G timeout 14400 scripts/in_cgroup.sh prep max $ZPY $L/convert_hf_to_gguf.py $1 --outtype bf16 --use-temp-file --outfile $2.part > $2.log 2>&1 && mv -f $2.part $2
  [ -s $2 ] || { say "stage 11: GGUF conversion FAILED for $1: $(grep -hE 'Error|error' $2.log | tail -1 | cut -c1-150)"; rm -f $2.part; return 1; }
}
LC(){  # LC <gguf> <tokenizer ckpt> <workload> : the runner up to --budget-gib/--out
  echo "$ZPY scripts/llamacpp_serve.py --server $BIN --gguf $1 --tokenizer $2 --workload results/WORKLOADS/$3.json"; }
for M in "mixtral8x7b /home/thor/kcj/models/mixtral8x7b_bf16 87.0 mmlu" \
         "qwen30b /home/thor/kcj/models/qwen3_30b_a3b 57.0 mmlu,sharegpt,longbench"; do
  set -- $M; m=$1; ck=$2; gb=$3; WLS=${4//,/ }; O=results/MATRIX5/$m; gg=$G/${m}_bf16.gguf
  if [ $m = qwen30b ] && [ ! -s $gg ] && [ $(free_gb) -lt 130 ] && [ -f $ST/ok_delete_gguf ]; then
    rm -f "$G/mixtral8x7b_bf16.gguf"; say "stage 11q: removed the Mixtral bf16 GGUF (done with it; approved) for the Qwen3 conversion"
  fi
  conv $ck $gg || { for w in $WLS; do for f in 0.25 0.45 0.65 1.08; do echo "NORUN budget=- reason=gguf-conversion-failed" > $O/$w/llamacpp_$f.txt; done; done; continue; }
  # 1 smoke
  settle; drop
  timeout 7200 scripts/in_cgroup.sh prep max $(LC $gg $ck mmlu) --budget-gib 20 --limit 2 --out $P/llamacpp/smoke_$m.json > $P/llamacpp/smoke_$m.log 2>&1
  if ! grep -q '^RESULT' $P/llamacpp/smoke_$m.log; then
    say "stage 11: llama.cpp could not serve $m: $(grep -hE 'HOSTGUARD|error|Error|exited' $P/llamacpp/smoke_$m.log | tail -1 | cut -c1-150)"
    note "- $m: smoke failed: $(grep -hE 'HOSTGUARD|error|Error|exited' $P/llamacpp/smoke_$m.log | tail -1 | cut -c1-200) (results/PREP/llamacpp/smoke_$m.log)"
    for w in $WLS; do for f in 0.25 0.45 0.65 1.08; do echo "NORUN budget=- reason=does-not-serve (results/PREP/LLAMACPP.md)" > $O/$w/llamacpp_$f.txt; done; done
    continue
  fi
  say "stage 11: smoke $m $(grep -h '^RESULT' $P/llamacpp/smoke_$m.log | cut -c1-140)"
  # 2 tokens (Qwen3: the two prompts of check_tokens.py against stock transformers)
  if [ $m = qwen30b ]; then
    python3 - <<'PY'
import json
ps = ["Explain why the sky is blue in two sentences.", "Write a Python function that returns the n-th Fibonacci number."]
json.dump([{"name": f"tok{i}", "prompt": p, "max_new": 24} for i, p in enumerate(ps)], open("results/PREP/llamacpp/tok_prompts.json", "w"))
PY
    settle; drop
    timeout 3600 scripts/in_cgroup.sh prep max $ZPY scripts/llamacpp_serve.py --server $BIN --gguf $gg --tokenizer $ck \
      --workload $P/llamacpp/tok_prompts.json --max-new 24 --budget-gib 20 --out $P/llamacpp/tok.json > $P/llamacpp/tok.log 2>&1
    TOK=$(python3 - <<'PY'
import json
try:
    a = json.load(open("results/PREP/llamacpp/tok.json"))["rows"]; b = json.load(open("results/PHASOR_HF/correct_stock.json"))
    ps = ["Explain why the sky is blue in two sentences.", "Write a Python function that returns the n-th Fibonacci number."]
    same = [sum(x == y for x, y in zip(r["out_ids"], b[p])) for r, p in zip(a, ps)]
    print("identical" if all(r["out_ids"] == b[p] for r, p in zip(a, ps)) else f"{same} of 24 tokens equal (CPU bf16 kernels)")
except Exception as e: print("n/a", e)
PY
)
    say "stage 11: llama.cpp vs stock tokens: $TOK"; note "- Tokens vs stock transformers (2 prompts x 24, greedy): $TOK"
  fi
  # 3/4 memcal (knob: the cgroup cap on anon + page cache) and E1
  FR="0.25 0.45 0.65 1.08"; [ $m = mixtral8x7b ] && FR="0.25 0.45 0.65"
  for f in $FR; do
    b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); c=$O/memcal/llamacpp_$b.json
    T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$b.json'))['peak_gib'],2))")
    cal $c $T $b $(LC $gg $ck mmlu) --budget-gib {B} --out {OUT} --limit 2
    say "memcal $m $f llamacpp: $(tail -1 $c.log)"
  done
  for w in $WLS; do for f in $FR; do
    nb=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); o=$O/$w/llamacpp_$f
    T=$(python3 -c "import json;print(round(json.load(open('$O/memcal/phasor_$nb.json'))['peak_gib'],2))")
    kb=$(python3 -c "import json;v=json.load(open('$O/memcal/llamacpp_$nb.json'))['budget_gib'];print(v if v else 'none')" 2>/dev/null || echo none)
    if [ "$kb" = none ]; then   # no cap within PHASOR's peak: the smallest (1 GiB) under 1.4 x that peak
      CAP_GIB=$(awk -v t=$T 'BEGIN{printf "%.2f", 1.4*t}') run "s11_${m}_${w}_llamacpp_${nb}_min" 1 $o $(LC $gg $ck $w) --budget-gib 1 --out $o.json
      over14 $o $T $nb
    else
      CAP_GIB=$(capfor $nb) run "s11_${m}_${w}_llamacpp_$nb" $kb $o $(LC $gg $ck $w) --budget-gib $kb --out $o.json
      if grep -q "^NORUN.*oom-under-cap" $o.txt 2>/dev/null; then
        for k in 0.85 0.7 0.55; do
          kk=$(awk -v a=$kb -v k=$k 'BEGIN{printf "%.2f", a*k}'); ok=$O/$w/llamacpp_${f}_k$k
          CAP_GIB=$(capfor $nb) run "s11_${m}_${w}_llamacpp_${nb}_k$k" $kk $ok $(LC $gg $ck $w) --budget-gib $kk --out $ok.json && break
        done
      fi
    fi
    because $o $nb
    say "E1 $m $w llamacpp $f: $(grep -hE '^(RESULT|NORUN)' $o.txt | tail -1 | cut -c1-150)"
  done; done
  if [ $m = mixtral8x7b ]; then
    cp -f $O/mmlu/phasor_1.08.txt $O/mmlu/llamacpp_1.08.txt 2>/dev/null && sed -i -n '/^NORUN/p' $O/mmlu/llamacpp_1.08.txt
    continue
  fi
  # E2: MMLU 20/15/10/5%, cap = budget (its own setting), stop at the first budget it cannot run at
  stop=""
  for f in 0.20 0.15 0.10 0.05; do
    b=$(awk -v g=$gb -v f=$f 'BEGIN{printf "%.2f", g*f}'); o=$O/mmlu/llamacpp_$f
    T=$(python3 -c "import json;print(round(json.load(open('$O/mmlu/phasor_$f.json'))['peak_gib'],2))")
    if [ -n "$stop" ]; then echo "NORUN budget=$b reason=not-tried (did not run at $stop)" > $o.txt; continue; fi
    CAP_GIB=$(capfor $b) run "s11_${m}_mmlu_llamacpp_$b" $b $o $(LC $gg $ck mmlu) --budget-gib $b --out $o.json
    because $o $b; flag14 $o $T $b
    grep -q '^RESULT' $o.txt 2>/dev/null || stop=$f
    say "E2 llamacpp $f: $(grep -hE '^(RESULT|NORUN)' $o.txt | tail -1 | cut -c1-150)"
  done
  # E3: batch 4 and 8 at 45% (MMLU, ShareGPT), E1's cap, run cap + 8 GiB
  b45=$(awk -v g=$gb 'BEGIN{printf "%.2f", g*0.45}'); mkdir -p $O/extras/e3
  kb45=$(python3 -c "import json;v=json.load(open('$O/memcal/llamacpp_$b45.json'))['budget_gib'];print(v if v else 'none')" 2>/dev/null || echo none)
  for B in 4 8; do for w in mmlu sharegpt; do
    o=$O/extras/e3/${w}_llamacpp_b$B
    if [ "$kb45" = none ]; then echo "NORUN budget=$b45 reason=no-setting-within-phasor-memory-at-45%" > $o.txt; continue; fi
    CAP_GIB=$(awk -v c=$(capfor $b45) 'BEGIN{printf "%.2f", c+8}') run "s11x_${m}_${w}_llamacpp_b$B" $kb45 $o $(LC $gg $ck $w) --budget-gib $kb45 --batch $B --out $o.json
    because $o $b45
  done; done
done
python3 scripts/summarize_matrix5.py > results/MATRIX5/SUMMARY.md 2>>"$LOG"
git add -f $P/LLAMACPP.md $P/llamacpp scripts/stage11_llamacpp.sh scripts/llamacpp_serve.py results/MATRIX5/*/*/llamacpp_* \
  results/MATRIX5/*/memcal/llamacpp_* results/MATRIX5/*/extras/e3/*llamacpp* results/MATRIX5/SUMMARY.md 2>/dev/null
git commit -q -m "llama.cpp (stock mmap offloading) baseline: Qwen3-30B E1/E2/E3, Mixtral MMLU

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" && timeout 300 git push -q origin HEAD
say "=== stage 11q done ==="
