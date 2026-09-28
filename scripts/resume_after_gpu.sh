#!/bin/bash
# Resume after the GPU is back (09-29): at 23:52 on 09-28 the Qwen3 GGUF conversion exhausted host
# memory (the host guard polls every 0.5 s; MemAvailable reached 7.9 GiB), the NVIDIA driver then
# failed to suspend the GPU ("GSP unload failed", dmesg) and the GPU has been unavailable since
# (nvidia-smi: "Unable to determine the device handle"; torch: "No CUDA GPUs are available").
# Runs the remaining chain in the order the user set (09-29 02:45):
#   stage 12b  MoE-Infinity 2024-08 release, Mixtral MMLU
#   stage 11q  llama.cpp Qwen3 E1 + E3, then Mixtral            (SKIP_E2)
#   stage 11q  llama.cpp Qwen3 E2                                (ONLY_E2)
set -u
R=/home/thor/kcj/thor_gtier; cd "$R"; . scripts/torch_env.sh
L=results/PIPELINE/pipeline.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
if ! timeout 120 $TORCH_VENV/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  say "resume_after_gpu: the GPU is still unavailable; not started"; exit 1
fi
say "resume_after_gpu: GPU available; stage 12b, then llama.cpp (Qwen3 E1+E3, Mixtral), then llama.cpp Qwen3 E2"
NOWAIT=1 SETTLE_GIB=${SETTLE_GIB:-75} bash scripts/stage12b_moeinf_legacy.sh >> results/PIPELINE/stage12b2_driver.out 2>&1
SKIP_E2=1 NOWAIT=1 SETTLE_GIB=${SETTLE_GIB:-75} bash scripts/stage11q_llamacpp.sh >> results/PIPELINE/stage11q_driver.out 2>&1
ONLY_E2=1 NOWAIT=1 SETTLE_GIB=${SETTLE_GIB:-75} bash scripts/stage11q_llamacpp.sh >> results/PIPELINE/stage11e2_driver.out 2>&1
say "resume_after_gpu: chain finished"
