#!/bin/bash
# Build the three MoE systems that are compared against, for sm_110.
#
# Every one of them names its CUDA architectures somewhere, and none of the
# lists contains Blackwell.  MoE-Infinity's setup.py hardcoded 80/90/120, so
# its CUTLASS GEMMs had no kernel to launch on this device and returned
# "Error Internal"; the same shape of fault is what to expect from the
# others.  Each build below names 110 explicitly and asks for PTX as well, so
# an arch nobody listed can still JIT rather than fail at launch.
set -u
R=/home/thor/kcj/thor_gtier
S=/home/thor/kcj
cd "$R"
. "$R/scripts/torch_env.sh"
LOG=$R/results/SOTA_build.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDAARCHS=110 TORCH_CUDA_ARCH_LIST="11.0" CMAKE_CUDA_ARCHITECTURES=110
export LIBRARY_PATH="/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/sbsa-linux/lib:${LIBRARY_PATH:-}"

# ---- Fiddler: pure PyTorch, no kernels to build -------------------------
# Its pins are from 2024 (torch 2.1.2, transformers 4.36.2) and this machine
# has 2.11; installing them would replace a working sm_110 PyTorch with a
# CPU-only wheel.  The code is orchestration, so it runs against the newer
# stack and --no-deps keeps the pins from being honoured.
say "fiddler"
$TORCH_VENV/bin/pip install -q --no-deps -e "$S/fiddler" 2>&1|tail -2|tee -a "$LOG"
$TORCH_VENV/bin/python -c "
import sys; sys.path.insert(0,'$S/fiddler/src/fiddler')
try:
    import mixtral; print('fiddler import ok')
except Exception as e: print('fiddler import FAILED:', type(e).__name__, e)
" 2>&1|tail -3|tee -a "$LOG"

# ---- Mixtral-offloading: needs HQQ, which has CUDA kernels --------------
say "hqq (for mixtral-offloading)"
$TORCH_VENV/bin/pip install -q --no-deps hqq 2>&1|tail -2|tee -a "$LOG"
$TORCH_VENV/bin/python -c "
try:
    import hqq; print('hqq ok', getattr(hqq,'__version__','?'))
except Exception as e: print('hqq FAILED:', type(e).__name__, e)
" 2>&1|tail -3|tee -a "$LOG"

# ---- Pre-gated MoE: CMake, and its arch list stops at Ampere ------------
say "pregated"
if [ -d "$S/Pregated_MoE" ]; then
  mkdir -p "$S/Pregated_MoE/build"
  ( cd "$S/Pregated_MoE/build" && \
    cmake -DSM_NUM=110 -DCMAKE_CUDA_ARCHITECTURES=110 \
          -DCMAKE_BUILD_TYPE=Release .. 2>&1 | tail -6 ) | tee -a "$LOG"
fi

say "=== sota build done ==="
