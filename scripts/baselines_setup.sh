#!/bin/bash
# Building the top-tier baselines on Jetson Thor (sm_110, CUDA 13.0).
#
# All three run here.  The obstacles were not that the systems are unsupported
# but four specific packaging problems, recorded so the setup is reproducible.
set -eu
. "$(dirname "$0")/torch_env.sh"
V=$TORCH_VENV
SRC=${SRC:-/home/thor/kcj}

# ---------------------------------------------------------------- PowerInfer
# Needs no PyTorch at all: it is a llama.cpp fork.  Its CMakeLists defaults to
# CUDA architectures 52/61/70, so sm_110 has to be named explicitly.
build_powerinfer() {
    cd "$SRC/PowerInfer"
    cmake -S . -B build -DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=110 \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j8
    cuobjdump --list-elf build/bin/main | grep -o 'sm_[0-9]*' | sort -u
}

# ------------------------------------------------------------------ FlexGen
# Plain Python.  Note the import name is flexllmgen, not flexgen.
build_flexgen() {
    cd "$SRC/FlexLLMGen"
    $V/bin/pip install -e .
    $V/bin/python -c "from flexllmgen.flex_opt import Policy; print('flexgen ok')"
}

# ------------------------------------------------------------- MoE-Infinity
# Four things have to be right, in this order:
#   1. moe-store is a separate repo (EfficientMoE/moe-store), not on PyPI, and
#      MoE-Infinity pins moe-store~=0.2.1 while that repo's main builds 0.0.0 --
#      relax the pin.
#   2. CUTLASS headers are not vendored; clone them and put them on CPATH.
#   3. nvtx3 ships with CUDA 13 under targets/sbsa-linux/include.
#   4. /usr/local/cuda points at 13.2 here, which has no lib64/libcublas, so the
#      link fails with "cannot find -lcublas" unless CUDA_HOME names 13.0.
build_moe_infinity() {
    [ -d "$SRC/cutlass" ] || git clone --depth 1 --branch v3.5.1 \
        https://github.com/NVIDIA/cutlass.git "$SRC/cutlass"
    cd "$SRC/moe-store" && $V/bin/pip install --no-build-isolation .

    cd "$SRC/MoE-Infinity"
    sed -i 's/moe-store~=0\.2\.1/moe-store/g' pyproject.toml requirements.txt setup.py 2>/dev/null || true
    export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
    export PATH=/usr/local/cuda-13.0/bin:$PATH
    export CUDAARCHS=110 TORCH_CUDA_ARCH_LIST="11.0" CMAKE_CUDA_ARCHITECTURES=110
    export CPATH="$SRC/cutlass/include:/usr/local/cuda-13.0/targets/sbsa-linux/include"
    export CPLUS_INCLUDE_PATH="$CPATH"
    export LIBRARY_PATH="/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/sbsa-linux/lib:${LIBRARY_PATH:-}"
    export NVCC_PREPEND_FLAGS="-I$SRC/cutlass/include -I/usr/local/cuda-13.0/targets/sbsa-linux/include"
    $V/bin/pip install --no-build-isolation .
    cd /tmp && $V/bin/python -c "
import moe_infinity, os, glob
d = os.path.dirname(moe_infinity.__file__)
print('moe_infinity ok:', [os.path.basename(x) for x in glob.glob(d + '/*.so')])"
}

case "${1:-all}" in
    powerinfer) build_powerinfer ;;
    flexgen)    build_flexgen ;;
    moe)        build_moe_infinity ;;
    all)        build_powerinfer; build_flexgen; build_moe_infinity ;;
esac
