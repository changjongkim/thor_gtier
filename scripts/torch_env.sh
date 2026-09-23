#!/bin/bash
# PyTorch with CUDA on Jetson Thor (sm_110).
#
# PyPI's aarch64 wheel is CPU-only; the CUDA build lives on NVIDIA's Jetson AI
# Lab index.  Passing --extra-index-url alongside it lets pip pick the PyPI
# wheel instead, since the versions match, so torch is installed --no-deps from
# that index alone and its dependencies come from PyPI separately.
#
# The SBSA wheel also links against NVPL and cuDSS without declaring them, so
# both have to be on LD_LIBRARY_PATH or `import torch` fails with
# libnvpl_lapack_lp64_gomp.so.0 / libcudss.so.0 not found.
export TORCH_VENV=${TORCH_VENV:-/home/thor/Envs/t26}
export NVPL_LIB=/home/thor/.cache/quiet-pantheon/venv/lib/python3.12/site-packages/nvpl/lib
export CUDSS_LIB=$TORCH_VENV/lib/python3.12/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH=$NVPL_LIB:$CUDSS_LIB:/usr/local/cuda-13.0/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}
export PY=$TORCH_VENV/bin/python

# Install, if the venv is not there yet:
#   python3 -m venv $TORCH_VENV
#   $TORCH_VENV/bin/pip install numpy filelock typing-extensions sympy networkx jinja2 fsspec
#   $TORCH_VENV/bin/pip install --no-deps --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130 torch
#   $TORCH_VENV/bin/pip install nvidia-cudss-cu13
