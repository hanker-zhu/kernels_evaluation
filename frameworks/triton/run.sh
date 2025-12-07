#!/bin/bash
# Triton GEMM Benchmark

set -e
cd "$(dirname "$0")"

echo "=== Triton GEMM Benchmark ==="

# 使用 CUDA 12.8
CUDA_PATH="/usr/local/cuda-12.8"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"


# 激活 conda 环境
source /home/yinyishu/miniconda3/etc/profile.d/conda.sh
conda activate kernels

python triton_gemm.py
