#!/bin/bash
# Triton GEMM Benchmark

set -e
cd "$(dirname "$0")"

echo "=== Triton GEMM Benchmark ==="

# 激活 conda 环境
source /data/hanker/miniconda3/etc/profile.d/conda.sh
conda activate compiler_kernel_eval

python triton_gemm.py
