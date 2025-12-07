#!/bin/bash
# TileLang GEMM Benchmark

set -e
cd "$(dirname "$0")"

echo "=== TileLang GEMM Benchmark ==="

# 激活 conda 环境
source /data/hanker/miniconda3/etc/profile.d/conda.sh
conda activate compiler_kernel_eval

python tilelang_gemm.py
