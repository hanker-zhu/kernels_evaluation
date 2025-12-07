#!/bin/bash

# TileLang GEMM 测试脚本

echo "=== Running TileLang GEMM ==="
python tilelang_gemm.py

echo ""
echo "=== Analyzing Generated Code ==="
echo "TileLang generates optimized CUDA code at runtime"
echo "Check the kernel compilation output above for details"
