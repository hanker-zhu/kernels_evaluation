#!/bin/bash

# Triton GEMM 测试脚本

echo "=== Running Triton GEMM ==="
python triton_gemm.py

echo ""
echo "=== Analyzing Generated Code ==="
echo "Triton JIT compiles Python code to optimized CUDA kernels"
echo "Check the kernel compilation output above for details"
