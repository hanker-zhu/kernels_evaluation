#!/bin/bash

# CUTLASS GEMM 测试脚本

echo "=== Building CUTLASS GEMM ==="
mkdir -p build
cd build
cmake ..
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "=== Running CUTLASS GEMM ==="
    ./cutlass_gemm

    echo ""
    echo "=== Analyzing Generated Code ==="
    echo "Extracting SASS code for analysis..."
    if command -v cuobjdump &> /dev/null; then
        cuobjdump -sass ./cutlass_gemm > ../cutlass_sass.txt
        echo "SASS code saved to cutlass_sass.txt"
    else
        echo "cuobjdump not found, skipping SASS extraction"
    fi
else
    echo "Build failed!"
    exit 1
fi