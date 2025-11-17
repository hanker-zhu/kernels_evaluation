#!/bin/bash

# cuBLAS GEMM 测试脚本

echo "=== Building cuBLAS GEMM ==="
mkdir -p build
cd build
cmake ..
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "=== Running cuBLAS GEMM ==="
    ./cublas_gemm
else
    echo "Build failed!"
    exit 1
fi
