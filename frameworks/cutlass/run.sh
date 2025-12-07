#!/bin/bash
# CUTLASS GEMM Benchmark - 带编译时间测量

set -e
cd "$(dirname "$0")"

# 使用 CUDA 12.8
CUDA_PATH="/usr/local/cuda-12.8"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

echo "=== CUTLASS GEMM Benchmark ==="

# 检查是否需要编译
if [ ! -f "build/cutlass_gemm" ]; then
    echo "编译中..."
    rm -rf build
    mkdir -p build && cd build
    
    # 测量编译时间
    COMPILE_START=$(date +%s%3N)
    cmake .. -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" > /dev/null 2>&1
    make -j$(nproc) > /dev/null 2>&1
    COMPILE_END=$(date +%s%3N)
    COMPILE_TIME=$((COMPILE_END - COMPILE_START))
    
    cd ..
    echo "编译时间: ${COMPILE_TIME} ms"
    echo "COMPILE_TIME_MS: ${COMPILE_TIME}"
else
    echo "使用已编译的二进制文件"
    echo "COMPILE_TIME_MS: 0"
fi

./build/cutlass_gemm
