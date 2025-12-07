#!/bin/bash

# CUTLASS GEMM 测试脚本

echo "=== Building CUTLASS GEMM ==="
mkdir -p build
cd build
cmake ..
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "=== Analyzing CUTLASS Binary ==="
    # 保存编译信息
    COMPILE_OUTPUT_DIR="/data/hanker/kernels/compile_outputs"
    mkdir -p "$COMPILE_OUTPUT_DIR"

    # 获取二进制文件大小
    BINARY_SIZE=$(stat -c%s ./cutlass_gemm)
    echo "CUTLASS binary size: $BINARY_SIZE bytes" > "$COMPILE_OUTPUT_DIR/cutlass_binary_info.txt"

    # 尝试提取SASS代码
    if command -v cuobjdump &> /dev/null; then
        echo "Generating CUTLASS SASS code..." >> "$COMPILE_OUTPUT_DIR/cutlass_binary_info.txt"
        cuobjdump -sass ./cutlass_gemm > "$COMPILE_OUTPUT_DIR/cutlass_gemm.sass" 2>/dev/null || echo "SASS extraction failed" >> "$COMPILE_OUTPUT_DIR/cutlass_binary_info.txt"
        echo "SASS code saved to $COMPILE_OUTPUT_DIR/cutlass_gemm.sass"
    else
        echo "cuobjdump not found, skipping SASS extraction" >> "$COMPILE_OUTPUT_DIR/cutlass_binary_info.txt"
    fi

    # 获取编译器版本信息
    echo "Compiler version:" >> "$COMPILE_OUTPUT_DIR/cutlass_binary_info.txt"
    nvcc --version >> "$COMPILE_OUTPUT_DIR/cutlass_binary_info.txt" 2>&1

    echo "=== Running CUTLASS GEMM ==="
    ./cutlass_gemm
else
    echo "Build failed!"
    exit 1
fi