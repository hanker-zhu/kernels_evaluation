#!/bin/bash

# cuBLAS GEMM 测试脚本

echo "=== Building cuBLAS GEMM ==="
mkdir -p build
cd build
cmake ..
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "=== Analyzing cuBLAS Binary ==="
    # 保存编译信息
    COMPILE_OUTPUT_DIR="/data/hanker/kernels/compile_outputs"
    mkdir -p "$COMPILE_OUTPUT_DIR"

    # 获取二进制文件大小
    BINARY_SIZE=$(stat -c%s ./cublas_gemm)
    echo "cuBLAS binary size: $BINARY_SIZE bytes" > "$COMPILE_OUTPUT_DIR/cublas_binary_info.txt"

    # 尝试反汇编（如果有objdump）
    if command -v objdump &> /dev/null; then
        echo "Generating cuBLAS disassembly..." >> "$COMPILE_OUTPUT_DIR/cublas_binary_info.txt"
        objdump -d ./cublas_gemm > "$COMPILE_OUTPUT_DIR/cublas_gemm.s" 2>/dev/null || echo "Disassembly failed" >> "$COMPILE_OUTPUT_DIR/cublas_binary_info.txt"
    fi

    # 获取编译器版本信息
    echo "Compiler version:" >> "$COMPILE_OUTPUT_DIR/cublas_binary_info.txt"
    nvcc --version >> "$COMPILE_OUTPUT_DIR/cublas_binary_info.txt" 2>&1

    echo "=== Running cuBLAS GEMM ==="
    ./cublas_gemm
else
    echo "Build failed!"
    exit 1
fi
