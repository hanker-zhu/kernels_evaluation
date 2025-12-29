#!/bin/bash
# 多核函数对比测试脚本
#
# 用法:
#   ./run_all_kernels.sh              # 运行所有框架的所有核函数
#   ./run_all_kernels.sh gemm         # 仅运行 GEMM
#   ./run_all_kernels.sh softmax      # 仅运行 Softmax
#   ./run_all_kernels.sh layernorm    # 仅运行 LayerNorm

set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
FRAMEWORKS_DIR="$BASE_DIR/frameworks"
LOG_DIR="$BASE_DIR/results/logs"

mkdir -p "$LOG_DIR"

# 支持的核函数
KERNELS=("gemm" "softmax" "layernorm")
FRAMEWORKS=("cublas" "cutlass" "triton" "tilelang")

# 运行单个核函数
run_kernel() {
    local kernel=$1
    local framework=$2
    local framework_dir="$FRAMEWORKS_DIR/$framework"
    local log="$LOG_DIR/${framework}_${kernel}_output.log"
    
    echo "  ► $framework ($kernel)"
    
    case "$framework" in
        triton)
            if [ -f "$framework_dir/triton_${kernel}.py" ]; then
                cd "$framework_dir"
                source /home/yinyishu/miniconda3/etc/profile.d/conda.sh
                conda activate kernels
                python "triton_${kernel}.py" 2>&1 | tee "$log"
            else
                echo "    ✗ triton_${kernel}.py not found"
            fi
            ;;
        tilelang)
            if [ -f "$framework_dir/tilelang_${kernel}.py" ]; then
                cd "$framework_dir"
                source /home/yinyishu/miniconda3/etc/profile.d/conda.sh
                conda activate kernels
                CUDA_PATH="/usr/local/cuda-12.8"
                export PATH="$CUDA_PATH/bin:$PATH"
                export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
                python "tilelang_${kernel}.py" 2>&1 | tee "$log"
            else
                echo "    ✗ tilelang_${kernel}.py not found"
            fi
            ;;
        cublas|cutlass)
            # 对于C++框架，目前只支持GEMM
            if [ "$kernel" = "gemm" ]; then
                cd "$framework_dir"
                chmod +x run.sh
                ./run.sh 2>&1 | tee "$log"
            else
                echo "    ⚠ $framework does not support $kernel yet"
            fi
            ;;
    esac
}

# 运行所有核函数
run_all_kernels() {
    local kernel_filter="${1:-all}"
    
    echo "======================================"
    echo "  多核函数框架对比测试"
    echo "======================================"
    
    for kernel in "${KERNELS[@]}"; do
        if [ "$kernel_filter" != "all" ] && [ "$kernel_filter" != "$kernel" ]; then
            continue
        fi
        
        echo ""
        echo "=== $kernel ==="
        
        for framework in "${FRAMEWORKS[@]}"; do
            run_kernel "$kernel" "$framework" || true
        done
    done
    
    echo ""
    echo "======================================"
    echo "  测试完成，日志: $LOG_DIR"
    echo "======================================"
}

# 主逻辑
case "${1:-all}" in
    gemm|softmax|layernorm)
        run_all_kernels "$1"
        ;;
    all|"")
        run_all_kernels "all"
        ;;
    -h|--help)
        echo "用法: $0 [gemm|softmax|layernorm|all]"
        ;;
    *)
        echo "未知选项: $1"
        echo "用法: $0 [gemm|softmax|layernorm|all]"
        exit 1
        ;;
esac

