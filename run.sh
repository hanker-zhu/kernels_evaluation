#!/bin/bash
# GEMM 算子框架对比测试
#
# 用法:
#   ./run.sh              # 运行所有框架
#   ./run.sh cublas       # 仅运行 cuBLAS
#   ./run.sh cutlass      # 仅运行 CUTLASS
#   ./run.sh triton       # 仅运行 Triton
#   ./run.sh tilelang     # 仅运行 TileLang
#   ./run.sh compare      # 生成对比报告

set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
FRAMEWORKS_DIR="$BASE_DIR/frameworks"
LOG_DIR="$BASE_DIR/results/logs"

mkdir -p "$LOG_DIR"

# 运行单个框架
run_framework() {
    local name=$1
    local dir="$FRAMEWORKS_DIR/$name"
    local log="$LOG_DIR/${name}_output.log"
    
    [ ! -d "$dir" ] && echo "✗ 框架 $name 不存在" && return 1
    
    echo "► $name"
    cd "$dir"
    chmod +x run.sh
    ./run.sh 2>&1 | tee "$log"
    echo "✓ $name 完成 (日志: $log)"
}

# 运行所有框架
run_all() {
    echo "======================================"
    echo "  GEMM 算子框架对比测试"
    echo "======================================"
    
    for fw in cublas cutlass triton tilelang; do
        echo ""
        run_framework "$fw" || true
    done
    
    echo ""
    echo "======================================"
    echo "  测试完成，日志: $LOG_DIR"
    echo "======================================"
}

# 生成对比报告
run_compare() {
    echo "► 生成对比报告..."
    cd "$BASE_DIR"
    python scripts/compare_frameworks.py
}

# 主逻辑
case "${1:-all}" in
    cublas|cutlass|triton|tilelang)
        run_framework "$1"
        ;;
    compare)
        run_compare
        ;;
    all|"")
        run_all
        ;;
    -h|--help)
        echo "用法: $0 [cublas|cutlass|triton|tilelang|compare|all]"
        ;;
    *)
        echo "未知选项: $1"
        echo "用法: $0 [cublas|cutlass|triton|tilelang|compare|all]"
        exit 1
        ;;
esac
