#!/bin/bash

# GEMM算子生成框架多尺寸对比测试 - 主运行脚本

echo "======================================================"
echo "  GEMM算子生成框架多尺寸深度对比分析"
echo "======================================================"
echo ""

BASE_DIR="/data/hanker/kernels"
RESULTS_DIR="$BASE_DIR/results"
mkdir -p "$RESULTS_DIR"

# 设置环境变量
export PYTHONPATH="/data/hanker/tilelang:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU

echo "测试环境信息:"
echo "- CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# 测试函数
run_benchmark() {
    local framework=$1
    local dir="$BASE_DIR/$framework"
    local framework_name=$(basename "$framework")
    local log_file="$RESULTS_DIR/logs/${framework_name}_output.log"

    echo "----------------------------------------"
    echo "运行 $framework 测试..."
    echo "----------------------------------------"

    cd "$dir"
    if [ -f "run.sh" ]; then
        chmod +x run.sh
        ./run.sh 2>&1 | tee "$log_file"
        local exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
            echo "✓ $framework 测试完成"
        else
            echo "✗ $framework 测试失败 (退出码: $exit_code)"
        fi
    else
        echo "✗ $framework 缺少 run.sh 脚本"
        return 1
    fi

    echo ""
    return $exit_code
}

# 检查依赖
check_dependencies() {
    echo "检查依赖..."

    # 检查CUDA
    if ! command -v nvcc &> /dev/null; then
        echo "警告: 未找到 nvcc，CUDA框架可能无法工作"
    fi

    # 检查Python
    if ! command -v python &> /dev/null; then
        echo "错误: 未找到 python"
        exit 1
    fi

    # 检查CMake
    if ! command -v cmake &> /dev/null; then
        echo "警告: 未找到 cmake，C++框架可能无法编译"
    fi

    # 检查Python包
    python -c "import torch; import triton" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "警告: 缺少 PyTorch 或 Triton"
    fi

    echo "依赖检查完成"
    echo ""
}

# 主测试流程
main() {
    check_dependencies

    # 运行各个框架的测试
    declare -a frameworks=("frameworks/cublas" "frameworks/cutlass" "frameworks/tilelang" "frameworks/triton")
    declare -A results

    for framework_path in "${frameworks[@]}"; do
        framework=$(basename "$framework_path")
        if run_benchmark "$framework_path"; then
            results[$framework]="success"
        else
            results[$framework]="failed"
        fi
    done

    # 生成对比报告
    echo "======================================================"
    echo "生成对比分析报告..."
    echo "======================================================"

    cd "$BASE_DIR"
    if [ -f "scripts/compare_frameworks.py" ]; then
        echo "解析测试日志并生成报告..."
        python scripts/compare_frameworks.py --parse-logs
    else
        echo "警告: 未找到 scripts/compare_frameworks.py"
    fi

    # 显示总结
    echo ""
    echo "======================================================"
    echo "多尺寸测试总结"
    echo "======================================================"

    for framework in "${frameworks[@]}"; do
        status=${results[$framework]}
        if [ "$status" = "success" ]; then
            echo "✓ $framework: 多尺寸测试成功"
        else
            echo "✗ $framework: 多尺寸测试失败"
        fi
    done

    echo ""
    echo "测试配置:"
    echo "- 矩阵尺寸: 128³ 到 4096³ (包括非方形矩阵)"
    echo "- 框架数量: ${#frameworks[@]}"
    echo "- 总测试点: 约30+个尺寸配置"
    echo ""
    echo "生成的文件:"
    echo "- 详细日志: $RESULTS_DIR/logs/"
    echo "- 多尺寸对比报告: $BASE_DIR/docs/comparison_report.md"
    echo "- 技术分析: $BASE_DIR/docs/technical_analysis.md"
    echo "- 性能趋势可视化: $RESULTS_DIR/performance_comparison.png (如果matplotlib可用)"
    echo "- 基准结果: $RESULTS_DIR/benchmark_results.json"

    echo ""
    echo "测试完成！请查看上述文件获取详细的多尺寸性能分析结果。"
}

# 执行主函数
main "$@"
