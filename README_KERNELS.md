# 多核函数框架对比测试

本项目扩展了原始的GEMM对比测试，支持多个核函数的性能对比。

## 支持的核函数

1. **GEMM** (矩阵乘法) - ✅ 完全支持所有框架
2. **Softmax** - ⚠️ 部分支持 (Triton/TileLang实现中，cuBLAS/CUTLASS待实现)
3. **LayerNorm** - ⚠️ 部分支持 (实现中)

## 使用方法

### 运行所有核函数

```bash
./run_all_kernels.sh all
```

### 运行特定核函数

```bash
./run_all_kernels.sh gemm      # 仅运行 GEMM
./run_all_kernels.sh softmax   # 仅运行 Softmax
./run_all_kernels.sh layernorm # 仅运行 LayerNorm
```

### 生成对比报告

```bash
source /home/yinyishu/miniconda3/etc/profile.d/conda.sh
conda activate kernels
python scripts/compare_all_kernels.py
```

## 当前状态

### GEMM ✅
- **cuBLAS**: ✅ 完全支持
- **CUTLASS**: ✅ 完全支持
- **Triton**: ✅ 完全支持
- **TileLang**: ✅ 完全支持

### Softmax ⚠️
- **Triton**: ✅ 基本支持 (部分尺寸有数值精度问题)
- **TileLang**: ⚠️ 实现中 (API需要调整)
- **cuBLAS/CUTLASS**: ⚠️ 需要cuDNN实现

### LayerNorm ⚠️
- **Triton**: ⚠️ 实现中 (需要测试)
- **TileLang**: ⚠️ 实现中 (API需要调整)
- **cuBLAS/CUTLASS**: ⚠️ 需要cuDNN实现

## 输出文件

- **报告**: `docs/all_kernels_comparison_report.md`
- **JSON结果**: `results/all_kernels_results.json`
- **可视化图表** (每个核函数单独一个):
  - `results/gemm_comparison.png` - GEMM性能对比
  - `results/softmax_comparison.png` - Softmax性能对比
  - `results/layernorm_comparison.png` - LayerNorm性能对比 (如果有数据)
- **日志**: `results/logs/{framework}_{kernel}_output.log`

## 框架对比总结

### GEMM性能 (平均延迟)
- cuBLAS: 0.232 ms (最快)
- CUTLASS: 0.238 ms
- TileLang: 0.262 ms
- Triton: 0.270 ms

### 编译时间
- **AOT框架** (cuBLAS/CUTLASS): 一次性编译，长期使用
- **JIT框架** (Triton/TileLang): 
  - Triton: ~1ms 每尺寸
  - TileLang: ~87ms 每尺寸

## 下一步工作

1. 完善 Softmax 和 LayerNorm 实现
2. 为 cuBLAS/CUTLASS 添加 cuDNN 支持
3. 添加更多核函数 (Conv2D, Attention等)
4. 优化数值精度问题

