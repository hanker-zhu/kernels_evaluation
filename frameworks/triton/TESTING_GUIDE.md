# Assembly-Aware Autotune 测试指南

## 快速测试

### 1. 功能验证测试

运行基本功能测试，验证代码能正常工作：

```bash
cd frameworks/triton
source /home/yinyishu/miniconda3/etc/profile.d/conda.sh
conda activate kernels
python test_asm_aware.py
```

这个测试会：
- 验证基本功能（256x256x256）
- 测试多个尺寸（128³, 256³, 512³, 1024³）
- 检查数值正确性

### 2. 性能对比基准测试

运行完整的性能对比测试，对比原有实现和新的 assembly-aware 实现：

```bash
cd frameworks/triton
./run_asm_aware_benchmark.sh
```

或者直接运行：

```bash
cd frameworks/triton
source /home/yinyishu/miniconda3/etc/profile.d/conda.sh
conda activate kernels
python benchmark_asm_aware.py
```

### 3. 单独运行 Assembly-Aware 基准测试

运行 assembly-aware kernel 的完整基准测试：

```bash
cd frameworks/triton
source /home/yinyishu/miniconda3/etc/profile.d/conda.sh
conda activate kernels
python triton_gemm_asm_aware.py
```

## 测试内容

### 功能测试 (test_asm_aware.py)

- ✅ 基本功能验证（256x256x256）
- ✅ 多尺寸测试（128³, 256³, 512³, 1024³）
- ✅ 数值正确性验证

### 性能基准测试 (benchmark_asm_aware.py)

对比以下实现：
1. **baseline_opt**: 原有的 `triton_matmul_opt_staticK` 实现
2. **baseline_smart**: 原有的 `triton_matmul_opt_smart` 实现（智能派发）
3. **asm_aware**: 新的 assembly-aware autotune 实现

测试尺寸：256³, 512³, 1024³, 2048³, 4096³

输出指标：
- `first_ms`: 首次运行时间（包含编译等一次性开销）
- `steady_ms`: 稳态运行时间
- `tflops`: TFLOPS 性能
- `speedup`: 相对于 baseline_opt 的加速比
- `improvement_pct`: 性能提升百分比

### Assembly-Aware 基准测试 (triton_gemm_asm_aware.py)

单独测试 assembly-aware 实现，输出各尺寸的性能数据。

## 预期结果

### 性能优化预期

根据 assembly-aware autotune 的设计目标：

1. **小尺寸（≤512）**：
   - 使用 heuristics，避免 autotune 开销
   - 应该与 baseline_smart 性能接近或更好

2. **大尺寸（>512）**：
   - 使用 assembly-aware autotune
   - 通过 early_config_prune 减少需要 benchmark 的配置数量
   - 应该能够找到更优的配置，性能提升可能达到 5-15%

3. **整体改进**：
   - 减少 autotune 时间（通过 early_config_prune）
   - 更好的配置选择（通过规则过滤和评分）

### 注意事项

1. **首次运行**：首次运行会有编译开销，后续运行会使用缓存
2. **GPU 占用**：确保 GPU 可用且没有被其他程序占用
3. **环境要求**：需要 CUDA 12.8 和 conda 环境 "kernels"

## 结果分析

测试结果会保存到 `results/asm_aware_benchmark.json`。

关键指标：
- **Speedup > 1.0**: assembly-aware 实现更快
- **Improvement > 0%**: 性能提升
- **TFLOPS**: 越高越好，可以对比不同尺寸的性能

## 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'torch'**
   - 解决：确保激活了正确的 conda 环境：`conda activate kernels`

2. **CUDA out of memory**
   - 解决：减小测试尺寸，或者关闭其他使用 GPU 的程序

3. **编译错误**
   - 检查 Triton 版本是否支持相关功能
   - 查看错误信息中的具体原因

4. **数值精度错误**
   - FP16 精度问题，如果误差在合理范围内（rtol=5e-2, atol=0.1）是正常的
   - 如果误差过大，检查 kernel 实现

## 进阶测试

### 启用详细日志

如果想看到 early_config_prune 的详细输出，可以修改 `triton_gemm_asm_aware.py` 中的 `early_config_prune` 函数，将 `verbose` 设置为 `True`。

### 测试不同的配置

可以修改 `_ASM_AWARE_CONFIGS` 列表，添加或删除配置，测试不同配置组合的效果。

### 测试 maxnreg 支持

如果 Triton 版本支持 maxnreg，可以使用 `generate_asm_aware_configs()` 函数生成包含 maxnreg 的配置。

