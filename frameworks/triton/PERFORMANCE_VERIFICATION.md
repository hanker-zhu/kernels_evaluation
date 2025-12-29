# Assembly-Aware Autotune 性能验证

## 已准备的测试工具

我已经为您准备了完整的性能验证工具：

### 1. 功能测试脚本
- **文件**: `test_asm_aware.py`
- **功能**: 验证 assembly-aware kernel 的基本功能和正确性
- **运行**: `python test_asm_aware.py`

### 2. 性能对比脚本
- **文件**: `benchmark_asm_aware.py`
- **功能**: 对比原有实现和新的 assembly-aware 实现的性能
- **运行**: `./run_asm_aware_benchmark.sh` 或 `python benchmark_asm_aware.py`

### 3. Assembly-Aware 独立基准测试
- **文件**: `triton_gemm_asm_aware.py` (main 函数)
- **功能**: 单独测试 assembly-aware 实现的性能
- **运行**: `python triton_gemm_asm_aware.py`

## 快速开始

### 步骤 1: 功能验证

首先运行功能测试，确保代码能正常工作：

```bash
cd /mnt/ssd4t/share/yys/kernels_evaluation/frameworks/triton

# 激活环境
source /home/yinyishu/miniconda3/etc/profile.d/conda.sh
conda activate kernels

# 运行功能测试
python test_asm_aware.py
```

**预期输出**: 所有测试应该通过（✓），如果有错误（✗），需要先修复。

### 步骤 2: 性能对比测试

运行性能对比测试，查看性能改进：

```bash
# 运行性能对比（会自动激活环境）
./run_asm_aware_benchmark.sh
```

或者手动运行：

```bash
source /home/yinyishu/miniconda3/etc/profile.d/conda.sh
conda activate kernels
python benchmark_asm_aware.py
```

**预期输出**:
- 每个尺寸的测试结果（baseline_opt, baseline_smart, asm_aware）
- 性能对比表格
- Speedup 和 Improvement 指标
- JSON 结果保存到 `results/asm_aware_benchmark.json`

## 结果解读

### 关键指标

1. **steady_ms**: 稳态运行时间（毫秒），越小越好
2. **TFLOPS**: 计算性能，越大越好
3. **Speedup**: 相对于 baseline_opt 的加速比
   - `> 1.0`: assembly-aware 更快
   - `= 1.0`: 性能相同
   - `< 1.0`: assembly-aware 较慢
4. **Improvement**: 性能提升百分比（%）
   - 正数: 性能提升
   - 负数: 性能下降

### 预期性能表现

根据 assembly-aware autotune 的设计：

1. **小尺寸（256³, 512³）**:
   - 使用 heuristics，避免 autotune 开销
   - 应该与 baseline_smart 性能接近
   - 可能略微提升（1-5%）

2. **中等尺寸（1024³, 2048³）**:
   - 使用 assembly-aware autotune
   - early_config_prune 减少需要 benchmark 的配置
   - 可能提升 5-15%

3. **大尺寸（4096³）**:
   - 使用 assembly-aware autotune
   - 规则过滤找到更好的配置
   - 可能提升 5-20%

### 结果示例

```
Size        Baseline (opt)      Assembly-Aware      Speedup    Improvement
--------------------------------------------------------------------------------
256³        0.0422 ms           0.0400 ms           1.055x     +5.21%
512³        0.0150 ms           0.0140 ms           1.071x     +6.67%
1024³       0.0500 ms           0.0450 ms           1.111x     +10.00%
2048³       0.2000 ms           0.1800 ms           1.111x     +10.00%
4096³       1.0000 ms           0.9000 ms           1.111x     +10.00%
```

## 分析优化效果

### 1. Autotune 时间减少

assembly-aware autotune 通过 `early_config_prune` 减少需要 benchmark 的配置数量（从全部配置减少到 top 20%），从而：

- **减少 autotune 时间**：首次运行时的配置选择时间减少
- **更快收敛**：只对最有希望的配置进行 benchmark

### 2. 配置质量提升

通过规则过滤：

- **Tensor Core 要求**：确保 BK 是 16 的倍数
- **Block Size 优化**：偏好更大的 block size（更好的 Tensor Core 利用率）
- **Warp/Stage 平衡**：考虑并行度和寄存器压力

### 3. 小尺寸优化

使用 heuristics 而不是 autotune：

- **避免启动开销**：小尺寸时 autotune 开销占主导
- **快速执行**：直接使用规则派发的配置

## 进一步优化建议

### 1. 如果性能没有提升

可能的原因和解决方案：

1. **配置列表需要调整**
   - 检查 `_ASM_AWARE_CONFIGS` 是否包含合适的配置
   - 尝试添加更多配置或调整现有配置

2. **规则过滤太严格**
   - 调整 `early_config_prune` 中的评分权重
   - 放宽某些过滤条件

3. **小尺寸阈值不合适**
   - 调整 `triton_matmul_asm_aware` 中的尺寸阈值（当前是 512）

### 2. 如果性能显著提升

可以考虑：

1. **扩展到更多尺寸**
   - 测试更多矩阵尺寸
   - 测试非方形矩阵

2. **优化评分函数**
   - 基于实际测试结果调整 PTX 评分权重
   - 如果能够获取 PTX，启用实际的 PTX 分析

3. **添加 maxnreg 维度**
   - 如果 Triton 版本支持，启用 maxnreg 作为 autotune 维度

### 3. 启用 PTX 分析

当前实现使用基于规则的剪枝。如果能够获取 PTX：

1. 实现 `get_ptx_from_kernel` 或使用 Triton 内部 API
2. 在 `early_config_prune` 中调用实际的 PTX 分析
3. 使用 `score_ptx` 函数对配置进行评分

## 调试和故障排除

### 启用详细日志

修改 `triton_gemm_asm_aware.py` 中的 `early_config_prune` 函数：

```python
verbose = True  # 改为 True
```

这样可以看到配置剪枝的详细过程。

### 检查编译产物

如果 Triton 支持，可以设置环境变量查看编译产物：

```bash
export TRITON_DUMP_IR=1
export TRITON_DUMP_PTX=1
python benchmark_asm_aware.py
```

### 对比配置选择

可以添加日志记录 autotune 最终选择的配置，对比 baseline 和 assembly-aware 选择的配置差异。

## 总结

通过运行这些测试，您可以：

1. ✅ 验证功能正确性
2. ✅ 对比性能改进
3. ✅ 分析优化效果
4. ✅ 找到进一步优化的方向

如果测试结果符合预期，说明 assembly-aware autotune 实现成功！如果还有改进空间，可以根据测试结果进行调整和优化。

