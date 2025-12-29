# 基准测试统计功能说明

## 新增统计功能

基准测试脚本 `benchmark_asm_aware.py` 现在包含以下增强的统计功能：

### 1. 编译时间统计

**指标**:
- `compile_time_ms`: 首次编译 kernel 所需的时间（毫秒）

**测量方法**:
- 记录首次调用 kernel 前到调用完成的时间
- 包含 Triton JIT 编译、PTX 编译、cubin 生成等所有编译阶段

**用途**:
- 评估编译开销
- 对比不同实现的编译时间
- 优化编译流程

### 2. 编译质量统计

**指标**:
- `first_steady_ratio`: 首次运行时间与稳态时间的比值
- `stability_cv`: 性能稳定性（变异系数，Coefficient of Variation）
- `quality_score`: 综合编译质量评分（0.0-1.0）

**评分标准**:

#### First-Steady Ratio 评分
- `ratio < 1.1`: 评分 1.0（优秀，首次运行接近稳态）
- `1.1 ≤ ratio < 1.5`: 评分 0.8（良好）
- `ratio ≥ 1.5`: 评分 0.5（一般，首次运行开销较大）

#### Stability CV 评分
- `CV < 0.01`: 评分 1.0（非常稳定）
- `0.01 ≤ CV < 0.05`: 评分 0.8（稳定）
- `CV ≥ 0.05`: 评分 0.6（不够稳定）

#### 综合质量评分
- `quality_score = first_steady_ratio_score * 0.6 + stability_score * 0.4`

**用途**:
- 评估编译后的代码质量
- 判断 JIT 优化效果
- 识别性能不稳定的问题

### 3. Autotune 质量统计（仅 Assembly-Aware）

**指标**:
- `total_configs`: 初始配置总数
- `pruned_configs`: 被剪枝掉的配置数量
- `benchmarked_configs`: 实际进行 benchmark 的配置数量
- `prune_time_ms`: 配置剪枝所需的时间（毫秒）

**说明**:
- 仅对 `asm_aware` 实现有效
- 小尺寸（≤512）使用 heuristics 而不是 autotune，统计为 N/A

**用途**:
- 评估 early_config_prune 的效果
- 计算剪枝比例（prune_ratio = pruned / total * 100）
- 优化 autotune 流程

## 输出格式

### 1. 性能对比表格

```
Size        Baseline (opt)      Assembly-Aware      Speedup    Improvement
--------------------------------------------------------------------------------
256³           0.0422 ms           0.0400 ms           1.055x     +5.21%
512³           0.0150 ms           0.0140 ms           1.071x     +6.67%
...
```

### 2. 编译时间统计表格

```
Size        Baseline Compile   ASM-Aware Compile   Compile Speedup
--------------------------------------------------------------------------------
256³           125.50 ms           98.30 ms           1.28x
512³           142.30 ms          115.20 ms           1.24x
...
```

### 3. 编译质量统计表格

```
Size        Baseline Quality   ASM-Aware Quality   Quality Δ
--------------------------------------------------------------------------------
256³             0.850             0.920            +0.070
512³             0.880             0.930            +0.050
...
```

### 4. Autotune 质量统计表格

```
Size        Total Configs   Pruned   Benchmarked   Prune Time   Prune Ratio
--------------------------------------------------------------------------------
256³        Heuristics      N/A      N/A           N/A          N/A
512³        Heuristics      N/A      N/A           N/A          N/A
1024³           8               6          2            0.15 ms       75.0%
2048³           8               6          2            0.18 ms       75.0%
4096³           8               6          2            0.20 ms       75.0%
```

## JSON 输出格式

所有统计数据都保存在 JSON 文件中，结构如下：

```json
{
  "M": 1024,
  "N": 1024,
  "K": 1024,
  "baseline_opt": {
    "first_ms": 0.0523,
    "steady_ms": 0.0500,
    "steady_std_ms": 0.0012,
    "compile_time_ms": 125.50,
    "compile_quality": {
      "first_steady_ratio": 1.046,
      "stability_cv": 0.0240,
      "quality_score": 0.850
    },
    "tflops": 41.94,
    "success": true
  },
  "asm_aware": {
    "first_ms": 0.0475,
    "steady_ms": 0.0450,
    "steady_std_ms": 0.0010,
    "compile_time_ms": 98.30,
    "compile_quality": {
      "first_steady_ratio": 1.056,
      "stability_cv": 0.0222,
      "quality_score": 0.920
    },
    "autotune_stats": {
      "total_configs": 8,
      "pruned_configs": 6,
      "benchmarked_configs": 2,
      "prune_time_ms": 0.15
    },
    "tflops": 46.60,
    "success": true
  },
  "speedup": 1.111,
  "improvement_pct": 10.00
}
```

## 解读统计结果

### 编译时间

- **低编译时间（< 100ms）**: 编译开销小，适合频繁调用
- **中等编译时间（100-500ms）**: 可接受，适合一次性编译多次使用
- **高编译时间（> 500ms）**: 编译开销大，考虑优化或预编译

### 编译质量评分

- **0.9-1.0**: 优秀，编译质量很好
- **0.8-0.9**: 良好，编译质量不错
- **0.7-0.8**: 一般，可能有优化空间
- **< 0.7**: 较差，需要检查编译选项或配置

### Autotune 剪枝比例

- **高剪枝比例（> 70%）**: early_config_prune 效果很好，大幅减少了需要 benchmark 的配置
- **中等剪枝比例（40-70%）**: 剪枝效果中等
- **低剪枝比例（< 40%）**: 剪枝效果有限，可能需要优化剪枝规则

### 性能稳定性

- **CV < 0.01**: 非常稳定，性能波动很小
- **0.01 ≤ CV < 0.05**: 稳定，性能波动在可接受范围
- **CV ≥ 0.05**: 不够稳定，可能需要进一步优化

## 使用建议

1. **编译时间优化**: 如果编译时间过长，考虑：
   - 减少 autotune 配置数量
   - 使用更激进的 early_config_prune
   - 对小尺寸使用 heuristics 而不是 autotune

2. **编译质量优化**: 如果质量评分较低，检查：
   - 首次运行时间是否异常高（可能是 JIT 缓存问题）
   - 性能稳定性是否足够（可能是 GPU 频率波动或系统负载）

3. **Autotune 优化**: 如果剪枝效果不好，考虑：
   - 优化 early_config_prune 的评分规则
   - 添加更多硬性过滤条件
   - 调整 top_k 比例

## 注意事项

1. **首次运行开销**: 首次运行可能包含一些非编译的开销（如 JIT 缓存、内存分配等），这些也会计入 `compile_time_ms`

2. **小尺寸使用 Heuristics**: 对于小尺寸（≤512），assembly-aware 实现使用 heuristics 而不是 autotune，所以 autotune_stats 为空

3. **性能稳定性**: 多次运行的结果可能会有波动，这是正常的。我们通过多次测量取平均值和标准差来评估稳定性

4. **编译时间测量**: 编译时间是在 Python 层面测量的，可能受到 Python 解释器开销的影响

