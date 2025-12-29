# Assembly-Aware Autotune for Triton GEMM

本文档说明如何使用 assembly-aware autotune 来优化 Triton GEMM kernel，目标是生成更高质量的汇编代码（接近 CUTLASS 的水平）。

## 核心思路

根据您提供的建议，我们实现了以下功能：

1. **PTX 分析和评分**：基于 PTX 中的指令（mma.sync、cp.async、bar.sync、bra）进行评分
2. **Early Config Pruning**：在 autotune 的早期阶段，基于规则和 PTX 分析剪枝配置
3. **maxnreg 维度**：将寄存器限制作为 autotune 维度（框架已提供，需要 Triton 版本支持）
4. **小尺寸 Heuristics**：对于小尺寸矩阵，使用 heuristics 而不是 autotune，避免启动开销

## 文件说明

- `triton_gemm_asm_aware.py`: 主实现文件，包含 assembly-aware autotune kernel

## 使用方法

### 基本使用

```python
import torch
from triton_gemm_asm_aware import triton_matmul_asm_aware

# 创建输入矩阵
A = torch.randn((1024, 1024), device='cuda', dtype=torch.float16)
B = torch.randn((1024, 1024), device='cuda', dtype=torch.float16)
C = torch.empty((1024, 1024), device='cuda', dtype=torch.float16)

# 执行 GEMM
triton_matmul_asm_aware(A, B, C)
```

### 运行基准测试

```bash
cd frameworks/triton
python triton_gemm_asm_aware.py
```

## 实现细节

### 1. PTX 评分函数

`score_ptx()` 函数分析 PTX 代码并给出评分：

- `mma.sync`: +10 分（Tensor Core 指令，最重要）
- `cp.async`: +2 分（异步拷贝，有利于隐藏延迟）
- `bar.sync`: -30 分（同步屏障，降低并行度）
- `bra/ret`: -5 分（分支指令，可能影响调度）

### 2. Early Config Pruning

`early_config_prune()` 函数基于以下规则过滤配置：

1. **Tensor Core 要求**：BK 必须是 16 的倍数
2. **Block Size 启发式**：偏好更大的 block size（有利于 Tensor Core 利用率）
3. **Warp 和 Stage 启发式**：考虑 num_warps 和 num_stages 的平衡

当前实现使用基于规则的剪枝（因为获取 PTX 需要额外工作）。未来可以扩展为基于实际 PTX 分析的剪枝。

### 3. maxnreg 支持

代码中提供了 `generate_asm_aware_configs()` 函数，它会为每个配置尝试添加不同的 `maxnreg` 值（80, 96, 112, 128, 144, 160）。

**注意**：`maxnreg` 的支持取决于 Triton 版本。如果当前版本不支持，代码会自动回退到不带 `maxnreg` 的配置。

### 4. 小尺寸优化

对于小尺寸矩阵（≤512），使用 `matmul_heuristics_kernel` 和 `@triton.heuristics` 装饰器，避免 autotune 的开销。

## 扩展方向

### 1. 实际 PTX 提取和分析

当前代码提供了 `score_ptx()` 和相关的框架，但实际获取 PTX 需要：

1. **方法 A**：使用 `triton.compile` 并访问编译产物
2. **方法 B**：使用环境变量 `TRITON_DUMP_IR`（如果支持）
3. **方法 C**：从 kernel 的 cache 中获取（需要 Triton 内部 API）

一旦能够获取 PTX，可以在 `early_config_prune()` 中调用实际的 PTX 分析。

### 2. ir_override 和 TRITON_KERNEL_OVERRIDE

如果需要真正的汇编级控制，可以使用：

- `triton.Config(..., ir_override="xxx.ptx")`：使用自定义 PTX 文件
- `TRITON_KERNEL_OVERRIDE` 环境变量：在编译管线各阶段覆盖 IR

这些功能需要手动生成或修改 PTX 文件，工程成本较高，但提供了最大的控制权。

### 3. 更精细的评分模型

可以根据实际测试结果调整评分权重，或者基于历史数据训练一个更精确的评分模型。

## 性能优化建议

根据您的分析，以下配置通常表现较好：

1. **大尺寸（>512）**：
   - BM/BN: 128-256
   - BK: 32-64
   - num_warps: 8
   - num_stages: 4-5
   - maxnreg: 96-128

2. **小尺寸（≤512）**：
   - BM/BN: 32-64
   - BK: 16-32
   - num_warps: 1-4
   - num_stages: 1-2
   - 使用 heuristics 而不是 autotune

## 注意事项

1. **Triton 版本兼容性**：某些功能（如 maxnreg）可能需要特定版本的 Triton
2. **PTX 获取**：当前实现主要基于规则，PTX 分析需要额外工作
3. **正确性验证**：在使用 ir_override 或修改 PTX 时，需要仔细验证正确性

## 参考

- [Triton Language Documentation](https://triton-lang.org/)
- [Triton Compiler Development Tips](https://www.lei.chat/posts/triton-compiler-development-tips/)
- [Triton Kernel Compilation Stages](https://pytorch.org/blog/triton-kernel-compilation-stages/)

