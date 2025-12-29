# Triton GEMM 小尺寸优化说明

## 问题分析

从测试结果可以看出，Triton 在小尺寸矩阵上表现不佳，但在大尺寸上表现良好：

| 矩阵尺寸 | cuBLAS (ms) | Triton (ms) | 性能差距 |
|---------|------------|------------|---------|
| 256³    | 0.0107     | 0.0422     | **4.0x** |
| 4096³   | 0.968      | 1.015      | **1.05x** |

### 问题根源

1. **块大小过大**: 原有配置使用 BM=128, BN=128，对于 256×256 矩阵只产生 2×2=4 个 block，GPU 利用率极低
2. **启动开销**: 小矩阵的计算时间短，无法掩盖 kernel 启动开销
3. **配置不当**: autotune 配置主要针对大尺寸优化，没有小尺寸专用配置

## 优化方案

### 1. 扩展 Autotune 配置

为原有 `matmul_opt_kernel_staticK` 添加了小尺寸专用配置：

```python
# 新增小尺寸配置
triton.Config({"BM":  32, "BN":  32, "BK": 16}, num_warps=2, num_stages=2),
triton.Config({"BM":  32, "BN":  64, "BK": 16}, num_warps=2, num_stages=2),
triton.Config({"BM":  64, "BN":  32, "BK": 16}, num_warps=2, num_stages=2),
triton.Config({"BM":  32, "BN":  32, "BK": 32}, num_warps=2, num_stages=3),
triton.Config({"BM":  64, "BN":  64, "BK": 16}, num_warps=4, num_stages=2),
```

**优化点**:
- 更小的 block 尺寸（32×32 替代 128×128），增加并行度
- 更少的 warps（1-2 替代 4-8），减少开销
- 更少的 stages（1-2 替代 3-5），降低延迟

### 2. 专用小尺寸 Kernel

创建了 `matmul_small_kernel_autotune`，专门针对小尺寸优化：

```python
@triton.autotune(
    configs=[
        triton.Config({"BM": 32, "BN": 32, "BK": 16}, num_warps=1, num_stages=1),
        triton.Config({"BM": 32, "BN": 32, "BK": 32}, num_warps=1, num_stages=2),
        # ... 更多小尺寸配置
    ],
    key=["M", "N", "K"],
)
```

**优化点**:
- 使用最少的 warps (1-2) 来减少启动开销
- 使用最少的 stages (1-2) 来降低延迟
- 简化的指针计算，减少指令数

### 3. 智能选择策略

实现了 `triton_matmul_opt_smart`，根据矩阵大小自动选择最优策略：

```python
def triton_matmul_opt_smart(A, B, C):
    M, K = A.shape
    _, N = B.shape
    
    # 极小尺寸: 使用 PyTorch（启动开销太大）
    if M <= 128 and N <= 128 and K <= 128:
        C.copy_((A @ B))
        return
    
    # 小尺寸: 使用专用小尺寸优化
    if M <= 512 and N <= 512 and K <= 512:
        triton_matmul_opt_small(A, B, C)
    else:
        # 大尺寸: 使用标准优化
        triton_matmul_opt_staticK(A, B, C)
```

**策略说明**:
- **≤128**: PyTorch fallback（启动开销占主导）
- **128-512**: 小尺寸专用 kernel（优化启动开销）
- **>512**: 标准优化 kernel（最大化吞吐量）

## 使用方法

### 运行基准测试

```bash
cd frameworks/triton
./run.sh
# 或直接运行
python matmul_triton_opt.py
```

### 测试不同实现

benchmark 函数现在支持以下实现：

- `baseline`: 原始实现
- `opt`: 标准优化版本（含小尺寸配置）
- `opt_small`: 专用小尺寸优化版本
- `smart`: 智能选择版本（推荐）
- `persistent`: persistent kernel 版本

## 预期改进

根据优化策略，预期在小尺寸上的改进：

1. **256×256×256**: 从 0.042ms 优化到 ~0.015ms（接近 cuBLAS 的 0.0107ms）
2. **512×512×512**: 从 0.040ms 优化到 ~0.014ms（接近 cuBLAS 的 0.013ms）
3. **大尺寸保持**: 4096×4096×4096 保持 ~1.015ms（已经接近最优）

## 进一步优化建议

1. **测试和验证**: 运行基准测试验证实际改进
2. **阈值调整**: 根据测试结果调整智能选择的阈值（128, 512）
3. **更多配置**: 可以添加更多中间尺寸的配置（如 64×64, 96×96）
4. **内存访问优化**: 对于小尺寸，可以考虑合并内存访问
5. **向量化**: 对于极小的 block，可以使用向量化加载/存储

## 技术细节

### Block Size 选择原则

- **小尺寸 (<512)**: 32-64 的 block，确保足够的并行度
- **大尺寸 (>512)**: 128+ 的 block，最大化计算密度

### Warp 数量选择

- **小尺寸**: 1-2 warps，减少调度开销
- **大尺寸**: 4-8 warps，提高并行度

### Stage 数量选择

- **小尺寸**: 1-2 stages，降低延迟
- **大尺寸**: 3-5 stages，隐藏内存延迟

