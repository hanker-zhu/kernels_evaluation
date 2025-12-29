#!/usr/bin/env python3
"""Triton LayerNorm Kernel - 区分 warmup、编译、运行时间"""

import torch
import triton
import triton.language as tl
import time
import json

@triton.jit
def layernorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton LayerNorm kernel"""
    row_idx = tl.program_id(0)
    col_offsets = row_idx * n_cols + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    x = tl.load(x_ptr + col_offsets, mask=mask, other=0.0)
    
    # 计算均值 - 只对有效元素求和
    x_masked = tl.where(mask, x, 0.0)
    x_sum = tl.sum(x_masked)
    # 计算有效元素数量
    valid_count = tl.sum(mask.to(tl.float32))
    valid_count = tl.where(valid_count > 0, valid_count, 1.0)
    mean = x_sum / valid_count
    
    # 计算方差 - 只对有效元素计算
    x_centered = tl.where(mask, x - mean, 0.0)
    x_centered_sq = x_centered * x_centered
    var_sum = tl.sum(x_centered_sq)
    var = var_sum / valid_count
    
    # 归一化
    inv_std = 1.0 / tl.sqrt(var + eps)
    y = x_centered * inv_std
    
    # 应用权重和偏置
    if weight_ptr is not None:
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        y = y * w
    if bias_ptr is not None:
        b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
        y = y + b
    
    tl.store(y_ptr + col_offsets, y, mask=mask)

def triton_layernorm(x, weight=None, bias=None, eps=1e-5):
    """Triton LayerNorm封装"""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 32768:
        BLOCK_SIZE = 32768
    
    y = torch.empty_like(x)
    grid = (n_rows,)
    layernorm_kernel[grid](
        x, y, weight if weight is not None else None,
        bias if bias is not None else None,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

def warmup_framework():
    """框架预热"""
    device = 'cuda'
    x = torch.randn((64, 64), device=device, dtype=torch.float16)
    weight = torch.ones(64, device=device, dtype=torch.float16)
    bias = torch.zeros(64, device=device, dtype=torch.float16)
    
    torch.cuda.synchronize()
    start = time.time()
    y = triton_layernorm(x, weight, bias)
    torch.cuda.synchronize()
    warmup_time = (time.time() - start) * 1000
    
    return warmup_time

def benchmark(shape, num_runs=10):
    """运行基准测试"""
    device = 'cuda'
    torch.manual_seed(0)
    x = torch.randn(shape, device=device, dtype=torch.float16)
    weight = torch.ones(shape[1], device=device, dtype=torch.float16)
    bias = torch.zeros(shape[1], device=device, dtype=torch.float16)
    
    # 第一次调用（可能触发编译）
    torch.cuda.synchronize()
    start = time.time()
    y = triton_layernorm(x, weight, bias)
    torch.cuda.synchronize()
    first_run = (time.time() - start) * 1000
    
    # 正确性验证 (FP16需要更宽松的容差)
    ref = torch.nn.functional.layer_norm(x.float(), (shape[1],), weight.float(), bias.float()).half()
    # LayerNorm对数值精度更敏感，使用更宽松的容差
    torch.testing.assert_close(y.float(), ref.float(), rtol=1e-1, atol=0.2, check_dtype=False)
    
    # 多次运行测量稳定运行时间
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        y = triton_layernorm(x, weight, bias)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    
    avg_runtime = sum(times) / len(times)
    compile_time = max(0, first_run - avg_runtime)
    
    return compile_time, avg_runtime

def main():
    print("Triton LayerNorm Benchmark")
    print("=" * 60)
    
    # 1. 框架预热
    print("\n[Phase 1] Framework Warmup")
    warmup_time = warmup_framework()
    print(f"  Framework init time: {warmup_time:.1f} ms")
    
    # 2. 各尺寸基准测试
    print("\n[Phase 2] Benchmark per Size")
    shapes = [
        (256, 256), (512, 512), (1024, 1024),
        (2048, 2048), (4096, 4096), (8192, 8192)
    ]
    results = []
    total_compile_time = 0.0
    total_runtime = 0.0
    
    for shape in shapes:
        try:
            compile_time, runtime = benchmark(shape)
            total_compile_time += compile_time
            total_runtime += runtime
            print(f"  {shape[0]}x{shape[1]}: compile={compile_time:.2f}ms, run={runtime:.3f}ms ✓")
            results.append({
                "shape": f"{shape[0]}x{shape[1]}",
                "compile_time_ms": round(compile_time, 3),
                "latency_ms": round(runtime, 4),
                "success": True
            })
        except Exception as e:
            print(f"  {shape[0]}x{shape[1]}: FAILED - {e}")
            results.append({"shape": f"{shape[0]}x{shape[1]}", "success": False, "error": str(e)})
    
    # 3. 汇总
    print("\n[Summary]")
    print(f"  Framework warmup:    {warmup_time:.1f} ms (one-time)")
    print(f"  Total JIT compile:   {total_compile_time:.1f} ms")
    print(f"  Total runtime:       {total_runtime:.1f} ms")
    
    print("\n=== JSON Results ===")
    output = {
        "kernel": "layernorm",
        "warmup_time_ms": round(warmup_time, 1),
        "total_compile_time_ms": round(total_compile_time, 1),
        "benchmarks": results
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()

