#!/usr/bin/env python3
"""Triton GEMM Kernel - 区分 warmup、编译、运行时间"""

import torch
import triton
import triton.language as tl
import time
import json

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Triton GEMM kernel"""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_matmul(A, B, C, M, N, K, block_m=128, block_n=128, block_k=32):
    """Triton GEMM封装"""
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )

def warmup_framework():
    """框架预热 - 测量初始化开销"""
    device = 'cuda'
    # 用小矩阵触发框架初始化
    A = torch.randn((64, 64), device=device, dtype=torch.float16)
    B = torch.randn((64, 64), device=device, dtype=torch.float16)
    C = torch.zeros((64, 64), device=device, dtype=torch.float16)
    
    torch.cuda.synchronize()
    start = time.time()
    triton_matmul(A, B, C, 64, 64, 64, block_m=32, block_n=32, block_k=32)
    torch.cuda.synchronize()
    warmup_time = (time.time() - start) * 1000
    
    return warmup_time

def benchmark(M, N, K, num_runs=10):
    """运行基准测试"""
    device = 'cuda'
    torch.manual_seed(0)
    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B = torch.randn((K, N), device=device, dtype=torch.float16)
    C = torch.zeros((M, N), device=device, dtype=torch.float16)
    
    # 第一次调用（可能触发编译）
    torch.cuda.synchronize()
    start = time.time()
    triton_matmul(A, B, C, M, N, K)
    torch.cuda.synchronize()
    first_run = (time.time() - start) * 1000
    
    # 正确性验证 (FP16需要更宽松的容差)
    ref = torch.matmul(A, B)
    # 对于FP16，使用更宽松的容差：相对误差5%，绝对误差0.1
    torch.testing.assert_close(C.float(), ref.float(), rtol=5e-2, atol=0.1)
    
    # 多次运行测量稳定运行时间
    times = []
    for _ in range(num_runs):
        C.zero_()
        torch.cuda.synchronize()
        start = time.time()
        triton_matmul(A, B, C, M, N, K)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    
    avg_runtime = sum(times) / len(times)
    compile_time = max(0, first_run - avg_runtime)
    tflops = (2.0 * M * N * K) / (avg_runtime * 1e6)
    
    return compile_time, avg_runtime, tflops

def main():
    print("Triton GEMM Benchmark")
    print("=" * 60)
    
    # 1. 框架预热（测量初始化开销）
    print("\n[Phase 1] Framework Warmup")
    warmup_time = warmup_framework()
    print(f"  Framework init time: {warmup_time:.1f} ms")
    
    # 2. 各尺寸基准测试
    print("\n[Phase 2] Benchmark per Size")
    sizes = [256, 512, 1024, 2048, 4096]
    results = []
    total_compile_time = 0.0
    total_runtime = 0.0
    
    for size in sizes:
        M, N, K = size, size, size
        try:
            compile_time, runtime, tflops = benchmark(M, N, K)
            total_compile_time += compile_time
            total_runtime += runtime
            print(f"  {M}x{N}x{K}: compile={compile_time:.2f}ms, run={runtime:.3f}ms, {tflops:.2f} TFLOPS ✓")
            results.append({
                "M": M, "N": N, "K": K,
                "compile_time_ms": round(compile_time, 3),
                "latency_ms": round(runtime, 4),
                "tflops": round(tflops, 2),
                "success": True
            })
        except Exception as e:
            print(f"  {M}x{N}x{K}: FAILED - {e}")
            results.append({"M": M, "N": N, "K": K, "success": False, "error": str(e)})
    
    # 3. 汇总
    print("\n[Summary]")
    print(f"  Framework warmup:    {warmup_time:.1f} ms (one-time)")
    print(f"  Total JIT compile:   {total_compile_time:.1f} ms")
    print(f"  Total runtime:       {total_runtime:.1f} ms")
    
    print("\n=== JSON Results ===")
    output = {
        "warmup_time_ms": round(warmup_time, 1),
        "total_compile_time_ms": round(total_compile_time, 1),
        "benchmarks": results
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
