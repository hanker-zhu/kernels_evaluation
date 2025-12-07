#!/usr/bin/env python3
"""TileLang GEMM Kernel - 区分 warmup、编译、运行时间"""

import tilelang
import tilelang.language as T
import torch
import json
import time

def matmul(M, N, K, block_M=128, block_N=128, block_K=32, dtype="float16", accum_dtype="float"):
    """TileLang GEMM函数定义"""
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            T.clear(C_local)
            
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * block_M, bx * block_N])
    return main

def warmup_framework():
    """框架预热 - 测量初始化开销"""
    start = time.time()
    # 用小矩阵触发框架初始化
    func = matmul(64, 64, 64, block_M=32, block_N=32, block_K=32)
    jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda")
    
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    c = jit_kernel(a, b)
    torch.cuda.synchronize()
    
    warmup_time = (time.time() - start) * 1000
    return warmup_time

def benchmark(M, N, K, num_runs=10):
    """运行基准测试"""
    # 编译
    compile_start = time.time()
    func = matmul(M, N, K)
    jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda")
    compile_time = (time.time() - compile_start) * 1000
    
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    
    # 热身并验证正确性
    c = jit_kernel(a, b)
    torch.cuda.synchronize()
    
    ref_c = a @ b
    # 对于FP16，使用更宽松的容差：相对误差5%，绝对误差0.1
    torch.testing.assert_close(c, ref_c, rtol=5e-2, atol=0.1)
    
    # 测量运行时间
    profiler = jit_kernel.get_profiler()
    runtime = profiler.do_bench()
    tflops = (2.0 * M * N * K) / (runtime * 1e6)
    
    return compile_time, runtime, tflops

def main():
    print("TileLang GEMM Benchmark")
    print("=" * 60)
    
    # 1. 框架预热
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
            print(f"  {M}x{N}x{K}: compile={compile_time:.1f}ms, run={runtime:.3f}ms, {tflops:.2f} TFLOPS ✓")
            results.append({
                "M": M, "N": N, "K": K,
                "compile_time_ms": round(compile_time, 1),
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
