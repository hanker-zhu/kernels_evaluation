#!/usr/bin/env python3
"""TileLang Softmax Kernel - 区分 warmup、编译、运行时间"""

import tilelang
import tilelang.language as T
import torch
import json
import time

def softmax_func(seq_len, hidden_dim, dtype="float16"):
    """TileLang Softmax函数定义"""
    @T.prim_func
    def main(
        X: T.Tensor((seq_len, hidden_dim), dtype),
        Y: T.Tensor((seq_len, hidden_dim), dtype),
    ):
        with T.Kernel(seq_len, threads=128) as (i,):
            x_local = T.alloc_fragment((hidden_dim,), dtype)
            T.copy(X[i, :], x_local)
            
            # 计算max - 使用alloc_var声明可变变量
            max_val = T.alloc_var(dtype)
            max_val[()] = x_local[0]
            for j in T.Serial(hidden_dim - 1):
                max_val[()] = T.max(max_val[()], x_local[j + 1])
            
            # 计算exp(x - max)
            exp_local = T.alloc_fragment((hidden_dim,), dtype)
            for j in T.Serial(hidden_dim):
                exp_local[j] = T.exp(x_local[j] - max_val[()])
            
            # 计算sum
            sum_val = T.alloc_var(dtype)
            sum_val[()] = exp_local[0]
            for j in T.Serial(hidden_dim - 1):
                sum_val[()] = sum_val[()] + exp_local[j + 1]
            
            # 归一化
            for j in T.Serial(hidden_dim):
                Y[i, j] = exp_local[j] / sum_val[()]
    return main

def warmup_framework():
    """框架预热"""
    start = time.time()
    func = softmax_func(64, 64)
    jit_kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    y = jit_kernel(x)
    torch.cuda.synchronize()
    
    warmup_time = (time.time() - start) * 1000
    return warmup_time

def benchmark(shape, num_runs=10):
    """运行基准测试"""
    seq_len, hidden_dim = shape
    
    # 编译
    compile_start = time.time()
    func = softmax_func(seq_len, hidden_dim)
    jit_kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    compile_time = (time.time() - compile_start) * 1000
    
    torch.manual_seed(0)
    x = torch.randn(seq_len, hidden_dim, device="cuda", dtype=torch.float16)
    
    # 热身并验证正确性
    y = jit_kernel(x)
    torch.cuda.synchronize()
    
    ref_y = torch.softmax(x.float(), dim=-1).half()
    torch.testing.assert_close(y, ref_y, rtol=5e-2, atol=0.1)
    
    # 测量运行时间
    profiler = jit_kernel.get_profiler()
    runtime = profiler.do_bench()
    
    return compile_time, runtime

def main():
    print("TileLang Softmax Benchmark")
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
            print(f"  {shape[0]}x{shape[1]}: compile={compile_time:.1f}ms, run={runtime:.3f}ms ✓")
            results.append({
                "shape": f"{shape[0]}x{shape[1]}",
                "compile_time_ms": round(compile_time, 1),
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
        "kernel": "softmax",
        "warmup_time_ms": round(warmup_time, 1),
        "total_compile_time_ms": round(total_compile_time, 1),
        "benchmarks": results
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()

