#!/usr/bin/env python3
"""TileLang LayerNorm Kernel - 区分 warmup、编译、运行时间"""

import tilelang
import tilelang.language as T
import torch
import json
import time

def layernorm_func(seq_len, hidden_dim, dtype="float16", eps=1e-5):
    """TileLang LayerNorm函数定义"""
    @T.prim_func
    def main(
        X: T.Tensor((seq_len, hidden_dim), dtype),
        Weight: T.Tensor((hidden_dim,), dtype),
        Bias: T.Tensor((hidden_dim,), dtype),
        Y: T.Tensor((seq_len, hidden_dim), dtype),
    ):
        with T.Kernel(seq_len, threads=128) as (i,):
            x_local = T.alloc_fragment((hidden_dim,), dtype)
            T.copy(X[i, :], x_local)
            
            # 计算均值 - 使用alloc_var声明可变变量
            sum_val = T.alloc_var(dtype)
            sum_val[()] = x_local[0]
            for j in T.Serial(hidden_dim - 1):
                sum_val[()] = sum_val[()] + x_local[j + 1]
            mean = sum_val[()] / hidden_dim
            
            # 计算方差
            x_centered = T.alloc_fragment((hidden_dim,), dtype)
            for j in T.Serial(hidden_dim):
                x_centered[j] = x_local[j] - mean
            
            x_centered_sq = T.alloc_fragment((hidden_dim,), dtype)
            for j in T.Serial(hidden_dim):
                x_centered_sq[j] = x_centered[j] * x_centered[j]
            
            # 计算方差和
            var_sum = T.alloc_var(dtype)
            var_sum[()] = x_centered_sq[0]
            for j in T.Serial(hidden_dim - 1):
                var_sum[()] = var_sum[()] + x_centered_sq[j + 1]
            var = var_sum[()] / hidden_dim
            
            # 归一化
            inv_std = 1.0 / T.sqrt(var + eps)
            y_local = T.alloc_fragment((hidden_dim,), dtype)
            for j in T.Serial(hidden_dim):
                y_local[j] = x_centered[j] * inv_std * Weight[j] + Bias[j]
            
            T.copy(y_local, Y[i, :])
    return main

def warmup_framework():
    """框架预热"""
    start = time.time()
    func = layernorm_func(64, 64)
    jit_kernel = tilelang.compile(func, out_idx=[3], target="cuda")
    
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    weight = torch.ones(64, device="cuda", dtype=torch.float16)
    bias = torch.zeros(64, device="cuda", dtype=torch.float16)
    y = jit_kernel(x, weight, bias)
    torch.cuda.synchronize()
    
    warmup_time = (time.time() - start) * 1000
    return warmup_time

def benchmark(shape, num_runs=10):
    """运行基准测试"""
    seq_len, hidden_dim = shape
    
    # 编译
    compile_start = time.time()
    func = layernorm_func(seq_len, hidden_dim)
    jit_kernel = tilelang.compile(func, out_idx=[3], target="cuda")
    compile_time = (time.time() - compile_start) * 1000
    
    torch.manual_seed(0)
    x = torch.randn(seq_len, hidden_dim, device="cuda", dtype=torch.float16)
    weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)
    bias = torch.zeros(hidden_dim, device="cuda", dtype=torch.float16)
    
    # 热身并验证正确性
    y = jit_kernel(x, weight, bias)
    torch.cuda.synchronize()
    
    ref_y = torch.nn.functional.layer_norm(x.float(), (hidden_dim,), weight.float(), bias.float()).half()
    torch.testing.assert_close(y, ref_y, rtol=5e-2, atol=0.1)
    
    # 测量运行时间
    profiler = jit_kernel.get_profiler()
    runtime = profiler.do_bench()
    
    return compile_time, runtime

def main():
    print("TileLang LayerNorm Benchmark")
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
        "kernel": "layernorm",
        "warmup_time_ms": round(warmup_time, 1),
        "total_compile_time_ms": round(total_compile_time, 1),
        "benchmarks": results
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()

