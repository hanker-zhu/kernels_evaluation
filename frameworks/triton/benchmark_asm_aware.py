#!/usr/bin/env python3
"""
性能对比脚本：对比原有的 Triton GEMM 实现和新的 Assembly-Aware Autotune 实现

增强功能：
- 编译时间统计
- 编译质量分析
- Autotune 环节质量统计
"""

import torch
import time
import json
import sys
import os
import statistics
from typing import Dict, Any, Optional

# 导入原有实现
sys.path.insert(0, os.path.dirname(__file__))
from matmul_triton_opt import (
    triton_matmul_opt_staticK,
    triton_matmul_opt_smart,
)
# from triton_gemm import triton_matmul

# 导入新的 assembly-aware 实现
from triton_gemm_asm_aware import triton_matmul_asm_aware, get_autotune_stats, reset_autotune_stats


def time_cuda_ms(fn, iters=10, warmup=5):
    """测量 CUDA kernel 执行时间"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def benchmark_one(M, N, K, impl_name, impl_fn, iters=50):
    """
    基准测试单个实现，包含编译时间和质量统计
    
    Args:
        M, N, K: 矩阵维度
        impl_name: 实现名称
        impl_fn: 实现函数，接受 (A, B, C) 参数
        iters: 迭代次数
    """
    device = "cuda"
    torch.manual_seed(0)
    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B = torch.randn((K, N), device=device, dtype=torch.float16)
    C = torch.empty((M, N), device=device, dtype=torch.float16)
    
    def call():
        impl_fn(A, B, C)
    
    # 重置 autotune 统计
    reset_autotune_stats()
    
    try:
        # 测量编译时间（首次调用）
        torch.cuda.synchronize()
        compile_start = time.time()
        call()
        torch.cuda.synchronize()
        compile_end = time.time()
        compile_time_ms = (compile_end - compile_start) * 1000
        
        # 首次运行（包含一些一次性开销，如 JIT cache 等）
        first_ms = time_cuda_ms(call, iters=1, warmup=0)
        
        # 稳态运行（多次测量以评估稳定性）
        steady_times = []
        for _ in range(3):  # 运行3次来评估稳定性
            steady_ms = time_cuda_ms(call, iters=iters, warmup=2)
            steady_times.append(steady_ms)
        
        steady_ms = statistics.mean(steady_times)
        steady_std = statistics.stdev(steady_times) if len(steady_times) > 1 else 0.0
        
        # 验证正确性
        ref = (A @ B).float()
        torch.testing.assert_close(C.float(), ref, rtol=5e-2, atol=0.1)
        
        # 计算 TFLOPS
        tflops = (2.0 * M * N * K) / (steady_ms * 1e6)
        
        # 编译质量指标
        # 1. 编译时间（越小越好）
        # 2. 首次运行与稳态的差距（越小说明编译质量越好，JIT 优化到位）
        compile_quality_score = 0.0
        if steady_ms > 0:
            # 如果首次运行和稳态差距小，说明编译质量好
            first_steady_ratio = first_ms / steady_ms if steady_ms > 0 else float('inf')
            # 理想情况下，首次运行应该接近稳态（ratio 接近 1.0）
            # 如果 ratio > 1.5，说明编译质量可能不够好
            if first_steady_ratio < 1.1:
                compile_quality_score = 1.0
            elif first_steady_ratio < 1.5:
                compile_quality_score = 0.8
            else:
                compile_quality_score = 0.5
        
        # 3. 性能稳定性（std 越小越好）
        stability_score = 1.0
        if steady_ms > 0:
            cv = steady_std / steady_ms  # 变异系数
            if cv < 0.01:
                stability_score = 1.0
            elif cv < 0.05:
                stability_score = 0.8
            else:
                stability_score = 0.6
        
        # 综合编译质量评分
        overall_compile_quality = (compile_quality_score * 0.6 + stability_score * 0.4)
        
        # 获取 autotune 统计（仅对 assembly-aware 实现有效）
        autotune_stats = get_autotune_stats() if impl_name == "asm_aware" else {}
        
        result = {
            "first_ms": round(first_ms, 4),
            "steady_ms": round(steady_ms, 4),
            "steady_std_ms": round(steady_std, 4),
            "first_over_steady_ms": round(first_ms - steady_ms, 4),
            "first_steady_ratio": round(first_ms / steady_ms, 3) if steady_ms > 0 else 0,
            "tflops": round(tflops, 2),
            "compile_time_ms": round(compile_time_ms, 2),
            "compile_quality": {
                "first_steady_ratio": round(first_ms / steady_ms, 3) if steady_ms > 0 else 0,
                "stability_cv": round(steady_std / steady_ms, 4) if steady_ms > 0 else 0,
                "quality_score": round(overall_compile_quality, 3),
            },
            "autotune_stats": autotune_stats,
            "success": True,
        }
        
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "autotune_stats": get_autotune_stats() if impl_name == "asm_aware" else {},
        }


def main():
    """主函数：运行性能对比测试"""
    print("=" * 80)
    print("Triton GEMM: Assembly-Aware Autotune 性能对比测试")
    print("=" * 80)
    
    # 测试尺寸
    sizes = [128, 256, 512, 1024, 2048]
    
    # 实现列表：(名称, 函数)
    implementations = [
        # ("baseline", triton_matmul),
        ("baseline_opt", triton_matmul_opt_staticK),
        ("baseline_smart", triton_matmul_opt_smart),
        ("asm_aware", triton_matmul_asm_aware),
    ]
    
    results = []
    
    for size in sizes:
        M = N = K = size
        print(f"\n{'='*80}")
        print(f"Testing {M}x{N}x{K}")
        print(f"{'='*80}")
        
        row = {
            "M": M,
            "N": N,
            "K": K,
        }
        
        for impl_name, impl_fn in implementations:
            print(f"\n  [{impl_name}] Running benchmark...")
            try:
                result = benchmark_one(M, N, K, impl_name, impl_fn)
                row[impl_name] = result
                
                if result.get("success", False):
                    print(f"    ✓ Steady: {result['steady_ms']:.4f} ms, "
                          f"TFLOPS: {result['tflops']:.2f}")
                    print(f"      Compile: {result.get('compile_time_ms', 0):.2f} ms, "
                          f"Quality: {result.get('compile_quality', {}).get('quality_score', 0):.2f}")
                    if impl_name == "asm_aware" and result.get("autotune_stats"):
                        stats = result["autotune_stats"]
                        if stats.get("total_configs", 0) > 0:
                            print(f"      Autotune: {stats.get('total_configs', 0)} configs, "
                                  f"pruned to {stats.get('benchmarked_configs', 0)}, "
                                  f"prune time: {stats.get('prune_time_ms', 0):.2f} ms")
                else:
                    print(f"    ✗ Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"    ✗ Exception: {e}")
                row[impl_name] = {"success": False, "error": str(e)}
        
        # 计算性能提升（相对于 baseline）
        if row.get("baseline_opt", {}).get("success") and row.get("asm_aware", {}).get("success"):
            baseline_time = row["baseline_opt"]["steady_ms"]
            asm_time = row["asm_aware"]["steady_ms"]
            speedup = baseline_time / asm_time if asm_time > 0 else 0
            improvement = (baseline_time - asm_time) / baseline_time * 100 if baseline_time > 0 else 0
            
            print(f"\n  [Performance Comparison]")
            print(f"    Baseline:     {baseline_time:.4f} ms")
            print(f"    Assembly-Aware:     {asm_time:.4f} ms")
            print(f"    Speedup:            {speedup:.3f}x")
            print(f"    Improvement:        {improvement:+.2f}%")
            
            row["speedup"] = round(speedup, 3)
            row["improvement_pct"] = round(improvement, 2)
        
        results.append(row)
    
    # 打印汇总 - 性能对比
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Size':<10} {'Baseline (opt)':<20} {'Assembly-Aware':<20} {'Speedup':<10} {'Improvement':<15}")
    print("-" * 80)
    
    for row in results:
        size = f"{row['M']}³"
        baseline = row.get("baseline_opt", {})
        asm = row.get("asm_aware", {})
        
        if baseline.get("success") and asm.get("success"):
            baseline_time = baseline["steady_ms"]
            asm_time = asm["steady_ms"]
            speedup = row.get("speedup", 0)
            improvement = row.get("improvement_pct", 0)
            
            print(f"{size:<10} {baseline_time:>8.4f} ms{'':>8} {asm_time:>8.4f} ms{'':>8} "
                  f"{speedup:>6.3f}x{'':>2} {improvement:>+6.2f}%")
        else:
            baseline_str = baseline.get("error", "N/A")[:15] if not baseline.get("success") else f"{baseline.get('steady_ms', 0):.4f} ms"
            asm_str = asm.get("error", "N/A")[:15] if not asm.get("success") else f"{asm.get('steady_ms', 0):.4f} ms"
            print(f"{size:<10} {baseline_str:<20} {asm_str:<20} {'N/A':<10} {'N/A':<15}")
    
    # 打印编译时间统计
    print(f"\n{'='*80}")
    print("COMPILATION TIME SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Size':<10} {'Baseline Compile':<18} {'ASM-Aware Compile':<18} {'Compile Speedup':<15}")
    print("-" * 80)
    
    for row in results:
        size = f"{row['M']}³"
        baseline = row.get("baseline_opt", {})
        asm = row.get("asm_aware", {})
        
        if baseline.get("success") and asm.get("success"):
            baseline_compile = baseline.get("compile_time_ms", 0)
            asm_compile = asm.get("compile_time_ms", 0)
            compile_speedup = baseline_compile / asm_compile if asm_compile > 0 else 0
            
            print(f"{size:<10} {baseline_compile:>10.2f} ms{'':>6} {asm_compile:>10.2f} ms{'':>6} "
                  f"{compile_speedup:>8.2f}x")
        else:
            baseline_str = baseline.get("error", "N/A")[:15] if not baseline.get("success") else f"{baseline.get('compile_time_ms', 0):.2f} ms"
            asm_str = asm.get("error", "N/A")[:15] if not asm.get("success") else f"{asm.get('compile_time_ms', 0):.2f} ms"
            print(f"{size:<10} {baseline_str:<18} {asm_str:<18} {'N/A':<15}")
    
    # 打印编译质量统计
    print(f"\n{'='*80}")
    print("COMPILATION QUALITY SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Size':<10} {'Baseline Quality':<18} {'ASM-Aware Quality':<18} {'Quality Δ':<15}")
    print("-" * 80)
    
    for row in results:
        size = f"{row['M']}³"
        baseline = row.get("baseline_opt", {})
        asm = row.get("asm_aware", {})
        
        if baseline.get("success") and asm.get("success"):
            baseline_quality = baseline.get("compile_quality", {}).get("quality_score", 0)
            asm_quality = asm.get("compile_quality", {}).get("quality_score", 0)
            quality_delta = asm_quality - baseline_quality
            
            print(f"{size:<10} {baseline_quality:>10.3f}{'':>6} {asm_quality:>10.3f}{'':>6} "
                  f"{quality_delta:>+8.3f}")
        else:
            baseline_str = baseline.get("error", "N/A")[:15] if not baseline.get("success") else f"{baseline.get('compile_quality', {}).get('quality_score', 0):.3f}"
            asm_str = asm.get("error", "N/A")[:15] if not asm.get("success") else f"{asm.get('compile_quality', {}).get('quality_score', 0):.3f}"
            print(f"{size:<10} {baseline_str:<18} {asm_str:<18} {'N/A':<15}")
    
    # 打印 Autotune 质量统计（仅对 asm_aware）
    print(f"\n{'='*80}")
    print("AUTOTUNE QUALITY SUMMARY (Assembly-Aware)")
    print(f"{'='*80}")
    print("Note: Small sizes (≤512) use heuristics instead of autotune, so stats are N/A")
    
    print(f"\n{'Size':<10} {'Total Configs':<15} {'Pruned':<10} {'Benchmarked':<13} {'Prune Time':<13} {'Prune Ratio':<13}")
    print("-" * 95)
    
    for row in results:
        size = f"{row['M']}³"
        asm = row.get("asm_aware", {})
        M = row.get("M", 0)
        
        if asm.get("success"):
            stats = asm.get("autotune_stats", {})
            total = stats.get("total_configs", 0)
            pruned = stats.get("pruned_configs", 0)
            benchmarked = stats.get("benchmarked_configs", 0)
            prune_time = stats.get("prune_time_ms", 0)
            prune_ratio = (pruned / total * 100) if total > 0 else 0
            
            if total > 0:
                print(f"{size:<10} {total:>13} {pruned:>8} {benchmarked:>11} "
                      f"{prune_time:>9.2f} ms {prune_ratio:>9.1f}%")
            else:
                # 小尺寸使用 heuristics
                if M <= 512:
                    print(f"{size:<10} {'Heuristics':<15} {'N/A':<10} {'N/A':<13} {'N/A':<13} {'N/A':<13}")
                else:
                    print(f"{size:<10} {'N/A':<15} {'N/A':<10} {'N/A':<13} {'N/A':<13} {'N/A':<13}")
        else:
            print(f"{size:<10} {'Error':<15} {'N/A':<10} {'N/A':<13} {'N/A':<13} {'N/A':<13}")
    
    # 保存结果到 JSON
    output_file = os.path.join(os.path.dirname(__file__), "..", "results", "asm_aware_benchmark.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # 返回 JSON 结果
    print("\n" + "=" * 80)
    print("JSON Results:")
    print("=" * 80)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

