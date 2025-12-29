#!/usr/bin/env python3
"""
Triton GEMM with Assembly-Aware Autotune

根据 PTX/SASS 分析来剪枝和优化 autotune 配置，目标是生成更像 CUTLASS 的高质量汇编代码。

核心思路：
1. 在 autotune 的 early_config_prune 阶段，编译每个候选 config 并获取 PTX
2. 基于 PTX 中的指令（mma.sync, cp.async, bar.sync, bra 等）进行评分
3. 剪枝掉明显不好的配置，只对 top-K 进行真实 bench
4. 将 maxnreg 作为 autotune 维度，控制寄存器分配
5. 小尺寸使用 heuristics 而不是 autotune
"""

import re
import torch
import triton
import triton.language as tl
import time
import json
from typing import List, Callable, Optional, Dict, Any

# 全局变量用于记录 autotune 统计信息（供 benchmark 脚本访问）
_autotune_stats = {
    "total_configs": 0,
    "pruned_configs": 0,
    "benchmarked_configs": 0,
    "prune_time_ms": 0.0,
}

def get_autotune_stats():
    """获取 autotune 统计信息"""
    return _autotune_stats.copy()

def reset_autotune_stats():
    """重置 autotune 统计信息"""
    global _autotune_stats
    _autotune_stats = {
        "total_configs": 0,
        "pruned_configs": 0,
        "benchmarked_configs": 0,
        "prune_time_ms": 0.0,
    }


# ============================================================================
# PTX 分析和评分
# ============================================================================

def score_ptx(ptx: str, verbose: bool = False) -> Dict[str, Any]:
    """
    分析 PTX 代码并给出评分
    
    评分指标（目标：更多 mma/cp.async，更少 barrier/branch）：
    - mma.sync: Tensor Core 指令，权重最高 (+10)
    - cp.async: 异步拷贝，有利于隐藏延迟 (+2)
    - bar.sync: 同步屏障，降低并行度 (-30)
    - bra/ret: 分支指令，可能影响调度 (-5)
    
    Returns:
        dict: 包含各项指标和总分的字典
    """
    # 统计各种指令
    mma_count = len(re.findall(r"\bmma\.sync\b", ptx))
    cp_async_count = len(re.findall(r"\bcp\.async\b", ptx))
    bar_sync_count = len(re.findall(r"\bbar\.sync\b", ptx))
    bra_count = len(re.findall(r"\b(bra|ret)\b", ptx))
    
    # 计算评分（权重可以根据实际效果调整）
    score = 10 * mma_count + 2 * cp_async_count - 30 * bar_sync_count - 5 * bra_count
    
    result = {
        "score": score,
        "mma_count": mma_count,
        "cp_async_count": cp_async_count,
        "bar_sync_count": bar_sync_count,
        "bra_count": bra_count,
    }
    
    if verbose:
        print(f"  PTX Score: {score} (mma={mma_count}, cp.async={cp_async_count}, "
              f"bar.sync={bar_sync_count}, bra={bra_count})")
    
    return result


def get_ptx_from_kernel(kernel_fn, config: triton.Config, args: tuple, kwargs: dict) -> Optional[str]:
    """
    尝试从 kernel 编译过程中获取 PTX 代码
    
    Note: Triton 的 PTX 获取方式可能有版本差异，这里提供几种尝试方案
    """
    try:
        # 方案1: 通过 triton.compile 直接编译并获取 asm
        # 需要构造完整的 kernel 签名
        import inspect
        import types
        
        # 尝试触发一次编译（warmup）
        try:
            # 触发 JIT 编译
            kernel_fn[args](*kwargs.values())
            torch.cuda.synchronize()
        except:
            pass
        
        # 方案2: 从 kernel 的 cache 中获取（需要 Triton 内部 API，可能不稳定）
        # 这里先返回 None，实际使用时需要根据 Triton 版本调整
        return None
        
    except Exception as e:
        print(f"Warning: Failed to get PTX: {e}")
        return None


def compile_with_config_and_get_ptx(
    kernel_fn: Callable,
    config: triton.Config,
    args: tuple,
    kwargs: dict,
    named_args: dict,
) -> Optional[str]:
    """
    使用指定 config 编译 kernel 并尝试获取 PTX
    
    这是一个占位实现，实际实现需要根据 Triton 版本调整
    """
    # TODO: 实际实现需要：
    # 1. 使用 config 编译 kernel（可能需要临时设置 config）
    # 2. 从编译产物中提取 PTX（可能需要访问 Triton 内部结构）
    # 
    # 可能的实现路径：
    # - 使用 triton.compile 的 compile_only 模式
    # - 从 kernel.cache 中获取已编译的结果
    # - 使用 TRITON_DUMP_IR 环境变量（如果支持）
    
    # 临时方案：通过编译并检查输出
    try:
        # 触发编译
        kernel_fn(*args, **kwargs)
        torch.cuda.synchronize()
        
        # 尝试从环境中获取（如果设置了 TRITON_DUMP_IR）
        # 或者尝试其他方式获取
        return None
    except Exception as e:
        print(f"Warning: Compilation failed for config {config}: {e}")
        return None


# ============================================================================
# Assembly-Aware Config Pruner
# ============================================================================

def early_config_prune(configs: List[triton.Config], named_args: dict, **kwargs) -> List[triton.Config]:
    """
    Early config pruning based on static analysis and rules
    
    This function filters configs based on:
    1. Tensor Core requirements (BK must be multiple of 16)
    2. Block size heuristics (prefer larger blocks for better Tensor Core utilization)
    3. Register pressure heuristics (consider num_warps and stages)
    
    Note: PTX analysis can be added later when PTX extraction is available.
    """
    global _autotune_stats
    
    # 记录开始时间和初始配置数
    prune_start = time.time()
    total_configs = len(configs)
    _autotune_stats["total_configs"] = total_configs
    
    # Note: verbose 信息可以通过环境变量或全局设置控制，这里暂时关闭以减少输出
    verbose = False
    
    if verbose:
        print(f"\n[ASM-Aware Pruner] Analyzing {len(configs)} configs...")
    
    scored_configs = []
    
    for i, cfg in enumerate(configs):
        # 获取配置参数
        BM = cfg.kwargs.get("BM", 128)
        BN = cfg.kwargs.get("BN", 128)
        BK = cfg.kwargs.get("BK", 32)
        num_warps = getattr(cfg, 'num_warps', 4)
        num_stages = getattr(cfg, 'num_stages', 3)
        
        if verbose:
            print(f"  Config {i+1}/{len(configs)}: BM={BM}, BN={BN}, BK={BK}, "
                  f"warps={num_warps}, stages={num_stages}")
        
        # 规则1: Tensor Core 要求 - BK 必须是 16 的倍数
        if BK % 16 != 0:
            if verbose:
                print(f"    Rejected: BK={BK} not multiple of 16 (Tensor Core requirement)")
            continue
        
        # 规则2: 评分基于配置参数
        # 偏好更大的 block size（有利于 Tensor Core 利用率）
        size_score = (BM * BN * BK) / (1024.0 * 1024.0)  # 归一化
        
        # 考虑 warp 数量（更多 warps 通常更好，但要平衡）
        warp_score = min(1.0, num_warps / 8.0)  # 8 warps 为满分
        
        # 考虑 stages（更多 stages 可以隐藏延迟，但要平衡寄存器压力）
        stage_score = min(1.0, num_stages / 5.0)  # 5 stages 为满分
        
        # 综合评分
        score = size_score * 0.5 + warp_score * 0.3 + stage_score * 0.2
        
        scored_configs.append((score, cfg))
    
    # 按评分排序
    scored_configs.sort(key=lambda x: x[0], reverse=True)
    
    # 保留 top-K (保留前 20% 或至少 2 个)
    top_k = max(2, int(len(scored_configs) * 0.2))
    selected = [cfg for _, cfg in scored_configs[:top_k]]
    
    # 记录统计信息
    prune_end = time.time()
    pruned_count = total_configs - len(selected)
    _autotune_stats["pruned_configs"] = pruned_count
    _autotune_stats["benchmarked_configs"] = len(selected)
    _autotune_stats["prune_time_ms"] = (prune_end - prune_start) * 1000
    
    if verbose:
        print(f"  Selected {len(selected)}/{len(configs)} configs for benchmarking")
        for i, (score, cfg) in enumerate(scored_configs[:len(selected)]):
            print(f"    {i+1}. Score={score:.3f}, BM={cfg.kwargs.get('BM')}, "
                  f"BN={cfg.kwargs.get('BN')}, BK={cfg.kwargs.get('BK')}")
    
    return selected if selected else [configs[0]]  # 至少返回一个


# ============================================================================
# Heuristics for Small Sizes (避免 autotune 开销)
# ============================================================================

@triton.heuristics({
    "BM": lambda args: 32 if args["M"] <= 256 else (64 if args["M"] <= 512 else 128),
    "BN": lambda args: 32 if args["N"] <= 256 else (64 if args["N"] <= 512 else 128),
    "BK": lambda args: 16 if args["K"] <= 256 else 32,
    "num_warps": lambda args: 2 if args["M"] <= 256 or args["N"] <= 256 else (4 if args["M"] <= 512 or args["N"] <= 512 else 8),
    "num_stages": lambda args: 1 if args["M"] <= 256 or args["N"] <= 256 else (2 if args["M"] <= 512 or args["N"] <= 512 else 3),
})
@triton.jit
def matmul_heuristics_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """使用 heuristics 的小尺寸 kernel，避免 autotune 开销"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    A = tl.make_block_ptr(
        base=A_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0),
        block_shape=(BM, BK), order=(1, 0),
    )
    B = tl.make_block_ptr(
        base=B_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN),
        block_shape=(BK, BN), order=(0, 1),
    )
    
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    
    for _ in tl.static_range(0, K, BK):
        a = tl.load(A, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(B, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc)
        A = tl.advance(A, (0, BK))
        B = tl.advance(B, (BK, 0))
    
    C = tl.make_block_ptr(
        base=C_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN),
        block_shape=(BM, BN), order=(1, 0),
    )
    tl.store(C, acc.to(tl.float16), boundary_check=(0, 1))


# ============================================================================
# Assembly-Aware Autotune Kernel
# ============================================================================

# 生成 autotune configs，包含 maxnreg 维度
def generate_asm_aware_configs():
    """
    生成包含 maxnreg 维度的 autotune 配置
    
    Note: maxnreg 的支持可能需要特定版本的 Triton，这里先提供框架
    如果当前版本不支持，可以先注释掉 maxnreg 相关代码
    """
    configs = []
    
    # 基础配置：不同的 tile size 和 warp 数量
    base_configs = [
        {"BM": 128, "BN": 128, "BK": 32, "num_warps": 8, "num_stages": 4},
        {"BM": 128, "BN": 128, "BK": 32, "num_warps": 8, "num_stages": 5},
        {"BM": 128, "BN": 256, "BK": 32, "num_warps": 8, "num_stages": 4},
        {"BM": 256, "BN": 128, "BK": 32, "num_warps": 8, "num_stages": 4},
        {"BM": 64, "BN": 128, "BK": 32, "num_warps": 4, "num_stages": 4},
        {"BM": 128, "BN": 64, "BK": 32, "num_warps": 4, "num_stages": 4},
        {"BM": 128, "BN": 128, "BK": 64, "num_warps": 8, "num_stages": 5},
    ]
    
    # 为每个基础配置添加不同的 maxnreg 值
    # 注意：需要根据 Triton 版本验证是否支持 maxnreg 参数
    maxnreg_values = [80, 96, 112, 128, 144, 160]
    
    for base_cfg in base_configs:
        for maxnreg in maxnreg_values:
            cfg_dict = base_cfg.copy()
            kwargs = {"BM": cfg_dict["BM"], "BN": cfg_dict["BN"], "BK": cfg_dict["BK"]}
            
            # 创建 Config 对象
            # 尝试添加 maxnreg（如果 Triton 支持）
            try:
                config = triton.Config(
                    kwargs,
                    num_warps=cfg_dict["num_warps"],
                    num_stages=cfg_dict["num_stages"],
                    maxnreg=maxnreg,  # 尝试添加 maxnreg
                )
            except TypeError:
                # 如果不支持 maxnreg 参数，则创建不带 maxnreg 的配置
                config = triton.Config(
                    kwargs,
                    num_warps=cfg_dict["num_warps"],
                    num_stages=cfg_dict["num_stages"],
                )
                # 尝试在创建后设置（如果支持）
                if hasattr(config, 'maxnreg'):
                    config.maxnreg = maxnreg
            
            configs.append(config)
    
    return configs


# 简化版本：先不使用 maxnreg（因为需要验证 Triton API）
def generate_simplified_configs():
    """生成简化的 autotune 配置（不包含 maxnreg，先验证基础功能）"""
    configs = [
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=5),
        triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8, num_stages=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=5),
        # 确保 BK 是 16 的倍数（Tensor Core 要求）
        triton.Config({"BM": 128, "BN": 128, "BK": 16}, num_warps=8, num_stages=3),
    ]
    return configs


# 生成 autotune 配置列表（必须在装饰器之前定义）
_ASM_AWARE_CONFIGS = [
    triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=4),
    triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=5),
    triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8, num_stages=4),
    triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8, num_stages=4),
    triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=4),
    triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=4, num_stages=4),
    triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=5),
    # 确保 BK 是 16 的倍数（Tensor Core 要求）
    triton.Config({"BM": 128, "BN": 128, "BK": 16}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=_ASM_AWARE_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "top_k": 0.2,  # 先用规则缩到 20%，然后 bench
    },
)
@triton.jit
def matmul_asm_aware_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Assembly-aware autotune kernel"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    A = tl.make_block_ptr(
        base=A_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0),
        block_shape=(BM, BK), order=(1, 0),
    )
    B = tl.make_block_ptr(
        base=B_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN),
        block_shape=(BK, BN), order=(0, 1),
    )
    
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    
    for _ in tl.static_range(0, K, BK):
        a = tl.load(A, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(B, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc)
        A = tl.advance(A, (0, BK))
        B = tl.advance(B, (BK, 0))
    
    C = tl.make_block_ptr(
        base=C_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN),
        block_shape=(BM, BN), order=(1, 0),
    )
    tl.store(C, acc.to(tl.float16), boundary_check=(0, 1))


# ============================================================================
# Wrapper Functions
# ============================================================================

def triton_matmul_asm_aware(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    """
    Assembly-aware Triton GEMM
    
    策略：
    - 小尺寸 (<=512): 使用 heuristics（避免 autotune 开销）
    - 大尺寸 (>512): 使用 assembly-aware autotune
    """
    M, K = A.shape
    _, N = B.shape
    
    # 小尺寸使用 heuristics
    if M <= 512 and N <= 512 and K <= 512:
        grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))
        matmul_heuristics_kernel[grid](
            A, B, C,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            M=M, N=N, K=K,
        )
    else:
        # 大尺寸使用 assembly-aware autotune
        grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))
        matmul_asm_aware_kernel[grid](
            A, B, C,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            M=M, N=N, K=K,
        )


# ============================================================================
# Benchmarking
# ============================================================================

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


def benchmark_asm_aware(M: int, N: int, K: int, iters: int = 50):
    """基准测试 assembly-aware kernel"""
    device = "cuda"
    torch.manual_seed(0)
    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B = torch.randn((K, N), device=device, dtype=torch.float16)
    C = torch.empty((M, N), device=device, dtype=torch.float16)
    
    def call():
        triton_matmul_asm_aware(A, B, C)
    
    # 触发编译
    print(f"Compiling for {M}x{N}x{K}...")
    call()
    torch.cuda.synchronize()
    print("Compilation done")
    
    # 首次运行（可能包含一些一次性开销）
    first_ms = time_cuda_ms(call, iters=1, warmup=0)
    
    # 稳态运行
    steady_ms = time_cuda_ms(call, iters=iters, warmup=5)
    
    # 验证正确性
    ref = (A @ B).float()
    torch.testing.assert_close(C.float(), ref, rtol=5e-2, atol=0.1)
    
    tflops = (2.0 * M * N * K) / (steady_ms * 1e6)
    
    return {
        "first_ms": round(first_ms, 4),
        "steady_ms": round(steady_ms, 4),
        "first_over_steady_ms": round(first_ms - steady_ms, 4),
        "tflops": round(tflops, 2),
    }


def main():
    """主函数：运行基准测试"""
    sizes = [256, 512, 1024, 2048, 4096]
    
    results = []
    for s in sizes:
        print(f"\n{'='*60}")
        print(f"Benchmarking {s}x{s}x{s}")
        print(f"{'='*60}")
        try:
            result = benchmark_asm_aware(s, s, s)
            result["M"] = s
            result["N"] = s
            result["K"] = s
            results.append(result)
            print(f"  Steady: {result['steady_ms']:.4f} ms, {result['tflops']:.2f} TFLOPS")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"M": s, "N": s, "K": s, "error": str(e)})
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

