#!/usr/bin/env python3
"""Triton GEMM Kernel - 区分 warmup、编译、运行时间"""

import torch
import triton
import triton.language as tl
import time
import json


def time_cuda_ms(fn, iters=10, warmup=5):
    # 先热身
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

@triton.autotune(
    configs=[
        # 大尺寸配置（保持原有）
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN":  64, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BM":  64, "BN": 128, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BM":  64, "BN":  64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=5),
        # 小尺寸优化配置（新增）
        triton.Config({"BM":  32, "BN":  32, "BK": 16}, num_warps=2, num_stages=2),
        triton.Config({"BM":  32, "BN":  64, "BK": 16}, num_warps=2, num_stages=2),
        triton.Config({"BM":  64, "BN":  32, "BK": 16}, num_warps=2, num_stages=2),
        triton.Config({"BM":  32, "BN":  32, "BK": 32}, num_warps=2, num_stages=3),
        triton.Config({"BM":  64, "BN":  64, "BK": 16}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_opt_kernel_staticK(
    A_ptr, B_ptr, C_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
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

    # ✅ 现在 K 是 constexpr，static_range 可以用
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



def triton_matmul_opt_staticK(A, B, C):
    M, K = A.shape
    _, N = B.shape

    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))
    matmul_opt_kernel_staticK[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        M=M, N=N, K=K,   # ✅ 作为 constexpr 传入
    )


# 针对极小尺寸的优化 kernel（减少开销）
@triton.jit
def matmul_small_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """针对小尺寸优化的简化 kernel"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 使用简单的指针计算，减少 overhead
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    
    # K 维度循环
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        mask_k = offs_k < K
        
        # Load A
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load B
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        acc += tl.dot(a, b)
    
    # Store C
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        # 小尺寸专用配置（更小的 block，更少的 warps）
        triton.Config({"BM": 32, "BN": 32, "BK": 16}, num_warps=1, num_stages=1),
        triton.Config({"BM": 32, "BN": 32, "BK": 32}, num_warps=1, num_stages=2),
        triton.Config({"BM": 64, "BN": 32, "BK": 16}, num_warps=2, num_stages=1),
        triton.Config({"BM": 32, "BN": 64, "BK": 16}, num_warps=2, num_stages=1),
        triton.Config({"BM": 64, "BN": 64, "BK": 16}, num_warps=2, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_small_kernel_autotune(
    A_ptr, B_ptr, C_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """小尺寸 autotune 版本"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        mask_k = offs_k < K
        
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        acc += tl.dot(a, b)
    
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul_opt_small(A, B, C):
    """小尺寸优化版本"""
    M, K = A.shape
    _, N = B.shape
    
    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))
    matmul_small_kernel_autotune[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        M=M, N=N, K=K,
    )


def triton_matmul_opt_smart(A, B, C):
    """
    智能选择：根据矩阵大小选择最优策略
    
    优化策略：
    1. 极小尺寸 (<=128): 使用 PyTorch，因为 kernel 启动开销太大
    2. 小尺寸 (<=512): 使用专用小尺寸优化 kernel (更小的 block，更少的 warps)
    3. 大尺寸 (>512): 使用标准优化版本 (大 block，更多 warps，更多 stages)
    """
    M, K = A.shape
    _, N = B.shape
    
    # 对于非常小的矩阵，使用 PyTorch（启动开销太大）
    if M <= 128 and N <= 128 and K <= 128:
        C.copy_((A @ B))
        return
    
    # 对于小尺寸，使用专用小尺寸优化版本
    if M <= 512 and N <= 512 and K <= 512:
        triton_matmul_opt_small(A, B, C)
    else:
        # 对于大尺寸，使用标准优化版本
        triton_matmul_opt_staticK(A, B, C)



@triton.jit
def matmul_persistent_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    REPEAT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    mask_m = offs_m < M
    mask_n = offs_n < N
    out_mask = mask_m[:, None] & mask_n[None, :]

    # 这里用普通指针也能工作；你也可以换成 make_block_ptr + advance
    for _rep in tl.static_range(0, REPEAT):
        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for k in tl.static_range(0, K, BK):
            offs_k = k + tl.arange(0, BK)
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)

        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16), mask=out_mask)


def triton_matmul_persistent(A, B, C, BM=64, BN=64, BK=32, repeat=32, num_warps=4, num_stages=3):
    M, K = A.shape
    _, N = B.shape
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_persistent_kernel[grid](
        A, B, C,
        M=M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bk=B.stride(0), stride_bn=B.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        BM=BM, BN=BN, BK=BK, REPEAT=repeat,
        num_warps=num_warps,
        num_stages=num_stages,
    )


import torch
import json
import triton
import time

def bench_one(M, N, K, impl="opt", iters=50):
    device = "cuda"
    torch.manual_seed(0)
    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B = torch.randn((K, N), device=device, dtype=torch.float16)
    C = torch.empty((M, N), device=device, dtype=torch.float16)

    def call():
        if impl == "baseline":
            # 你原来的 triton_matmul
            triton_matmul(A, B, C, M, N, K)
        elif impl == "opt":
            triton_matmul_opt_staticK(A, B, C)
        elif impl == "opt_small":
            triton_matmul_opt_small(A, B, C)
        elif impl == "smart":
            triton_matmul_opt_smart(A, B, C)
        elif impl == "persistent":
            triton_matmul_persistent(A, B, C, BM=64, BN=64, BK=32, repeat=32)
        else:
            raise ValueError(impl)

    print("Triggering compilation...")
    # 触发编译（不计入 steady）
    call()
    torch.cuda.synchronize()
    print("Compilation triggered")
    # 首发延迟（包含一些一次性开销：加载、cache、可能二次编译等）
    first_ms = time_cuda_ms(call, iters=1, warmup=0)

    # 稳态
    steady_ms = time_cuda_ms(call, iters=iters, warmup=5)
    print("Steady state reached")
    # 校验（只做一次）
    ref = (A @ B).float()
    torch.testing.assert_close(C.float(), ref, rtol=5e-2, atol=0.1)
    print("Verification passed")
    tflops = (2.0 * M * N * K) / (steady_ms * 1e6)
    print(f"TFLOPS: {tflops:.2f}")
    return {
        "first_ms": round(first_ms, 4),
        "steady_ms": round(steady_ms, 4),
        "first_over_steady_ms": round(first_ms - steady_ms, 4),
        "tflops": round(tflops, 2),
    }

def main():
    sizes = [256, 512, 1024, 2048, 4096]
    # sizes = [256, 512]
    impls = ["baseline", "opt", "opt_small", "smart", "persistent"]

    results = []
    for s in sizes:
        row = {"M": s, "N": s, "K": s}
        for impl in impls:
            try:
                print(f"Benchmarking {impl} for {s}x{s}x{s}")
                row[impl] = bench_one(s, s, s, impl=impl)
            except Exception as e:
                row[impl] = {"error": str(e)}
        results.append(row)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
