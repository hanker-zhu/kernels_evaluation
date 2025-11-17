# triton_gemm.py
import torch
import triton
import triton.language as tl
import time

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Triton GEMM kernel - 基于官方教程的标准实现"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算当前block的起始位置
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 遍历K维度
    for k in range(0, K, BLOCK_SIZE_K):
        # 加载A的tile: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # 加载B的tile: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 矩阵乘法累加
        accumulator = tl.dot(a, b, accumulator)

    # 存储结果
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

def triton_matmul(A, B, C, M, N, K):
    """Triton GEMM封装函数"""
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
    )

def run_triton_gemm():
    """运行Triton GEMM测试"""
    M = N = K = 4096
    device = 'cuda'

    # 初始化数据
    torch.manual_seed(0)
    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B = torch.randn((K, N), device=device, dtype=torch.float16)
    C = torch.zeros((M, N), device=device, dtype=torch.float16)

    # 热身
    triton_matmul(A, B, C, M, N, K)
    torch.cuda.synchronize()

    # 正确性验证
    ref = torch.matmul(A, B)
    torch.testing.assert_close(C.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print("Triton kernel output matches PyTorch reference.")

    # 性能测试
    num_runs = 10
    times = []

    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        triton_matmul(A, B, C, M, N, K)
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    avg_time = sum(times) / len(times)
    tflops = (2.0 * M * N * K) / (avg_time * 1e6)

    print(f"Triton GEMM: {avg_time:.2f} ms (avg of {num_runs} runs), {tflops:.2f} TFLOPS")

    # 代码结构分析
    print("\n=== Triton Kernel Analysis ===")
    print("- Programming Model: Python-based JIT compilation")
    print("- Memory Hierarchy: Programmed shared memory via tl.load/tl.store")
    print("- Parallelism: Block-level parallelism with thread cooperation")
    print("- Tiling Strategy: Fixed 128x128x32 block sizes")
    print("- Pipeline Stages: Manual loop unrolling for K dimension")

if __name__ == "__main__":
    run_triton_gemm()
