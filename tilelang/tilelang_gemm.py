# tilelang_gemm.py
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout
import torch
import time

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Optional layout hints (commented out by default)
            # T.annotate_layout({
            #     A_shared: make_mma_swizzle_layout(A_shared),
            #     B_shared: make_mma_swizzle_layout(B_shared),
            # })

            # Optional: Enabling swizzle-based rasterization
            # T.use_swizzle(panel_size=10, enable=True)

            # Clear local accumulation
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A from global to shared memory
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Parallel copy tile of B from global to shared memory
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]

                # Perform a tile-level GEMM
                T.gemm(A_shared, B_shared, C_local)

            # Copy result from local (register fragment) to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

def run_tilelang_gemm_single(M, N, K, verbose=True):
    """运行单个TileLang GEMM测试"""
    block_M, block_N, block_K = 128, 128, 32

    # 1. Create the TileLang function
    func = matmul(M, N, K, block_M, block_N, block_K)

    # 2. JIT-compile the kernel for NVIDIA GPU
    jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda")

    # 3. Prepare input tensors in PyTorch
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # 4. Invoke the JIT-compiled kernel (warmup)
    c = jit_kernel(a, b)
    torch.cuda.synchronize()

    # 5. Validate correctness
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    if verbose:
        print(f"TileLang kernel ({M}x{N}x{K}) output matches PyTorch reference.")

    # 6. Profile performance
    profiler = jit_kernel.get_profiler()
    latency = profiler.do_bench()
    tflops = (2.0 * M * N * K) / (latency * 1e6)

    if verbose:
        print(f"TileLang GEMM ({M}x{N}x{K}): {latency:.2f} ms, {tflops:.2f} TFLOPS")

    return latency, tflops

def run_tilelang_gemm():
    """运行TileLang GEMM测试 - 支持多种尺寸"""
    print("TileLang GEMM Multi-Size Benchmark")
    print("=" * 50)

    # 测试尺寸范围：从128到4096
    sizes = [128, 256, 512, 1024, 2048, 4096]

    results = []

    for size in sizes:
        try:
            # 测试方形矩阵
            latency, tflops = run_tilelang_gemm_single(size, size, size, verbose=True)
            results.append({
                'M': size, 'N': size, 'K': size,
                'latency_ms': latency,
                'tflops': tflops,
                'success': True
            })

            # 测试非方形矩阵（如果不是太小）
            if size >= 512:
                # 测试MxN矩形矩阵
                rect_sizes = [(size, size//2, size), (size//2, size, size), (size, size, size//2)]
                for M, N, K in rect_sizes:
                    try:
                        latency, tflops = run_tilelang_gemm_single(M, N, K, verbose=False)
                        results.append({
                            'M': M, 'N': N, 'K': K,
                            'latency_ms': latency,
                            'tflops': tflops,
                            'success': True
                        })
                        print(f"TileLang GEMM ({M}x{N}x{K}): {latency:.2f} ms, {tflops:.2f} TFLOPS")
                    except Exception as e:
                        print(f"TileLang GEMM ({M}x{N}x{K}) failed: {e}")
                        results.append({
                            'M': M, 'N': N, 'K': K,
                            'latency_ms': None,
                            'tflops': None,
                            'success': False,
                            'error': str(e)
                        })

        except Exception as e:
            print(f"TileLang GEMM ({size}x{size}x{size}) failed: {e}")
            results.append({
                'M': size, 'N': size, 'K': size,
                'latency_ms': None,
                'tflops': None,
                'success': False,
                'error': str(e)
            })

    # 代码结构分析
    print("\n=== TileLang Kernel Analysis ===")
    print("- Programming Model: TensorIR-based functional programming")
    print("- Memory Hierarchy: Explicit shared/fragment memory allocation")
    print("- Parallelism: Kernel-level thread orchestration")
    print("- Tiling Strategy: Fixed block sizes (128x128x32)")
    print("- Pipeline Stages: 3-stage pipelining for K dimension")
    print("- Optimization: Automatic memory coalescing and instruction scheduling")

    return results

if __name__ == "__main__":
    run_tilelang_gemm()
