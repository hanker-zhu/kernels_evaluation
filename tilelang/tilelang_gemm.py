# tilelang_gemm.py
import tilelang
import tilelang.language as T
import torch
import time

@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def matmul_kernel(A: T.Tensor((M, K), dtype),
                      B: T.Tensor((K, N), dtype),
                      C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j]  # no-op
            T.copy(C_local, C[by * block_M, bx * block_N])
        return matmul_kernel

if __name__ == "__main__":
    M = N = K = 1024  # 4096 可能需要更大显存或分批测试
    block_M, block_N, block_K = 128, 128, 32
    kernel = matmul(M, N, K, block_M, block_N, block_K)

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)

    # Warmup + correctness
    kernel(a, b, c)
    ref = a @ b
    torch.testing.assert_close(c.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print("TileLang kernel output matches PyTorch reference.")

    # Benchmark (example)
    import time
    torch.cuda.synchronize()
    t0 = time.time()
    kernel(a, b, c)
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000.0
    tflops = (2.0*M*N*K) / (ms * 1e6)
    print(f"TileLang matmul: {ms:.2f} ms, {tflops:.2f} TFLOPS")
