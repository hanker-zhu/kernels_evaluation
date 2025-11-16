// main_cublas_gemm.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms; cudaEventElapsedTime(&ms, a, b); return ms;
}
int main() {
    int M=4096, N=4096, K=4096;
    size_t szA = (size_t)M*K;
    size_t szB = (size_t)K*N;
    size_t szC = (size_t)M*N;
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, szA * sizeof(half));
    cudaMalloc(&d_B, szB * sizeof(half));
    cudaMalloc(&d_C, szC * sizeof(half));
    // fill with random using host then copy (omitted for brevity)
    // use cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta = 0.0f;
    // warmup
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16F, N,
                 d_A, CUDA_R_16F, K,
                 &beta,
                 d_C, CUDA_R_16F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16F, N,
                 d_A, CUDA_R_16F, K,
                 &beta,
                 d_C, CUDA_R_16F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = elapsed_ms(start, stop);
    double gflops = (2.0 * (double)M * (double)N * (double)K) / (ms * 1e6);
    std::cout << "cuBLAS GEMM: " << ms << " ms, " << gflops << " TFLOPS\n";
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
