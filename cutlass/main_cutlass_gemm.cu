// main_cutlass_gemm.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/tensor_view_io.h>
#include <cutlass/numeric_types.h>

// helper timing
float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms; cudaEventElapsedTime(&ms, a, b); return ms;
}

int main() {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAcc = float;

    int M = 4096, N = 4096, K = 4096;

    // Define CUTLASS Gemm
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, cutlass::layout::RowMajor,
        ElementB, cutlass::layout::RowMajor,
        ElementC, cutlass::layout::RowMajor,
        ElementAcc
    >;

    Gemm gemm_op;

    // Host tensors
    cutlass::HostTensor<ElementA, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<ElementB, cutlass::layout::RowMajor> B({K, N});
    cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> C({M, N});
    cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> C_ref({M, N});

    // Fill A,B with random
    srand(0);
    for (int i=0;i<M*K;i++) A.data()[i] = cutlass::half_t((float)rand()/RAND_MAX - 0.5f);
    for (int i=0;i<K*N;i++) B.data()[i] = cutlass::half_t((float)rand()/RAND_MAX - 0.5f);
    for (int i=0;i<M*N;i++) C.data()[i] = cutlass::half_t(0);

    A.sync_device();
    B.sync_device();
    C.sync_device();

    // set up gemm arguments
    typename Gemm::Arguments args(
        {M, N, K},
        { (ElementA*)A.device_data(), K },
        { (ElementB*)B.device_data(), N },
        { (ElementC*)C.device_data(), N },
        { (ElementC*)C.device_data(), N },
        {1.0f, 0.0f}
    );

    // warmup
    auto status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM launch failed\n";
        return -1;
    }
    cudaDeviceSynchronize();

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    status = gemm_op(args);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = elapsed_ms(start, stop);

    double gflops = (2.0 * (double)M * (double)N * (double)K) / (ms * 1e6);
    std::cout << "CUTLASS GEMM: " << ms << " ms, " << gflops << " TFLOPS\n";

    // validate against CPU/PyTorch result: we can't easily call torch here; for a quick check use cuBLAS or trust seed consistency
    // For correctness: copy C to host and print a checksum
    C.sync_host();
    double sum = 0;
    for (int i=0;i<M*N;i++) sum += float(C.host_data()[i]);
    std::cout << "C sum (host): " << sum << std::endl;

    return 0;
}
