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

struct BenchmarkResult {
    int M, N, K;
    float latency_ms;
    double tflops;
    double checksum;
    bool success;
};

template<typename Gemm>
bool run_cutlass_benchmark_single(int M, int N, int K, BenchmarkResult& result) {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;

    result.M = M;
    result.N = N;
    result.K = K;

    try {
        Gemm gemm_op;

        // Host tensors
        cutlass::HostTensor<ElementA, cutlass::layout::RowMajor> A({M, K});
        cutlass::HostTensor<ElementB, cutlass::layout::RowMajor> B({K, N});
        cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> C({M, N});

        // Fill A,B with random (using same seed as other implementations for consistency)
        srand(0);
        for (int i = 0; i < M * K; i++) A.host_data()[i] = cutlass::half_t((float)rand() / RAND_MAX - 0.5f);
        for (int i = 0; i < K * N; i++) B.host_data()[i] = cutlass::half_t((float)rand() / RAND_MAX - 0.5f);
        for (int i = 0; i < M * N; i++) C.host_data()[i] = cutlass::half_t(0);

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
            return false;
        }
        cudaDeviceSynchronize();

        // timing
        int num_runs = 10;
        std::vector<float> times;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < num_runs; ++i) {
            cudaEventRecord(start);
            status = gemm_op(args);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            if (status != cutlass::Status::kSuccess) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return false;
            }
            times.push_back(elapsed_ms(start, stop));
        }

        // calculate average time
        float total_time = 0.0f;
        for (float t : times) total_time += t;
        float avg_time = total_time / num_runs;

        double tflops = (2.0 * (double)M * (double)N * (double)K) / (avg_time * 1e6);

        // calculate checksum for correctness validation
        C.sync_host();
        double sum = 0.0;
        for (int i = 0; i < M * N; i++) sum += static_cast<double>(C.host_data()[i]);

        result.latency_ms = avg_time;
        result.tflops = tflops;
        result.checksum = sum;
        result.success = true;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return true;

    } catch (...) {
        result.success = false;
        result.latency_ms = 0.0f;
        result.tflops = 0.0;
        result.checksum = 0.0;
        return false;
    }
}

int main() {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAcc = float;

    // Define CUTLASS Gemm (using a standard configuration)
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, cutlass::layout::RowMajor,
        ElementB, cutlass::layout::RowMajor,
        ElementC, cutlass::layout::RowMajor,
        ElementAcc,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>
    >;

    std::cout << "CUTLASS GEMM Multi-Size Benchmark" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // 测试指定的尺寸：1024*512*1024 和 4096*2048*4096
    std::vector<BenchmarkResult> results;

    // 测试第一个尺寸：1024*512*1024
    {
        int M = 1024, N = 512, K = 1024;
        BenchmarkResult result;
        if (run_cutlass_benchmark_single<Gemm>(M, N, K, result)) {
            results.push_back(result);
            std::cout << "CUTLASS GEMM (" << result.M << "x" << result.N << "x" << result.K << "): "
                      << result.latency_ms << " ms (avg of 10 runs), "
                      << result.tflops << " TFLOPS" << std::endl;
            std::cout << "Result checksum: " << result.checksum << std::endl;
        } else {
            result.M = M; result.N = N; result.K = K;
            result.success = false;
            results.push_back(result);
            std::cout << "CUTLASS GEMM (" << M << "x" << N << "x" << K << ") failed" << std::endl;
        }
    }

    // 测试第二个尺寸：4096*2048*4096
    {
        int M = 4096, N = 2048, K = 4096;
        BenchmarkResult result;
        if (run_cutlass_benchmark_single<Gemm>(M, N, K, result)) {
            results.push_back(result);
            std::cout << "CUTLASS GEMM (" << result.M << "x" << result.N << "x" << result.K << "): "
                      << result.latency_ms << " ms (avg of 10 runs), "
                      << result.tflops << " TFLOPS" << std::endl;
            std::cout << "Result checksum: " << result.checksum << std::endl;
        } else {
            result.M = M; result.N = N; result.K = K;
            result.success = false;
            results.push_back(result);
            std::cout << "CUTLASS GEMM (" << M << "x" << N << "x" << K << ") failed" << std::endl;
        }
    }

    // 输出JSON格式的结果供Python脚本解析
    std::cout << "\n=== BENCHMARK RESULTS JSON ===" << std::endl;
    std::cout << "[" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "  {" << std::endl;
        std::cout << "    \"M\": " << r.M << "," << std::endl;
        std::cout << "    \"N\": " << r.N << "," << std::endl;
        std::cout << "    \"K\": " << r.K << "," << std::endl;
        std::cout << "    \"latency_ms\": " << r.latency_ms << "," << std::endl;
        std::cout << "    \"tflops\": " << r.tflops << "," << std::endl;
        std::cout << "    \"checksum\": " << r.checksum << "," << std::endl;
        std::cout << "    \"success\": " << (r.success ? "true" : "false") << std::endl;
        std::cout << "  }";
        if (i < results.size() - 1) std::cout << ",";
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;

    // code structure analysis
    std::cout << "\n=== CUTLASS Analysis ===" << std::endl;
    std::cout << "- Programming Model: Template metaprogramming with C++ templates" << std::endl;
    std::cout << "- Implementation: Highly configurable GEMM template library" << std::endl;
    std::cout << "- Memory Hierarchy: Explicit layout and tiling specification" << std::endl;
    std::cout << "- Parallelism: Architecture-specific warp/thread orchestration" << std::endl;
    std::cout << "- Tiling Strategy: 128x128x32 threadblock, 64x64x32 warp, 16x8x16 instruction" << std::endl;
    std::cout << "- Algorithm: Optimized for Ampere architecture (SM80)" << std::endl;
    std::cout << "- Precision: Mixed precision (FP16 input, FP32 accumulation)" << std::endl;

    return 0;
}
