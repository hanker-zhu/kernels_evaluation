// cublas_softmax.cu - cuDNN Softmax implementation
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <chrono>

float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms; cudaEventElapsedTime(&ms, a, b); return ms;
}

struct BenchmarkResult {
    int seq_len, hidden_dim;
    float latency_ms;
    bool success;
};

void run_softmax_benchmark(int seq_len, int hidden_dim, BenchmarkResult& result) {
    result.seq_len = seq_len;
    result.hidden_dim = hidden_dim;
    
    try {
        size_t sz = (size_t)seq_len * hidden_dim;
        float *d_x, *d_y;
        cudaMalloc(&d_x, sz * sizeof(float));
        cudaMalloc(&d_y, sz * sizeof(float));
        
        // 初始化随机数据
        std::vector<float> h_x(sz);
        for (size_t i = 0; i < sz; ++i) {
            h_x[i] = (float)rand() / RAND_MAX;
        }
        cudaMemcpy(d_x, h_x.data(), sz * sizeof(float), cudaMemcpyHostToDevice);
        
        // 创建 cuDNN 句柄
        cudnnHandle_t cudnn;
        cudnnCreate(&cudnn);
        
        // 创建张量描述符
        cudnnTensorDescriptor_t x_desc, y_desc;
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnCreateTensorDescriptor(&y_desc);
        cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   seq_len, hidden_dim, 1, 1);
        cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   seq_len, hidden_dim, 1, 1);
        
        // 创建 Softmax 描述符
        cudnnSoftmaxDescriptor_t softmax_desc;
        cudnnCreateSoftmaxDescriptor(&softmax_desc);
        cudnnSetSoftmaxDescriptor(softmax_desc, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);
        
        // 热身
        cudnnSoftmaxForward(cudnn, softmax_desc, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                           &(float){1.0f}, x_desc, d_x,
                           &(float){0.0f}, y_desc, d_y);
        cudaDeviceSynchronize();
        
        // 性能测试
        int num_runs = 10;
        std::vector<float> times;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        for (int i = 0; i < num_runs; ++i) {
            cudaEventRecord(start);
            cudnnSoftmaxForward(cudnn, softmax_desc, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                               &(float){1.0f}, x_desc, d_x,
                               &(float){0.0f}, y_desc, d_y);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            times.push_back(elapsed_ms(start, stop));
        }
        
        float total_time = 0.0f;
        for (float t : times) total_time += t;
        float avg_time = total_time / num_runs;
        
        result.latency_ms = avg_time;
        result.success = true;
        
        cudnnDestroySoftmaxDescriptor(softmax_desc);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroy(cudnn);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
    } catch (...) {
        result.success = false;
        result.latency_ms = 0.0f;
    }
}

int main() {
    std::cout << "cuDNN Softmax Benchmark" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    std::vector<std::pair<int, int>> shapes = {
        {256, 256}, {512, 512}, {1024, 1024},
        {2048, 2048}, {4096, 4096}, {8192, 8192}
    };
    std::vector<BenchmarkResult> results;
    
    for (auto& shape : shapes) {
        BenchmarkResult result;
        run_softmax_benchmark(shape.first, shape.second, result);
        results.push_back(result);
        
        if (result.success) {
            std::cout << "Softmax (" << result.seq_len << "x" << result.hidden_dim << "): "
                      << result.latency_ms << " ms (avg of 10 runs)" << std::endl;
        } else {
            std::cout << "Softmax (" << shape.first << "x" << shape.second << ") failed" << std::endl;
        }
    }
    
    // 输出JSON格式
    std::cout << "\n=== BENCHMARK RESULTS JSON ===" << std::endl;
    std::cout << "{" << std::endl;
    std::cout << "  \"kernel\": \"softmax\"," << std::endl;
    std::cout << "  \"benchmarks\": [" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "    {" << std::endl;
        std::cout << "      \"shape\": \"" << r.seq_len << "x" << r.hidden_dim << "\"," << std::endl;
        std::cout << "      \"latency_ms\": " << r.latency_ms << "," << std::endl;
        std::cout << "      \"success\": " << (r.success ? "true" : "false") << std::endl;
        std::cout << "    }";
        if (i < results.size() - 1) std::cout << ",";
        std::cout << std::endl;
    }
    std::cout << "  ]" << std::endl;
    std::cout << "}" << std::endl;
    
    return 0;
}

