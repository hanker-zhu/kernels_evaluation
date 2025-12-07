cmake_minimum_required(VERSION 3.18)
project(cublas_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
find_package(CUDA REQUIRED)

# cuBLAS 需要链接 cublas 库
add_executable(cublas_gemm main_cublas_gemm.cu)

target_compile_options(cublas_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>)
target_link_libraries(cublas_gemm PRIVATE cublas cudart curand)
