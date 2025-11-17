cmake_minimum_required(VERSION 3.18)
project(cutlass_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
find_package(CUDA REQUIRED)

# CUTLASS 源码路径
set(CUTLASS_DIR "/data/hanker/cutlass")

include_directories(${CUTLASS_DIR})
include_directories(${CUTLASS_DIR}/include)
include_directories(${CUTLASS_DIR}/tools/util/include)
link_directories(${CUTLASS_DIR}/lib)

add_executable(cutlass_gemm main_cutlass_gemm.cu)

# 链接 CUDA runtime (如果需要其他 libs 可加)
target_compile_options(cutlass_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>)
target_link_libraries(cutlass_gemm PRIVATE cudart)
