mkdir build && cd build
cmake .. -DCUTLASS_DIR=/your/path/to/cutlass
make -j4
./cutlass_gemm


# 如何查看生成的 PTX / SASS

# 生成的二进制中可用 cuobjdump / nvdisasm 提取 SASS：
# cuobjdump -sass ./cutlass_gemm > cutlass_sass.txt
# 或 nvdisasm --dump-sass ./cutlass_gemm > cutlass_nvdisasm.sass
# （不同 CUDA 工具链的命令略有差别）