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

def run_triton_gemm_single(M, N, K, verbose=True):
    """运行单个Triton GEMM测试并保存编译结果"""
    device = 'cuda'

    # 初始化数据
    torch.manual_seed(0)
    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B = torch.randn((K, N), device=device, dtype=torch.float16)
    C = torch.zeros((M, N), device=device, dtype=torch.float16)

    # 保存编译后的代码
    import os
    compile_output_dir = "/data/hanker/kernels/compile_outputs"
    os.makedirs(compile_output_dir, exist_ok=True)

    # 热身并获取编译信息
    triton_matmul(A, B, C, M, N, K)
    torch.cuda.synchronize()

    # 获取编译后的PTX和SASS代码 - 在kernel运行后立即检查
    if verbose:
        print("Inspecting Triton kernel attributes...")

    try:
        # 打印所有非私有属性
        attrs = [attr for attr in dir(matmul_kernel) if not attr.startswith('_')]
        if verbose:
            print(f"Available kernel attributes: {attrs}")

        # 检查asm属性
        if hasattr(matmul_kernel, 'asm') and matmul_kernel.asm:
            asm_dict = matmul_kernel.asm
            if verbose:
                print(f"Found asm dict with keys: {list(asm_dict.keys())}")

            if "ptx" in asm_dict and asm_dict["ptx"]:
                ptx_code = asm_dict["ptx"]
                ptx_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}.ptx"
                with open(ptx_filename, 'w') as f:
                    f.write(ptx_code)
                print(f"Saved Triton PTX code to: {ptx_filename}")
                print(f"PTX code size: {len(ptx_code)} characters")

        # 检查其他可能的属性，特别是device_caches
        for attr in ['device_caches', '_cache', 'cache', '_triton_kernel']:
            if hasattr(matmul_kernel, attr):
                cache_obj = getattr(matmul_kernel, attr)
                if verbose:
                    print(f"Found {attr}: {type(cache_obj)}")

                if attr == 'device_caches' and cache_obj:
                    # device_caches通常是一个字典，包含不同设备的缓存
                    if verbose:
                        print(f"device_caches keys: {list(cache_obj.keys())}")
                    for device_key, device_cache in cache_obj.items():
                        if verbose:
                            print(f"Device {device_key} cache: {type(device_cache)}")
                            if isinstance(device_cache, tuple):
                                print(f"Tuple length: {len(device_cache)}")
                                for i, item in enumerate(device_cache):
                                    print(f"Tuple item {i}: {type(item)}")
                                    if isinstance(item, dict):
                                        print(f"  Dict keys: {list(item.keys())}")
                                        # 检查dict中的值
                                        for dict_key, dict_value in item.items():
                                            print(f"    Key: {dict_key}")
                                            print(f"    Value type: {type(dict_value)}")
                                            if hasattr(dict_value, 'asm') and dict_value.asm:
                                                print(f"    Found asm in dict value!")
                                                if "ptx" in dict_value.asm:
                                                    ptx_code = dict_value.asm["ptx"]
                                                    ptx_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}_dict.ptx"
                                                    with open(ptx_filename, 'w') as f:
                                                        f.write(ptx_code)
                                                    print(f"Saved Triton PTX from dict to: {ptx_filename}")
                                                if "sass" in dict_value.asm:
                                                    sass_code = dict_value.asm["sass"]
                                                    sass_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}_dict.sass"
                                                    with open(sass_filename, 'w') as f:
                                                        f.write(sass_code)
                                                    print(f"Saved Triton SASS from dict to: {sass_filename}")
                                            elif isinstance(dict_value, dict):
                                                print(f"    Nested dict keys: {list(dict_value.keys())}")
                                                # 检查嵌套dict
                                                if 'asm' in dict_value and dict_value['asm']:
                                                    asm_dict = dict_value['asm']
                                                    if "ptx" in asm_dict:
                                                        ptx_code = asm_dict["ptx"]
                                                        ptx_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}_nested.ptx"
                                                        with open(ptx_filename, 'w') as f:
                                                            f.write(ptx_code)
                                                        print(f"Saved Triton PTX from nested dict to: {ptx_filename}")

                        if hasattr(device_cache, 'asm') and device_cache.asm:
                            if "ptx" in device_cache.asm:
                                ptx_code = device_cache.asm["ptx"]
                                ptx_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}_device{device_key}.ptx"
                                with open(ptx_filename, 'w') as f:
                                    f.write(ptx_code)
                                print(f"Saved Triton PTX from device_caches to: {ptx_filename}")
                                break
                        elif isinstance(device_cache, tuple):
                            # 检查tuple中的每个item
                            for i, item in enumerate(device_cache):
                                if hasattr(item, 'asm') and item.asm:
                                    if "ptx" in item.asm:
                                        ptx_code = item.asm["ptx"]
                                        ptx_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}_device{device_key}_item{i}.ptx"
                                        with open(ptx_filename, 'w') as f:
                                            f.write(ptx_code)
                                        print(f"Saved Triton PTX from device_caches tuple item {i} to: {ptx_filename}")
                                        break
                        elif isinstance(device_cache, dict):
                            # 如果device_cache是字典，遍历查找asm
                            for cache_key, cache_entry in device_cache.items():
                                if hasattr(cache_entry, 'asm') and cache_entry.asm:
                                    if "ptx" in cache_entry.asm:
                                        ptx_code = cache_entry.asm["ptx"]
                                        ptx_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}_device{device_key}_{cache_key}.ptx"
                                        with open(ptx_filename, 'w') as f:
                                            f.write(ptx_code)
                                        print(f"Saved Triton PTX from device_caches entry to: {ptx_filename}")
                                        break

                elif cache_obj and hasattr(cache_obj, 'asm') and cache_obj.asm:
                    if "ptx" in cache_obj.asm:
                        ptx_code = cache_obj.asm["ptx"]
                        ptx_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}_{attr}.ptx"
                        with open(ptx_filename, 'w') as f:
                            f.write(ptx_code)
                        print(f"Saved Triton PTX from {attr} to: {ptx_filename}")
                        break

    except Exception as e:
        if verbose:
            print(f"Direct kernel inspection failed: {e}")

    # 尝试使用Triton的compile函数来获取编译后的代码
    try:
        # 使用Triton的compile函数来获取编译后的代码
        import triton.language as tl

        # 创建一个编译后的kernel对象
        compiled_kernel = triton.compile(
            matmul_kernel,
            # 使用简化的signature
            signature="*fp16, *fp16, *fp16, i32, i32, i32, i32, i32, i32, i32, i32, i32",
            device=device
        )

        # 获取asm信息
        if hasattr(compiled_kernel, 'asm') and compiled_kernel.asm:
            asm_dict = compiled_kernel.asm

            # 保存PTX代码
            if "ptx" in asm_dict and asm_dict["ptx"]:
                ptx_code = asm_dict["ptx"]
                ptx_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}.ptx"
                with open(ptx_filename, 'w') as f:
                    f.write(ptx_code)
                if verbose:
                    print(f"Saved Triton PTX code to: {ptx_filename}")
                    print(f"PTX code size: {len(ptx_code)} characters")

            # 保存SASS代码
            if "sass" in asm_dict and asm_dict["sass"]:
                sass_code = asm_dict["sass"]
                sass_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}.sass"
                with open(sass_filename, 'w') as f:
                    f.write(sass_code)
                if verbose:
                    print(f"Saved Triton SASS code to: {sass_filename}")
                    print(f"SASS code size: {len(sass_code)} characters")

            # 保存其他可用的编译信息
            for key, value in asm_dict.items():
                if key not in ["ptx", "sass"] and isinstance(value, str) and value:
                    other_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}.{key}"
                    with open(other_filename, 'w') as f:
                        f.write(value)
                    if verbose:
                        print(f"Saved Triton {key} code to: {other_filename}")

        else:
            if verbose:
                print("Warning: Compiled kernel has no asm attribute")

    except Exception as e:
        if verbose:
            print(f"Warning: Could not save Triton compiled code using triton.compile: {e}")
            # 尝试备用方法 - 检查kernel运行后的属性
            try:
                # 运行kernel后检查是否有编译缓存
                if hasattr(matmul_kernel, '_cache') and matmul_kernel._cache:
                    print(f"Found kernel cache with {len(matmul_kernel._cache)} entries")
                    for cache_key, cache_value in matmul_kernel._cache.items():
                        print(f"Cache key: {cache_key}")
                        if hasattr(cache_value, 'asm'):
                            print(f"Found asm in cache!")
                            # 尝试保存这个
                            asm_dict = cache_value.asm
                            if "ptx" in asm_dict and asm_dict["ptx"]:
                                ptx_filename = f"{compile_output_dir}/triton_gemm_{M}x{N}x{K}_fallback.ptx"
                                with open(ptx_filename, 'w') as f:
                                    f.write(asm_dict["ptx"])
                                print(f"Saved Triton PTX from fallback method to: {ptx_filename}")
            except Exception as fallback_e:
                print(f"Fallback method also failed: {fallback_e}")

    # 正确性验证
    ref = torch.matmul(A, B)
    torch.testing.assert_close(C.float(), ref.float(), rtol=1e-2, atol=1e-2)
    if verbose:
        print(f"Triton kernel ({M}x{N}x{K}) output matches PyTorch reference.")

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

    if verbose:
        print(f"Triton GEMM ({M}x{N}x{K}): {avg_time:.2f} ms (avg of {num_runs} runs), {tflops:.2f} TFLOPS")

    return avg_time, tflops

def run_triton_gemm():
    """运行Triton GEMM测试 - 测试指定尺寸"""
    print("Triton GEMM Specific Size Benchmark")
    print("=" * 50)

    # 测试指定的尺寸：1024*512*1024 和 4096*2048*4096
    test_configs = [
        (1024, 512, 1024),   # 用户指定的第一个尺寸
        (4096, 2048, 4096)   # 用户指定的第二个尺寸
    ]

    results = []

    for M, N, K in test_configs:
        try:
            latency, tflops = run_triton_gemm_single(M, N, K, verbose=True)
            results.append({
                'M': M, 'N': N, 'K': K,
                'latency_ms': latency,
                'tflops': tflops,
                'success': True
            })
        except Exception as e:
            print(f"Triton GEMM ({M}x{N}x{K}) failed: {e}")
            results.append({
                'M': M, 'N': N, 'K': K,
                'latency_ms': None,
                'tflops': None,
                'success': False,
                'error': str(e)
            })

    # 代码结构分析
    print("\n=== Triton Kernel Analysis ===")
    print("- Programming Model: Python-based JIT compilation")
    print("- Memory Hierarchy: Programmed shared memory via tl.load/tl.store")
    print("- Parallelism: Block-level parallelism with thread cooperation")
    print("- Tiling Strategy: Fixed 128x128x32 block sizes")
    print("- Pipeline Stages: Manual loop unrolling for K dimension")

    return results

if __name__ == "__main__":
    run_triton_gemm()
