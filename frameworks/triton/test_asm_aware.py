#!/usr/bin/env python3
"""
简单的测试脚本，验证 assembly-aware autotune kernel 是否正常工作
"""

import torch
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from triton_gemm_asm_aware import triton_matmul_asm_aware

def test_basic():
    """基本功能测试"""
    print("Testing basic functionality...")
    
    device = "cuda"
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return False
    
    # 创建测试矩阵
    M, N, K = 256, 256, 256
    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B = torch.randn((K, N), device=device, dtype=torch.float16)
    C = torch.empty((M, N), device=device, dtype=torch.float16)
    
    try:
        # 执行 kernel
        triton_matmul_asm_aware(A, B, C)
        torch.cuda.synchronize()
        
        # 验证正确性
        ref = (A @ B).float()
        torch.testing.assert_close(C.float(), ref, rtol=5e-2, atol=0.1)
        
        print(f"✓ Test passed for {M}x{N}x{K}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_sizes():
    """测试多个尺寸"""
    print("\nTesting multiple sizes...")
    
    device = "cuda"
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return False
    
    sizes = [(128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
    
    results = []
    for M, N, K in sizes:
        A = torch.randn((M, K), device=device, dtype=torch.float16)
        B = torch.randn((K, N), device=device, dtype=torch.float16)
        C = torch.empty((M, N), device=device, dtype=torch.float16)
        
        try:
            triton_matmul_asm_aware(A, B, C)
            torch.cuda.synchronize()
            
            # 验证正确性
            ref = (A @ B).float()
            torch.testing.assert_close(C.float(), ref, rtol=5e-2, atol=0.1)
            
            print(f"✓ Test passed for {M}x{N}x{K}")
            results.append(True)
        except Exception as e:
            print(f"✗ Test failed for {M}x{N}x{K}: {e}")
            results.append(False)
    
    return all(results)

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Assembly-Aware Autotune GEMM")
    print("=" * 60)
    
    success = True
    success = test_basic() and success
    success = test_multiple_sizes() and success
    
    print("\n" + "=" * 60)
    if success:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)
    
    sys.exit(0 if success else 1)

