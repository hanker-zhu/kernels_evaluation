#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEMM框架环境验证脚本
"""

import sys
import subprocess
import importlib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def check_cmd(cmd: str, name: str) -> bool:
    """检查命令是否可用"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ {name}")
            return True
        print(f"✗ {name}")
        return False
    except Exception:
        print(f"✗ {name}")
        return False


def check_pkg(name: str) -> bool:
    """检查Python包"""
    try:
        importlib.import_module(name)
        print(f"✓ {name}")
        return True
    except ImportError:
        print(f"✗ {name}")
        return False


def check_files() -> bool:
    """检查项目文件"""
    required = [
        "README.md",
        "run.sh",
        "frameworks/cublas/main_cublas_gemm.cu",
        "frameworks/cublas/run.sh",
        "frameworks/cutlass/main_cutlass_gemm.cu",
        "frameworks/cutlass/run.sh",
        "frameworks/triton/triton_gemm.py",
        "frameworks/triton/run.sh",
        "frameworks/tilelang/tilelang_gemm.py",
        "frameworks/tilelang/run.sh",
    ]
    
    ok = True
    for f in required:
        path = BASE_DIR / f
        if path.exists():
            print(f"✓ {f}")
        else:
            print(f"✗ {f}")
            ok = False
    return ok


def main():
    print("=" * 40)
    print("GEMM 框架环境验证")
    print("=" * 40)
    
    ok = True
    
    print("\n[系统工具]")
    ok &= check_cmd("nvcc --version", "CUDA Compiler")
    ok &= check_cmd("cmake --version", "CMake")
    ok &= check_cmd("make --version", "Make")
    
    print("\n[Python 包]")
    ok &= check_pkg("torch")
    ok &= check_pkg("triton")
    check_pkg("tilelang")  # 可选
    check_pkg("matplotlib")  # 可选
    
    print("\n[项目文件]")
    ok &= check_files()
    
    print("\n" + "=" * 40)
    if ok:
        print("✓ 环境检查通过")
        print("\n运行测试: ./run.sh")
    else:
        print("✗ 部分检查失败，请修复后重试")
    print("=" * 40)
    
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
