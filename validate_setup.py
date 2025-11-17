#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEMM框架对比项目环境验证脚本
检查所有依赖和代码是否正确设置
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_command(cmd, description):
    """检查命令是否可用"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description}: 可用")
            return True
        else:
            print(f"✗ {description}: 失败 ({result.stderr.strip()})")
            return False
    except Exception as e:
        print(f"✗ {description}: 错误 ({str(e)})")
        return False

def check_python_package(package_name):
    """检查Python包是否安装"""
    try:
        importlib.import_module(package_name)
        print(f"✓ Python包 {package_name}: 已安装")
        return True
    except ImportError:
        print(f"✗ Python包 {package_name}: 未安装")
        return False

def check_cuda():
    """检查CUDA环境"""
    print("检查CUDA环境...")

    # 检查nvcc
    if not check_command("nvcc --version", "NVIDIA CUDA Compiler"):
        return False

    # 检查cuBLAS
    if not check_command("pkg-config --exists cublas", "cuBLAS library"):
        print("  注意: cuBLAS通常随CUDA安装")
    else:
        print("✓ cuBLAS library: 可用")

    return True

def check_files():
    """检查项目文件"""
    print("检查项目文件...")

    base_dir = Path("/data/hanker/kernels")
    required_files = [
        "README.md",
        "technical_analysis.md",
        "compare_frameworks.py",
        "run_all_benchmarks.sh",
        "cublas/main_cublas_gemm.cu",
        "cublas/CMakeLists.txt",
        "cublas/run.sh",
        "cutlass/main_cutlass_gemm.cu",
        "cutlass/CMakeLists.txt",
        "cutlass/run.sh",
        "tilelang/tilelang_gemm.py",
        "tilelang/run.sh",
        "triton/triton_gemm.py",
        "triton/run.sh"
    ]

    all_present = True
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}: 存在")
        else:
            print(f"✗ {file_path}: 缺失")
            all_present = False

    return all_present

def check_compilation():
    """检查代码是否能编译"""
    print("检查代码编译...")

    base_dir = Path("/data/hanker/kernels")

    # 检查cuBLAS编译
    cublas_dir = base_dir / "cublas"
    if cublas_dir.exists():
        print("测试cuBLAS编译...")
        try:
            # 创建构建目录
            build_dir = cublas_dir / "build"
            build_dir.mkdir(exist_ok=True)

            # 运行cmake
            result = subprocess.run(
                "cmake ..",
                cwd=str(build_dir),
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # 运行make
                result = subprocess.run(
                    "make -j2",
                    cwd=str(build_dir),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    print("✓ cuBLAS: 编译成功")
                else:
                    print(f"✗ cuBLAS编译失败: {result.stderr[:200]}...")
                    return False
            else:
                print(f"✗ cuBLAS CMake配置失败: {result.stderr[:200]}...")
                return False
        except subprocess.TimeoutExpired:
            print("✗ cuBLAS编译超时")
            return False
        except Exception as e:
            print(f"✗ cuBLAS编译错误: {str(e)}")
            return False

    return True

def main():
    """主验证函数"""
    print("=" * 50)
    print("GEMM框架对比项目环境验证")
    print("=" * 50)
    print()

    all_checks_pass = True

    # 检查系统工具
    print("1. 检查系统工具...")
    checks = [
        ("gcc --version", "GNU C Compiler"),
        ("g++ --version", "GNU C++ Compiler"),
        ("cmake --version", "CMake"),
        ("make --version", "Make"),
        ("python --version", "Python"),
    ]

    for cmd, desc in checks:
        if not check_command(cmd, desc):
            all_checks_pass = False
    print()

    # 检查CUDA
    print("2. 检查CUDA环境...")
    if not check_cuda():
        all_checks_pass = False
    print()

    # 检查Python包
    print("3. 检查Python包...")
    python_packages = [
        "torch",
        "torchvision",
        "triton",
        "tilelang"
    ]

    for package in python_packages:
        if not check_python_package(package):
            all_checks_pass = False
    print()

    # 检查项目文件
    print("4. 检查项目文件...")
    if not check_files():
        all_checks_pass = False
    print()

    # 检查编译
    print("5. 检查代码编译...")
    if not check_compilation():
        all_checks_pass = False
    print()

    # 总结
    print("=" * 50)
    if all_checks_pass:
        print("✓ 所有检查通过！项目环境配置正确。")
        print()
        print("下一步:")
        print("  ./run_all_benchmarks.sh    # 运行完整对比测试")
        print("  python compare_frameworks.py  # 生成对比报告")
    else:
        print("✗ 部分检查失败。请修复上述问题后重试。")
        print()
        print("常见解决方案:")
        print("  CUDA问题: 安装/更新NVIDIA驱动和CUDA工具包")
        print("  依赖缺失: pip install torch triton tilelang")
        print("  编译失败: 检查CUDA路径和库链接")

    print("=" * 50)
    return 0 if all_checks_pass else 1

if __name__ == "__main__":
    sys.exit(main())
