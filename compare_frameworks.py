#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEMM算子生成框架对比分析脚本

对比 cuBLAS, CUTLASS, TileLang, Triton 在GEMM算子生成方面的差异：
1. 设计出发点
2. 面向的问题
3. 生成算法
4. 性能效果
"""

import subprocess
import re
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd

class FrameworkComparator:
    """算子生成框架对比器"""

    def __init__(self, base_dir: str = "/data/hanker/kernels"):
        self.base_dir = Path(base_dir)
        self.frameworks = {
            'cublas': {
                'name': 'cuBLAS',
                'dir': self.base_dir / 'cublas',
                'build_cmd': 'make',
                'run_cmd': './cublas_gemm',
                'source': 'main_cublas_gemm.cu'
            },
            'cutlass': {
                'name': 'CUTLASS',
                'dir': self.base_dir / 'cutlass',
                'build_cmd': 'make',
                'run_cmd': './cutlass_gemm',
                'source': 'main_cutlass_gemm.cu'
            },
            'tilelang': {
                'name': 'TileLang',
                'dir': self.base_dir / 'tilelang',
                'build_cmd': None,  # Python脚本，无需编译
                'run_cmd': 'python tilelang_gemm.py',
                'source': 'tilelang_gemm.py'
            },
            'triton': {
                'name': 'Triton',
                'dir': self.base_dir / 'triton',
                'build_cmd': None,  # Python脚本，无需编译
                'run_cmd': 'python triton_gemm.py',
                'source': 'triton_gemm.py'
            }
        }

        self.results = {}

    def run_command(self, cmd: str, cwd: Optional[str] = None,
                   timeout: int = 300) -> Tuple[str, str, int]:
        """运行命令并返回输出"""
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=cwd,
                capture_output=True, text=True, timeout=timeout
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", f"Command timed out after {timeout}s", -1
        except Exception as e:
            return "", str(e), -1

    def build_framework(self, framework: str) -> bool:
        """编译框架代码"""
        fw = self.frameworks[framework]
        if fw['build_cmd'] is None:
            print(f"✓ {fw['name']}: No build required (Python script)")
            return True

        print(f"Building {fw['name']}...")
        stdout, stderr, ret = self.run_command(fw['build_cmd'], cwd=str(fw['dir']))

        if ret == 0:
            print(f"✓ {fw['name']}: Build successful")
            return True
        else:
            print(f"✗ {fw['name']}: Build failed")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return False

    def run_framework(self, framework: str) -> Dict:
        """运行框架测试并解析结果"""
        fw = self.frameworks[framework]
        print(f"Running {fw['name']}...")

        stdout, stderr, ret = self.run_command(fw['run_cmd'], cwd=str(fw['dir']))

        result = {
            'framework': fw['name'],
            'success': ret == 0,
            'stdout': stdout,
            'stderr': stderr,
            'returncode': ret,
            'time_ms': None,
            'tflops': None,
            'checksum': None,
            'analysis': {}
        }

        if ret == 0:
            # 解析性能数据
            time_match = re.search(r'(\w+) GEMM: ([0-9.]+) ms.*?, ([0-9.]+) TFLOPS', stdout)
            if time_match:
                result['time_ms'] = float(time_match.group(2))
                result['tflops'] = float(time_match.group(3))

            # 解析校验和
            checksum_match = re.search(r'Result checksum: ([0-9.e+-]+)', stdout)
            if checksum_match:
                result['checksum'] = float(checksum_match.group(1))

            # 解析分析信息
            analysis_section = re.search(r'=== .* Analysis ===(.*?)(?=\n\n|\Z)', stdout, re.DOTALL)
            if analysis_section:
                analysis_lines = analysis_section.group(1).strip().split('\n')
                for line in analysis_lines:
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        key = key.strip('- ')
                        result['analysis'][key] = value.strip()

            print(".2f"                   f"{result['tflops']:.2f}" if result['tflops'] else "N/A")
        else:
            print(f"✗ {fw['name']}: Run failed")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)

        return result

    def run_all(self) -> Dict:
        """运行所有框架的测试"""
        print("=" * 60)
        print("GEMM算子生成框架对比测试")
        print("=" * 60)

        for fw_name in self.frameworks.keys():
            # 编译
            if not self.build_framework(fw_name):
                self.results[fw_name] = {'framework': self.frameworks[fw_name]['name'], 'success': False}
                continue

            # 运行测试
            self.results[fw_name] = self.run_framework(fw_name)
            print()

        return self.results

    def analyze_results(self) -> Dict:
        """分析测试结果"""
        analysis = {
            'performance_comparison': {},
            'correctness_validation': {},
            'framework_characteristics': {},
            'summary': {}
        }

        # 性能对比
        successful_runs = {k: v for k, v in self.results.items() if v.get('success', False)}

        if successful_runs:
            best_tflops = max(r['tflops'] for r in successful_runs.values() if r['tflops'])
            worst_tflops = min(r['tflops'] for r in successful_runs.values() if r['tflops'])

            analysis['performance_comparison'] = {
                'best_performer': max(successful_runs.items(), key=lambda x: x[1]['tflops'])[0],
                'worst_performer': min(successful_runs.items(), key=lambda x: x[1]['tflops'])[0],
                'performance_range': f"{worst_tflops:.2f} - {best_tflops:.2f} TFLOPS",
                'speedup_ratio': best_tflops / worst_tflops if worst_tflops > 0 else float('inf')
            }

        # 正确性验证
        checksums = {k: v['checksum'] for k, v in successful_runs.items() if v['checksum'] is not None}
        if len(checksums) > 1:
            checksum_values = list(checksums.values())
            max_diff = max(checksums.values()) - min(checksums.values())
            analysis['correctness_validation'] = {
                'checksum_consistency': max_diff < 1e-3,  # 允许小误差
                'max_checksum_diff': max_diff,
                'checksums': checksums
            }

        # 框架特性对比
        analysis['framework_characteristics'] = {}
        for fw_name, result in successful_runs.items():
            analysis['framework_characteristics'][fw_name] = result.get('analysis', {})

        return analysis

    def generate_report(self) -> str:
        """生成对比报告"""
        if not self.results:
            return "No results available. Please run tests first."

        analysis = self.analyze_results()

        report = []
        report.append("# GEMM算子生成框架对比分析报告")
        report.append("")
        report.append("## 测试配置")
        report.append("- 矩阵尺寸: 4096x4096x4096")
        report.append("- 数据类型: FP16输入, FP32累加")
        report.append("- 测试运行次数: 10次取平均")
        report.append("")

        report.append("## 性能对比")
        successful_runs = {k: v for k, v in self.results.items() if v.get('success', False)}

        if successful_runs:
            report.append("| 框架 | 时间(ms) | 性能(TFLOPS) | 校验和 |")
            report.append("|------|----------|-------------|--------|")

            for fw_name, result in successful_runs.items():
                time_str = ".2f" if result['time_ms'] else "N/A"
                tflops_str = ".2f" if result['tflops'] else "N/A"
                checksum_str = ".2e" if result['checksum'] else "N/A"
                report.append(f"| {result['framework']} | {time_str} | {tflops_str} | {checksum_str} |")

            if 'performance_comparison' in analysis:
                perf = analysis['performance_comparison']
                report.append("")
                report.append(f"- 最快框架: {self.frameworks[perf['best_performer']]['name']}")
                report.append(f"- 最慢框架: {self.frameworks[perf['worst_performer']]['name']}")
                report.append(".1f")
        else:
            report.append("无成功运行的框架")

        report.append("")
        report.append("## 框架特性对比")
        report.append("")

        # 设计出发点对比
        report.append("### 设计出发点")
        design_philosophies = {
            'cublas': "高度优化的厂商库，专注于生产环境的极致性能",
            'cutlass': "可配置的模板库，为研究者和开发者提供灵活的算子定制能力",
            'tilelang': "基于TensorIR的函数式编程，提供自动优化的声明式算子描述",
            'triton': "Python JIT编译框架，降低CUDA编程门槛的同时保持高性能"
        }

        for fw_name, desc in design_philosophies.items():
            fw = self.frameworks[fw_name]
            report.append(f"**{fw['name']}**: {desc}")

        report.append("")

        # 编程模型对比
        report.append("### 编程模型")
        for fw_name, result in successful_runs.items():
            fw = self.frameworks[fw_name]
            if 'Programming Model' in result.get('analysis', {}):
                model = result['analysis']['Programming Model']
                report.append(f"**{fw['name']}**: {model}")

        report.append("")

        # 算法特点
        report.append("### 核心算法特点")
        algorithm_features = {
            'cublas': "专有优化算法，针对具体硬件深度调优",
            'cutlass': "模板元编程，精确控制tiling和并行策略",
            'tilelang': "声明式编程 + 自动优化调度",
            'triton': "基于tile的矩阵乘法，结合软件流水线优化"
        }

        for fw_name, desc in algorithm_features.items():
            fw = self.frameworks[fw_name]
            report.append(f"**{fw['name']}**: {desc}")

        report.append("")

        # 内存层次
        report.append("### 内存层次管理")
        for fw_name, result in successful_runs.items():
            fw = self.frameworks[fw_name]
            if 'Memory Hierarchy' in result.get('analysis', {}):
                mem = result['analysis']['Memory Hierarchy']
                report.append(f"**{fw['name']}**: {mem}")

        report.append("")

        # 平行化策略
        report.append("### 并行化策略")
        for fw_name, result in successful_runs.items():
            fw = self.frameworks[fw_name]
            if 'Parallelism' in result.get('analysis', {}):
                par = result['analysis']['Parallelism']
                report.append(f"**{fw['name']}**: {par}")

        report.append("")

        report.append("## 总结与建议")
        report.append("")
        report.append("### 适用场景")
        report.append("- **生产环境**: 推荐使用cuBLAS，性能最稳定可靠")
        report.append("- **研究开发**: 推荐使用CUTLASS，可定制性强")
        report.append("- **快速原型**: 推荐使用Triton或TileLang，开发效率高")
        report.append("")
        report.append("### 技术洞察")
        report.append("1. **库 vs 编译器**: cuBLAS/CUTLASS偏向预优化库，Triton/TileLang偏向JIT编译")
        report.append("2. **抽象层次**: 从低到高为cuBLAS < CUTLASS < TileLang < Triton")
        report.append("3. **优化策略**: 库关注运行时效率，编译器关注代码生成优化")
        report.append("4. **开发体验**: Python框架显著降低了CUDA编程门槛")

        return "\n".join(report)

    def save_results(self, filename: str = "benchmark_results.json"):
        """保存测试结果"""
        with open(self.base_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")

    def create_visualization(self):
        """创建性能对比可视化"""
        successful_runs = {k: v for k, v in self.results.items()
                          if v.get('success', False) and v.get('tflops')}

        if not successful_runs:
            print("No successful runs to visualize")
            return

        frameworks = [self.frameworks[k]['name'] for k in successful_runs.keys()]
        tflops = [v['tflops'] for v in successful_runs.values()]
        times = [v['time_ms'] for v in successful_runs.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # TFLOPS对比
        bars1 = ax1.bar(frameworks, tflops, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax1.set_ylabel('Performance (TFLOPS)')
        ax1.set_title('GEMM Performance Comparison')
        ax1.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, val in zip(bars1, tflops):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    '.1f', ha='center', va='bottom')

        # 时间对比
        bars2 = ax2.bar(frameworks, times, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('GEMM Execution Time Comparison')
        ax2.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, val in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    '.1f', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.base_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to performance_comparison.png")

def main():
    """主函数"""
    comparator = FrameworkComparator()

    # 运行所有测试
    results = comparator.run_all()

    # 生成报告
    report = comparator.generate_report()
    print("\n" + "="*60)
    print(report)

    # 保存结果
    comparator.save_results()

    # 创建可视化
    try:
        comparator.create_visualization()
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")

    # 保存详细报告
    with open(comparator.base_dir / 'comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("Detailed report saved to comparison_report.md")

if __name__ == "__main__":
    main()
