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
try:
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd
    HAS_MATPLOTLIB = True
    
    matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'
    matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    
except ImportError:
    HAS_MATPLOTLIB = False

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

            print(f"✓ {fw['name']}: {result['time_ms']:.2f} ms, {result['tflops']:.2f} TFLOPS")
        else:
            print(f"✗ {fw['name']}: Run failed")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)

        return result

    def parse_log_file(self, framework: str) -> List[Dict]:
        """从日志文件解析测试结果 - 支持多尺寸测试"""
        fw = self.frameworks[framework]
        log_file = self.base_dir / 'results' / f'{framework}_output.log'

        if not log_file.exists():
            return [{
                'framework': fw['name'],
                'success': False,
                'error': f'Log file not found: {log_file}',
                'M': None, 'N': None, 'K': None
            }]

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return [{
                'framework': fw['name'],
                'success': False,
                'error': f'Failed to read log file: {e}',
                'M': None, 'N': None, 'K': None
            }]

        # 检查是否是C++实现的JSON输出格式
        json_match = re.search(r'=== BENCHMARK RESULTS JSON ===(.*?)\]', content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1) + ']'
                results = json.loads(json_str)
                for result in results:
                    result['framework'] = fw['name']
                return results
            except json.JSONDecodeError:
                pass

        # 检查是否是Python实现的列表格式
        if 'return results' in content or 'run_' in content:
            # 这是Python脚本，直接返回空列表，让脚本自己运行
            return []

        # 尝试解析多尺寸输出
        multi_size_results = self.parse_python_script_output(content, framework)
        if multi_size_results and len(multi_size_results) > 1:
            # 如果解析到了多个结果，返回多尺寸结果
            return multi_size_results

        # 回退到旧的单尺寸解析格式
        result = {
            'framework': fw['name'],
            'success': True,
            'stdout': content,
            'stderr': '',
            'returncode': 0,
            'time_ms': None,
            'tflops': None,
            'checksum': None,
            'analysis': {},
            'M': 4096, 'N': 4096, 'K': 4096  # 默认尺寸
        }

        # 解析性能数据 - 根据不同框架的输出格式
        framework_patterns = {
            'cublas': r'cuBLAS GEMM: ([0-9.]+) ms.*?, ([0-9.]+) TFLOPS',
            'cutlass': r'CUTLASS GEMM: ([0-9.]+) ms.*?, ([0-9.]+) TFLOPS',
            'tilelang': r'TileLang GEMM: ([0-9.]+) ms, ([0-9.]+) TFLOPS',
            'triton': r'Triton GEMM: ([0-9.]+) ms.*?, ([0-9.]+) TFLOPS'
        }

        pattern = framework_patterns.get(framework)
        if pattern:
            time_match = re.search(pattern, content)
            if time_match:
                result['time_ms'] = float(time_match.group(1))
                result['tflops'] = float(time_match.group(2))

        # 解析校验和
        checksum_match = re.search(r'Result checksum: ([0-9.e+-]+)', content)
        if checksum_match:
            result['checksum'] = float(checksum_match.group(1))

        # 解析分析信息
        analysis_section = re.search(r'=== .* Analysis ===(.*?)(?=\n\n|\Z)', content, re.DOTALL)
        if analysis_section:
            analysis_lines = analysis_section.group(1).strip().split('\n')
            for line in analysis_lines:
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    key = key.strip('- ')
                    result['analysis'][key] = value.strip()

        return [result]

    def parse_all_logs(self) -> Dict:
        """解析所有日志文件 - 支持多尺寸测试"""
        print("=" * 60)
        print("解析GEMM算子生成框架多尺寸对比测试日志")
        print("=" * 60)

        all_results = []
        framework_results = {}

        for fw_name in self.frameworks.keys():
            results = self.parse_log_file(fw_name)

            if not results:
                # Python脚本需要直接运行
                try:
                    if fw_name in ['tilelang', 'triton']:
                        # 运行Python脚本
                        fw = self.frameworks[fw_name]
                        stdout, stderr, ret = self.run_command(fw['run_cmd'], cwd=str(fw['dir']))

                        if ret == 0:
                            # 尝试解析Python脚本返回的结果
                            results = self.parse_python_script_output(stdout, fw_name)
                            if not results:
                                # 如果解析失败，提供基本信息
                                results = [{
                                    'framework': fw['name'],
                                    'success': True,
                                    'error': None,
                                    'M': None, 'N': None, 'K': None
                                }]
                        else:
                            results = [{
                                'framework': fw['name'],
                                'success': False,
                                'error': stderr if stderr else 'Script execution failed',
                                'M': None, 'N': None, 'K': None
                            }]
                except Exception as e:
                    results = [{
                        'framework': fw['name'],
                        'success': False,
                        'error': str(e),
                        'M': None, 'N': None, 'K': None
                    }]

            framework_results[fw_name] = results
            all_results.extend(results)

            # 统计信息
            successful_count = sum(1 for r in results if r.get('success', False))
            total_count = len(results)
            print(f"✓ {fw_name}: 解析完成 {successful_count}/{total_count} 个测试点")

        self.results = framework_results
        self.all_results = all_results
        return framework_results

    def parse_python_script_output(self, stdout: str, framework: str) -> List[Dict]:
        """解析Python脚本的输出"""
        results = []
        fw_name = self.frameworks[framework]['name']

        # 解析逻辑 - 查找性能输出行
        lines = stdout.split('\n')

        for line in lines:
            # 查找性能结果行，支持不同的格式
            # TileLang: "TileLang GEMM (128x128x128): 0.02 ms, 254.51 TFLOPS"
            # Triton: "Triton GEMM (128x128x128): 0.07 ms (avg of 10 runs), 58.96 TFLOPS"
            pattern = rf'{fw_name} GEMM \((\d+)x(\d+)x(\d+)\): ([0-9.]+) ms.*?, ([0-9.]+) TFLOPS'
            match = re.search(pattern, line)
            if match:
                M, N, K = int(match.group(1)), int(match.group(2)), int(match.group(3))
                latency = float(match.group(4))
                tflops = float(match.group(5))

                result = {
                    'framework': fw_name,
                    'success': True,
                    'M': M, 'N': N, 'K': K,
                    'latency_ms': latency,
                    'tflops': tflops,
                    'checksum': None
                }
                results.append(result)
            else:
                # 检查是否是失败的行
                fail_pattern = rf'{fw_name} GEMM \((\d+)x(\d+)x(\d+)\) failed:'
                fail_match = re.search(fail_pattern, line)
                if fail_match:
                    M, N, K = int(fail_match.group(1)), int(fail_match.group(2)), int(fail_match.group(3))

                    result = {
                        'framework': fw_name,
                        'success': False,
                        'M': M, 'N': N, 'K': K,
                        'latency_ms': None,
                        'tflops': None,
                        'checksum': None,
                        'error': 'Test failed - precision issues'
                    }
                    results.append(result)

        return results if results else [{
            'framework': fw_name,
            'success': False,
            'error': 'Failed to parse Python output',
            'M': None, 'N': None, 'K': None
        }]

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
        """生成多尺寸对比报告"""
        if not hasattr(self, 'all_results') or not self.all_results:
            return "No results available. Please run tests first."

        report = []
        report.append("# GEMM算子生成框架多尺寸对比分析报告")
        report.append("")
        report.append("## 测试配置")
        report.append("- 矩阵尺寸范围: 128x128x128 到 4096x4096x4096 (包括非方形矩阵)")
        report.append("- 数据类型: FP16输入, FP32累加")
        report.append("- 测试运行次数: 10次取平均")
        report.append("- 测试框架: cuBLAS, CUTLASS, TileLang, Triton")
        report.append("")

        # 汇总所有结果
        all_successful_results = [r for r in self.all_results if r.get('success', False) and r.get('M')]

        if all_successful_results:
            # 按尺寸分组统计
            size_groups = {}
            for result in all_successful_results:
                size_key = f"{result['M']}x{result['N']}x{result['K']}"
                if size_key not in size_groups:
                    size_groups[size_key] = []
                size_groups[size_key].append(result)

            report.append("## 多尺寸性能对比表")
            report.append("")
            report.append("| 矩阵尺寸 | 框架 | 时间(ms) | 性能(TFLOPS) | 校验和 |")
            report.append("|----------|------|----------|-------------|--------|")

            # 按尺寸排序
            sorted_sizes = sorted(size_groups.keys(),
                                key=lambda x: [int(d) for d in x.split('x')],
                                reverse=True)

            for size_key in sorted_sizes:
                results = size_groups[size_key]
                # 按性能排序，处理None值
                results.sort(key=lambda x: x.get('tflops', 0) or 0, reverse=True)

                for result in results:
                    time_str = f"{result['latency_ms']:.2f}" if result.get('latency_ms') else "N/A"
                    tflops_str = f"{result['tflops']:.2f}" if result.get('tflops') else "N/A"
                    checksum_str = f"{result['checksum']:.2e}" if result.get('checksum') else "N/A"
                    report.append(f"| {size_key} | {result['framework']} | {time_str} | {tflops_str} | {checksum_str} |")

                report.append("| | | | | |")  # 空行分隔不同尺寸

            # 性能趋势分析
            report.append("")
            report.append("## 性能趋势分析")
            report.append("")

            # 方形矩阵性能趋势
            square_sizes = [128, 256, 512, 1024, 2048, 4096]
            report.append("### 方形矩阵性能 (MxMxM)")
            report.append("")
            report.append("| 尺寸 | cuBLAS | CUTLASS | TileLang | Triton |")
            report.append("|------|--------|---------|----------|--------|")

            for size in square_sizes:
                size_key = f"{size}x{size}x{size}"
                if size_key in size_groups:
                    results = {r['framework']: r for r in size_groups[size_key]}
                    cublas_tflops = results.get('cuBLAS', {}).get('tflops', None)
                    cutlass_tflops = results.get('CUTLASS', {}).get('tflops', None)
                    tilelang_tflops = results.get('TileLang', {}).get('tflops', None)
                    triton_tflops = results.get('Triton', {}).get('tflops', None)

                    def format_tflops(val):
                        return f"{val:.1f}" if val else "N/A"

                    report.append(f"| {size}x{size}x{size} | {format_tflops(cublas_tflops)} | {format_tflops(cutlass_tflops)} | {format_tflops(tilelang_tflops)} | {format_tflops(triton_tflops)} |")
                else:
                    report.append(f"| {size}x{size}x{size} | N/A | N/A | N/A | N/A |")

            # 非方形矩阵示例
            report.append("")
            report.append("### 非方形矩阵性能示例")
            report.append("")
            rect_examples = ["4096x2048x4096", "2048x4096x4096", "4096x4096x2048"]
            for size_key in rect_examples:
                if size_key in size_groups:
                    results = size_groups[size_key]
                    results.sort(key=lambda x: x.get('tflops', 0) or 0, reverse=True)
                    report.append(f"**{size_key}**:")
                    for result in results[:3]:  # 取前3名
                        if result.get('tflops'):
                            report.append(f"- {result['framework']}: {result['tflops']:.2f} TFLOPS")

        # 框架特性对比（保持原有内容）
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

        report.append("## 总结与建议")
        report.append("")
        report.append("### 多尺寸测试洞察")
        report.append("- **性能一致性**: 观察各框架在不同尺寸下的性能表现稳定性")
        report.append("- **扩展性**: 大尺寸矩阵上的性能扩展趋势")
        report.append("- **非方形矩阵**: 不同宽高比矩阵的性能特点")
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
        """创建多尺寸性能对比可视化"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available, skipping visualization")
            return

        if not hasattr(self, 'all_results') or not self.all_results:
            print("No results available for visualization")
            return

        # 收集所有成功的测试结果
        all_successful_results = [r for r in self.all_results if r.get('success', False) and r.get('M')]

        if not all_successful_results:
            print("No successful runs to visualize")
            return

        # 按框架分组
        framework_data = {}
        for result in all_successful_results:
            fw = result['framework']
            if fw not in framework_data:
                framework_data[fw] = []
            framework_data[fw].append(result)

        # 方形矩阵尺寸
        square_sizes = [128, 256, 512, 1024, 2048, 4096]

        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 颜色映射
        colors = {'cuBLAS': 'blue', 'CUTLASS': 'green', 'TileLang': 'orange', 'Triton': 'red'}

        # 1. 方形矩阵性能趋势
        ax1.set_title('方形矩阵性能趋势 (TFLOPS)', fontsize=12)
        ax1.set_xlabel('矩阵尺寸')
        ax1.set_ylabel('性能 (TFLOPS)')
        ax1.set_xticks(range(len(square_sizes)))
        ax1.set_xticklabels([f'{s}³' for s in square_sizes])

        # 2. 方形矩阵时间趋势
        ax2.set_title('方形矩阵执行时间趋势 (ms)', fontsize=12)
        ax2.set_xlabel('矩阵尺寸')
        ax2.set_ylabel('时间 (ms)')
        ax2.set_xticks(range(len(square_sizes)))
        ax2.set_xticklabels([f'{s}³' for s in square_sizes])

        # 3. 非方形矩阵性能对比
        ax3.set_title('非方形矩阵性能对比1 (TFLOPS)', fontsize=12)
        ax3.set_xlabel('矩阵配置')
        ax3.set_ylabel('性能 (TFLOPS)')

        # 4. 各框架性能分布折线图
        ax4.set_title('非方形矩阵性能对比2 (TFLOPS)', fontsize=12)
        ax4.set_xlabel('矩阵配置')
        ax4.set_ylabel('性能 (TFLOPS)')

        # 绘制方形矩阵趋势线
        for fw_name, results in framework_data.items():
            # 方形矩阵数据
            square_tflops = []
            square_times = []

            for size in square_sizes:
                size_key = f"{size}x{size}x{size}"
                matching_results = [r for r in results if f"{r['M']}x{r['N']}x{r['K']}" == size_key]

                if matching_results:
                    # 取平均值（如果有多个结果）
                    avg_tflops = sum(r.get('tflops', 0) for r in matching_results) / len(matching_results)
                    avg_time = sum(r.get('latency_ms', 0) for r in matching_results) / len(matching_results)
                    square_tflops.append(avg_tflops)
                    square_times.append(avg_time)
                else:
                    square_tflops.append(0)
                    square_times.append(0)

            # 绘制趋势线
            color = colors.get(fw_name, 'gray')
            ax1.plot(range(len(square_sizes)), square_tflops, 'o-', label=fw_name, color=color, linewidth=2, markersize=6)
            ax2.plot(range(len(square_sizes)), square_times, 'o-', label=fw_name, color=color, linewidth=2, markersize=6)

        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 非方形矩阵对比
        rect_configs = ["1024x512x1024", "512x1024x1024", "1024x1024x512"]
        x_positions = range(len(rect_configs))

        for i, config in enumerate(rect_configs):
            config_results = []
            labels = []

            for fw_name, results in framework_data.items():
                matching_results = [r for r in results if f"{r['M']}x{r['N']}x{r['K']}" == config]
                if matching_results:
                    avg_tflops = sum(r.get('tflops', 0) for r in matching_results) / len(matching_results)
                    config_results.append(avg_tflops)
                    labels.append(fw_name)

            if config_results:
                bars = ax3.bar([i + j*0.2 for j in range(len(config_results))], config_results,
                              width=0.15, label=labels if i == 0 else "",
                              color=[colors.get(l, 'gray') for l in labels])
                # 添加数值标签
                for bar, val in zip(bars, config_results):
                    if val > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax3.set_xticks([i + 0.2 for i in x_positions])
        ax3.set_xticklabels([c.replace('1024', '1K') for c in rect_configs])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 非方形矩阵对比
        rect_configs = ["4096x2048x4096", "2048x4096x4096", "4096x4096x2048"]
        x_positions = range(len(rect_configs))

        for i, config in enumerate(rect_configs):
            config_results = []
            labels = []

            for fw_name, results in framework_data.items():
                matching_results = [r for r in results if f"{r['M']}x{r['N']}x{r['K']}" == config]
                if matching_results:
                    avg_tflops = sum(r.get('tflops', 0) for r in matching_results) / len(matching_results)
                    config_results.append(avg_tflops)
                    labels.append(fw_name)

            if config_results:
                bars = ax4.bar([i + j*0.2 for j in range(len(config_results))], config_results,
                              width=0.15, label=labels if i == 0 else "",
                              color=[colors.get(l, 'gray') for l in labels])
                # 添加数值标签
                for bar, val in zip(bars, config_results):
                    if val > 0:
                        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax4.set_xticks([i + 0.2 for i in x_positions])
        ax4.set_xticklabels([c.replace('2048', '2K').replace('4096', '4K') for c in rect_configs])
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.base_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        print("Multi-size visualization saved to performance_comparison.png")

def main():
    """主函数"""
    import sys

    comparator = FrameworkComparator()

    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--parse-logs':
        # 解析现有日志文件
        results = comparator.parse_all_logs()
    else:
        # 运行所有测试
        results = comparator.run_all()

    # 生成报告
    report = comparator.generate_report()
    print("\n" + "="*60)
    print(report)

    # 保存结果
    comparator.save_results()

    # 创建可视化
    comparator.create_visualization()

    # 保存详细报告
    with open(comparator.base_dir / 'comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("Detailed report saved to comparison_report.md")

if __name__ == "__main__":
    main()
