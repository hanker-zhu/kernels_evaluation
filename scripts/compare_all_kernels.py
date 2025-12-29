#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多核函数框架对比分析脚本
支持 GEMM, Softmax, LayerNorm 等核函数
"""

import re
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BASE_DIR = Path(__file__).parent.parent
FRAMEWORKS = ['cublas', 'cutlass', 'triton', 'tilelang']
FRAMEWORK_NAMES = {'cublas': 'cuBLAS', 'cutlass': 'CUTLASS', 'triton': 'Triton', 'tilelang': 'TileLang'}
KERNELS = ['gemm', 'softmax', 'layernorm']


def parse_log(framework: str, kernel: str) -> Dict:
    """解析框架日志文件"""
    log_file = BASE_DIR / 'results' / 'logs' / f'{framework}_{kernel}_output.log'
    if kernel == 'gemm':
        # GEMM 使用旧的日志格式
        log_file = BASE_DIR / 'results' / 'logs' / f'{framework}_output.log'
    
    fw_name = FRAMEWORK_NAMES[framework]
    
    result = {
        'framework': fw_name,
        'kernel': kernel,
        'warmup_time_ms': 0.0,
        'compile_time_ms': 0.0,
        'benchmarks': []
    }
    
    if not log_file.exists():
        return result
    
    content = log_file.read_text(encoding='utf-8', errors='ignore')
    
    # 解析 AOT 编译时间 (C++ 框架)
    compile_match = re.search(r'COMPILE_TIME_MS:\s*(\d+)', content)
    if compile_match:
        result['compile_time_ms'] = float(compile_match.group(1))
    
    # 解析 warmup 时间 (Python JIT 框架)
    warmup_match = re.search(r'Framework init time:\s*([0-9.]+)\s*ms', content)
    if warmup_match:
        result['warmup_time_ms'] = float(warmup_match.group(1))
    
    # 解析 JSON 格式
    json_markers = ['=== BENCHMARK RESULTS JSON ===', '=== JSON Results ===']
    for marker in json_markers:
        if marker in content:
            start_idx = content.find(marker) + len(marker)
            remaining = content[start_idx:].strip()
            
            # 尝试解析为对象
            if remaining.startswith('{'):
                bracket_count = 0
                end_idx = 0
                for i, c in enumerate(remaining):
                    if c == '{': bracket_count += 1
                    elif c == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                if end_idx > 0:
                    try:
                        data = json.loads(remaining[:end_idx])
                        result['warmup_time_ms'] = data.get('warmup_time_ms', 0)
                        result['compile_time_ms'] = data.get('total_compile_time_ms', data.get('compile_time_ms', 0))
                        for r in data.get('benchmarks', []):
                            r['framework'] = fw_name
                            r['kernel'] = kernel
                        result['benchmarks'] = data.get('benchmarks', [])
                        return result
                    except: pass
            
            # 尝试解析为数组 (旧格式)
            elif remaining.startswith('['):
                bracket_count = 0
                end_idx = 0
                for i, c in enumerate(remaining):
                    if c == '[': bracket_count += 1
                    elif c == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                if end_idx > 0:
                    try:
                        data = json.loads(remaining[:end_idx])
                        for r in data:
                            r['framework'] = fw_name
                            r['kernel'] = kernel
                        result['benchmarks'] = data
                        result['compile_time_ms'] = sum(b.get('compile_time_ms', 0) for b in data)
                        return result
                    except: pass
    
    return result


def parse_all_logs() -> Dict[str, Dict[str, Dict]]:
    """解析所有框架和核函数的日志"""
    print("解析测试日志...")
    all_results = defaultdict(lambda: defaultdict(dict))
    
    for kernel in KERNELS:
        for fw in FRAMEWORKS:
            result = parse_log(fw, kernel)
            all_results[kernel][fw] = result
            
            count = len([b for b in result['benchmarks'] if b.get('success', False)])
            warmup = result['warmup_time_ms']
            compile_t = result['compile_time_ms']
            if count > 0 or warmup > 0 or compile_t > 0:
                print(f"  {FRAMEWORK_NAMES[fw]} ({kernel}): {count}点, warmup={warmup:.0f}ms, compile={compile_t:.0f}ms")
    
    return all_results


def generate_report(all_results: Dict) -> str:
    """生成报告"""
    lines = ["# 多核函数框架性能对比报告\n"]
    lines.append("本报告对比 **GEMM**、**Softmax**、**LayerNorm** 等核函数在不同框架下的性能。\n")
    
    # 为每个核函数生成报告
    for kernel in KERNELS:
        kernel_results = all_results.get(kernel, {})
        if not kernel_results:
            continue
        
        lines.append(f"\n## {kernel.upper()} 核函数对比\n")
        
        # 编译性能
        lines.append("### 编译性能\n")
        lines.append("| 框架 | 类型 | Warmup(ms) | 编译时间(ms) |")
        lines.append("|------|------|------------|--------------|")
        
        for fw in FRAMEWORKS:
            r = kernel_results.get(fw, {})
            fw_name = FRAMEWORK_NAMES[fw]
            warmup = r.get('warmup_time_ms', 0)
            compile_t = r.get('compile_time_ms', 0)
            
            if fw in ['cublas', 'cutlass']:
                lines.append(f"| {fw_name} | AOT | - | {compile_t:.0f} |")
            else:
                lines.append(f"| {fw_name} | JIT | {warmup:.0f} | {compile_t:.0f} |")
        
        # 运行性能
        lines.append("\n### 运行性能\n")
        all_benchmarks = []
        for r in kernel_results.values():
            all_benchmarks.extend([b for b in r.get('benchmarks', []) if b.get('success')])
        
        if all_benchmarks:
            # 按尺寸分组
            by_size = defaultdict(list)
            for b in all_benchmarks:
                size_key = b.get('shape') or f"{b.get('M', '?')}x{b.get('N', '?')}x{b.get('K', '?')}"
                by_size[size_key].append(b)
            
            lines.append("| 尺寸 | 框架 | 延迟(ms) | TFLOPS |")
            lines.append("|------|------|----------|--------|")
            
            for size in sorted(by_size.keys()):
                results = sorted(by_size[size], key=lambda x: x.get('latency_ms', float('inf')))
                for r in results:
                    rt = f"{r['latency_ms']:.3f}" if r.get('latency_ms') else "N/A"
                    tp = f"{r['tflops']:.2f}" if r.get('tflops') else "N/A"
                    lines.append(f"| {size} | {r['framework']} | {rt} | {tp} |")
                lines.append("| | | | |")
    
    # 汇总对比
    lines.append("\n## 综合对比\n")
    lines.append("| 核函数 | 框架 | 平均延迟(ms) | 最高性能 |")
    lines.append("|--------|------|--------------|----------|")
    
    for kernel in KERNELS:
        kernel_results = all_results.get(kernel, {})
        for fw in FRAMEWORKS:
            r = kernel_results.get(fw, {})
            fw_name = FRAMEWORK_NAMES[fw]
            benchmarks = [b for b in r.get('benchmarks', []) if b.get('success')]
            if benchmarks:
                avg_latency = sum(b.get('latency_ms', 0) for b in benchmarks) / len(benchmarks)
                max_tflops = max((b.get('tflops', 0) for b in benchmarks), default=0)
                lines.append(f"| {kernel} | {fw_name} | {avg_latency:.3f} | {max_tflops:.2f} |")
    
    return "\n".join(lines)


def create_visualization(all_results: Dict):
    """为每个核函数生成单独的对比图表"""
    if not HAS_MATPLOTLIB:
        print("matplotlib 不可用，跳过可视化")
        return
    
    colors = {'cuBLAS': '#2ecc71', 'CUTLASS': '#3498db', 'Triton': '#e74c3c', 'TileLang': '#f39c12'}
    
    for kernel in KERNELS:
        kernel_results = all_results.get(kernel, {})
        if not kernel_results:
            continue
        
        all_benchmarks = []
        for r in kernel_results.values():
            all_benchmarks.extend([b for b in r.get('benchmarks', []) if b.get('success')])
        
        if not all_benchmarks:
            continue
        
        # 为每个核函数创建单独的图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: 延迟对比
        ax1 = axes[0]
        by_fw = defaultdict(list)
        for b in all_benchmarks:
            by_fw[b['framework']].append(b)
        
        for fw_name, fw_b in by_fw.items():
            # 获取尺寸并排序
            sizes = []
            for b in fw_b:
                size_key = b.get('shape') or f"{b.get('M', '?')}x{b.get('N', '?')}x{b.get('K', '?')}"
                if size_key not in sizes:
                    sizes.append(size_key)
            
            # 按尺寸大小排序
            def sort_key(s):
                # 提取第一个数字进行排序
                import re
                match = re.search(r'(\d+)', s)
                return int(match.group(1)) if match else 0
            sizes = sorted(sizes, key=sort_key)
            
            latencies = []
            for size in sizes:
                match = [b for b in fw_b if (b.get('shape') == size or 
                                            f"{b.get('M', '')}x{b.get('N', '')}x{b.get('K', '')}" == size or
                                            f"{b.get('M', '')}x{b.get('N', '')}" == size)]
                if match:
                    latencies.append(match[0].get('latency_ms', 0))
                else:
                    latencies.append(0)
            
            if latencies:
                ax1.plot(range(len(sizes)), latencies, 'o-', label=fw_name, 
                        color=colors.get(fw_name, 'gray'), lw=2, markersize=6)
        
        ax1.set_title(f'{kernel.upper()} - Latency Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(sizes)))
        ax1.set_xticklabels(sizes, rotation=45, ha='right')
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_xlabel('Matrix Size', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 右图: 性能对比 (如果有TFLOPS) 或编译时间
        ax2 = axes[1]
        has_tflops = any(b.get('tflops') for b in all_benchmarks)
        
        if has_tflops:
            # 显示TFLOPS
            for fw_name, fw_b in by_fw.items():
                sizes = []
                for b in fw_b:
                    size_key = b.get('shape') or f"{b.get('M', '?')}x{b.get('N', '?')}x{b.get('K', '?')}"
                    if size_key not in sizes:
                        sizes.append(size_key)
                
                def sort_key(s):
                    import re
                    match = re.search(r'(\d+)', s)
                    return int(match.group(1)) if match else 0
                sizes = sorted(sizes, key=sort_key)
                
                tflops = []
                for size in sizes:
                    match = [b for b in fw_b if (b.get('shape') == size or 
                                                f"{b.get('M', '')}x{b.get('N', '')}x{b.get('K', '')}" == size)]
                    if match and match[0].get('tflops'):
                        tflops.append(match[0].get('tflops', 0))
                    else:
                        tflops.append(0)
                
                if any(tflops):
                    ax2.plot(range(len(sizes)), tflops, 'o-', label=fw_name,
                            color=colors.get(fw_name, 'gray'), lw=2, markersize=6)
            
            ax2.set_title(f'{kernel.upper()} - Performance (TFLOPS)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('TFLOPS', fontsize=12)
        else:
            # 显示编译时间
            fw_names = []
            compile_times = []
            warmup_times = []
            
            for fw in FRAMEWORKS:
                r = kernel_results.get(fw, {})
                if r.get('compile_time_ms', 0) > 0 or r.get('warmup_time_ms', 0) > 0:
                    fw_names.append(FRAMEWORK_NAMES[fw])
                    compile_times.append(r.get('compile_time_ms', 0))
                    warmup_times.append(r.get('warmup_time_ms', 0))
            
            if fw_names:
                x = range(len(fw_names))
                width = 0.35
                ax2.bar([i - width/2 for i in x], warmup_times, width, label='Warmup', color='orange', alpha=0.8)
                ax2.bar([i + width/2 for i in x], compile_times, width, label='Compile', color='blue', alpha=0.8)
                ax2.set_xticks(x)
                ax2.set_xticklabels(fw_names)
                ax2.set_ylabel('Time (ms)', fontsize=12)
                ax2.legend(fontsize=10)
                ax2.set_title(f'{kernel.upper()} - Compile & Warmup Time', fontsize=14, fontweight='bold')
        
        if has_tflops:
            ax2.set_xticks(range(len(sizes)))
            ax2.set_xticklabels(sizes, rotation=45, ha='right')
            ax2.set_xlabel('Matrix Size', fontsize=12)
        else:
            ax2.set_xlabel('Framework', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = BASE_DIR / 'results' / f'{kernel}_comparison.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  {kernel.upper()} 图表已保存: {out_path}")
        plt.close()  # 关闭当前图表，释放内存


def main():
    print("=" * 60)
    print("多核函数框架对比分析")
    print("=" * 60)
    
    all_results = parse_all_logs()
    
    report = generate_report(all_results)
    report_dir = BASE_DIR / 'docs'
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / 'all_kernels_comparison_report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f"\n报告已保存: {report_path}")
    
    json_path = BASE_DIR / 'results' / 'all_kernels_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {json_path}")
    
    create_visualization(all_results)
    print("\n完成!")


if __name__ == "__main__":
    main()

