#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEMM算子框架对比分析脚本 - 区分 warmup、编译、运行时间
"""

import re
import json
from pathlib import Path
from typing import Dict, List

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


def parse_log(framework: str) -> Dict:
    """解析框架日志文件"""
    log_file = BASE_DIR / 'results' / 'logs' / f'{framework}_output.log'
    fw_name = FRAMEWORK_NAMES[framework]
    
    result = {
        'framework': fw_name,
        'warmup_time_ms': 0.0,
        'compile_time_ms': 0.0,  # AOT 或 总 JIT 编译时间
        'benchmarks': []
    }
    
    if not log_file.exists():
        return result
    
    content = log_file.read_text(encoding='utf-8')
    
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
            if remaining.startswith('['):
                # 旧格式：直接数组
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
                        result['benchmarks'] = data
                        # 累计 JIT 编译时间
                        result['compile_time_ms'] = sum(b.get('compile_time_ms', 0) for b in data)
                        return result
                    except: pass
            elif remaining.startswith('{'):
                # 新格式：包含 warmup_time_ms 的对象
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
                        result['compile_time_ms'] = data.get('total_compile_time_ms', 0)
                        for r in data.get('benchmarks', []):
                            r['framework'] = fw_name
                        result['benchmarks'] = data.get('benchmarks', [])
                        return result
                    except: pass
    
    return result


def parse_all_logs() -> Dict[str, Dict]:
    """解析所有框架日志"""
    print("解析测试日志...")
    all_results = {}
    for fw in FRAMEWORKS:
        result = parse_log(fw)
        all_results[fw] = result
        count = len([b for b in result['benchmarks'] if b.get('success', False)])
        warmup = result['warmup_time_ms']
        compile_t = result['compile_time_ms']
        print(f"  {FRAMEWORK_NAMES[fw]}: {count}点, warmup={warmup:.0f}ms, compile={compile_t:.0f}ms")
    return all_results


def generate_report(all_results: Dict) -> str:
    """生成报告"""
    lines = ["# GEMM算子框架性能对比报告\n"]
    lines.append("本报告区分 **Warmup时间**、**编译时间** 和 **运行时间**。\n")
    
    # ========== 编译性能 ==========
    lines.append("## 一、编译性能对比\n")
    lines.append("| 框架 | 类型 | Warmup(ms) | 编译时间(ms) | 说明 |")
    lines.append("|------|------|------------|--------------|------|")
    
    for fw in FRAMEWORKS:
        r = all_results.get(fw, {})
        fw_name = FRAMEWORK_NAMES[fw]
        warmup = r.get('warmup_time_ms', 0)
        compile_t = r.get('compile_time_ms', 0)
        
        if fw in ['cublas', 'cutlass']:
            lines.append(f"| {fw_name} | AOT | - | {compile_t:.0f} | cmake+make 一次性编译 |")
        else:
            lines.append(f"| {fw_name} | JIT | {warmup:.0f} | {compile_t:.0f} | 每尺寸编译（不含warmup） |")
    
    # 编译时间可视化
    lines.append("\n### 时间分布图\n")
    lines.append("```")
    lines.append("框架       | Warmup        | 编译          | 说明")
    lines.append("-" * 70)
    for fw in FRAMEWORKS:
        r = all_results.get(fw, {})
        fw_name = FRAMEWORK_NAMES[fw]
        warmup = r.get('warmup_time_ms', 0)
        compile_t = r.get('compile_time_ms', 0)
        
        w_bar = "█" * int(warmup / 50) if warmup > 0 else ""
        c_bar = "▓" * int(compile_t / 50) if compile_t > 0 else ""
        
        if fw in ['cublas', 'cutlass']:
            lines.append(f"{fw_name:10} |               | {c_bar} {compile_t:.0f}ms | AOT")
        else:
            lines.append(f"{fw_name:10} | {w_bar} {warmup:.0f}ms | {c_bar} {compile_t:.0f}ms | JIT")
    lines.append("```\n")
    
    # ========== 运行性能 ==========
    lines.append("## 二、运行性能对比\n")
    
    all_benchmarks = []
    for r in all_results.values():
        all_benchmarks.extend([b for b in r.get('benchmarks', []) if b.get('success') and b.get('M')])
    
    if all_benchmarks:
        by_size = {}
        for b in all_benchmarks:
            key = f"{b['M']}x{b['N']}x{b['K']}"
            by_size.setdefault(key, []).append(b)
        
        lines.append("### 各尺寸性能\n")
        lines.append("| 矩阵尺寸 | 框架 | 运行(ms) | TFLOPS | 单尺寸编译(ms) |")
        lines.append("|----------|------|----------|--------|----------------|")
        
        for size in sorted(by_size.keys(), key=lambda x: -int(x.split('x')[0])):
            results = sorted(by_size[size], key=lambda x: -(x.get('tflops') or 0))
            for r in results:
                rt = f"{r['latency_ms']:.3f}" if r.get('latency_ms') else "N/A"
                tp = f"{r['tflops']:.2f}" if r.get('tflops') else "N/A"
                ct = f"{r.get('compile_time_ms', 0):.1f}"
                lines.append(f"| {size} | {r['framework']} | {rt} | {tp} | {ct} |")
            lines.append("| | | | | |")
        
        # 汇总
        lines.append("\n### 性能汇总\n")
        lines.append("| 框架 | 平均TFLOPS | 最高TFLOPS | 平均延迟(ms) |")
        lines.append("|------|------------|------------|--------------|")
        
        for fw in FRAMEWORKS:
            fw_name = FRAMEWORK_NAMES[fw]
            fw_b = [b for b in all_benchmarks if b['framework'] == fw_name]
            if fw_b:
                avg_t = sum(b.get('tflops', 0) for b in fw_b) / len(fw_b)
                max_t = max(b.get('tflops', 0) for b in fw_b)
                avg_l = sum(b.get('latency_ms', 0) for b in fw_b) / len(fw_b)
                lines.append(f"| {fw_name} | {avg_t:.2f} | {max_t:.2f} | {avg_l:.3f} |")
    
    # ========== 综合分析 ==========
    lines.append("\n## 三、综合分析\n")
    lines.append("### AOT vs JIT 对比\n")
    lines.append("| 特性 | AOT (cuBLAS/CUTLASS) | JIT (Triton/TileLang) |")
    lines.append("|------|----------------------|----------------------|")
    lines.append("| Warmup | 无 | 首次启动 300-1800ms |")
    lines.append("| 编译 | 一次性 (8-17s) | 每尺寸 20-30ms |")
    lines.append("| 运行 | 极致性能 | 接近AOT |")
    lines.append("| 适用 | 生产环境 | 研究开发 |")
    
    lines.append("\n### 关键洞察\n")
    lines.append("1. **Warmup开销**：JIT框架首次启动需要加载编译器，约300-1800ms")
    lines.append("2. **编译开销**：AOT一次编译长期使用；JIT每个新尺寸需编译20-30ms")
    lines.append("3. **运行性能**：大矩阵时 JIT 接近 AOT；小矩阵 AOT 优势明显")
    lines.append("4. **总体建议**：生产用AOT，原型开发用JIT")
    
    return "\n".join(lines)


def create_visualization(all_results: Dict):
    """生成图表"""
    if not HAS_MATPLOTLIB:
        print("matplotlib 不可用，跳过可视化")
        return
    
    all_benchmarks = []
    for r in all_results.values():
        all_benchmarks.extend([b for b in r.get('benchmarks', []) if b.get('success') and b.get('M')])
    
    if not all_benchmarks:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sizes = [256, 512, 1024, 2048, 4096]
    colors = {'cuBLAS': '#2ecc71', 'CUTLASS': '#3498db', 'Triton': '#e74c3c', 'TileLang': '#f39c12'}
    
    by_fw = {}
    for b in all_benchmarks:
        by_fw.setdefault(b['framework'], []).append(b)
    
    # 图1: Warmup + 编译时间
    ax1 = axes[0, 0]
    fw_names = []
    warmup_times = []
    compile_times = []
    for fw in FRAMEWORKS:
        r = all_results.get(fw, {})
        fw_names.append(FRAMEWORK_NAMES[fw])
        warmup_times.append(r.get('warmup_time_ms', 0))
        compile_times.append(r.get('compile_time_ms', 0))
    
    x = range(len(fw_names))
    ax1.bar([i - 0.2 for i in x], warmup_times, 0.35, label='Warmup', color='orange', alpha=0.8)
    ax1.bar([i + 0.2 for i in x], compile_times, 0.35, label='Compile', color='blue', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(fw_names)
    ax1.set_title('Warmup & Compile Time', fontsize=12)
    ax1.set_ylabel('Time (ms)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: TFLOPS
    ax2 = axes[0, 1]
    for fw_name, fw_b in by_fw.items():
        data = []
        for size in sizes:
            key = f"{size}x{size}x{size}"
            match = [b for b in fw_b if f"{b['M']}x{b['N']}x{b['K']}" == key]
            data.append(match[0].get('tflops', 0) if match else 0)
        ax2.plot(range(len(sizes)), data, 'o-', label=fw_name, color=colors.get(fw_name, 'gray'), lw=2)
    ax2.set_title('Runtime Performance (TFLOPS)', fontsize=12)
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([f'{s}³' for s in sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 延迟
    ax3 = axes[1, 0]
    for fw_name, fw_b in by_fw.items():
        data = []
        for size in sizes:
            key = f"{size}x{size}x{size}"
            match = [b for b in fw_b if f"{b['M']}x{b['N']}x{b['K']}" == key]
            data.append(match[0].get('latency_ms', 0) if match else 0)
        ax3.plot(range(len(sizes)), data, 'o-', label=fw_name, color=colors.get(fw_name, 'gray'), lw=2)
    ax3.set_title('Runtime Latency (ms)', fontsize=12)
    ax3.set_xticks(range(len(sizes)))
    ax3.set_xticklabels([f'{s}³' for s in sizes])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: JIT 每尺寸编译时间
    ax4 = axes[1, 1]
    bar_width = 0.35
    for i, (fw_name, fw_b) in enumerate(by_fw.items()):
        if fw_name not in ['Triton', 'TileLang']:
            continue
        data = []
        for size in sizes:
            key = f"{size}x{size}x{size}"
            match = [b for b in fw_b if f"{b['M']}x{b['N']}x{b['K']}" == key]
            data.append(match[0].get('compile_time_ms', 0) if match else 0)
        offset = -0.2 if fw_name == 'Triton' else 0.2
        ax4.bar([j + offset for j in range(len(sizes))], data, bar_width, 
               label=fw_name, color=colors.get(fw_name, 'gray'), alpha=0.8)
    ax4.set_title('JIT Compile Time per Size (ms)', fontsize=12)
    ax4.set_xticks(range(len(sizes)))
    ax4.set_xticklabels([f'{s}³' for s in sizes])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = BASE_DIR / 'results' / 'performance_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {out_path}")


def main():
    print("=" * 60)
    print("GEMM 算子框架对比分析（Warmup + 编译 + 运行）")
    print("=" * 60)
    
    all_results = parse_all_logs()
    
    report = generate_report(all_results)
    report_dir = BASE_DIR / 'docs'
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / 'comparison_report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f"\n报告已保存: {report_path}")
    
    json_path = BASE_DIR / 'results' / 'benchmark_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {json_path}")
    
    create_visualization(all_results)
    print("\n完成!")


if __name__ == "__main__":
    main()
