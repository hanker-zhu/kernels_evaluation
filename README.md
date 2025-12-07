# GEMM 算子框架对比分析

对比 **cuBLAS**、**CUTLASS**、**Triton**、**TileLang** 四个 GEMM 算子框架的性能与特性。

## 项目结构

```
kernels/
├── run.sh                    # 主运行脚本
├── frameworks/               # 框架实现
│   ├── cublas/               # cuBLAS (NVIDIA 官方库)
│   ├── cutlass/              # CUTLASS (NVIDIA 模板库)
│   ├── triton/               # Triton (JIT 编译)
│   └── tilelang/             # TileLang (声明式)
├── scripts/                  # 工具脚本
│   ├── compare_frameworks.py # 对比分析脚本
│   └── validate_setup.py     # 环境验证脚本
├── results/                  # 测试结果
│   ├── logs/                 # 运行日志
│   └── *.json, *.png         # 分析结果
├── docs/                     # 文档
└── artifacts/                # 编译产物
```

## 快速开始

### 1. 环境验证

```bash
python scripts/validate_setup.py
```

### 2. 运行测试

```bash
# 运行所有框架
./run.sh

# 运行单个框架
./run.sh cublas
./run.sh cutlass
./run.sh triton
./run.sh tilelang

# 生成对比报告
./run.sh compare
```

### 3. 查看结果

- **日志**: `results/logs/`
- **报告**: `docs/comparison_report.md`
- **图表**: `results/performance_comparison.png`

## 框架对比

| 框架 | 类型 | 编程模型 | 适用场景 |
|------|------|----------|----------|
| **cuBLAS** | 厂商库 | 库调用 | 生产环境，极致性能 |
| **CUTLASS** | 模板库 | C++ 模板 | 算子定制，研究开发 |
| **Triton** | JIT | Python DSL | 快速原型，Python 开发 |
| **TileLang** | 声明式 | 函数式 | 自动优化，学术研究 |

## 环境要求

- CUDA 11.0+
- CMake 3.18+
- Python 3.8+
- PyTorch 2.0+
- Triton 2.0+

```bash
pip install torch triton tilelang matplotlib
```

## 测试配置

- **矩阵尺寸**: 128³ ~ 4096³ (含非方形)
- **数据类型**: FP16 输入, FP32 累加
- **测试次数**: 10 次取平均
