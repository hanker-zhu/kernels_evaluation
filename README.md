# GEMM算子生成框架对比分析

这个项目对四个主流的GEMM算子生成框架进行了深入的技术对比：**cuBLAS**、**CUTLASS**、**TileLang** 和 **Triton**。

## 项目结构

```
kernels/
├── frameworks/              # 框架实现
│   ├── cublas/              # NVIDIA cuBLAS实现
│   │   ├── main_cublas_gemm.cu
│   │   ├── CMakeLists.txt
│   │   └── run.sh
│   ├── cutlass/             # NVIDIA CUTLASS实现
│   │   ├── main_cutlass_gemm.cu
│   │   ├── CMakeLists.txt
│   │   └── run.sh
│   ├── tilelang/            # TileLang实现
│   │   ├── tilelang_gemm.py
│   │   └── run.sh
│   └── triton/              # Triton实现
│       ├── triton_gemm.py
│       └── run.sh
├── scripts/                 # 脚本文件
│   ├── compare_frameworks.py    # 统一对比分析脚本
│   ├── run_all_benchmarks.sh    # 主运行脚本
│   └── validate_setup.py        # 环境验证脚本
├── docs/                    # 文档
│   ├── technical_analysis.md                    # 详细技术分析报告
│   ├── PROJECT_SUMMARY.md                       # 项目总结
│   ├── comparison_report.md                     # 对比分析报告
│   ├── detailed_performance_analysis.md         # 详细性能分析
│   ├── detailed_assembly_performance_analysis.md # 汇编性能分析
│   ├── compilation_performance_analysis.md      # 编译性能分析
│   ├── compile_code_performance_analysis.md     # 编译代码性能分析
│   └── main.tex                                  # LaTeX文档
├── results/                  # 测试结果
│   ├── benchmark_results.json       # 基准测试结果
│   ├── compile_analysis_report.json # 编译分析报告
│   ├── performance_comparison.png   # 性能对比图
│   └── logs/                          # 日志文件
│       ├── cublas_output.log
│       ├── cutlass_output.log
│       ├── tilelang_output.log
│       └── triton_output.log
├── artifacts/               # 编译产物
│   ├── binaries/            # 可执行文件
│   └── assembly/            # 汇编代码
│       ├── cublas_gemm.s
│       ├── cutlass_gemm.sass
│       ├── tilelang_gemm_*.tl
│       └── triton_gemm_*.ptx
└── README.md
```

## 框架简介

### 1. cuBLAS (NVIDIA官方库)
- **特点**: 高度优化的生产级BLAS库
- **优势**: 极致性能，生产就绪
- **适用**: 生产环境，性能敏感应用

### 2. CUTLASS (NVIDIA模板库)
- **特点**: 可配置的C++模板库
- **优势**: 高度可定制，接近cuBLAS性能
- **适用**: 算子开发，硬件架构研究

### 3. TileLang (声明式框架)
- **特点**: 基于TensorIR的函数式编程
- **优势**: 开发效率高，自动优化
- **适用**: 快速原型，学术研究

### 4. Triton (JIT编译框架)
- **特点**: Python嵌入式CUDA编程
- **优势**: 编程友好，降低开发门槛
- **适用**: 研究开发，快速迭代

## 快速开始

### 环境要求

- **CUDA**: 11.0+
- **GCC**: 9.0+
- **CMake**: 3.18+
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Triton**: 2.0+
- **TileLang**: 最新版

### 安装依赖

```bash
# CUDA和cuBLAS (通常随NVIDIA驱动安装)
# 确认CUDA安装
nvcc --version

# Python依赖
pip install torch torchvision torchaudio
pip install triton
pip install tilelang
```

### 运行完整测试

```bash
# 给主脚本执行权限
chmod +x scripts/run_all_benchmarks.sh

# 运行所有框架的对比测试
./scripts/run_all_benchmarks.sh
```

### 单独运行某个框架

```bash
# cuBLAS
cd frameworks/cublas && ./run.sh

# CUTLASS
cd frameworks/cutlass && ./run.sh

# TileLang
cd frameworks/tilelang && ./run.sh

# Triton
cd frameworks/triton && ./run.sh
```

### 生成对比报告

```bash
# 运行对比分析脚本
python scripts/compare_frameworks.py
```

## 测试配置

- **矩阵尺寸**: 4096×4096×4096
- **数据类型**: FP16输入，FP32累加
- **测试次数**: 10次取平均
- **GPU**: 第一个可见GPU (CUDA_VISIBLE_DEVICES=0)

## 输出文件

运行测试后会生成以下文件：

1. **对比报告**: `docs/comparison_report.md`
   - 性能数据表格
   - 框架特性对比
   - 适用场景建议

2. **技术分析**: `docs/technical_analysis.md`
   - 深入的技术对比
   - 算法详解
   - 性能分析

3. **性能可视化**: `results/performance_comparison.png`
   - TFLOPS对比柱状图
   - 执行时间对比

4. **详细日志**: `results/logs/` 目录
   - 每个框架的完整输出日志

5. **基准结果**: `results/benchmark_results.json`
   - 结构化测试数据

6. **编译产物**: `artifacts/` 目录
   - 可执行文件 (`artifacts/binaries/`)
   - 汇编代码 (`artifacts/assembly/`)

## 核心发现

### 性能对比 (理论排序)
1. **cuBLAS**: 最接近硬件理论峰值，经过厂商深度优化
2. **CUTLASS**: 与cuBLAS性能相当，可定制性强
3. **Triton**: 接近手工优化水平，编译器优化优秀
4. **TileLang**: 自动优化水平，性能稳健

### 开发效率对比
1. **cuBLAS**: 调用简单，但难以定制
2. **Triton**: Python编程，学习曲线平缓
3. **TileLang**: 声明式编程，抽象层次高
4. **CUTLASS**: C++模板，学习曲线陡峭

### 适用场景建议

| 场景 | 推荐框架 | 理由 |
|------|----------|------|
| 生产应用 | cuBLAS | 性能最稳定可靠 |
| 算子开发 | CUTLASS | 可定制性强，性能优秀 |
| 快速原型 | Triton/TileLang | 开发效率高 |
| 学术研究 | TileLang | 自动优化，易集成 |

## 技术特点对比

| 框架 | 编程模型 | 编译方式 | 优化策略 | 学习曲线 |
|------|----------|----------|----------|----------|
| cuBLAS | 库调用 | AOT | 厂商优化 | 平缓 |
| CUTLASS | 模板元编程 | AOT | 可配置优化 | 陡峭 |
| TileLang | 函数式编程 | JIT | 自动优化 | 中等 |
| Triton | Python DSL | JIT | 编译器优化 | 平缓 |

## 深入分析

详细的技术分析请参考 `docs/technical_analysis.md`，内容包括：

- 各框架的设计出发点和哲学
- 面向的具体问题和解决方案
- GEMM算法的详细实现
- 性能优化策略分析
- 代码结构和可维护性对比

更多文档请查看 `docs/` 目录：
- `PROJECT_SUMMARY.md` - 项目总结
- `comparison_report.md` - 对比分析报告
- `detailed_performance_analysis.md` - 详细性能分析
- `detailed_assembly_performance_analysis.md` - 汇编性能分析
- `compilation_performance_analysis.md` - 编译性能分析

## 故障排除

### 常见问题

1. **CUDA相关错误**
   ```bash
   # 检查CUDA安装
   nvcc --version
   nvidia-smi
   ```

2. **编译失败**
   ```bash
   # 清理构建目录重新编译
   cd frameworks/cublas  # 或 cutlass
   rm -rf build/
   ./run.sh
   ```

3. **Python依赖缺失**
   ```bash
   pip install --upgrade torch triton tilelang
   ```

4. **内存不足**
   - 减小矩阵尺寸 (修改代码中的M,N,K值)
   - 使用更小的GPU

### 调试建议

- 查看详细日志：`results/logs/` 目录下的各框架日志
- 检查GPU状态：`nvidia-smi -l 1`
- 验证CUDA安装：`cuda-gdb --version`
- 验证环境：`python scripts/validate_setup.py`

## 贡献指南

欢迎提交问题和改进建议！

1. **报告问题**: 请提供完整的错误信息和环境描述
2. **性能优化**: 欢迎分享针对特定硬件的优化配置
3. **新框架**: 如果有其他值得对比的框架，欢迎提出

## 许可证

本项目仅用于技术研究和教育目的。
