# LLM-HPC: LLM 辅助 HPC 性能分析与 GPU 转换

## 项目简介

本项目研究使用大型语言模型（LLM）辅助高性能计算（HPC）代码的性能分析和 GPU 转换。

主要研究问题：
1. LLM 能否准确识别 HPC 代码的性能瓶颈？
2. 不同 prompt 策略对分析准确率的影响？
3. LLM 能否辅助将 CPU 热点代码转换为 GPU 代码？
4. **[新增]** 级联分析方案能否提高分析准确性？

## 🆕 最新进展 (2026-02-24)

### 级联分析方案 (Cascaded Pipeline)

利用不同 LLM 的互补优势，通过两阶段验证提高分析准确性：
```
源代码 ──→ [Stage 1: GPT-4o] ──→ [Stage 2: GPT-5.2] ──→ 最终结果
            快速初筛              深度验证+优化建议
```

**实验结果**:

| 程序 | Stage 1 (GPT-4o) | Stage 2 (GPT-5.2) | 结果 |
|------|------------------|-------------------|------|
| miniMD | compute | memory/latency + sync | ❌ 修正 |
| HPCG SPMV | memory | memory | ✅ 确认 |
| HPCG SYMGS | memory | memory | ✅ 确认 |
| Abinit | compute | memory + allocation | ❌ 修正 |

- **瓶颈修正率**: 50% (2/4)
- **优化建议提升**: 2.1x (平均 2.75 → 5.75 条)
- **修正准确性**: 100%

### CUDA 代码生成

| 模型 | 首次成功 | 加速比 | 误差 | 花费 |
|------|----------|--------|------|------|
| GPT-4o | ✅ | **11.72x** | 3.98e-13 | $0.009 |
| GPT-5.2 | ✅ | **11.58x** | 7.28e-12 | $0.027 |

## 支持的基准程序

| 程序 | 领域 | 语言 | 主要热点 | 瓶颈类型 |
|------|------|------|----------|----------|
| miniMD | 分子动力学 | C++ | ForceLJ::compute | memory/sync |
| HPCG SPMV | 稀疏矩阵向量乘 | C++ | ComputeSPMV_ref | memory |
| HPCG SYMGS | 对称高斯-赛德尔 | C++ | ComputeSYMGS_ref | memory |
| Abinit | DFT 计算 | Fortran | nonlop_ylm | memory/allocation |

其中 miniMD 和 HPCG 做完整分析 + GPU 转换，Abinit 仅做 LLM 分析对比。

## 快速开始

### 环境要求

- Python 3.8+
- OpenAI API Key (支持 GPT-4o, GPT-5.2)
- (可选) VTune 用于 profiling
- (可选) CUDA Toolkit 12.6+ 用于 GPU 转换
- (可选) WSL2 Ubuntu 用于 CUDA 编译

### 安装
```bash
git clone https://github.com/yourusername/llm-hpc-project.git
cd llm-hpc-project

pip install -r requirements.txt

# 设置 API Key
export OPENAI_API_KEY="your-key"  # Linux/Mac
$env:OPENAI_API_KEY="your-key"   # Windows PowerShell
```

### 运行测试
```bash
# 本地测试（不调用 API）
python test_local.py

# 单程序分析
python test_api.py           # miniMD
python test_hpcg.py          # HPCG SPMV
python test_hpcg_symgs.py    # HPCG SYMGS
python test_abinit.py        # Abinit

# 级联分析（所有程序）
python test_cascaded.py

# CUDA 优化代码生成
python test_cuda_optimization_full.py
```

## 项目结构
```
llm-hpc-project/
├── src/
│   ├── analyzer.py              # LLM 性能分析
│   ├── benchmark_config.py      # 基准程序配置（支持4个程序）
│   ├── generalized_evaluator.py # 通用评估器
│   ├── vtune_integration.py     # VTune 数据解析
│   ├── analysis_pipeline.py     # 端到端流水线
│   ├── cascaded_pipeline.py     # 🆕 级联分析流水线
│   ├── converter.py             # GPU 转换
│   └── llm_client.py            # OpenAI API 封装
│
├── prompts/
│   ├── zero_shot.txt            # 无示例
│   ├── few_shot.txt             # 带示例（原版）
│   ├── few_shot_v2.txt          # 带示例（泛化版）
│   └── contextual.txt           # 带 profiling 数据
│
├── configs/
│   ├── config.yaml              # 基础配置
│   └── config_v2.yaml           # 支持4个程序的配置
│
├── benchmarks/                  # 待分析的代码
│   ├── minimd/
│   ├── hpcg/
│   └── abinit/
│
├── results/
│   ├── analysis/                # LLM 分析结果
│   ├── evaluation/              # 评估结果
│   ├── vtune/                   # VTune 数据
│   ├── reports/                 # 汇总报告
│   ├── cascaded/                # 🆕 级联分析结果
│   └── cuda_optimization/       # 🆕 CUDA 生成结果
│
├── test_cascaded.py             # 🆕 级联分析测试
├── test_cuda_optimization_full.py # 🆕 CUDA 完整测试
│
└── docs/
    ├── IMPROVEMENT_NOTES.md     # 改进说明
    └── daily_log_20260224.md    # 🆕 开发日志
```

## 使用方法

### 方法1：使用集成流水线
```python
from analysis_pipeline import IntegratedAnalysisPipeline

pipeline = IntegratedAnalysisPipeline()

# 使用手动输入的 VTune 数据
results = pipeline.run_with_manual_vtune_data(
    code_path='benchmarks/minimd/force_lj.cpp',
    benchmark_name='minimd',
    hotspots=[
        {"name": "ForceLJ::compute", "time": 3.685, "percentage": 73.7},
        {"name": "Neighbor::build", "time": 0.859, "percentage": 17.2},
    ],
    total_time=5.0,
    cpu_utilization=9.5
)

print(f"最佳 prompt: {results['summary']['best_prompt_type']}")
print(f"最高分: {results['summary']['best_score']}")
```

### 方法2：使用 VTune 报告文件
```python
results = pipeline.run_with_vtune_report(
    code_path='benchmarks/minimd/force_lj.cpp',
    vtune_report_path='vtune_hotspots.csv',
    benchmark_name='minimd'
)
```

### 方法3：使用预定义配置
```python
results = pipeline.run_with_config(
    code_path='benchmarks/minimd/force_lj.cpp',
    benchmark_name='minimd'
)
```

### 方法4：级联分析 🆕
```python
from src.cascaded_pipeline import CascadedAnalysisPipeline

pipeline = CascadedAnalysisPipeline()
result = pipeline.analyze(
    code_path='benchmarks/minimd/force_lj.cpp',
    benchmark_name='minimd'
)

print(f"Stage 1 瓶颈: {result['stage1']['bottleneck_type']}")
print(f"Stage 2 验证: {result['stage2']['verification']}")
print(f"优化建议: {result['stage2']['optimizations']}")
```

### 方法5：CUDA 代码生成测试 🆕
```python
from test_cuda_optimization_full import CUDAOptimizationTester

tester = CUDAOptimizationTester()
results = tester.run_full_test()

# 自动测试 GPT-4o 和 GPT-5.2
# 编译失败会让 GPT-5.2 修复
# 输出 CPU vs GPU 加速比
```

### 快捷函数
```python
from analysis_pipeline import run_minimd_analysis, run_hpcg_analysis

results = run_minimd_analysis()
results = run_hpcg_analysis()
```

## 实验结果

### miniMD 单模型分析 (2026-02-17)

| Prompt | 分数 | 热点数 | 花费 |
|--------|------|--------|------|
| zero_shot | 91.71 | 4 | $0.018 |
| few_shot | **96.17** | 4 | $0.022 |
| contextual | 93.09 | 1 | $0.019 |

### HPCG 单模型分析 (2026-02-17)

| Prompt | 分数 | 热点数 | 花费 |
|--------|------|--------|------|
| zero_shot | **85.20** | 2 | $0.006 |
| few_shot | 82.40 | 1 | $0.009 |
| contextual | 82.20 | 1 | $0.009 |

### 多模型对比 (2026-02-24) 🆕

| 指标 | GPT-4o | GPT-5.2 |
|------|--------|---------|
| 瓶颈判断准确率 | **100%** (4/4) | 50% (2/4) |
| 优化建议数量 | 2-3 条 | **5-7 条** |
| 分析详细度 | 中等 | **高** |
| 响应时间 | **~10s** | ~35s |
| 成本 | **$0.02** | $0.04 |

### 级联方案效果 🆕

| 指标 | 单模型 | 级联方案 |
|------|--------|----------|
| 瓶颈准确率 | 50-100% | **100%** |
| 优化建议 | 2-3 条 | **5-7 条** |
| 验证机制 | ❌ | ✅ |
| 成本 | $0.02-0.04 | $0.08 |

### CUDA 生成性能 🆕

测试环境: RTX 3060, CUDA 12.6, 100K 原子

| 模型 | CPU (ms) | GPU (ms) | 加速比 | 正确性 |
|------|----------|----------|--------|--------|
| GPT-4o | 17.00 | 1.45 | **11.72x** | ✅ (err: 3.98e-13) |
| GPT-5.2 | 16.85 | 1.45 | **11.58x** | ✅ (err: 7.28e-12) |

## 关键发现 🆕

1. **级联方案有效**: GPT-5.2 修正了 50% 的 GPT-4o 瓶颈判断错误
2. **模型互补**: GPT-4o 快速准确，GPT-5.2 深入详细
3. **CUDA 生成可行**: 两个模型都能生成 ~11.7x 加速的正确代码
4. **GPT-5.2 知识盲区**: 错误使用 `__ldg()` 于 `double4` 类型

## VTune 使用
```bash
# 运行热点分析
vtune -collect hotspots -result-dir vtune_result -- ./minimd < in.lj

# 导出 CSV
vtune -report hotspots -result-dir vtune_result -format csv -report-output hotspots.csv
```

支持的 VTune 报告格式：CSV、TXT、JSON

## 配置说明

`configs/config_v2.yaml` 包含4个基准程序的配置：
```yaml
# 完整分析 + GPU 转换
primary_benchmarks:
  - minimd
  - hpcg

# 仅 LLM 分析
secondary_benchmarks:
  - abinit
  - cp2k
```

## 成本估算

| 任务 | GPT-4o | GPT-5.2 | 级联 |
|------|--------|---------|------|
| 单程序分析 | $0.02 | $0.04 | $0.08 |
| 4 程序完整测试 | $0.08 | $0.15 | $0.31 |
| CUDA 生成 | $0.01 | $0.03 | - |

## 许可证

MIT License

## 作者

Yusei - 本科毕业设计项目 (2026)