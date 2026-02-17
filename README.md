# LLM-HPC: LLM 辅助 HPC 性能分析与 GPU 转换

## 项目简介

本项目研究使用大型语言模型（LLM）辅助高性能计算（HPC）代码的性能分析和 GPU 转换。

主要研究问题：
1. LLM 能否准确识别 HPC 代码的性能瓶颈？
2. 不同 prompt 策略对分析准确率的影响？
3. LLM 能否辅助将 CPU 热点代码转换为 GPU 代码？

## 支持的基准程序

| 程序 | 领域 | 语言 | 主要热点 | 瓶颈类型 |
|------|------|------|----------|----------|
| miniMD | 分子动力学 | C++ | ForceLJ::compute | compute |
| HPCG | 稀疏线性代数 | C++ | ComputeSYMGS/SPMV | memory |
| Abinit | DFT 计算 | Fortran | nonlop/fourwf | compute/memory |
| CP2K | 量子化学 | Fortran | grid_integrate/dbcsr | compute |

其中 miniMD 和 HPCG 做完整分析 + GPU 转换，Abinit 和 CP2K 仅做 LLM 分析对比。

## 快速开始

### 环境要求

- Python 3.8+
- OpenAI API Key
- (可选) VTune 用于 profiling
- (可选) CUDA Toolkit 用于 GPU 转换

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

# miniMD 完整分析
python test_api.py

# HPCG 完整分析
python test_hpcg.py
```

## 项目结构

```
llm-hpc-project/
├── src/
│   ├── analyzer.py           # LLM 性能分析
│   ├── benchmark_config.py   # 基准程序配置（支持4个程序）
│   ├── generalized_evaluator.py  # 通用评估器
│   ├── vtune_integration.py  # VTune 数据解析
│   ├── analysis_pipeline.py  # 端到端流水线
│   ├── converter.py          # GPU 转换
│   └── llm_client.py         # OpenAI API 封装
│
├── prompts/
│   ├── zero_shot.txt         # 无示例
│   ├── few_shot.txt          # 带示例（原版）
│   ├── few_shot_v2.txt       # 带示例（泛化版）
│   └── contextual.txt        # 带 profiling 数据
│
├── configs/
│   ├── config.yaml           # 基础配置
│   └── config_v2.yaml        # 支持4个程序的配置
│
├── benchmarks/               # 待分析的代码
│   ├── minimd/
│   └── hpcg/
│
├── results/
│   ├── analysis/             # LLM 分析结果
│   ├── evaluation/           # 评估结果
│   ├── vtune/                # VTune 数据
│   └── reports/              # 汇总报告
│
└── docs/
    └── IMPROVEMENT_NOTES.md  # 改进说明
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

### 快捷函数

```python
from analysis_pipeline import run_minimd_analysis, run_hpcg_analysis

results = run_minimd_analysis()
results = run_hpcg_analysis()
```

## 实验结果

### miniMD (2026-02-17)

| Prompt | 分数 | 热点数 | 花费 |
|--------|------|--------|------|
| zero_shot | 91.71 | 4 | $0.018 |
| few_shot | **96.17** | 4 | $0.022 |
| contextual | 93.09 | 1 | $0.019 |

### HPCG (2026-02-17)

| Prompt | 分数 | 热点数 | 花费 |
|--------|------|--------|------|
| zero_shot | **85.20** | 2 | $0.006 |
| few_shot | 82.40 | 1 | $0.009 |
| contextual | 82.20 | 1 | $0.009 |

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

## 许可证

MIT License

## 作者

Yusei - 毕业设计项目
