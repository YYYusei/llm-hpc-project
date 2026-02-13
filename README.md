# LLM-HPC: LLM 辅助 HPC 性能分析与 GPU 转换

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 项目简介

本项目研究使用大型语言模型（LLM）辅助高性能计算（HPC）代码的性能分析和 GPU 转换。

### 研究问题

1. **LLM 能否准确识别 HPC 代码的性能瓶颈？**
2. **LLM 能否辅助将 CPU 代码转换为 GPU 代码？**

### 测试基准

| 基准程序 | 领域 | 主要热点 | 瓶颈类型 |
|----------|------|----------|----------|
| miniMD | 分子动力学 | ForceLJ::compute (80%) | compute/latency |
| HPCG | 稀疏线性代数 | SpMV (60-70%) | memory |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- OpenAI API Key
- (可选) CUDA Toolkit 12.0+

### 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/llm-hpc-project.git
cd llm-hpc-project

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp configs/config.example.yaml configs/config.yaml
# 编辑 config.yaml，填入你的 API Key
```

### 运行测试

```bash
# 运行 LLM 性能分析
python src/main.py analyze --code minimd

# 运行 GPU 转换
python src/main.py convert --code minimd --function compute_fullneigh

# 运行完整实验
python src/main.py experiment --all
```

## 📁 项目结构

```
llm-hpc-project/
├── README.md                 # 项目说明
├── requirements.txt          # Python 依赖
├── setup.py                  # 安装配置
├── .gitignore               # Git 忽略文件
├── .env.example             # 环境变量示例
│
├── configs/                  # 配置文件
│   ├── config.yaml          # 主配置
│   └── prompts.yaml         # Prompt 配置
│
├── src/                      # 源代码
│   ├── __init__.py
│   ├── main.py              # 主入口
│   ├── llm_client.py        # LLM API 客户端
│   ├── analyzer.py          # 性能分析模块
│   ├── converter.py         # GPU 转换模块
│   ├── evaluator.py         # 结果评估模块
│   └── utils.py             # 工具函数
│
├── prompts/                  # Prompt 模板
│   ├── zero_shot.txt
│   ├── few_shot.txt
│   └── contextual.txt
│
├── benchmarks/               # 基准代码
│   ├── minimd/
│   │   └── force_lj.cpp
│   └── hpcg/
│       └── ComputeSPMV_ref.cpp
│
├── results/                  # 实验结果
│   ├── analysis/            # 性能分析结果
│   └── conversion/          # GPU 转换结果
│
├── tests/                    # 测试代码
│   ├── test_analyzer.py
│   └── test_converter.py
│
├── docs/                     # 文档
│   ├── baseline_report.md   # D1: Baseline 报告
│   ├── llm_experiment.md    # D2: LLM 实验报告
│   └── gpu_conversion.md    # D3: GPU 转换报告
│
└── scripts/                  # 部署脚本
    ├── deploy.sh            # 服务器部署
    └── run_experiment.sh    # 批量实验
```

## 📊 实验结果

### LLM 性能分析准确率

| 指标 | Zero-shot | Few-shot | Contextual |
|------|-----------|----------|------------|
| 热点识别 | 100% | 100% | 100% |
| 瓶颈类型 | 83% | 100% | 100% |
| 平均得分 | 95 | 100 | 116.5 |

### GPU 转换效果

| 代码 | CPU 时间 | GPU 时间 | 加速比 |
|------|----------|----------|--------|
| miniMD force | TBD | TBD | TBD |
| HPCG SpMV | TBD | TBD | TBD |

## 📝 使用示例

### Python API

```python
from src.analyzer import HPCAnalyzer
from src.converter import GPUConverter

# 性能分析
analyzer = HPCAnalyzer(api_key="your-key")
result = analyzer.analyze(
    code_path="benchmarks/minimd/force_lj.cpp",
    prompt_type="contextual",
    profiling_data=profiling_data
)
print(result.hotspots)
print(result.bottleneck_type)

# GPU 转换
converter = GPUConverter(api_key="your-key")
cuda_code = converter.convert(
    code_path="benchmarks/minimd/force_lj.cpp",
    function_name="compute_fullneigh"
)
cuda_code.save("output/force_lj.cu")
```

### 命令行

```bash
# 分析单个文件
python src/main.py analyze \
    --code benchmarks/minimd/force_lj.cpp \
    --prompt contextual \
    --output results/analysis/

# 批量实验
python src/main.py experiment \
    --config configs/experiment.yaml \
    --output results/
```

## 🔧 配置说明

### config.yaml

```yaml
llm:
  provider: openai
  model: gpt-4o
  temperature: 0
  max_tokens: 4096

analysis:
  prompt_types:
    - zero_shot
    - few_shot
    - contextual

benchmarks:
  minimd:
    path: benchmarks/minimd/force_lj.cpp
    hotspot: ForceLJ::compute
    bottleneck: compute
  hpcg:
    path: benchmarks/hpcg/ComputeSPMV_ref.cpp
    hotspot: SpMV inner loop
    bottleneck: memory
```

## 🚢 部署

### 服务器部署

```bash
# 上传到服务器
scp -r llm-hpc-project user@server:/path/to/

# SSH 登录并运行
ssh user@server
cd /path/to/llm-hpc-project
./scripts/deploy.sh
```

### Docker 部署

```bash
docker build -t llm-hpc .
docker run -e OPENAI_API_KEY=your-key llm-hpc
```

## 📚 文档

- [Baseline 报告 (D1)](docs/baseline_report.md)
- [LLM 实验报告 (D2)](docs/llm_experiment.md)
- [GPU 转换报告 (D3)](docs/gpu_conversion.md)
- [API 文档](docs/api.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 👤 作者

- Yusei - 毕业设计项目
