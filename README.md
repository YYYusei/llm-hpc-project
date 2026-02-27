# LLM-HPC: LLM 辅助 HPC 性能分析与 GPU 转换

## 项目简介

本项目研究使用大型语言模型（LLM）辅助高性能计算（HPC）代码的性能分析和 GPU 转换。

主要研究问题：
1. LLM 能否准确识别 HPC 代码的性能瓶颈？
2. 不同 prompt 策略对分析准确率的影响？
3. LLM 能否生成高质量的 GPU 优化代码？
4. 级联分析方案能否提高分析准确性？
5. **[新增]** 直接生成 vs 完整流程，哪种方法更好？

## 🆕 最新进展 (2026-02-27)

### 三个 GPU Kernel 全部完成

| Kernel | 类型 | 最佳方法 | 加速比 | 难度 |
|--------|------|----------|--------|------|
| miniMD LJ Force | 分子动力学 | 直接生成 | **14.34x** | 中等 |
| HPCG SPMV | 稀疏矩阵 | 直接生成 | **10.30x** | 简单 |
| HPCG SYMGS | 迭代求解 | 完整流程 | **5.61x** | 困难 |

### 关键发现

1. **简单并行代码**: 直接生成更好（SPMV: 10.30x vs 完整流程 6.18x）
2. **复杂依赖代码**: 完整流程必须（SYMGS: 0.02x → 5.61x）
3. **过度分析可能有害**: SPMV 完整流程因错误假设数据格式而失败

### 级联分析方案 (Cascaded Pipeline)
```
源代码 ──→ [Stage 1: GPT-4o] ──→ [Stage 2: GPT-5.2] ──→ 最终结果
            快速初筛              深度验证+优化建议
```

**实验结果**:

| 程序 | Stage 1 (GPT-4o) | Stage 2 (GPT-5.2) | 结果 |
|------|------------------|-------------------|------|
| miniMD | compute | memory/latency + sync | ❌ 修正 |
| HPCG SPMV | memory | memory | ✅ 确认 |
| HPCG SYMGS | memory | memory + dependency | ✅ 确认 |
| Abinit | compute | memory + allocation | ❌ 修正 |

- **瓶颈修正率**: 50% (2/4)
- **优化建议提升**: 67% (平均 3 → 5 条)

### 完整流程对比实验

| Kernel | 直接生成 | 完整流程 | 差异 | 推荐 |
|--------|----------|----------|------|------|
| miniMD | 14.34x | 15.59x | +8.7% | 直接生成 (性价比) |
| SPMV | **10.30x** | 6.18x ❌ | -40% | **直接生成** |
| SYMGS | 0.02x ❌ | **5.61x** | +28000% | **完整流程** |

## 支持的基准程序

| 程序 | 领域 | 语言 | 主要热点 | 瓶颈类型 |
|------|------|------|----------|----------|
| miniMD | 分子动力学 | C++ | ForceLJ::compute | memory/sync |
| HPCG SPMV | 稀疏矩阵向量乘 | C++ | ComputeSPMV_ref | memory |
| HPCG SYMGS | 对称高斯-赛德尔 | C++ | ComputeSYMGS_ref | memory + dependency |
| Abinit | DFT 计算 | Fortran | nonlop_ylm | memory/allocation |

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

# CUDA 代码生成
python test_cuda_optimization_full.py  # miniMD
python test_cuda_spmv.py               # SPMV
python test_cuda_symgs.py              # SYMGS (直接生成)
python test_cuda_symgs_v2.py           # SYMGS (策略提示)
python test_cuda_symgs_full_pipeline.py # SYMGS (完整流程)

# 完整流程对比
python test_full_pipeline_all.py       # miniMD + SPMV 完整流程
## 项目结构
```
llm-hpc-project/
├── src/
│   ├── analyzer.py              # LLM 性能分析器
│   ├── benchmark_config.py      # 基准程序配置
│   ├── cascaded_pipeline.py     # 级联分析流水线
│   ├── analysis_pipeline.py     # 端到端分析流水线
│   ├── generalized_evaluator.py # 通用评估器
│   ├── vtune_integration.py     # VTune 数据解析
│   ├── converter.py             # GPU 转换
│   ├── llm_client.py            # OpenAI API 封装
│   ├── main.py                  # 主程序入口
│   └── utils.py                 # 工具函数
│
├── prompts/
│   ├── zero_shot.txt            # 无示例 prompt
│   ├── few_shot_v3.txt          # 带示例 (stencil + sparse matrix)
│   └── contextual.txt           # 带 VTune 数据的 prompt
│
├── benchmarks/                  # 待分析的 HPC 代码
│   ├── minimd/
│   ├── hpcg/
│   └── abinit/
│
├── configs/                     # 配置文件
│   └── config.yaml
│
├── results/                     # 测试结果
│   ├── analysis/                # LLM 分析结果
│   ├── cascaded/                # 级联分析结果
│   ├── cuda_optimization/       # miniMD CUDA 结果
│   ├── cuda_spmv/               # SPMV CUDA 结果
│   ├── cuda_symgs/              # SYMGS V1 结果
│   ├── cuda_symgs_v2/           # SYMGS V2 (策略提示)
│   ├── cuda_symgs_pipeline/     # SYMGS 完整流程结果
│   ├── full_pipeline_comparison/ # 完整流程对比结果
│   ├── vtune/                   # VTune 工作流验证
│   ├── evaluation/              # 评估结果
│   └── reports/                 # 汇总报告
│
├── docs/
│   ├── case_studies.md          # 7 个 Case Study
│   ├── daily_log_20260224.md    # 开发日志
│   └── daily_log_20260226.md    # 开发日志
│
├── gpu_conversion/              # GPU 转换相关
├── scripts/                     # 辅助脚本
├── tests/                       # 单元测试
├── logs/                        # 运行日志
│
├── test_cascaded.py             # 级联分析测试
├── test_cuda_optimization_full.py # miniMD CUDA 测试
├── test_cuda_spmv.py            # SPMV CUDA 测试
├── test_cuda_symgs.py           # SYMGS V1 (直接生成)
├── test_cuda_symgs_v2.py        # SYMGS V2 (策略提示)
├── test_cuda_symgs_full_pipeline.py # SYMGS 完整流程
├── test_full_pipeline_all.py    # 完整流程对比测试
├── test_vtune_workflow.py       # VTune 工作流验证
├── test_api.py                  # miniMD 单程序分析
├── test_hpcg.py                 # HPCG SPMV 分析
├── test_hpcg_symgs.py           # HPCG SYMGS 分析
├── test_abinit.py               # Abinit 分析
├── test_local.py                # 本地测试 (不调用 API)
│
├── README.md
├── requirements.txt
├── setup.py
├── Dockerfile
└── .gitignore
```

## Case Studies

本项目包含 7 个详细的 Case Study，见 `docs/case_studies.md`：

| # | 案例 | 类型 | 关键发现 |
|---|------|------|----------|
| 1 | miniMD → CUDA | ✅ 成功 | 14.34x 加速，GPT-4o 更优 |
| 2 | SPMV → CUDA | ✅ 成功 | 10.30x 加速，GPT-5.2 更优 |
| 3 | miniMD 级联修正 | ⚠️ 部分正确 | compute → memory/sync |
| 4 | GPT-4o SPMV 参数错误 | ❌ 失败 | 参数顺序不匹配 |
| 5 | Abinit 级联修正 | ⚠️ 部分正确 | compute → memory/allocation |
| 6 | SYMGS 数据依赖 | ✅ 成功 | 完整流程 5.61x，直接生成 0.02x |
| 7 | 完整流程对比 | 🔬 实验 | 简单代码直接生成更好 |

## 最佳实践

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 简单并行，无依赖 | **直接生成** | 更快、更便宜、更可靠 |
| 复杂依赖（如 GS） | **完整流程** | 需要分析才能找到正确策略 |
| 数据格式复杂 | **明确指定格式** | 避免 LLM 错误假设 |
| 首次失败 | **修复机制** | GPT-5.2 修复成功率高 |

## 成本估算

| 任务 | 花费 |
|------|------|
| 单程序级联分析 | $0.07 |
| 简单 CUDA 直接生成 | $0.01-0.02 |
| 复杂 CUDA 完整流程 | $0.04-0.09 |
| 代码修复 | $0.01 |
| **项目总计** | **~$0.70** |

## 实验结果汇总

### CUDA 生成性能

| Kernel | 模型 | 方法 | 加速比 | 误差 |
|--------|------|------|--------|------|
| miniMD | GPT-4o | 直接 | **14.34x** | 3.98e-13 |
| miniMD | GPT-5.2 | 完整 | 15.59x | 1.16e-10 |
| SPMV | GPT-5.2 | 直接 | **10.30x** | 7.11e-15 |
| SYMGS | GPT-5.2 | 完整 | **5.61x** | 3.42e-02 |

### 级联分析效果

| 指标 | 结果 |
|------|------|
| 瓶颈修正率 | 50% (2/4) |
| 优化建议提升 | 67% |
| 修正准确性 | 100% |

## 论文贡献

1. **提出级联分析方案**: 两阶段 LLM 分析，修正率 50%
2. **验证 LLM CUDA 生成能力**: 三个 kernel 达到 5-15x 加速
3. **发现方法适用边界**: 简单并行 → 直接生成，复杂依赖 → 完整流程
4. **识别 LLM 局限性**: 不能自动发明并行化策略，可能错误假设数据格式
5. **提供最佳实践指南**: 根据代码特性选择合适方法

## 许可证

MIT License

## 作者

Yusei - 本科毕业设计项目 (2026)