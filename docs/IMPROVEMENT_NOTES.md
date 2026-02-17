# LLM-HPC 项目泛化性改进说明

## 📋 改进概述

### 原有问题
1. **硬编码关键词** - 只针对 miniMD 和 HPCG，无法支持新程序
2. **配置不完整** - 只有2个基准程序的 ground truth
3. **Few-shot 过拟合** - 示例与测试代码高度相似
4. **评估逻辑耦合** - 假设特定代码结构

### 改进方案

| 组件 | 原实现 | 改进后 |
|------|--------|--------|
| 基准配置 | config.yaml 硬编码 | benchmark_config.py 模块化注册表 |
| 关键词系统 | 固定列表 | 配置驱动 + 正则表达式 |
| 评估器 | analyzer.py 内嵌 | generalized_evaluator.py 独立模块 |
| Few-shot | 领域特定示例 | 通用 HPC 模式示例 |
| 程序支持 | 2个 (miniMD, HPCG) | 4个 (+ Abinit, CP2K) |

---

## 📁 新增文件

### 1. `src/benchmark_config.py`
**功能**: 基准程序配置注册表

```python
# 获取基准程序定义
from benchmark_config import get_benchmark, get_registry

bench = get_benchmark("minimd")
print(bench.hotspots)  # 热点定义列表
print(bench.get_all_keywords())  # 所有关键词

# 自动检测代码属于哪个基准程序
registry = get_registry()
name = registry.find_matching_benchmark(code_content)
```

**支持的基准程序**:
- `minimd` - 分子动力学 (C++, compute-bound)
- `hpcg` - 共轭梯度 (C++, memory-bound)  
- `abinit` - DFT计算 (Fortran, compute/memory mixed)
- `cp2k` - 量子化学 (Fortran, compute-bound)

### 2. `src/generalized_evaluator.py`
**功能**: 通用评估器

```python
from generalized_evaluator import evaluate_analysis, GeneralizedEvaluator

# 快速评估
result = evaluate_analysis(llm_output, "minimd", "zero_shot")
print(f"Score: {result.total_score}/100")

# 自定义评估器
evaluator = GeneralizedEvaluator(
    weights={"hotspot": 0.5, "bottleneck": 0.25, "gpu": 0.15, "suggestions": 0.1},
    tolerance=0.25  # 百分比匹配容差
)
result = evaluator.evaluate(llm_output, "abinit")
```

**评估维度**:
- `hotspot_score` - 热点识别准确度 (正则+关键词+相似度)
- `bottleneck_score` - 瓶颈类型判断
- `gpu_score` - GPU适合度评估
- `suggestions_score` - 优化建议质量

### 3. `configs/config_v2.yaml`
**功能**: 支持4个程序的配置文件

```yaml
# 完整分析 + GPU 转换
primary_benchmarks:
  - minimd
  - hpcg

# 仅 LLM 分析对比
secondary_benchmarks:
  - abinit
  - cp2k
```

### 4. `prompts/few_shot_v2.txt`
**功能**: 泛化的 few-shot 模板

**改进点**:
- 示例1: Stencil 计算 (通用 compute-bound)
- 示例2: 间接索引 Gather (通用 memory-bound)
- 示例3: 条件归约 (Fortran, mixed)

不再使用与测试代码高度相似的 N-body 和 SpMV 示例。

---

## 🔧 使用方法

### 1. 更新现有代码使用新评估器

```python
# 原代码
from analyzer import HPCAnalyzer
analyzer = HPCAnalyzer()
result = analyzer.analyze(code_path, prompt_type="zero_shot")
analyzer.evaluate(result, ground_truth)  # 旧的评估方法

# 新代码
from analyzer import HPCAnalyzer
from generalized_evaluator import evaluate_analysis

analyzer = HPCAnalyzer()
result = analyzer.analyze(code_path, prompt_type="zero_shot")

# 使用新评估器
eval_result = evaluate_analysis(
    {
        "hotspots": result.hotspots,
        "bottleneck_type": result.bottleneck_type,
        "gpu_suitability": result.gpu_suitability,
        "optimization_suggestions": result.optimization_suggestions
    },
    benchmark_name="minimd",
    prompt_type="zero_shot"
)
print(f"Score: {eval_result.total_score}/100")
```

### 2. 添加新基准程序

在 `benchmark_config.py` 中添加:

```python
BENCHMARK_DEFINITIONS["new_benchmark"] = BenchmarkDefinition(
    name="new_benchmark",
    full_name="New HPC Application",
    language="cpp",  # 或 "fortran"
    domain="your_domain",
    hotspots=[
        HotspotDefinition(
            name="main_hotspot",
            location_patterns=[r"pattern1", r"pattern2"],
            time_percentage=70.0,
            bottleneck_type="compute",
            loop_keywords=["inner loop", "main loop"],
            memory_patterns=["sequential", "streaming"]
        )
    ],
    gpu_suitable=True,
    function_keywords=["func1", "func2"],
    structure_keywords=["algorithm", "pattern"]
)
```

### 3. 运行实验 (方案B)

```bash
# Primary benchmarks: 完整流程
python src/main.py --benchmark minimd --full
python src/main.py --benchmark hpcg --full

# Secondary benchmarks: 仅 LLM 分析
python src/main.py --benchmark abinit --llm-only
python src/main.py --benchmark cp2k --llm-only
```

---

## 📊 泛化性改进效果

### Before (硬编码)
```python
generic_keywords = {
    "forcelj", "compute", "spmv", "computespmv", ...  # 固定列表
}
```

### After (配置驱动)
```python
# 从配置自动获取
keywords = benchmark.get_all_keywords()
for pattern in hotspot.location_patterns:
    if re.search(pattern, location, re.IGNORECASE):
        score = 0.9  # 正则匹配
```

### 评估覆盖度

| 基准程序 | 原评估 | 新评估 |
|----------|--------|--------|
| miniMD | ✅ 完整 | ✅ 完整 |
| HPCG | ✅ 完整 | ✅ 完整 (含 SYMGS) |
| Abinit | ❌ 不支持 | ✅ 基础支持 |
| CP2K | ❌ 不支持 | ✅ 基础支持 |

---

## ⚠️ 注意事项

1. **Abinit 和 CP2K 的 profiling 数据需要补充**
   - 当前使用估计值
   - 需要实际运行并收集 VTune/gprof 数据

2. **Few-shot 模板切换**
   ```python
   # 使用新模板
   analyzer = HPCAnalyzer(prompts_dir="prompts")
   # 手动指定使用 few_shot_v2.txt
   ```

3. **保持原有接口兼容**
   - `analyzer.evaluate()` 仍然可用
   - 推荐使用 `generalized_evaluator.evaluate_analysis()` 获得更详细的评估

---

---

## 🔧 VTune 集成

### 新增文件

| 文件 | 作用 |
|------|------|
| `src/vtune_integration.py` | VTune 报告解析、数据转换 |
| `src/analysis_pipeline.py` | 端到端分析流水线 |

### VTune 数据使用方式

#### 方式1: 解析 VTune 报告文件

```python
from analysis_pipeline import IntegratedAnalysisPipeline

pipeline = IntegratedAnalysisPipeline()

# 使用 VTune CSV 报告
results = pipeline.run_with_vtune_report(
    code_path="benchmarks/minimd/force_lj.cpp",
    vtune_report_path="vtune_results/hotspots.csv",
    benchmark_name="minimd",
    gpu_suitable=True
)
```

支持的 VTune 报告格式:
- **CSV**: `vtune -report hotspots -format csv`
- **TXT**: `vtune -report hotspots -format text`
- **JSON**: 自定义格式

#### 方式2: 手动输入 VTune 数据

```python
from analysis_pipeline import IntegratedAnalysisPipeline

pipeline = IntegratedAnalysisPipeline()

results = pipeline.run_with_manual_vtune_data(
    code_path="benchmarks/minimd/force_lj.cpp",
    benchmark_name="minimd",
    hotspots=[
        {"name": "ForceLJ::compute", "time": 3.685, "percentage": 73.7},
        {"name": "Neighbor::build", "time": 0.859, "percentage": 17.2},
    ],
    total_time=5.0,
    cpu_utilization=9.5,
    system_info={
        "cpu": "Intel Core i7-11800H",
        "cores": 8,
        "threads": 16
    },
    gpu_suitable=True
)
```

#### 方式3: 快捷函数

```python
from analysis_pipeline import run_minimd_analysis, run_hpcg_analysis

# 使用预定义的 VTune 数据
results = run_minimd_analysis()
results = run_hpcg_analysis()
```

### VTune 数据结构

```python
# 输入格式
hotspots = [
    {
        "name": "ForceLJ::compute",    # 函数名
        "time": 3.685,                  # CPU 时间（秒）
        "percentage": 73.7,             # 时间占比（%）
        "module": "minimd.exe",         # 可选：模块名
        "cpi_rate": 1.2,                # 可选：CPI
        "source_file": "force_lj.cpp",  # 可选：源文件
        "start_line": 88                # 可选：起始行号
    },
    ...
]

# 输出格式（用于 contextual prompt）
profiling_data = {
    "source": "vtune",
    "total_elapsed_time": 5.0,
    "cpu_utilization": 9.5,
    "hotspots": {
        "forcelj_compute": {"time": 3.685, "percentage": 73.7},
        "neighbor_build": {"time": 0.859, "percentage": 17.2}
    },
    "system_info": {...}
}
```

### 运行 VTune 的命令参考

```bash
# 热点分析
vtune -collect hotspots -result-dir vtune_result -- ./minimd < in.lj

# 生成 CSV 报告
vtune -report hotspots -result-dir vtune_result -format csv -report-output hotspots.csv

# 生成文本报告
vtune -report hotspots -result-dir vtune_result -format text -report-output hotspots.txt

# 内存访问分析
vtune -collect memory-access -result-dir vtune_mem -- ./hpcg
```

---

## 📈 下一步建议

1. **收集 Abinit/CP2K 的实际 VTune profiling 数据**
2. **测试新评估器在4个程序上的表现**
3. **调整关键词和正则表达式以优化匹配率**
4. **考虑添加更多通用 HPC 模式到 few-shot 示例**
