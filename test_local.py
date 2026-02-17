# test_local.py
import sys
sys.path.insert(0, 'src')

from benchmark_config import get_benchmark, get_registry
from generalized_evaluator import GeneralizedEvaluator
from vtune_integration import create_manual_vtune_report

# 1. 测试配置
print("=== 测试 benchmark_config ===")
registry = get_registry()
print(f"支持的程序: {registry.list_all()}")

minimd = get_benchmark("minimd")
print(f"miniMD 热点: {[h.name for h in minimd.hotspots]}")
print(f"miniMD 关键词数: {len(minimd.get_all_keywords())}")

# 2. 测试 VTune 数据创建
print("\n=== 测试 vtune_integration ===")
report = create_manual_vtune_report(
    hotspots=[
        {"name": "ForceLJ::compute", "time": 3.685, "percentage": 73.7},
        {"name": "Neighbor::build", "time": 0.859, "percentage": 17.2},
    ],
    total_time=5.0,
    cpu_utilization=9.5
)
print(f"Top 热点: {[h.function_name for h in report.get_top_hotspots(2)]}")
print(f"Ground truth: {report.to_ground_truth()}")

# 3. 测试评估器（模拟 LLM 输出）
print("\n=== 测试 generalized_evaluator ===")
mock_llm_result = {
    "hotspots": [{"location": "ForceLJ::compute inner k-loop", "estimated_time_percentage": "75%"}],
    "bottleneck_type": {"primary": "compute"},
    "gpu_suitability": {"suitable": True, "reasoning": "Parallel force calculation"},
    "optimization_suggestions": [{"target": "k-loop", "suggestion": "SIMD vectorization"}]
}

evaluator = GeneralizedEvaluator()
result = evaluator.evaluate(mock_llm_result, "minimd", "zero_shot")

print(f"总分: {result.total_score:.2f}/100")
print(f"  热点: {result.hotspot_score:.2f}")
print(f"  瓶颈: {result.bottleneck_score:.2f}")
print(f"  GPU: {result.gpu_score:.2f}")
print(f"错误: {result.errors}")
print(f"警告: {result.warnings}")

print("\n✅ 所有本地测试通过！")