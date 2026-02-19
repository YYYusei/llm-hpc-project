import sys
import logging
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, 'src')
from analysis_pipeline import IntegratedAnalysisPipeline

pipeline = IntegratedAnalysisPipeline()

# HPCG SYMGS 分析
results = pipeline.run_with_manual_vtune_data(
    code_path='benchmarks/hpcg/ComputeSYMGS_ref.cpp',
    benchmark_name='hpcg',
    hotspots=[
        {"name": "ComputeSYMGS_ref", "time": 45.731, "percentage": 67.3},
        {"name": "ComputeSPMV_ref", "time": 18.834, "percentage": 27.7},
        {"name": "ComputeDotProduct_ref", "time": 1.358, "percentage": 2.0},
    ],
    total_time=67.977,
    cpu_utilization=11.6,
    system_info={
        "cpu": "Intel Core i7-11800H",
        "cores": 8,
        "threads": 16
    },
    gpu_suitable=True
)

print("\n" + "="*50)
print("HPCG SYMGS Results")
print("="*50)
print(f"Best prompt: {results['summary']['best_prompt_type']}")
print(f"Best score: {results['summary']['best_score']:.2f}")
print(f"Total cost: ${results['summary']['total_cost']:.4f}")

print("\nPrompt comparison:")
for pt, score in results['summary']['prompt_comparison'].items():
    print(f"  {pt}: {score:.2f}")