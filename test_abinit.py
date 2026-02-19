import sys
import logging
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, 'src')
from analysis_pipeline import IntegratedAnalysisPipeline

pipeline = IntegratedAnalysisPipeline()

# 使用 Fortran 源码
results = pipeline.run_with_manual_vtune_data(
    code_path='benchmarks/abinit/m_nonlop_ylm.F90',  # 改成 Fortran 源码
    benchmark_name='abinit',
    hotspots=[
        {"name": "nonlop_ylm", "time": 2.70, "percentage": 28.0},
        {"name": "opernl4b", "time": 1.21, "percentage": 12.6},
        {"name": "fourwf", "time": 0.99, "percentage": 10.3},
        {"name": "zgemm", "time": 0.46, "percentage": 4.8},
    ],
    total_time=9.65,
    cpu_utilization=11.0,
    gpu_suitable=True
)

print("\n" + "="*50)
print("Abinit (Fortran source) Results")
print("="*50)
print(f"Best prompt: {results['summary']['best_prompt_type']}")
print(f"Best score: {results['summary']['best_score']:.2f}")
print(f"Total cost: ${results['summary']['total_cost']:.4f}")

print("\nPrompt comparison:")
for pt, score in results['summary']['prompt_comparison'].items():
    print(f"  {pt}: {score:.2f}")