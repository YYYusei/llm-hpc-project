import sys
import logging
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, 'src')
from analysis_pipeline import run_hpcg_analysis

results = run_hpcg_analysis(code_path='benchmarks/hpcg/ComputeSPMV_ref.cpp')

print("\n" + "="*50)
print(" HPCG 结果汇总")
print("="*50)
print(f"最佳 prompt: {results['summary']['best_prompt_type']}")
print(f"最高分: {results['summary']['best_score']:.2f}")
print(f"总花费: ${results['summary']['total_cost']:.4f}")

print("\n各 Prompt 对比:")
for pt, score in results['summary']['prompt_comparison'].items():
    print(f"  {pt}: {score:.2f}")