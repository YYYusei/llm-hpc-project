import sys
import logging
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, 'src')
from analysis_pipeline import run_minimd_analysis

results = run_minimd_analysis(code_path='benchmarks/minimd/force_lj.cpp')

print("\n" + "="*50)
print(" 结果汇总")
print("="*50)
print(f"最佳 prompt: {results['summary']['best_prompt_type']}")
print(f"最高分: {results['summary']['best_score']:.2f}")
print(f"平均分: {results['summary']['average_score']:.2f}")
print(f"总花费: ${results['summary']['total_cost']:.4f}")

print("\n各 Prompt 对比:")
for pt, score in results['summary']['prompt_comparison'].items():
    print(f"  {pt}: {score:.2f}")