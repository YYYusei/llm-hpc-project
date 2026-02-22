import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, 'src')

from cascaded_pipeline import run_cascaded_analysis

# 测试配置
tests = [
    {"name": "minimd", "code_path": "benchmarks/minimd/force_lj.cpp"},
    {"name": "hpcg_spmv", "code_path": "benchmarks/hpcg/ComputeSPMV_ref.cpp"},
    {"name": "hpcg_symgs", "code_path": "benchmarks/hpcg/ComputeSYMGS_ref.cpp"},
    {"name": "abinit", "code_path": "benchmarks/abinit/m_nonlop_ylm.F90"},
]

all_results = {}
total_cost = 0

for test in tests:
    print(f"\n{'='*60}")
    print(f"测试: {test['name']}")
    print('='*60)
    
    try:
        results = run_cascaded_analysis(
            code_path=test['code_path'],
            code_name=test['name'],
            prompt_types=["zero_shot"]
        )
        
        all_results[test['name']] = results
        
        for pt, result in results.items():
            s1 = result['stage1']
            s2 = result['stage2']
            cost = result['total_cost']
            total_cost += cost
            
            print(f"\nStage 1 (GPT-4o):")
            print(f"  热点数: {len(s1['hotspots'])}")
            print(f"  瓶颈: {s1['bottleneck_type'].get('primary', 'N/A')}")
            print(f"  GPU适合: {s1['gpu_suitability'].get('suitable', 'N/A')}")
            print(f"  花费: ${s1['cost']:.4f}")
            
            print(f"Stage 2 (GPT-5.2):")
            validation = s2.get('validation', {})
            correct = validation.get('bottleneck_correct', False)
            print(f"  瓶颈验证: {'正确' if correct else '修正'}")
            if not correct:
                corrected = validation.get('corrected_bottleneck', 'N/A')
                print(f"  修正为: {corrected[:60]}...")
            print(f"  优化建议数: {len(s2.get('detailed_optimizations', []))}")
            print(f"  花费: ${s2['cost']:.4f}")
            
            print(f"总花费: ${cost:.4f}")
            
    except Exception as e:
        print(f"失败: {e}")
        all_results[test['name']] = {"error": str(e)}

print(f"\n{'='*60}")
print(f"全部测试完成，总花费: ${total_cost:.4f}")
print('='*60)

with open('results/cascaded/cascaded_all_results.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print("结果已保存到 results/cascaded/cascaded_all_results.json")