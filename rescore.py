"""
重评分脚本：用更新后的 benchmark_config GT 重新计算 S1 评分
不调 API，只读取已有结果文件重新跑 evaluator

用法:
    cd llm-hpc-project
    python rescore.py

输出:
    打印新旧分数对比表
    保存到 results/extended_cascaded/rescore_comparison.json
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generalized_evaluator import GeneralizedEvaluator
from extended_benchmark_config import register_extended_benchmarks

# 注册更新后的 benchmark 配置
register_extended_benchmarks()

SUMMARY_PATH = "results/extended_cascaded/summary.json"

CONFIG_NAME_MAP = {
    "minimd":     "minimd",
    "hpcg_spmv":  "hpcg",
    "hpcg_symgs": "hpcg",
    "abinit":     "abinit",
    "hotspot":    "hotspot",
    "srad":       "srad",
    "lulesh":     "lulesh",
    "nas_cg":     "nas_cg",
    "jacobi2d":   "jacobi2d",
}


def main():
    with open(SUMMARY_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    evaluator = GeneralizedEvaluator()

    print(f"\n{'='*90}")
    print("RESCORE: Using updated benchmark_config GT (VTune-aligned)")
    print(f"{'='*90}")
    print(f"{'Program':<14} {'Old Score':>10} {'New Score':>10} {'Delta':>8} {'S1 BT':>10} {'Config GT':>12}")
    print("-" * 90)

    results = []
    for r in data['results']:
        prog = r['program']
        config_name = CONFIG_NAME_MAP[prog]
        old_score = r['stage1']['eval_score']

        # Re-evaluate S1 with updated GT
        s1 = r['full_result']['stage1']
        new_eval = evaluator.evaluate(
            identified={
                "hotspots": s1["hotspots"],
                "bottleneck_type": s1["bottleneck_type"],
                "gpu_suitability": s1["gpu_suitability"],
                "optimization_suggestions": [],
            },
            benchmark_name=config_name,
            prompt_type="stage1_zero_shot"
        )
        new_score = new_eval.total_score
        delta = new_score - old_score

        s1_bt = s1.get("bottleneck_type", {}).get("primary", "?")

        entry = {
            "program": prog,
            "old_score": old_score,
            "new_score": new_score,
            "delta": delta,
            "s1_bottleneck": s1_bt,
        }
        results.append(entry)

        sign = "+" if delta > 0 else ""
        print(f"{prog:<14} {old_score:>10.1f} {new_score:>10.1f} {sign}{delta:>7.1f} {s1_bt:>10}")

    print("-" * 90)
    avg_old = sum(r['old_score'] for r in results) / len(results)
    avg_new = sum(r['new_score'] for r in results) / len(results)
    avg_delta = avg_new - avg_old
    sign = "+" if avg_delta > 0 else ""
    print(f"{'Average':<14} {avg_old:>10.1f} {avg_new:>10.1f} {sign}{avg_delta:>7.1f}")

    # Save
    output = {
        "description": "Rescore comparison after updating benchmark_config bottleneck_type to VTune GT",
        "results": results,
        "average_old": avg_old,
        "average_new": avg_new,
    }
    out_path = "results/extended_cascaded/rescore_comparison.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
