"""
GPT-5.4 单阶段瓶颈分类测试
只跑 S1（不跑 S2，不跑 GPU 生成），收集瓶颈分类结果

用法:
    cd llm-hpc-project
    python test_model_bias.py --model gpt-5.4
    python test_model_bias.py --model gpt-5.4 --programs minimd hotspot srad lulesh

也可以用来重跑其他模型:
    python test_model_bias.py --model gpt-4o
    python test_model_bias.py --model gpt-5.2
"""

import sys
import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyzer import HPCAnalyzer
from generalized_evaluator import GeneralizedEvaluator
from extended_benchmark_config import register_extended_benchmarks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/model_bias_test.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

BENCHMARK_SOURCES = {
    "minimd":     "benchmarks/minimd/force_lj.cpp",
    "hpcg_spmv":  "benchmarks/hpcg/ComputeSPMV_ref.cpp",
    "hpcg_symgs": "benchmarks/hpcg/ComputeSYMGS_ref.cpp",
    "abinit":     "benchmarks/abinit/m_nonlop_ylm.F90",
    "hotspot":    "benchmarks/hotspot/hotspot.c",
    "srad":       "benchmarks/srad/srad.c",
    "lulesh":     "benchmarks/lulesh/lulesh_simplified.c",
    "nas_cg":     "benchmarks/nas_cg/cg.c",
    "jacobi2d":   "benchmarks/jacobi2d/jacobi2d.c",
}

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

# VTune-aligned ground truth
GROUND_TRUTH = {
    "minimd":     "compute",
    "hpcg_spmv":  "memory",
    "hpcg_symgs": "memory",
    "abinit":     "memory",
    "hotspot":    "memory",
    "srad":       "memory",
    "lulesh":     "memory",
    "nas_cg":     "memory",
    "jacobi2d":   "memory",
}


def main():
    parser = argparse.ArgumentParser(description="Model Bias Test - S1 only")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to test (e.g. gpt-4o, gpt-5.2, gpt-5.4)")
    parser.add_argument("--programs", nargs="+",
                        choices=list(BENCHMARK_SOURCES.keys()),
                        help="Programs to test (default: all 9)")
    parser.add_argument("--output-dir", type=str, default="results/model_bias",
                        help="Output directory")
    args = parser.parse_args()

    programs = args.programs or list(BENCHMARK_SOURCES.keys())
    model = args.model

    register_extended_benchmarks()
    os.makedirs("logs", exist_ok=True)

    output_dir = Path(args.output_dir) / model.replace(".", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Testing model: {model}")
    logger.info(f"Programs: {programs}")

    analyzer = HPCAnalyzer(model=model)
    evaluator = GeneralizedEvaluator()

    results = []
    total_cost = 0

    for prog in programs:
        source_path = BENCHMARK_SOURCES[prog]
        config_name = CONFIG_NAME_MAP[prog]
        gt = GROUND_TRUTH[prog]

        if not os.path.exists(source_path):
            logger.error(f"Source not found: {source_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"[{model}] {prog}")
        logger.info(f"{'='*50}")

        try:
            result = analyzer.analyze(
                code_path=source_path,
                code_name=config_name,
                prompt_type="zero_shot"
            )

            s1_bt = result.bottleneck_type.get("primary", "unknown") if isinstance(result.bottleneck_type, dict) else str(result.bottleneck_type)
            s1_reasoning = result.bottleneck_type.get("reasoning", "") if isinstance(result.bottleneck_type, dict) else ""
            cost = result.cost
            total_cost += cost

            # Evaluate
            eval_result = evaluator.evaluate(
                identified={
                    "hotspots": result.hotspots,
                    "bottleneck_type": result.bottleneck_type,
                    "gpu_suitability": result.gpu_suitability,
                    "optimization_suggestions": [],
                },
                benchmark_name=config_name,
                prompt_type="zero_shot"
            )

            correct = s1_bt == gt
            entry = {
                "program": prog,
                "model": model,
                "s1_bottleneck": s1_bt,
                "s1_reasoning": s1_reasoning[:200],
                "ground_truth": gt,
                "correct": correct,
                "eval_score": eval_result.total_score,
                "cost": cost,
            }
            results.append(entry)

            mark = "✓" if correct else "✗"
            logger.info(f"  S1={s1_bt} | GT={gt} | {mark} | score={eval_result.total_score:.1f} | ${cost:.4f}")

            # Save per-program
            with open(output_dir / f"{prog}_result.json", 'w', encoding='utf-8') as f:
                json.dump(entry, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"MODEL BIAS TEST: {model}")
    print(f"{'='*70}")
    print(f"{'Program':<14} {'S1 BT':>10} {'GT':>10} {'Match':>6} {'Score':>8}")
    print("-" * 70)

    correct_count = 0
    compute_count = 0
    memory_count = 0

    for r in results:
        mark = "✓" if r['correct'] else "✗"
        print(f"{r['program']:<14} {r['s1_bottleneck']:>10} {r['ground_truth']:>10} {mark:>6} {r['eval_score']:>8.1f}")
        if r['correct']:
            correct_count += 1
        if r['s1_bottleneck'] == 'compute':
            compute_count += 1
        elif r['s1_bottleneck'] == 'memory':
            memory_count += 1

    total = len(results)
    print("-" * 70)
    print(f"Accuracy: {correct_count}/{total}")
    print(f"Predicted compute: {compute_count}/{total}")
    print(f"Predicted memory:  {memory_count}/{total}")
    print(f"Total cost: ${total_cost:.4f}")

    # Save summary
    summary = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "programs_tested": programs,
        "accuracy": f"{correct_count}/{total}",
        "compute_predictions": compute_count,
        "memory_predictions": memory_count,
        "total_cost": total_cost,
        "results": results,
    }
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
