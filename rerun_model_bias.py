"""
Re-run the single-stage model-bias test with full result capture.

Motivation:
    The original `test_model_bias.py` saved only `s1_bottleneck` and
    `eval_score`, not the full `hotspots` array. That makes post-hoc
    rescoring impossible without re-running the API. This script produces
    the same experimental measurements but saves the complete S1 output
    (including hotspots), so that rescore_eval_all.py can later re-evaluate
    against updated ground truth without additional API calls.

Usage:
    cd llm-hpc-project
    python rerun_model_bias.py --model gpt-5.4
    python rerun_model_bias.py --model gpt-5.4 --programs minimd hotspot
    python rerun_model_bias.py --model gpt-5.4 --force   # re-run even if JSON exists

Cost estimate (9 programs):
    gpt-5.4:  ~$0.28

Output:
    results/model_bias_v2/{model}/{prog}_result.json      per-program
    results/model_bias_v2/{model}/summary.json            per-model summary

The JSON schema matches what rescore_eval_all.py expects, so after running
this script you can immediately run:
    python rescore_eval_all.py
and the new numbers will appear (model_bias_v2 is NOT in CONFIGS today —
you can either add it, or I will add it as part of the thesis_v8 package).
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyzer import HPCAnalyzer
from generalized_evaluator import GeneralizedEvaluator
from extended_benchmark_config import register_extended_benchmarks
from bottleneck_taxonomy import primary, GROUND_TRUTH as TAXONOMY_GT


# ==================================================================
# Reuse the BENCHMARK_SOURCES and CONFIG_NAME_MAP from run_ablation.py
# so this script is consistent with the rest of the refactored pipeline.
# ==================================================================
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


def run_one(analyzer, evaluator, prog, logger):
    source_path = BENCHMARK_SOURCES[prog]
    config_name = CONFIG_NAME_MAP[prog]
    gt = TAXONOMY_GT[prog]

    if not os.path.exists(source_path):
        logger.error(f"Source not found: {source_path}")
        return None

    logger.info(f"\n{'='*50}")
    logger.info(f"[single-stage] {prog}")
    logger.info(f"{'='*50}")

    try:
        result = analyzer.analyze(
            code_path=source_path,
            code_name=config_name,
            prompt_type="zero_shot"
        )

        s1_bt_raw = (
            result.bottleneck_type.get("primary", "")
            if isinstance(result.bottleneck_type, dict)
            else str(result.bottleneck_type or "")
        )
        s1_reasoning = (
            result.bottleneck_type.get("reasoning", "")
            if isinstance(result.bottleneck_type, dict)
            else ""
        )
        s1_primary = primary(s1_bt_raw)

        # Evaluate with current (VTune-aligned) GT
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

        # Save full S1 output so rescore_eval_all.py can re-evaluate later
        model_name = analyzer.client.model
        entry = {
            "program": prog,
            "model": model_name,  # e.g. "gpt-5.4"
            "s1_bottleneck_raw": s1_bt_raw,
            "s1_primary": s1_primary,
            "s1_reasoning": s1_reasoning[:500],
            "ground_truth": gt,
            "correct": s1_primary == gt,
            "eval_score": eval_result.total_score,
            "cost": result.cost,
            # Key addition: full S1 output for rescore
            "stage1": {
                "model": model_name,
                "hotspots": result.hotspots,
                "bottleneck_type": result.bottleneck_type,
                "gpu_suitability": result.gpu_suitability,
                "cost": result.cost,
                "bottleneck_raw": s1_bt_raw,
            },
        }
        mark = "✓" if entry["correct"] else "✗"
        logger.info(f"  S1={s1_primary} | GT={gt} | {mark} | score={eval_result.total_score:.1f} | ${result.cost:.4f}")
        return entry

    except Exception as e:
        logger.error(f"Error on {prog}: {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Re-run single-stage model bias test with full output capture")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to test (gpt-4o / gpt-5.2 / gpt-5.4)")
    parser.add_argument("--programs", nargs="+",
                        choices=list(BENCHMARK_SOURCES.keys()),
                        help="Programs to test (default: all 9)")
    parser.add_argument("--output-dir", type=str, default="results/model_bias_v2",
                        help="Output directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if per-program result JSON exists")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/model_bias_v2.log', mode='a')
        ]
    )
    logger = logging.getLogger(__name__)

    programs = args.programs or list(BENCHMARK_SOURCES.keys())
    register_extended_benchmarks()

    output_dir = Path(args.output_dir) / args.model.replace(".", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {args.model}")
    logger.info(f"Programs: {programs}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    analyzer = HPCAnalyzer(model=args.model)
    evaluator = GeneralizedEvaluator()

    results = []
    total_cost = 0
    for prog in programs:
        out_file = output_dir / f"{prog}_result.json"
        if out_file.exists() and not args.force:
            logger.info(f"[skip] {prog}: result already exists (use --force to re-run)")
            try:
                results.append(json.load(open(out_file, encoding='utf-8')))
            except Exception:
                pass
            continue
        entry = run_one(analyzer, evaluator, prog, logger)
        if entry is None:
            continue
        total_cost += entry.get("cost", 0)
        results.append(entry)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=2, ensure_ascii=False, default=str)

    # Summary
    summary = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "programs_tested": programs,
        "n_correct": sum(1 for r in results if r.get("correct")),
        "n_total": len(results),
        "total_cost": round(total_cost, 4),
        "results": results,
    }
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Print compact stdout summary
    print("\n" + "="*72)
    print(f"MODEL-BIAS V2 SUMMARY: {args.model}")
    print("="*72)
    print(f"{'Program':<14} {'S1 primary':>12} {'GT':>8} {'Match':>6} {'Score':>7}")
    print("-"*72)
    for r in results:
        mark = "✓" if r.get("correct") else "✗"
        print(f"{r['program']:<14} {r['s1_primary']:>12} {r['ground_truth']:>8} {mark:>6} {r['eval_score']:>7.1f}")
    print("-"*72)
    print(f"Accuracy: {summary['n_correct']}/{summary['n_total']}")
    print(f"Total cost: ${summary['total_cost']:.4f}")
    print("="*72)
    print(f"\nResults saved to: {output_dir}/")
    print("Next: (optional) extend rescore_eval_all.py to include this config,")
    print("      or just read the saved scores directly from {prog}_result.json.")


if __name__ == "__main__":
    main()