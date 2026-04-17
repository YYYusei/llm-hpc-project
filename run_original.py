"""
Original cascade configuration: S1=GPT-4o, S2=GPT-5.2 with the default
"validate/correct" prompt.

This is the baseline configuration against which all ablations (A/B/C/D)
and role-swap variants (V1/V3) are compared.

2026-04-17 refactor:
    - Replaces the previous test_extended_cascaded.py entry point.
    - Uses the same ConfigurableCascadedPipeline as ablations, so Original and
      ablations are truly identical except for S1/S2 model choice.
    - Output JSON schema matches ablations (results/ablation/ablation_original/).
      This way rescore_all.py can treat Original as just another config.
    - Resume support: skip programs that already have a result JSON.

Usage:
    python run_original.py                # run all 9 programs
    python run_original.py --programs minimd hotspot
    python run_original.py --force        # re-run even if result JSON exists
"""

import sys
import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from configurable_pipeline import ConfigurableCascadedPipeline
from generalized_evaluator import GeneralizedEvaluator
from extended_benchmark_config import register_extended_benchmarks
from run_ablation import BENCHMARK_SOURCES, CONFIG_NAME_MAP, run_on_program


ORIGINAL_CONFIG = {
    "s1_model": "gpt-4o",
    "s2_model": "gpt-5.2",
    "label": "Original (4o→5.2)",
}


def main():
    parser = argparse.ArgumentParser(description="Original cascade (S1=GPT-4o, S2=GPT-5.2)")
    parser.add_argument("--programs", nargs="+",
                        choices=list(BENCHMARK_SOURCES.keys()),
                        help="Specific programs (default: all 9)")
    parser.add_argument("--output-dir", default="results/ablation",
                        help="Output parent directory (config subdir 'ablation_original' will be created)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result JSON already exists")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/original.log', mode='a')
        ]
    )
    logger = logging.getLogger(__name__)
    
    programs = args.programs or list(BENCHMARK_SOURCES.keys())
    
    register_extended_benchmarks()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    evaluator = GeneralizedEvaluator()
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"Running {ORIGINAL_CONFIG['label']}")
    logger.info(f"{'#'*60}")
    
    pipeline = ConfigurableCascadedPipeline(
        s1_model=ORIGINAL_CONFIG['s1_model'],
        s2_model=ORIGINAL_CONFIG['s2_model']
    )
    
    config_dir = output_dir / "ablation_original"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_results = {}
    for prog in programs:
        out_file = config_dir / f"{prog}_result.json"
        if out_file.exists() and not args.force:
            logger.info(f"[skip] {prog}: result already exists (use --force to re-run)")
            config_results[prog] = json.load(open(out_file, encoding='utf-8'))
            continue
        
        summary = run_on_program(pipeline, evaluator, prog,
                                 ORIGINAL_CONFIG['label'], logger)
        if summary:
            config_results[prog] = summary
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    # Save summary
    config_summary = {
        "config": ORIGINAL_CONFIG,
        "timestamp": datetime.now().isoformat(),
        "programs_tested": programs,
        "results": config_results,
    }
    with open(config_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(config_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # Stats
    changes = sum(1 for r in config_results.values()
                  if r.get('classification', {}).get('changed'))
    correction = sum(1 for r in config_results.values()
                     if r.get('classification', {}).get('correction_type') == 'correction')
    total = len(config_results)
    total_cost = sum(r.get('cost', 0) for r in config_results.values())
    
    logger.info(f"\n{ORIGINAL_CONFIG['label']} summary:")
    logger.info(f"  changes: {changes}/{total}")
    logger.info(f"  of which corrections: {correction}")
    logger.info(f"  total cost: ${total_cost:.4f}")
    logger.info(f"\nResults saved to: {config_dir}/")
    logger.info("Run rescore_all.py for canonical cross-config summary.")


if __name__ == "__main__":
    main()
