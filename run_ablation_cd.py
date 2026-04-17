"""
Ablation C / D: Extending the bias spectrum with GPT-5.4 as S2.

Configurations:
    Ablation C: S1=GPT-4o,  S2=GPT-5.4  — does GPT-5.4 correct GPT-4o's errors?
    Ablation D: S1=GPT-5.4, S2=GPT-5.4  — does GPT-5.4 self-validate differently?

2026-04-17 refactor: same changes as run_ablation.py —
    - Classification comes from ConfigurableCascadedPipeline.result['classification']
    - No hardcoded cross-config numbers
    - Resume support via per-program JSON check

Usage:
    python run_ablation_cd.py                  # run C and D on all 9
    python run_ablation_cd.py --config C
    python run_ablation_cd.py --programs minimd hotspot
    python run_ablation_cd.py --force
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

# Reuse the shared benchmark registry
from run_ablation import (
    BENCHMARK_SOURCES, CONFIG_NAME_MAP, run_on_program as run_one_program
)


ABLATION_CD_CONFIGS = {
    "C": {"s1_model": "gpt-4o",  "s2_model": "gpt-5.4", "label": "Ablation C (4o→5.4)"},
    "D": {"s1_model": "gpt-5.4", "s2_model": "gpt-5.4", "label": "Ablation D (5.4→5.4)"},
}


def main():
    parser = argparse.ArgumentParser(description="Ablation C/D (GPT-5.4 as S2)")
    parser.add_argument("--programs", nargs="+",
                        choices=list(BENCHMARK_SOURCES.keys()),
                        help="Specific programs (default: all 9)")
    parser.add_argument("--config", choices=["C", "D", "CD"], default="CD")
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result JSON already exists")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/ablation_cd.log', mode='a')
        ]
    )
    logger = logging.getLogger(__name__)
    
    programs = args.programs or list(BENCHMARK_SOURCES.keys())
    configs_to_run = list(args.config)
    
    register_extended_benchmarks()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    evaluator = GeneralizedEvaluator()
    
    for config_key in configs_to_run:
        cfg = ABLATION_CD_CONFIGS[config_key]
        logger.info(f"\n{'#'*60}")
        logger.info(f"Running {cfg['label']}: S1={cfg['s1_model']}, S2={cfg['s2_model']}")
        logger.info(f"{'#'*60}")
        
        pipeline = ConfigurableCascadedPipeline(
            s1_model=cfg['s1_model'],
            s2_model=cfg['s2_model']
        )
        
        config_dir = output_dir / f"ablation_{config_key}"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_results = {}
        for prog in programs:
            out_file = config_dir / f"{prog}_result.json"
            if out_file.exists() and not args.force:
                logger.info(f"[skip] {prog}: {out_file.name} exists (use --force to re-run)")
                config_results[prog] = json.load(open(out_file, encoding='utf-8'))
                continue
            
            summary = run_one_program(pipeline, evaluator, prog, cfg['label'], logger)
            if summary:
                config_results[prog] = summary
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        config_summary = {
            "config": cfg,
            "timestamp": datetime.now().isoformat(),
            "programs_tested": programs,
            "results": config_results,
        }
        with open(config_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(config_summary, f, indent=2, ensure_ascii=False, default=str)
        
        changes = sum(1 for r in config_results.values()
                      if r.get('classification', {}).get('changed'))
        correction = sum(1 for r in config_results.values()
                         if r.get('classification', {}).get('correction_type') == 'correction')
        total = len(config_results)
        total_cost = sum(r.get('cost', 0) for r in config_results.values())
        
        logger.info(f"\n{cfg['label']} summary:")
        logger.info(f"  changes: {changes}/{total}")
        logger.info(f"  of which corrections: {correction}")
        logger.info(f"  total cost: ${total_cost:.4f}")
    
    logger.info(f"\nAll results saved to: {output_dir}/")
    logger.info("Run rescore_all.py for canonical cross-config summary.")


if __name__ == "__main__":
    main()
