"""
Ablation A / B: Isolate pipeline structure vs model capability.

Configurations (only 9-program bottleneck-classification task, no GPU code gen):
    Ablation A: S1=GPT-5.2, S2=GPT-5.2  — does S2 need S1 context, or is model enough?
    Ablation B: S1=GPT-5.2, S2=GPT-4o   — is GPT-5.2 as S2 essential?

2026-04-17 refactor:
    - Classification ("change" / "correction") comes from ConfigurableCascadedPipeline
      via the result['classification'] field. No per-script primary() logic.
    - No hardcoded Original numbers — Original is a separate script (run_original.py)
      and cross-config comparison is produced by rescore_all.py.
    - Output JSON follows the unified schema defined by ConfigurableCascadedPipeline.
    - Resume support: if a per-program result file already exists, skip
      (pass --force to re-run).

Usage:
    python run_ablation.py                      # run A and B on all 9 programs
    python run_ablation.py --config A           # only Ablation A
    python run_ablation.py --programs minimd hotspot
    python run_ablation.py --force              # re-run even if result JSON exists
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


# ==================================================================
# Experiment registry — shared program metadata
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

# Config name used for prompt building / benchmark registry
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

ABLATION_CONFIGS = {
    "A": {"s1_model": "gpt-5.2", "s2_model": "gpt-5.2", "label": "Ablation A (5.2→5.2)"},
    "B": {"s1_model": "gpt-5.2", "s2_model": "gpt-4o",  "label": "Ablation B (5.2→4o)"},
}


# ==================================================================
# Per-program runner
# ==================================================================

def run_on_program(pipeline, evaluator, program_key, config_label, logger):
    """Run the pipeline on one program and return a JSON-ready summary dict."""
    source_path = BENCHMARK_SOURCES[program_key]
    config_name = CONFIG_NAME_MAP[program_key]
    
    if not os.path.exists(source_path):
        logger.error(f"Source file not found: {source_path}")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"[{config_label}] Program: {program_key}")
    logger.info(f"{'='*60}")
    
    try:
        result = pipeline.analyze(
            code_path=source_path,
            code_name=config_name,
            prompt_type="zero_shot",
            program_key=program_key,
        )
        
        # Stage 1 evaluation (composite score against benchmark config)
        stage1_eval = evaluator.evaluate(
            identified={
                "hotspots": result["stage1"]["hotspots"],
                "bottleneck_type": result["stage1"]["bottleneck_type"],
                "gpu_suitability": result["stage1"]["gpu_suitability"],
                "optimization_suggestions": [],
            },
            benchmark_name=config_name,
            prompt_type="stage1_zero_shot"
        )
        
        cls = result["classification"]
        
        summary = {
            "program": program_key,
            "config": config_label,
            "s1_model": result["stage1"]["model"],
            "s2_model": result["stage2"]["model"],
            "s1_bottleneck_raw": result["stage1"]["bottleneck_raw"],
            "s2_bottleneck_raw": result["stage2"]["bottleneck_raw"],
            "s1_score": stage1_eval.total_score,
            "classification": cls,
            "cost": result.get("total_cost", 0),
            "full_result": result,
        }
        
        logger.info(f"  S1 ({result['stage1']['model']}): "
                    f"{cls['s1_primary']} (score: {summary['s1_score']:.1f})")
        logger.info(f"  S2 ({result['stage2']['model']}): "
                    f"{cls['s2_primary']} | type={cls['correction_type']} | "
                    f"changed={cls['changed']}")
        logger.info(f"  GT={cls['ground_truth']}  |  Cost: ${summary['cost']:.4f}")
        
        return summary
    
    except Exception as e:
        logger.error(f"Error processing {program_key}: {e}", exc_info=True)
        return None


# ==================================================================
# Main
# ==================================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation A/B: varying S1/S2 model pairing")
    parser.add_argument("--programs", nargs="+",
                        choices=list(BENCHMARK_SOURCES.keys()),
                        help="Specific programs to test (default: all 9)")
    parser.add_argument("--config", choices=["A", "B", "AB"], default="AB",
                        help="Which ablation: A, B, or AB (default)")
    parser.add_argument("--output-dir", default="results/ablation",
                        help="Output directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if per-program result JSON already exists")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/ablation.log', mode='a')
        ]
    )
    logger = logging.getLogger(__name__)
    
    programs = args.programs or list(BENCHMARK_SOURCES.keys())
    configs_to_run = list(args.config)  # "AB" → ["A", "B"]
    
    register_extended_benchmarks()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    evaluator = GeneralizedEvaluator()
    
    for config_key in configs_to_run:
        cfg = ABLATION_CONFIGS[config_key]
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
                logger.info(f"[skip] {prog}: {out_file.name} already exists (use --force to re-run)")
                config_results[prog] = json.load(open(out_file, encoding='utf-8'))
                continue
            
            summary = run_on_program(pipeline, evaluator, prog, cfg['label'], logger)
            if summary:
                config_results[prog] = summary
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Save per-config summary
        config_summary = {
            "config": cfg,
            "timestamp": datetime.now().isoformat(),
            "programs_tested": programs,
            "results": config_results,
        }
        with open(config_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(config_summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Quick per-config stats (using taxonomy classification)
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
    logger.info("Run rescore_all.py to generate canonical cross-config summary.")


if __name__ == "__main__":
    main()
