"""
One-shot orchestrator: run all seven cascade configurations in sequence.

Configurations run:
    Original   : S1=GPT-4o,  S2=GPT-5.2  (standard validate/correct prompt)
    Ablation A : S1=GPT-5.2, S2=GPT-5.2
    Ablation B : S1=GPT-5.2, S2=GPT-4o
    Ablation C : S1=GPT-4o,  S2=GPT-5.4
    Ablation D : S1=GPT-5.4, S2=GPT-5.4
    Role-Swap V1: S1=GPT-4o, S2=GPT-5.2 (neutral prompt)
    Role-Swap V3: S1=GPT-4o, S2=GPT-5.2 (biased-agreement prompt)

Each configuration is idempotent (per-program JSON files are skipped if they
already exist). If the orchestrator is interrupted, re-running it resumes
from where it left off.

Total estimated cost (from prior runs): ~$4.42 across all configs.

Usage:
    python run_all_experiments.py                 # run everything
    python run_all_experiments.py --stages ABCD   # only A/B/C/D
    python run_all_experiments.py --force         # re-run everything from scratch
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


STAGES = {
    "O":  ("Original",        [sys.executable, "run_original.py"]),
    "A":  ("Ablation A",      [sys.executable, "run_ablation.py",    "--config", "A"]),
    "B":  ("Ablation B",      [sys.executable, "run_ablation.py",    "--config", "B"]),
    "C":  ("Ablation C",      [sys.executable, "run_ablation_cd.py", "--config", "C"]),
    "D":  ("Ablation D",      [sys.executable, "run_ablation_cd.py", "--config", "D"]),
    "V1": ("Role-Swap V1",    [sys.executable, "run_role_swap.py",   "--variant", "V1"]),
    "V3": ("Role-Swap V3",    [sys.executable, "run_role_swap.py",   "--variant", "V3"]),
}

DEFAULT_ORDER = ["O", "A", "B", "C", "D", "V1", "V3"]


def run_stage(key, label, cmd, extra_args, logger):
    full_cmd = cmd + extra_args
    logger.info("")
    logger.info("=" * 72)
    logger.info(f"Stage {key}: {label}")
    logger.info(f"Command: {' '.join(full_cmd)}")
    logger.info("=" * 72)
    
    result = subprocess.run(full_cmd)
    if result.returncode != 0:
        logger.error(f"Stage {key} ({label}) FAILED with exit code {result.returncode}")
        return False
    logger.info(f"Stage {key} ({label}) completed.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all seven cascade configurations")
    parser.add_argument("--stages", default="OABCDV1V3",
                        help="Which stages to run. Use the keys: O, A, B, C, D, V1, V3. "
                             "Default: OABCDV1V3 (all seven). "
                             "V1/V3 are two-character keys and parsed left-to-right.")
    parser.add_argument("--programs", nargs="+", default=None,
                        help="Limit to specific programs (passed through to each stage)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if per-program result JSONs exist")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop orchestration if any stage fails (default: continue)")
    args = parser.parse_args()
    
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/run_all_experiments.log', mode='a')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Parse the --stages string into ordered keys.
    # V1 and V3 are two characters — consume them as pairs when seen.
    raw = args.stages
    order = []
    i = 0
    while i < len(raw):
        if raw[i] == 'V' and i + 1 < len(raw) and raw[i+1] in ('1', '3'):
            order.append(raw[i:i+2])
            i += 2
        else:
            order.append(raw[i])
            i += 1
    
    for k in order:
        if k not in STAGES:
            logger.error(f"Unknown stage key: {k!r}. Valid: {list(STAGES.keys())}")
            sys.exit(2)
    
    # Extra args to pass through
    extra = []
    if args.force:
        extra.append("--force")
    if args.programs:
        extra.append("--programs")
        extra.extend(args.programs)
    
    logger.info(f"Orchestrator start: {datetime.now().isoformat()}")
    logger.info(f"Stages to run: {order}")
    
    successes, failures = [], []
    for k in order:
        label, cmd = STAGES[k]
        ok = run_stage(k, label, cmd, extra, logger)
        (successes if ok else failures).append(k)
        if not ok and args.stop_on_error:
            logger.error("Stopping orchestration due to --stop-on-error.")
            break
    
    logger.info("")
    logger.info("=" * 72)
    logger.info("ORCHESTRATOR SUMMARY")
    logger.info("=" * 72)
    logger.info(f"  Completed: {successes}")
    if failures:
        logger.warning(f"  Failed:    {failures}")
    logger.info("")
    logger.info("Next step: python rescore_all.py")
    logger.info("(reads all per-program JSON results and generates the pf-unified summary)")


if __name__ == "__main__":
    main()
