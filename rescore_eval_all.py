"""
Re-score all S1 evaluation scores using the updated VTune-aligned benchmark config.

2026-04-17 (problem #2):
    After updating extended_benchmark_config.py to use VTune-measured
    time_percentage values, every stored S1 `eval_score` in per-program result
    JSONs is now stale — it was computed against the old (estimated)
    time_percentage GT.

    This script re-runs the evaluator on the S1 hotspots stored in each
    per-program JSON, using the NEW (VTune-aligned) GT. No API calls.

    It runs the re-scoring TWICE — once with the default evaluator tolerance
    of 0.3, and once with a stricter 0.15 — so you can compare.

Inputs (read-only):
    results/ablation/ablation_original/{prog}_result.json
    results/ablation/ablation_A/{prog}_result.json
    results/ablation/ablation_B/{prog}_result.json
    results/ablation/ablation_C/{prog}_result.json
    results/ablation/ablation_D/{prog}_result.json
    results/role_swap/V1/{prog}_result.json
    results/role_swap/V3/{prog}_result.json
    (falls back to results/extended_cascaded/* for legacy Original data)

Outputs:
    results/pf_summary/eval_rescore_comparison.md
    results/pf_summary/eval_rescore.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from extended_benchmark_config import register_extended_benchmarks
from generalized_evaluator import GeneralizedEvaluator

register_extended_benchmarks()


PROGRAMS = ['minimd', 'hpcg_spmv', 'hpcg_symgs', 'abinit',
            'hotspot', 'srad', 'lulesh', 'nas_cg', 'jacobi2d']


# Canonical config -> (label, candidate dirs, filename templates)
# Same schema as rescore_all.py
CONFIGS = [
    ("original", "Original (4o→5.2)",
        ["results/ablation/ablation_original", "results/extended_cascaded"]),
    ("A", "Ablation A (5.2→5.2)", ["results/ablation/ablation_A"]),
    ("B", "Ablation B (5.2→4o)",  ["results/ablation/ablation_B"]),
    ("C", "Ablation C (4o→5.4)",  ["results/ablation/ablation_C"]),
    ("D", "Ablation D (5.4→5.4)", ["results/ablation/ablation_D"]),
    ("V1", "Role-Swap V1 Neutral", ["results/role_swap/V1"]),
    ("V3", "Role-Swap V3 Biased",  ["results/role_swap/V3"]),
    # Single-stage model-bias tests (v2 with full S1 output capture)
    ("MB-5.4", "Model-bias single-stage GPT-5.4", ["results/model_bias_v2/gpt-5_4"]),
]

FILENAME_TEMPLATES = ["{prog}_result.json", "{prog}_cascaded_result.json"]


# Map program_key to the name expected by the benchmark registry.
# hpcg_spmv / hpcg_symgs both use the 'hpcg' benchmark definition but we
# want them as separate rows in the output.
def benchmark_name_for(program_key):
    if program_key in ('hpcg_spmv', 'hpcg_symgs'):
        return 'hpcg'
    return program_key


def _resolve_file(candidate_dirs, prog):
    for d in candidate_dirs:
        dp = Path(d)
        if not dp.is_dir():
            continue
        for tmpl in FILENAME_TEMPLATES:
            f = dp / tmpl.format(prog=prog)
            if f.exists():
                return f
    return None


def _extract_s1_hotspots(data):
    """
    Pull S1 hotspots out of a per-program JSON, handling both new schema
    (result['stage1']['hotspots']) and legacy schemas.
    """
    # Modern: top-level stage1 (as in extended_cascaded format)
    s1 = data.get('stage1')
    if isinstance(s1, dict):
        hs = s1.get('hotspots')
        if hs is not None:
            return s1.get('hotspots', []), s1.get('bottleneck_type', {}), s1.get('gpu_suitability', {})
    
    # Modern: ablation/role_swap schema — full_result.stage1
    full = data.get('full_result', {})
    s1 = full.get('stage1')
    if isinstance(s1, dict):
        return s1.get('hotspots', []), s1.get('bottleneck_type', {}), s1.get('gpu_suitability', {})
    
    return None, None, None


def rescore_config(label, candidate_dirs, evaluator, logger):
    """Re-score one config. Returns per-program dict of new S1 scores."""
    per_prog = {}
    for prog in PROGRAMS:
        fp = _resolve_file(candidate_dirs, prog)
        if fp is None:
            per_prog[prog] = None
            continue
        try:
            data = json.load(open(fp, encoding='utf-8'))
        except Exception as e:
            logger.warning(f"  {prog}: parse error: {e}")
            per_prog[prog] = None
            continue
        
        hs, bt, gpu = _extract_s1_hotspots(data)
        if hs is None:
            logger.warning(f"  {prog}: no S1 hotspots in {fp.name}")
            per_prog[prog] = None
            continue
        
        # Find old score for comparison
        old_score = None
        # extended_cascaded format
        if isinstance(data.get('stage1'), dict):
            old_score = data['stage1'].get('eval_score')
        # ablation/role_swap format
        if old_score is None:
            old_score = data.get('s1_score')
        
        # Re-evaluate using NEW GT
        result = evaluator.evaluate(
            identified={
                'hotspots': hs,
                'bottleneck_type': bt if isinstance(bt, dict) else {'primary': str(bt or '')},
                'gpu_suitability': gpu or {},
                'optimization_suggestions': [],
            },
            benchmark_name=benchmark_name_for(prog),
            prompt_type="s1_rescore",
        )
        
        per_prog[prog] = {
            'old_score': round(old_score, 2) if old_score is not None else None,
            'new_score': round(result.total_score, 2),
            'hotspot_score': round(result.hotspot_score, 3),
            'bottleneck_score': round(result.bottleneck_score, 3),
            'gpu_score': round(result.gpu_score, 3),
            'details_score': round(result.details_score, 3),
        }
    return per_prog


def run_with_tolerance(tolerance, logger):
    """Run re-score across all configs with one tolerance setting."""
    evaluator = GeneralizedEvaluator(tolerance=tolerance)
    out = {}
    for (key, label, candidate_dirs) in CONFIGS:
        logger.info(f"[tolerance={tolerance}] Re-scoring {label}...")
        out[key] = {
            'label': label,
            'per_program': rescore_config(label, candidate_dirs, evaluator, logger),
        }
    return out


def _fmt_score(v):
    if v is None:
        return '—'
    return f"{v:.1f}"


def _fmt_delta(old, new):
    if old is None or new is None:
        return '—'
    d = new - old
    if abs(d) < 0.05:
        return "0.0"
    return f"{d:+.1f}"


def write_markdown(data_03, data_15, out_path):
    lines = []
    lines.append("# S1 Eval Score Re-scoring after VTune Alignment")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("**Context:** `extended_benchmark_config.py` `time_percentage` values")
    lines.append("have been re-aligned to VTune-measured values (thesis §1.4).")
    lines.append("This document shows the impact on S1 eval scores for every program")
    lines.append("in every config, under two evaluator tolerance settings.")
    lines.append("")
    lines.append("- **old_score**: S1 eval score as it was stored in the result JSON")
    lines.append("  (computed against the OLD, estimated `time_percentage` GT).")
    lines.append("- **new_score (tol=0.3)**: same inputs, re-evaluated with NEW VTune GT")
    lines.append("  and the default evaluator tolerance of 0.3.")
    lines.append("- **new_score (tol=0.15)**: same inputs, re-evaluated with NEW VTune GT")
    lines.append("  and stricter tolerance of 0.15.")
    lines.append("")

    # Per config, side-by-side comparison
    for (key, label, _) in CONFIGS:
        p03 = data_03[key]['per_program']
        p15 = data_15[key]['per_program']
        
        lines.append(f"## {label}")
        lines.append("")
        lines.append(
            "| Program | Old score | New score (tol=0.3) | Δ | New score (tol=0.15) | Δ |"
        )
        lines.append(
            "|---------|:---------:|:-------------------:|:-:|:--------------------:|:-:|"
        )
        for prog in PROGRAMS:
            r03 = p03.get(prog) or {}
            r15 = p15.get(prog) or {}
            old = r03.get('old_score')
            new3 = r03.get('new_score')
            new15 = r15.get('new_score')
            d3 = _fmt_delta(old, new3)
            d15 = _fmt_delta(old, new15)
            lines.append(
                f"| {prog} | {_fmt_score(old)} | **{_fmt_score(new3)}** | {d3} "
                f"| **{_fmt_score(new15)}** | {d15} |"
            )
        
        # averages
        def avg(scores):
            vals = [s for s in scores if s is not None]
            return sum(vals) / len(vals) if vals else None
        
        old_vals = [p03[p]['old_score'] for p in PROGRAMS if p03.get(p) and p03[p].get('old_score') is not None]
        new3_vals = [p03[p]['new_score'] for p in PROGRAMS if p03.get(p) and p03[p].get('new_score') is not None]
        new15_vals = [p15[p]['new_score'] for p in PROGRAMS if p15.get(p) and p15[p].get('new_score') is not None]
        old_avg = avg(old_vals)
        new3_avg = avg(new3_vals)
        new15_avg = avg(new15_vals)
        lines.append(
            f"| **average** | {_fmt_score(old_avg)} | **{_fmt_score(new3_avg)}** "
            f"| {_fmt_delta(old_avg, new3_avg)} | **{_fmt_score(new15_avg)}** "
            f"| {_fmt_delta(old_avg, new15_avg)} |"
        )
        lines.append("")

    # Comparison headline
    lines.append("## Overall impact summary")
    lines.append("")
    lines.append("| Config | Avg old | Avg new (tol=0.3) | Δ0.3 | Avg new (tol=0.15) | Δ0.15 |")
    lines.append("|--------|:-------:|:-----------------:|:----:|:------------------:|:-----:|")
    for (key, label, _) in CONFIGS:
        p03 = data_03[key]['per_program']
        p15 = data_15[key]['per_program']
        old_vals = [p03[p]['old_score'] for p in PROGRAMS if p03.get(p) and p03[p].get('old_score') is not None]
        new3_vals = [p03[p]['new_score'] for p in PROGRAMS if p03.get(p) and p03[p].get('new_score') is not None]
        new15_vals = [p15[p]['new_score'] for p in PROGRAMS if p15.get(p) and p15[p].get('new_score') is not None]
        o = sum(old_vals) / len(old_vals) if old_vals else None
        n3 = sum(new3_vals) / len(new3_vals) if new3_vals else None
        n15 = sum(new15_vals) / len(new15_vals) if new15_vals else None
        lines.append(
            f"| {label} | {_fmt_score(o)} | **{_fmt_score(n3)}** "
            f"| {_fmt_delta(o, n3)} | **{_fmt_score(n15)}** | {_fmt_delta(o, n15)} |"
        )

    out_path.write_text("\n".join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Re-score S1 eval scores with VTune-aligned GT")
    parser.add_argument("--output-dir", default="results/pf_summary")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 72)
    logger.info("S1 eval score re-scoring with VTune-aligned GT")
    logger.info("=" * 72)
    logger.info("")
    
    data_03 = run_with_tolerance(0.3, logger)
    logger.info("")
    data_15 = run_with_tolerance(0.15, logger)
    
    # Write JSON
    with open(out_dir / "eval_rescore.json", 'w', encoding='utf-8') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'tolerance_03': data_03,
            'tolerance_15': data_15,
        }, f, indent=2, ensure_ascii=False)
    
    # Write markdown
    write_markdown(data_03, data_15, out_dir / "eval_rescore_comparison.md")
    
    logger.info("")
    logger.info(f"Written: {out_dir / 'eval_rescore.json'}")
    logger.info(f"Written: {out_dir / 'eval_rescore_comparison.md'}")
    
    # Print a concise summary to stdout
    print("\n" + "=" * 90)
    print("S1 EVAL SCORE — RE-SCORING SUMMARY")
    print("=" * 90)
    print(f"{'Config':<25} {'avg_old':>8} {'avg_new(0.3)':>13} {'Δ0.3':>6} {'avg_new(0.15)':>14} {'Δ0.15':>7}")
    print("-" * 90)
    for (key, label, _) in CONFIGS:
        p03 = data_03[key]['per_program']
        p15 = data_15[key]['per_program']
        old_vals = [p03[p]['old_score'] for p in PROGRAMS if p03.get(p) and p03[p].get('old_score') is not None]
        new3_vals = [p03[p]['new_score'] for p in PROGRAMS if p03.get(p) and p03[p].get('new_score') is not None]
        new15_vals = [p15[p]['new_score'] for p in PROGRAMS if p15.get(p) and p15[p].get('new_score') is not None]
        o = sum(old_vals) / len(old_vals) if old_vals else None
        n3 = sum(new3_vals) / len(new3_vals) if new3_vals else None
        n15 = sum(new15_vals) / len(new15_vals) if new15_vals else None
        def f(v): return f"{v:.1f}" if v is not None else "—"
        d3 = (n3 - o) if (o is not None and n3 is not None) else None
        d15 = (n15 - o) if (o is not None and n15 is not None) else None
        def d(v): return f"{v:+.1f}" if v is not None else "—"
        print(f"{label:<25} {f(o):>8} {f(n3):>13} {d(d3):>6} {f(n15):>14} {d(d15):>7}")
    print("=" * 90)


if __name__ == "__main__":
    main()
