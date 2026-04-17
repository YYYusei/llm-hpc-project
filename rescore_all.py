"""
Aggregate all per-program result JSONs into a canonical cross-config summary.

Reads from:
    results/ablation/ablation_original/{prog}_result.json
    results/ablation/ablation_A/{prog}_result.json
    results/ablation/ablation_B/{prog}_result.json
    results/ablation/ablation_C/{prog}_result.json
    results/ablation/ablation_D/{prog}_result.json
    results/role_swap/V1/{prog}_result.json
    results/role_swap/V3/{prog}_result.json

Every per-program JSON is assumed to have been produced by
ConfigurableCascadedPipeline and therefore contains a `classification` block
(via bottleneck_taxonomy). This script does NOT re-implement the primary()
logic — it trusts the classification already computed at generation time.

If a JSON does NOT have a classification block (legacy data from before the
2026-04-17 refactor), this script treats it as missing and warns. Re-run the
corresponding experiment to regenerate.

Writes:
    results/pf_summary/summary_table.json
    results/pf_summary/summary_table.md
    results/pf_summary/per_config_details.json
"""

import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from bottleneck_taxonomy import GROUND_TRUTH, primary, classify


PROGRAMS = ['minimd', 'hpcg_spmv', 'hpcg_symgs', 'abinit',
            'hotspot', 'srad', 'lulesh', 'nas_cg', 'jacobi2d']


# Single registry of experiment configurations, mirrored across run_*.py files.
# (config_key, label, list-of-candidate-directories, s1_model, s2_model)
# The list-of-candidate-directories is searched in order; the first existing
# one wins. This allows rescore_all.py to read both (a) new results from the
# refactored pipeline in `results/ablation/ablation_original/` and (b) legacy
# results from `results/extended_cascaded/` without modification.
CONFIGS = [
    ("original", "Original (4o→5.2)",
        ["results/ablation/ablation_original", "results/extended_cascaded"],
        "gpt-4o",  "gpt-5.2"),
    ("A",        "Ablation A (5.2→5.2)",       ["results/ablation/ablation_A"], "gpt-5.2", "gpt-5.2"),
    ("B",        "Ablation B (5.2→4o)",        ["results/ablation/ablation_B"], "gpt-5.2", "gpt-4o"),
    ("C",        "Ablation C (4o→5.4)",        ["results/ablation/ablation_C"], "gpt-4o",  "gpt-5.4"),
    ("D",        "Ablation D (5.4→5.4)",       ["results/ablation/ablation_D"], "gpt-5.4", "gpt-5.4"),
    ("V1",       "Role-Swap V1 Neutral",       ["results/role_swap/V1"],        "gpt-4o",  "gpt-5.2"),
    ("V3",       "Role-Swap V3 Biased",        ["results/role_swap/V3"],        "gpt-4o",  "gpt-5.2"),
]


def _resolve_dir(candidate_dirs):
    """Pick the first existing directory from candidates. Returns (path, fallback_used)."""
    for d in candidate_dirs:
        p = Path(d)
        if p.exists() and p.is_dir():
            return p, (candidate_dirs[0] != d)
    return Path(candidate_dirs[0]), False  # none exist; summarise will return empty


# ==================================================================
# Legacy JSON support
# ==================================================================

def _extract_legacy_s1_s2(data, program_key):
    """
    Given a legacy JSON (produced by run_ablation.py / run_role_swap.py before
    the 2026-04-17 refactor), recover raw S1 and S2 bottleneck strings.
    
    Returns (s1_raw, s2_raw) or (None, None) if structure is unrecognised.
    """
    # Variant A — extended_cascaded-style: top-level stage1/stage2 with
    # stage1.bottleneck_type (str) and stage2.bottleneck_corrected / corrected_type.
    if 'stage1' in data and 'stage2' in data:
        s1 = data['stage1'].get('bottleneck_type', '')
        if isinstance(s1, dict):
            s1 = s1.get('primary', '') or ''
        s2_info = data['stage2']
        if 'bottleneck_corrected' in s2_info:
            if s2_info.get('bottleneck_corrected'):
                s2 = s2_info.get('corrected_type', s1)
            else:
                s2 = s1
            return s1, s2 or s1
        # Alternatively, configurable_pipeline-style with validation block
        val = s2_info.get('validation', {})
        if val:
            if val.get('bottleneck_correct', True):
                return s1, s1
            corrected = val.get('corrected_bottleneck', '') or s1
            return s1, corrected
    
    # Variant B — ablation/role-swap summary: flat dict with s1_bottleneck field.
    if 's1_bottleneck' in data:
        s1 = data.get('s1_bottleneck', '')
        if 's2_corrected' in data:
            if data.get('s2_corrected') and data.get('s2_type') not in (None, 'N/A', ''):
                return s1, data['s2_type']
            return s1, s1
        if 's2_bottleneck_raw' in data:
            s2 = data.get('s2_bottleneck_raw', '') or s1
            return s1, s2
    
    return None, None


def _build_classification_from_legacy(data, program_key, logger):
    """Recover a classification dict from legacy JSON using the pf taxonomy."""
    s1_raw, s2_raw = _extract_legacy_s1_s2(data, program_key)
    if s1_raw is None:
        return None
    
    s1_p = primary(s1_raw)
    s2_p = primary(s2_raw)
    gt = GROUND_TRUTH.get(program_key, 'unknown')
    cls_obj = classify(s1_p, s2_p, gt)
    return cls_obj.as_dict()


def load_classification(result_json_path, program_key, logger):
    """
    Load a per-program result JSON and return (classification_dict, cost).
    
    First tries the modern `classification` field. If absent, falls back to
    recovering classification from legacy fields using the pf taxonomy —
    result is functionally equivalent to what the refactored pipeline would
    have produced.
    """
    if not result_json_path.exists():
        return None
    try:
        data = json.load(open(result_json_path, encoding='utf-8'))
    except Exception as e:
        logger.warning(f"Failed to parse {result_json_path}: {e}")
        return None
    
    # Modern: direct classification field
    cls = data.get('classification')
    if cls is None:
        # Might be nested under full_result
        full = data.get('full_result', {})
        cls = full.get('classification')
    
    if cls is None:
        # Legacy JSON — recover via pf taxonomy on raw S1/S2 strings
        cls = _build_classification_from_legacy(data, program_key, logger)
        if cls is None:
            logger.warning(f"{result_json_path}: could not recover classification "
                           f"from legacy fields; skipping.")
            return None
        logger.info(f"  [legacy-recover] {program_key}: "
                    f"s1={cls['s1_primary']} s2={cls['s2_primary']} type={cls['correction_type']}")
    
    cost = data.get('cost', data.get('total_cost', 0))
    return cls, cost


def summarise_config(config_key, label, candidate_dirs, s1_m, s2_m, logger):
    dir_path, used_fallback = _resolve_dir(candidate_dirs)
    per_prog = {}
    total_cost = 0.0
    source_dir = str(dir_path)
    
    # Filename patterns to try in order (refactored naming first, legacy second)
    candidate_filename_templates = [
        "{prog}_result.json",
        "{prog}_cascaded_result.json",
    ]
    
    for prog in PROGRAMS:
        result_file = None
        for tmpl in candidate_filename_templates:
            f = dir_path / tmpl.format(prog=prog)
            if f.exists():
                result_file = f
                break
        if result_file is None:
            per_prog[prog] = None
            continue
        
        loaded = load_classification(result_file, prog, logger)
        if loaded is None:
            per_prog[prog] = None
            continue
        cls, cost = loaded
        per_prog[prog] = cls
        total_cost += cost
    
    present = [p for p in PROGRAMS if per_prog.get(p) is not None]
    if not present:
        return {
            'label': label, 's1_model': s1_m, 's2_model': s2_m,
            'dir': source_dir, 'status': 'missing',
            'n': 0, 'per_program': per_prog, 'cost': 0,
            'changes': 0, 'corrections': 0, 'over_corrections': 0,
            'lateral_changes': 0, 'no_changes': 0,
            's1_accuracy': 0, 's2_accuracy': 0,
        }
    
    def count(pred):
        return sum(1 for p in present if pred(per_prog[p]))
    
    return {
        'label': label, 's1_model': s1_m, 's2_model': s2_m,
        'dir': source_dir,
        'status': 'complete' if len(present) == len(PROGRAMS) else 'partial',
        'n': len(present),
        'cost': round(total_cost, 4),
        'changes':          count(lambda c: c.get('changed')),
        'corrections':      count(lambda c: c.get('correction_type') == 'correction'),
        'over_corrections': count(lambda c: c.get('correction_type') == 'over-correction'),
        'lateral_changes':  count(lambda c: c.get('correction_type') == 'lateral-change'),
        'no_changes':       count(lambda c: c.get('correction_type') == 'no-change'),
        's1_accuracy':      count(lambda c: c.get('s1_matches_gt')),
        's2_accuracy':      count(lambda c: c.get('s2_matches_gt')),
        'per_program':      per_prog,
    }


def write_markdown(all_summaries, out_path):
    lines = []
    lines.append("# PF Unified Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("Taxonomy: **pf** (position-based primary match, `bottleneck_taxonomy.primary`).")
    lines.append("")
    lines.append("**Definitions:**")
    lines.append("- **change** = S2's primary bottleneck category differs from S1's.")
    lines.append("- **correction** = change where S2 moves S1 toward ground truth.")
    lines.append("- **over-correction** = change where S2 moves S1 AWAY from ground truth.")
    lines.append("- **lateral** = change where neither S1 nor S2 matches ground truth.")
    lines.append("- **no-change** = S1 and S2 agree on primary category.")
    lines.append("")
    
    # Main summary
    lines.append("## Cross-configuration summary")
    lines.append("")
    lines.append("| Config | S1 → S2 | N | Changes | Corrections | Over-corr. | Lateral | No-change | S1 acc | S2 acc | Cost |")
    lines.append("|--------|---------|:-:|:-------:|:-----------:|:----------:|:-------:|:---------:|:------:|:------:|:----:|")
    for s in all_summaries:
        n = s['n']
        if n == 0:
            lines.append(f"| **{s['label']}** | {s['s1_model']} → {s['s2_model']} | 0 | (no data) | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| **{s['label']}** | {s['s1_model']} → {s['s2_model']} "
            f"| {n} | **{s['changes']}/{n}** | {s['corrections']} "
            f"| {s['over_corrections']} | {s['lateral_changes']} | {s['no_changes']} "
            f"| {s['s1_accuracy']}/{n} | {s['s2_accuracy']}/{n} "
            f"| ${s['cost']:.2f} |"
        )
    lines.append("")
    
    # Per-config details
    for s in all_summaries:
        if s['n'] == 0:
            continue
        lines.append(f"## {s['label']} — per-program detail")
        lines.append("")
        lines.append("| Program | S1 primary | S2 primary | GT | S1 ✓ | S2 ✓ | Type |")
        lines.append("|---------|-----------|-----------|-----|:----:|:----:|------|")
        for prog in PROGRAMS:
            cls = s['per_program'].get(prog)
            if cls is None:
                lines.append(f"| {prog} | — | — | — | — | — | (missing) |")
                continue
            s1_ok = "✓" if cls['s1_matches_gt'] else "✗"
            s2_ok = "✓" if cls['s2_matches_gt'] else "✗"
            lines.append(f"| {prog} | {cls['s1_primary']} | {cls['s2_primary']} "
                         f"| {cls['ground_truth']} | {s1_ok} | {s2_ok} | {cls['correction_type']} |")
        lines.append("")
    
    # Cross-program diff highlights
    lines.append("## Role-Swap finding")
    lines.append("")
    lines.append("To quickly check the Phase-3 claim (prompt structure drives S2 correction),")
    lines.append("compare Original / V1 Neutral / V3 Biased change counts:")
    lines.append("")
    orig = next((s for s in all_summaries if s['label'].startswith('Original')), None)
    v1 = next((s for s in all_summaries if 'V1' in s['label']), None)
    v3 = next((s for s in all_summaries if 'V3' in s['label']), None)
    if orig and v1 and v3 and orig['n'] and v1['n'] and v3['n']:
        lines.append(f"- Original (validate/correct prompt):  **{orig['changes']}/{orig['n']}** changes")
        lines.append(f"- V1 Neutral (no validation language):  **{v1['changes']}/{v1['n']}** changes")
        lines.append(f"- V3 Biased (confirm-the-analysis):     **{v3['changes']}/{v3['n']}** changes")
        lines.append("")
        lines.append("All three use S1=GPT-4o, S2=GPT-5.2, temperature=0. Only the S2 prompt varies.")
    else:
        lines.append("(One or more of Original / V1 / V3 has not been run yet.)")
    lines.append("")
    
    out_path.write_text("\n".join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Aggregate all per-program JSONs into pf canonical summary")
    parser.add_argument("--output-dir", default="results/pf_summary")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_summaries = []
    for (key, label, candidate_dirs, s1_m, s2_m) in CONFIGS:
        logger.info(f"Aggregating: {label}  (searching {candidate_dirs})")
        s = summarise_config(key, label, candidate_dirs, s1_m, s2_m, logger)
        all_summaries.append(s)
        logger.info(
            f"  n={s['n']} changes={s['changes']} "
            f"corrections={s['corrections']} over={s['over_corrections']} "
            f"S1_acc={s['s1_accuracy']} S2_acc={s['s2_accuracy']} "
            f"cost=${s['cost']:.2f}"
        )
    
    # JSON: top-level summary (counts only)
    summary_json = {
        'generated': datetime.now().isoformat(),
        'taxonomy': 'pf (position-based primary match)',
        'ground_truth': GROUND_TRUTH,
        'configs': [
            {k: v for k, v in s.items() if k != 'per_program'}
            for s in all_summaries
        ],
    }
    with open(out_dir / "summary_table.json", 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    
    # JSON: per-config full detail (with per_program)
    with open(out_dir / "per_config_details.json", 'w', encoding='utf-8') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'configs': all_summaries,
        }, f, indent=2, ensure_ascii=False)
    
    # Markdown
    write_markdown(all_summaries, out_dir / "summary_table.md")
    
    logger.info("")
    logger.info(f"Written: {out_dir / 'summary_table.json'}")
    logger.info(f"Written: {out_dir / 'per_config_details.json'}")
    logger.info(f"Written: {out_dir / 'summary_table.md'}")
    
    # Print a concise stdout table
    print("\n" + "="*90)
    print("PF CANONICAL SUMMARY (cross-configuration)")
    print("="*90)
    print(f"{'Config':<28} {'S1→S2':<20} {'N':>4} {'Chg':>5} {'Corr':>5} {'Over':>5} {'S1✓':>5} {'S2✓':>5} {'Cost':>8}")
    print("-"*90)
    for s in all_summaries:
        n = s['n']
        if n == 0:
            print(f"{s['label']:<28} {s['s1_model']+'→'+s['s2_model']:<20} {'0':>4} {'-':>5} {'-':>5} {'-':>5} {'-':>5} {'-':>5} {'-':>8}")
        else:
            print(f"{s['label']:<28} {s['s1_model']+'→'+s['s2_model']:<20} "
                  f"{n:>4} {s['changes']:>5} {s['corrections']:>5} {s['over_corrections']:>5} "
                  f"{s['s1_accuracy']:>5} {s['s2_accuracy']:>5} ${s['cost']:>7.2f}")
    print("="*90)


if __name__ == "__main__":
    main()
