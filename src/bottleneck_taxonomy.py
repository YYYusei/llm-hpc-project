"""
Bottleneck Taxonomy — single source of truth for all correction/change counting.

This module exists because the project previously mixed three different counting methods
(bc = validation.bottleneck_correct flag; pb = buggy list-order primary match;
pf = position-based primary match), producing contradictory numbers across documents.

Policy (as of 2026-04-17, adopted for thesis final version):
    - `pf` (position-based primary match) is the OFFICIAL definition of "change".
    - `bc` (validation flag) is retained only for backward-compat JSON reading.
    - `pb` is removed — it was a bug.

Terminology (use these words consistently everywhere):
    change         = S2's primary bottleneck category differs from S1's
                     (umbrella term, measured by pf)
    correction     = change where S2 moves S1's wrong label to the correct one
    over-correction = change where S2 moves S1's correct label to a wrong one
    modification   = same primary category, but S2 adds qualifiers
                     (e.g. 'memory' → 'memory + dependency')
                     NOTE: modifications are NOT counted as 'changes' under pf.

Usage:
    from bottleneck_taxonomy import primary, classify, CATEGORIES

    p1 = primary(s1_bottleneck_text)   # => 'compute' / 'memory' / 'communication' / 'unknown'
    p2 = primary(s2_bottleneck_text)
    cls = classify(p1, p2, ground_truth)
    # cls.changed, cls.correction_type, etc.
"""

from dataclasses import dataclass
from typing import Optional

# The recognised bottleneck category keywords, in no particular order.
# Ordering does not matter for pf semantics.
CATEGORIES = ('compute', 'memory', 'communication')

# Secondary keywords that qualify but do NOT override the primary category.
# These are used for descriptive matching only (e.g. 'memory/latency' is primary=memory).
# The rule is: if a string contains both a category and a secondary qualifier,
# the primary category wins even if the qualifier appears first.
SECONDARY_QUALIFIERS = ('latency', 'bandwidth', 'sync', 'synchronization',
                        'dependency', 'allocation', 'cache', 'branch')


def primary(text: Optional[str]) -> str:
    """
    Return the primary bottleneck category for a free-text S1 or S2 bottleneck string.

    Definition (pf — position-based):
        Among the three CATEGORIES that appear in the lowercased text,
        return the one that appears EARLIEST (smallest character position).
        If none appear, return 'unknown'.

    Rationale:
        LLM output like 'memory/latency + sync bound' or 'mixed (memory + compute)
        with dominant allocation overhead' clearly communicates memory as primary.
        The earliest-position rule captures this because writers put the primary
        bottleneck first in natural language. The old list-order rule instead
        scanned for 'compute' first regardless of position, which misclassified
        'memory + compute' as compute.

    Examples:
        primary('memory/latency + synchronization bound')    -> 'memory'
        primary('mixed (memory + compute) with allocation')  -> 'memory'
        primary('compute-bound due to FLOP intensity')       -> 'compute'
        primary('memory-bandwidth / control-flow bound')     -> 'memory'
        primary('')                                          -> 'unknown'
        primary(None)                                        -> 'unknown'
    """
    if not text:
        return 'unknown'
    s = text.lower()
    positions = {}
    for cat in CATEGORIES:
        idx = s.find(cat)
        if idx >= 0:
            positions[cat] = idx
    if not positions:
        return 'unknown'
    return min(positions, key=positions.get)


@dataclass
class Classification:
    """Result of comparing S1, S2 against ground truth."""
    s1_primary: str
    s2_primary: str
    ground_truth: str

    # Measurable fields
    changed: bool             # pf-definition: S2's primary differs from S1's
    s1_matches_gt: bool       # S1's primary matches ground truth
    s2_matches_gt: bool       # S2's primary matches ground truth

    # Derived classification
    correction_type: str      # 'correction', 'over-correction',
                              # 'lateral-change', 'no-change', 'no-change-both-wrong',
                              # 'unknown'

    def as_dict(self):
        return {
            's1_primary': self.s1_primary,
            's2_primary': self.s2_primary,
            'ground_truth': self.ground_truth,
            'changed': self.changed,
            's1_matches_gt': self.s1_matches_gt,
            's2_matches_gt': self.s2_matches_gt,
            'correction_type': self.correction_type,
        }


def classify(s1_primary_str: str, s2_primary_str: str, ground_truth: str) -> Classification:
    """
    Given normalised primary categories, classify the change.

    Parameters:
        s1_primary_str: e.g. 'compute' / 'memory' / 'unknown'
        s2_primary_str: e.g. 'compute' / 'memory' / 'unknown'
                        (if S2 didn't disagree, pass the same as s1_primary_str)
        ground_truth:   e.g. 'compute' / 'memory'

    Correction taxonomy:
        - correction         : S1 wrong, S2 right, primary changed
        - over-correction    : S1 right, S2 wrong, primary changed
        - lateral-change     : primary changed but neither matches GT
                               (or both unknown)
        - no-change          : S1 == S2 primary
        - unknown            : either primary == 'unknown'
    """
    changed = (s1_primary_str != s2_primary_str) and s2_primary_str != 'unknown'
    s1_ok = s1_primary_str == ground_truth
    s2_ok = s2_primary_str == ground_truth

    if s1_primary_str == 'unknown' or s2_primary_str == 'unknown':
        correction_type = 'unknown'
    elif not changed:
        correction_type = 'no-change'
    elif not s1_ok and s2_ok:
        correction_type = 'correction'
    elif s1_ok and not s2_ok:
        correction_type = 'over-correction'
    else:
        # Both right (impossible if changed) or both wrong
        correction_type = 'lateral-change'

    return Classification(
        s1_primary=s1_primary_str,
        s2_primary=s2_primary_str,
        ground_truth=ground_truth,
        changed=changed,
        s1_matches_gt=s1_ok,
        s2_matches_gt=s2_ok,
        correction_type=correction_type,
    )


# Canonical ground truth for the 9 programs.
# Used by re-scoring scripts. Keep in sync with benchmark_config.py / extended_benchmark_config.py.
GROUND_TRUTH = {
    'minimd':     'compute',
    'hpcg_spmv':  'memory',
    'hpcg_symgs': 'memory',
    'abinit':     'memory',
    'hotspot':    'memory',
    'srad':       'memory',
    'lulesh':     'memory',
    'nas_cg':     'memory',
    'jacobi2d':   'memory',
    'gemm':        'compute',
    '2mm':         'compute',
    '3mm':         'compute',
    'syrk':        'compute',
    'syr2k':       'compute',
    'doitgen':     'compute',
    'gramschmidt': 'compute',
}


if __name__ == "__main__":
    # Self-tests: run `python bottleneck_taxonomy.py` to validate
    tests = [
        ("memory/latency + synchronization bound", 'memory'),
        ("mixed (memory + compute) with dominant overhead from allocation", 'memory'),
        ("memory-bandwidth / control-flow bound", 'memory'),
        ("compute-bound due to FLOP intensity", 'compute'),
        ("mixed (memory + synchronization/atomics) with compute-heavy kernels", 'memory'),
        ("", 'unknown'),
        (None, 'unknown'),
        ("communication-bound MPI halo exchange", 'communication'),
        ("compute", 'compute'),
        ("memory", 'memory'),
    ]
    all_ok = True
    for inp, expected in tests:
        got = primary(inp)
        ok = got == expected
        all_ok = all_ok and ok
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] primary({inp!r:<70}) = {got!r:<15} (expected {expected!r})")
    print()

    # Classify examples
    print("Classification examples:")
    examples = [
        ('compute', 'memory', 'compute'),   # minimd S1 right, S2 wrong → over-correction
        ('compute', 'memory', 'memory'),    # abinit S1 wrong, S2 right → correction
        ('memory', 'memory', 'memory'),     # spmv no change
    ]
    for s1, s2, gt in examples:
        c = classify(s1, s2, gt)
        print(f"  S1={s1}, S2={s2}, GT={gt} -> changed={c.changed}, type={c.correction_type}")

    print()
    print("All primary() tests passed." if all_ok else "SOME TESTS FAILED.")
