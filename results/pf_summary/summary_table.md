# PF Unified Summary

Generated: 2026-04-17T14:24:06.122674

Taxonomy: **pf** (position-based primary match, `bottleneck_taxonomy.primary`).

**Definitions:**
- **change** = S2's primary bottleneck category differs from S1's.
- **correction** = change where S2 moves S1 toward ground truth.
- **over-correction** = change where S2 moves S1 AWAY from ground truth.
- **lateral** = change where neither S1 nor S2 matches ground truth.
- **no-change** = S1 and S2 agree on primary category.

## Cross-configuration summary

| Config | S1 → S2 | N | Changes | Corrections | Over-corr. | Lateral | No-change | S1 acc | S2 acc | Cost |
|--------|---------|:-:|:-------:|:-----------:|:----------:|:-------:|:---------:|:------:|:------:|:----:|
| **Original (4o→5.2)** | gpt-4o → gpt-5.2 | 9 | **5/9** | 4 | 1 | 0 | 4 | 5/9 | 8/9 | $0.61 |
| **Ablation A (5.2→5.2)** | gpt-5.2 → gpt-5.2 | 9 | **0/9** | 0 | 0 | 0 | 9 | 8/9 | 8/9 | $0.79 |
| **Ablation B (5.2→4o)** | gpt-5.2 → gpt-4o | 9 | **0/9** | 0 | 0 | 0 | 9 | 8/9 | 8/9 | $0.45 |
| **Ablation C (4o→5.4)** | gpt-4o → gpt-5.4 | 9 | **5/9** | 4 | 1 | 0 | 4 | 5/9 | 8/9 | $0.62 |
| **Ablation D (5.4→5.4)** | gpt-5.4 → gpt-5.4 | 9 | **1/9** | 1 | 0 | 0 | 8 | 7/9 | 8/9 | $0.81 |
| **Role-Swap V1 Neutral** | gpt-4o → gpt-5.2 | 9 | **4/9** | 3 | 1 | 0 | 5 | 5/9 | 7/9 | $0.58 |
| **Role-Swap V3 Biased** | gpt-4o → gpt-5.2 | 9 | **1/9** | 1 | 0 | 0 | 8 | 5/9 | 6/9 | $0.56 |

## Original (4o→5.2) — per-program detail

| Program | S1 primary | S2 primary | GT | S1 ✓ | S2 ✓ | Type |
|---------|-----------|-----------|-----|:----:|:----:|------|
| minimd | compute | memory | compute | ✓ | ✗ | over-correction |
| hpcg_spmv | memory | memory | memory | ✓ | ✓ | no-change |
| hpcg_symgs | memory | memory | memory | ✓ | ✓ | no-change |
| abinit | compute | memory | memory | ✗ | ✓ | correction |
| hotspot | compute | memory | memory | ✗ | ✓ | correction |
| srad | compute | memory | memory | ✗ | ✓ | correction |
| lulesh | compute | memory | memory | ✗ | ✓ | correction |
| nas_cg | memory | memory | memory | ✓ | ✓ | no-change |
| jacobi2d | memory | memory | memory | ✓ | ✓ | no-change |

## Ablation A (5.2→5.2) — per-program detail

| Program | S1 primary | S2 primary | GT | S1 ✓ | S2 ✓ | Type |
|---------|-----------|-----------|-----|:----:|:----:|------|
| minimd | memory | memory | compute | ✗ | ✗ | no-change |
| hpcg_spmv | memory | memory | memory | ✓ | ✓ | no-change |
| hpcg_symgs | memory | memory | memory | ✓ | ✓ | no-change |
| abinit | memory | memory | memory | ✓ | ✓ | no-change |
| hotspot | memory | memory | memory | ✓ | ✓ | no-change |
| srad | memory | memory | memory | ✓ | ✓ | no-change |
| lulesh | memory | memory | memory | ✓ | ✓ | no-change |
| nas_cg | memory | memory | memory | ✓ | ✓ | no-change |
| jacobi2d | memory | memory | memory | ✓ | ✓ | no-change |

## Ablation B (5.2→4o) — per-program detail

| Program | S1 primary | S2 primary | GT | S1 ✓ | S2 ✓ | Type |
|---------|-----------|-----------|-----|:----:|:----:|------|
| minimd | memory | memory | compute | ✗ | ✗ | no-change |
| hpcg_spmv | memory | memory | memory | ✓ | ✓ | no-change |
| hpcg_symgs | memory | memory | memory | ✓ | ✓ | no-change |
| abinit | memory | memory | memory | ✓ | ✓ | no-change |
| hotspot | memory | memory | memory | ✓ | ✓ | no-change |
| srad | memory | memory | memory | ✓ | ✓ | no-change |
| lulesh | memory | memory | memory | ✓ | ✓ | no-change |
| nas_cg | memory | memory | memory | ✓ | ✓ | no-change |
| jacobi2d | memory | memory | memory | ✓ | ✓ | no-change |

## Ablation C (4o→5.4) — per-program detail

| Program | S1 primary | S2 primary | GT | S1 ✓ | S2 ✓ | Type |
|---------|-----------|-----------|-----|:----:|:----:|------|
| minimd | compute | memory | compute | ✓ | ✗ | over-correction |
| hpcg_spmv | memory | memory | memory | ✓ | ✓ | no-change |
| hpcg_symgs | memory | memory | memory | ✓ | ✓ | no-change |
| abinit | compute | memory | memory | ✗ | ✓ | correction |
| hotspot | compute | memory | memory | ✗ | ✓ | correction |
| srad | compute | memory | memory | ✗ | ✓ | correction |
| lulesh | compute | memory | memory | ✗ | ✓ | correction |
| nas_cg | memory | memory | memory | ✓ | ✓ | no-change |
| jacobi2d | memory | memory | memory | ✓ | ✓ | no-change |

## Ablation D (5.4→5.4) — per-program detail

| Program | S1 primary | S2 primary | GT | S1 ✓ | S2 ✓ | Type |
|---------|-----------|-----------|-----|:----:|:----:|------|
| minimd | memory | memory | compute | ✗ | ✗ | no-change |
| hpcg_spmv | memory | memory | memory | ✓ | ✓ | no-change |
| hpcg_symgs | memory | memory | memory | ✓ | ✓ | no-change |
| abinit | memory | memory | memory | ✓ | ✓ | no-change |
| hotspot | memory | memory | memory | ✓ | ✓ | no-change |
| srad | memory | memory | memory | ✓ | ✓ | no-change |
| lulesh | compute | memory | memory | ✗ | ✓ | correction |
| nas_cg | memory | memory | memory | ✓ | ✓ | no-change |
| jacobi2d | memory | memory | memory | ✓ | ✓ | no-change |

## Role-Swap V1 Neutral — per-program detail

| Program | S1 primary | S2 primary | GT | S1 ✓ | S2 ✓ | Type |
|---------|-----------|-----------|-----|:----:|:----:|------|
| minimd | compute | memory | compute | ✓ | ✗ | over-correction |
| hpcg_spmv | memory | memory | memory | ✓ | ✓ | no-change |
| hpcg_symgs | memory | memory | memory | ✓ | ✓ | no-change |
| abinit | compute | memory | memory | ✗ | ✓ | correction |
| hotspot | compute | memory | memory | ✗ | ✓ | correction |
| srad | compute | memory | memory | ✗ | ✓ | correction |
| lulesh | compute | compute | memory | ✗ | ✗ | no-change |
| nas_cg | memory | memory | memory | ✓ | ✓ | no-change |
| jacobi2d | memory | memory | memory | ✓ | ✓ | no-change |

## Role-Swap V3 Biased — per-program detail

| Program | S1 primary | S2 primary | GT | S1 ✓ | S2 ✓ | Type |
|---------|-----------|-----------|-----|:----:|:----:|------|
| minimd | compute | compute | compute | ✓ | ✓ | no-change |
| hpcg_spmv | memory | memory | memory | ✓ | ✓ | no-change |
| hpcg_symgs | memory | memory | memory | ✓ | ✓ | no-change |
| abinit | compute | compute | memory | ✗ | ✗ | no-change |
| hotspot | compute | memory | memory | ✗ | ✓ | correction |
| srad | compute | compute | memory | ✗ | ✗ | no-change |
| lulesh | compute | compute | memory | ✗ | ✗ | no-change |
| nas_cg | memory | memory | memory | ✓ | ✓ | no-change |
| jacobi2d | memory | memory | memory | ✓ | ✓ | no-change |

## Role-Swap finding

To quickly check the Phase-3 claim (prompt structure drives S2 correction),
compare Original / V1 Neutral / V3 Biased change counts:

- Original (validate/correct prompt):  **5/9** changes
- V1 Neutral (no validation language):  **4/9** changes
- V3 Biased (confirm-the-analysis):     **1/9** changes

All three use S1=GPT-4o, S2=GPT-5.2, temperature=0. Only the S2 prompt varies.
