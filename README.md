# LLM-HPC: Cascaded LLM Analysis for HPC Bottleneck Classification and CUDA Generation

> Undergraduate thesis project (COMP3931, University of Leeds, 2025/26).
> Companion code repository for the thesis _"Meta-Prompt Design for
> LLM-Assisted HPC Performance Analysis: When Prompt Structure
> Outweighs Model Choice"_.

This repository contains the experimental infrastructure for a
two-stage cascaded LLM pipeline applied to (a) HPC bottleneck
classification and (b) CUDA kernel generation. The thesis investigates
whether the cascade's correction behaviour is driven by model choice
or by prompt structure, and reports findings from nine benchmark
programs across seven experimental configurations.

---

## TL;DR

The thesis answers four research questions:

**RQ1.** Single-stage LLM bottleneck classification accuracy is not
uniform: GPT-4o reaches 5/9, GPT-5.2 reaches 8/9, GPT-5.4 reaches
7/9. Errors are systematic and directional — GPT-4o predicts
compute-bound for memory-bound programs; the GPT-5.x family does the
opposite, predicting memory-bound for the only compute-bound program.
Newer is not more accurate.

**RQ2.** A two-stage cascade improves classification only when
Stage 2's directional bias opposes Stage 1's. Pairing a compute-biased
GPT-4o with a memory-biased GPT-5.x produces 5/9 changes (4
corrections + 1 over-correction) and reaches an 8/9 ceiling. Pairing
two memory-biased models, or reversing the direction, produces 0–1
changes. Stage 2 model identity does not predict cascade behaviour;
two configurations with different Stage 2 models but the same bias
direction produce identical change sets.

**RQ3.** A controlled prompt-variation experiment (Role-Swap) holding
the model pair constant at GPT-4o → GPT-5.2 and varying only the
Stage 2 prompt produces a near-complete collapse of changes: 5/9
under the validate/correct prompt, 4/9 under a neutral prompt, and
1/9 under a confirm/extend prompt. **Prompt structure produces
effects on cascade behaviour comparable in magnitude to model-pair
selection.** Two distinct mechanisms operate at different scales: a
*validation-removal effect* (cost: 1 correction) and an
*agreement-induction effect* (cost: 3 more changes).

**RQ4 (secondary).** The cascade's analytical outputs translate into
functional CUDA kernels for workloads whose algorithmic structure
suits the generation method. Embarrassingly parallel kernels (miniMD)
yield high speedups under direct generation alone (14.34×). Kernels
with serial data dependencies (HPCG SYMGS) require cascade-derived
strategy hints to produce correct GPU code at all (direct generation
runs 50× *slower* than CPU). The cascade's value extends to
architectural decisions: for Abinit's FFT hotspot, Stage 2 correctly
recommended replacing hand-written FFT routines with cuFFT rather
than generating a custom kernel.

The findings integrate into a **three-condition model** of when
cascade-induced correction occurs: directional Stage 1 bias,
opposing Stage 2 bias, and a non-directive Stage 2 prompt. Each
condition is necessary; violating any one drops correction to at
most 1/9.

---

## Repository Structure

```
llm-hpc-project/
├── src/                                  # Core library
│   ├── configurable_pipeline.py          # Parametrised S1/S2 pipeline (all 7 configs)
│   ├── bottleneck_taxonomy.py            # Position-based primary category classification
│   ├── generalized_evaluator.py          # Five-dimension scoring against VTune ground truth
│   ├── benchmark_config.py               # Ground truth for original 4 programs
│   ├── extended_benchmark_config.py      # Ground truth for HotSpot/SRAD/LULESH/NAS CG/Jacobi-2D
│   ├── analyzer.py                       # Stage 1 wrapper around OpenAI Chat API
│   ├── llm_client.py                     # OpenAI API client with cost tracking
│   └── converter.py                      # CUDA generation + compile/run harness
│
├── prompts/                              # Stage 1 + Stage 2 prompt templates
│
├── benchmarks/                           # Source code for nine evaluation programs
│   ├── minimd/        hpcg/        abinit/         # Original three (HPCG counted twice: SPMV + SYMGS)
│   ├── hotspot/       srad/        lulesh/         # Extended five programs
│   └── nas_cg/        jacobi2d/    omp_comparison.c
│
├── results/                              # Stored JSON outputs from all experiments
│
├── run_original.py                       # Baseline cascade: GPT-4o → GPT-5.2 (validate/correct prompt)
├── run_role_swap.py                      # Central experiment: NeutralPipeline + BiasedAgreementPipeline
├── run_ablation.py                       # Ablations A (5.2→5.2) + B (5.2→4o)
├── run_ablation_cd.py                    # Ablations C (4o→5.4) + D (5.4→5.4)
├── run_all_experiments.py                # One-shot orchestrator for all 7 configurations
├── rerun_model_bias.py                   # Single-stage bias test (full S1 capture for rescoring)
├── rescore_all.py                        # Regenerate change counts from stored JSONs (zero-cost)
├── rescore_eval_all.py                   # Regenerate Stage 1 evaluation scores from stored JSONs
├── test_cuda_*.py                        # CUDA generation experiments per kernel
├── test_abinit_cufft.py                  # Abinit cuFFT case study
└── stage2_manual_optimization_test.py    # Human-implemented CUDA from Stage 2 suggestions
```

The single-class design point: `ConfigurableCascadedPipeline` accepts
S1 and S2 model identifiers as constructor parameters. All seven
configurations are instantiated from this one class; `run_role_swap.py`
defines two subclasses (`NeutralPipeline`, `BiasedAgreementPipeline`)
that override only `_build_stage2_prompt()` and
`_extract_s2_bottleneck()`. This separation is what makes the
Role-Swap experiment a controlled comparison: the Stage 2 prompt is
the only variable that changes between the Original and the V1/V3
subclasses.

---

## Benchmark Suite

| Program | Source | Top hotspot | Runtime % | GT bottleneck |
|---|---|---|:-:|---|
| miniMD | Mantevo | `ForceLJ::compute` | 75.0 | compute |
| HPCG SPMV | HPCG 3.1 | `ComputeSPMV_ref` | 27.2 | memory |
| HPCG SYMGS | HPCG 3.1 | `ComputeSYMGS_ref` | 67.7 | memory + dep. |
| Abinit | ABINIT 9.10.4 | `sg_ffty` | 40.7 | memory |
| HotSpot | Rodinia | `compute_tran_temp` | 100.0 | memory |
| SRAD | Rodinia | `srad_kernel` | 98.1 | memory |
| LULESH | LLNL | `CalcFBHourglass...` | 48.9 | memory |
| NAS CG | NAS Parallel | `sparse_matvec` | 83.6 | memory |
| Jacobi-2D | PolyBench | `jacobi_kernel` | 99.0 | memory |

Ground truth was determined with Intel VTune Profiler 2025.9 in
user-mode sampling mode. The methodology is documented in §2.3 of
the thesis. The 8:1 memory-to-compute ratio reflects the empirical
prevalence of memory-bound kernels in production HPC workloads;
implications for the GPT-5.x memory-bias claim are discussed in
§5.2.

---

## Headline Results

### Hotspot identification

The pipeline correctly identifies the dominant hotspot for all
nine programs. Bottleneck classification is the open question —
LLMs reliably *locate* hotspots; whether they correctly *classify*
the bottleneck type is what RQ1–RQ3 investigate.

### Single-stage classification (RQ1)

```
            compute-bias  memory-bias    accuracy   error pattern
GPT-4o            ✓                       5/9       4 mem→compute
GPT-5.2                       ✓           8/9       1 compute→mem (miniMD)
GPT-5.4                       ✓           7/9       1 compute→mem + 1 LULESH
```

### Cascaded pipeline baseline (RQ2)

```
Configuration     S1 bias    S2 bias    Changes    S1 acc   S2 acc   Δ
Original          compute    memory     5/9        5/9      8/9      +3
Ablation A        memory     memory     0/9        8/9      8/9       0
Ablation B        memory     compute    0/9        8/9      8/9       0
Ablation C        compute    memory     5/9        5/9      8/9      +3   (same model identity)
Ablation D        memory     memory     1/9        7/9      8/9      +1
```

All five configurations converge on an 8/9 ceiling. The single
program no configuration classifies correctly is miniMD, the only
compute-bound program in the suite — every memory-biased Stage 2
over-corrects miniMD's correct compute label to memory.

### Role-Swap (RQ3)

```
S2 prompt              Changes   S2 accuracy
Original (V/C)         5/9       8/9
V1 Neutral             4/9       7/9
V3 Biased Agreement    1/9       6/9
```

S1=GPT-4o, S2=GPT-5.2, temperature=0 held constant across all three.
Only the Stage 2 prompt template varies. Removing the validation
directive costs one correction (LULESH reverts to S1's compute
label); inducing agreement collapses changes by three more
(Abinit/SRAD revert, miniMD spuriously matches S1).

### CUDA generation (RQ4)

```
Kernel              Direct gen   Strategy hint   Full pipeline   S2 manual
miniMD ForceLJ      14.34×       —               15.59×          16.97×
HPCG SPMV           10.30×       —               6.18× (wrong)    7.95×
HPCG SYMGS          0.02× (fail) 3.14×           5.61×            1.96×†
```

†Multi-colour Gauss–Seidel changes the iteration ordering.

For Abinit, Stage 2 correctly recommended cuFFT library replacement
over a custom CUDA kernel — an architectural decision rather than a
code-generation task.

---

## Reproducibility

Every numerical claim in the thesis traces to one of three sources:
a VTune measurement, a rescore-script output applied to stored JSON,
or a cited reference. No number is inferred or back-derived.

### Stored results and zero-cost rescore

Every pipeline run serialises its full result — Stage 1 hotspots,
Stage 2 response, taxonomy classification, and per-call API cost —
to JSON in `results/`. The two rescore scripts regenerate all
derived quantities from these files without API calls:

- **`rescore_all.py`** — recomputes change counts under the
  position-based taxonomy across all seven configurations, writing
  `results/pf_summary/summary_table.md`. Runtime: under 2 seconds,
  cost: zero.

- **`rescore_eval_all.py`** — re-runs the five-dimension evaluator
  on stored Stage 1 hotspot data using the current benchmark
  configuration as ground truth. Produces canonical scores at
  tolerance 0.3 and at stricter 0.15 for sensitivity comparison.

### Single source of truth for bottleneck classification

All scripts (experiment runners and rescore scripts) import
`primary()` and `classify()` from `src/bottleneck_taxonomy.py`. No
script-local heuristic is used anywhere in the codebase. This was
introduced after an early counting bug: list-ordered keyword checks
classified compound Stage 2 outputs such as "memory-bandwidth with
compute-heavy kernels" as compute-bound, recording 0 changes for
the V1 Neutral configuration. The position-based rule (whichever
keyword's earliest character index is lowest wins) records the
correct value of 4. The §3.3 thesis section documents this
provenance.

---

## Experimental Workflow

The thesis experiments were conducted in the following order. Each
phase produces stored results that subsequent phases reference; no
phase needs to re-execute earlier phases.

### Phase 1: Ground truth construction

```bash
# Profile each benchmark with VTune (user-mode sampling)
vtune -collect hotspots -knob sampling-mode=sw \
    -result-dir results/vtune/<program> -- ./<program>
```

Hotspot identification + Roofline classification documented per
program. Ground truth recorded in `src/benchmark_config.py` (original
four programs) and `src/extended_benchmark_config.py` (extended
five). The methodology including the gprof → VTune transition is
described in §2.3 of the thesis.

### Phase 2: Single-stage bias characterisation (RQ1)

```bash
python rerun_model_bias.py --model gpt-4o
python rerun_model_bias.py --model gpt-5.2
python rerun_model_bias.py --model gpt-5.4
```

Establishes per-model directional bias (GPT-4o → compute, GPT-5.x
→ memory). `rerun_model_bias.py` saves the full Stage 1 output
(including hotspots) so that `rescore_eval_all.py` can later
re-evaluate against updated ground truth without additional API
calls. Output stored in `results/model_bias_v2/`.

### Phase 3: Cascaded pipeline baseline (RQ2)

```bash
# Original cascade (baseline: GPT-4o → GPT-5.2 across all 9 programs)
python run_original.py

# Ablations A/B (model-pair variation, same-bias and reversed)
python run_ablation.py             # 5.2→5.2, 5.2→4o

# Ablations C/D (alternative S2 model, self-cascade with higher accuracy)
python run_ablation_cd.py          # 4o→5.4, 5.4→5.4
```

The five configurations all use the validate/correct prompt; only
the model pair varies. Output stored in `results/<configuration>/`.

### Phase 4: Role-Swap (RQ3 — central experiment)

```bash
# Both V1 Neutral and V3 Biased, model pair held at GPT-4o → GPT-5.2
python run_role_swap.py

# Or run individual variants:
python run_role_swap.py --variant V1
python run_role_swap.py --variant V3

# Or subset of programs:
python run_role_swap.py --programs minimd hotspot
```

The Stage 2 prompt is the only variable that changes between
Original, V1, and V3. This is the controlled comparison that
isolates prompt structure as an independent variable.

### Phase 5: Aggregation and rescore (zero cost)

```bash
# Cross-configuration change counts under position-based taxonomy
python rescore_all.py              # → results/pf_summary/summary_table.md

# Stage 1 evaluation scores under current ground truth + tolerance sensitivity
python rescore_eval_all.py         # → results/pf_summary/eval_rescore_comparison.md
```

These scripts read stored JSONs from Phases 2–4 and regenerate all
derived numbers without API calls. They are the canonical source
of the headline counts reported in the thesis.

### Phase 6: CUDA generation (RQ4)

```bash
# Per-kernel generation methods
python test_cuda_optimization_full.py    # miniMD ForceLJ (direct + full pipeline)
python test_cuda_spmv.py                 # HPCG SPMV
python test_cuda_symgs.py                # SYMGS direct generation (fails)
python test_cuda_symgs_v2.py             # SYMGS with strategy hint
python test_cuda_symgs_full_pipeline.py  # SYMGS full pipeline (succeeds)

# Architectural decision case study
python test_abinit_cufft.py              # Abinit FFT → cuFFT recommendation

# Manual CUDA from Stage 2 suggestions (upper-bound implementation)
python stage2_manual_optimization_test.py
```

Generated kernels are compiled with `nvcc -O2` and benchmarked
against serial CPU baselines on the RTX 3060. Speedup numbers are
reported in §4.2 of the thesis.

The phases are independent in execution: Phase 4 (Role-Swap) can
be re-run without re-executing Phase 3 ablations, and Phase 5
(rescore) reads stored JSONs from any subset of phases. This
decoupling is the basis of the zero-cost rescore methodology
(§3.5 of the thesis).

---

## Using the Pipeline on a New Program

The pipeline is designed to be applied to HPC programs beyond the
nine in the benchmark suite. The workflow for analysing a new
program is as follows.

### Step 1: Profile with VTune

```bash
# Compile the program with debug symbols
gcc -O2 -g -o myprogram myprogram.c

# Run VTune hotspot analysis
vtune -collect hotspots -knob sampling-mode=sw \
    -result-dir results/vtune/myprogram -- ./myprogram

# Inspect hotspots
vtune -report hotspots -result-dir results/vtune/myprogram
```

Identify the top hotspot function and its runtime percentage.

### Step 2: Determine ground truth bottleneck

Apply Roofline classification to the hotspot's arithmetic intensity:

- Above the platform's ridge-point (~10 FLOPs/byte for typical
  consumer GPUs) → **compute-bound**
- Below the ridge-point → **memory-bound**

For ambiguous cases (e.g., dependency-heavy kernels), inspect the
algorithmic structure: serial dependencies indicate "memory + dep"
classification.

### Step 3: Add to benchmark configuration

Edit `src/extended_benchmark_config.py` (or create a new
configuration file) with the program's metadata:

```python
BENCHMARKS = {
    "myprogram": {
        "source_path": "benchmarks/myprogram/main.c",
        "top_hotspot": "my_hotspot_function",
        "runtime_percentage": 85.0,
        "ground_truth_bottleneck": "memory",
        "compile_command": "gcc -O2 -o myprogram main.c",
    },
}
```

### Step 4: Run the cascade

```bash
# Single-stage analysis (Stage 1 only)
python -c "
from src.analyzer import HPCAnalyzer
a = HPCAnalyzer(model='gpt-4o')
result = a.analyze('benchmarks/myprogram/main.c')
print(result)
"

# Two-stage cascade (Stage 1 + Stage 2)
python -c "
from src.configurable_pipeline import ConfigurableCascadedPipeline
p = ConfigurableCascadedPipeline(s1_model='gpt-4o', s2_model='gpt-5.2')
result = p.analyze('benchmarks/myprogram/main.c')
print(result)
"
```

The cascade output includes:

- Hotspot identification (function name, line range)
- Bottleneck classification (compute / memory / communication)
- GPU suitability judgment
- Optimisation suggestions (target, technique, expected speedup,
  difficulty)
- Stage 2 validation/correction of Stage 1's classification
- (When applicable) GPU implementation plan

### Step 5: Interpret results in light of the three-condition model

The thesis's three-condition model (§4.1.4) establishes that the
cascade produces reliable corrections only when:

1. **Stage 1 has a directional bias** — verified by single-stage
   accuracy patterns.
2. **Stage 2's bias opposes Stage 1's** — pair compute-biased and
   memory-biased models.
3. **The Stage 2 prompt does not actively induce agreement** — use
   the validate/correct (Original) prompt rather than V3 Biased.

If your program produces a Stage 2 output that "confirms" Stage 1
without modification, this may reflect (a) Stage 1 being correct,
(b) the model pair being same-biased, or (c) prompt-induced
agreement. The three-condition model does not certify cascade
correctness for individual programs; it characterises when
correction *can* occur.

### Step 6: (Optional) Generate CUDA

For programs whose Stage 2 analysis produces a GPU implementation
plan:

```bash
python -c "
from src.converter import CUDAConverter
c = CUDAConverter(model='gpt-5.2')
cuda_code = c.convert('benchmarks/myprogram/main.c', method='full_pipeline')
print(cuda_code)
"
```

Generation methods (per §3.6 of the thesis):

- `direct`: source code → CUDA, no analysis context
- `strategy_hint`: explicit parallelisation strategy provided
- `full_pipeline`: complete Stage 1 + Stage 2 cascade output

The optimal generation method depends on the program's algorithmic
structure (§4.2). Embarrassingly parallel kernels benefit from
direct generation; kernels with serial dependencies require
strategy hints; structurally simple memory-bound kernels can be
harmed by over-specification of optimisation strategy.

### Step 7: Compile, validate, benchmark

Generated CUDA kernels need correctness validation against the CPU
baseline before benchmarking is meaningful:

```bash
# Compile CUDA kernel
nvcc -O2 -o myprogram_cuda gpu_conversion/myprogram.cu

# Run with correctness check
./myprogram_cuda --validate

# Benchmark against CPU baseline
./myprogram_cuda --benchmark
```

### Cost estimate for a new program

Single program through the full cascade pipeline costs approximately
**$0.05–$0.15** depending on source code length, with Stage 2 typically
2–3× more expensive than Stage 1 due to longer context. CUDA
generation adds approximately **$0.05–$0.10** per kernel.

---

## Re-running Thesis Experiments from Scratch

```bash
# Setup
git clone <repo-url>
cd llm-hpc-project
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."

# All seven cascade configurations in one shot (Phases 3 + 4).
# Idempotent: per-program JSON files are skipped if already present,
# so the orchestrator resumes cleanly after any interruption.
python run_all_experiments.py

# Or run individual phases / a subset of stages:
#   python run_all_experiments.py --stages OABCD   # only model-pair (Phase 3)
#   python run_all_experiments.py --stages V1V3    # only Role-Swap (Phase 4)
#   python run_original.py                         # just the baseline
#   python run_ablation.py                         # just A/B
#   python run_ablation_cd.py                      # just C/D
#   python run_role_swap.py                        # just V1/V3

# Rescore (Phase 5 — zero cost)
python rescore_all.py
python rescore_eval_all.py

# CUDA generation (Phase 6)
python test_cuda_optimization_full.py
python test_cuda_spmv.py
python test_cuda_symgs_full_pipeline.py
python test_abinit_cufft.py
```

A full re-execution of all seven cascade configurations across nine
programs costs approximately $4.43 at the API pricing in effect
during the project. The full project total, including single-stage
bias tests and GPU generation, was approximately $5.05.

---

## Environment

| Component | Details |
|---|---|
| CPU | Intel Tiger Lake H, 2.304 GHz, 8 physical / 16 logical cores |
| GPU | NVIDIA RTX 3060, 6 GB, SM 86, CUDA 12.6 |
| OS | WSL2 Ubuntu 24.04 on Windows 11 |
| Profiler | Intel VTune Profiler 2025.9 (user-mode sampling) |
| LLM models | GPT-4o, GPT-5.2, GPT-5.4 (OpenAI Chat Completions API, temperature=0) |
| Compilers | gcc 13.2 (`-O2 -fopenmp`); nvcc 12.6 (`-O2`) |
| Python | 3.11 |

Full hardware/software details are in Appendix C of the thesis.

---

## Use of External Material

The nine benchmark programs are obtained from publicly available
HPC benchmark suites (Mantevo, HPCG, ABINIT, Rodinia, LLNL, NAS,
PolyBench) under their respective licenses (BSD or GNU GPL). All
were used in their unmodified released form for profiling purposes
only; no derivative works were created or distributed. Standard
third-party tools (VTune, OpenAI API, CUDA toolkit, gcc) were used
in their released form. Full disclosure is provided in
**Appendix B (External Material)** of the thesis.

---

## Citation

```bibtex
@thesis{llm-hpc-2026,
  author = {Wenshou Zhong}
  title  = {Meta-Prompt Design for LLM-Assisted HPC Performance
            Analysis: When Prompt Structure Outweighs Model Choice},
  school = {University of Leeds},
  year   = {2026},
  type   = {BSc Computer Science Individual Project (COMP3931)}
}
```

---

## License

Code: MIT.

Benchmark programs retain their respective licenses (see source
directories under `benchmarks/`).

---

## Acknowledgements

Benchmark programs courtesy of the Mantevo project, the HPCG
consortium, the ABINIT developers, the Rodinia benchmark suite,
Lawrence Livermore National Laboratory, the NASA Advanced
Supercomputing Division, and the PolyBench/C developers. Intel
VTune Profiler used under the oneAPI community licence. Project
supervised by Dr. Xia at the School of Computing,
University of Leeds.
