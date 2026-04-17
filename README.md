# LLM-HPC: LLM-Assisted HPC Performance Analysis and GPU Code Generation

> Undergraduate thesis project (2026). Investigates whether large language models can (a) correctly identify performance bottlenecks in scientific computing codes and (b) generate functional, performant CUDA kernels from CPU source — and why a cascaded two-stage pipeline works when it works.

> **Canonical data snapshot:** this README reflects the 2026-04-17 data revision (v8). All headline counts use the **pf taxonomy** (position-based primary bottleneck match; see `src/bottleneck_taxonomy.py`). All S1 evaluation scores are computed against VTune-aligned `time_percentage` ground truth. See the Provenance & Reproducibility section below for how every number is regenerated.

---

## TL;DR

Three questions, three answers from this project:

**Q1. Can a single LLM reliably classify HPC bottlenecks?**
No. GPT-4o achieves 5/9 on VTune-validated ground truth, with all errors in one direction (memory-bound codes mislabelled as compute-bound). GPT-5.2 achieves 8/9 but overcorrects in the opposite direction (9/9 predictions are "memory", including one wrong call on miniMD). Every model tested exhibits a **directional bias**; newer ≠ more accurate (GPT-5.4 scores 7/9, slightly below GPT-5.2).

**Q2. Does a cascaded pipeline (S1: GPT-4o → S2: GPT-5.2) help?**
Yes — it converts 5/9 correct classifications into 9/9 for bottleneck type, and turns a catastrophic 0.02× CUDA generation failure (SYMGS direct generation) into a working 5.61× speedup. But the *reason* it works is not what we initially thought.

**Q3. Why does the cascade actually work?**
A Role-Swap experiment (fix S1=GPT-4o, S2=GPT-5.2, vary only the S2 prompt) shows that S2's independent-judgment changes drop from 5/9 to 4/9 when the prompt's "validate/correct" language is removed, and drop further to 1/9 when the prompt instead induces agreement. **The cascade's correction behaviour is strongly prompt-driven**, with an effect size (5/9 → 1/9, i.e. 5×) comparable to model-pair selection under identical S1/S2/temperature. Prompt structure operates through two distinct mechanisms: removing skepticism is a *weak* effect (1 program flips), but adding agreement is a *strong* effect (4 programs flip). This reframes the cascade's value as *"prompt structure preserves S2's independent judgment when no agreement cue is present"* rather than *"a stronger model fixes a weaker model."*

---

## Thesis Context

This project addresses three tasks from the thesis brief:

1. Deploy state-of-the-art AI4Science applications on HPC platforms and identify performance bottlenecks
2. Build an LLM-driven agent that automatically analyses which code segments are suitable for GPU acceleration
3. Use LLMs to transform C/Fortran code into CUDA/Triton and evaluate GPU performance

All three are covered end-to-end with nine benchmark programs, VTune-validated ground truth, cascaded analysis, multiple ablation studies, and benchmarked CUDA kernel generation.

---

## Key Findings

### Finding 1: Prompt structure, not model capability, drives S2 correction behaviour

The Role-Swap experiment fixes S1=GPT-4o and S2=GPT-5.2, varying only the S2 prompt. All other factors (model, temperature, S1 output, source code) are held constant.

| Prompt variant | Framing | S2 changed S1's primary bottleneck |
|----------------|---------|-----------------------------------|
| Original | "Validate/correct the initial analysis" | **5/9** |
| V1 Neutral | "Provide a deep performance analysis" (no validation language) | **4/9** |
| V3 Biased agreement | "Confirm the analysis and extend it with implementation details" | **1/9** |

The effect operates through two distinct mechanisms:
- **Weak effect — removing the "validate/correct" directive** drops changes from 5/9 to 4/9. Only LULESH flips from "corrected to memory" to "kept as compute"; the other four S1 errors (Abinit, HotSpot, SRAD, miniMD) still get changed by S2 even without explicit skepticism instruction.
- **Strong effect — adding explicit agreement framing** drops changes from 5/9 to 1/9. The `agree_with_initial` phrasing overrides S2's independent judgment on 4 of the 5 programs where it would otherwise have disagreed.

This is stronger evidence than prior ablations about model choice, because prompt is an explicit, reproducible control variable under identical model/temperature settings. The 5× effect size (5/9 → 1/9) is at least as large as any cross-model effect measured in this project.

**Historical note:** an earlier version of the Role-Swap analysis (2026-04-16) reported V1 Neutral as 0/9 changes. That number came from a buggy bottleneck-taxonomy function that list-ordered `[compute, memory, ...]` and misclassified S2 outputs such as "memory+sync bound (…not pure compute-bound)" as `compute`. The corrected pf taxonomy (position-based primary match; see `src/bottleneck_taxonomy.py`) yields 4/9. The narrative here reflects the corrected data.

### Finding 2: LLM bottleneck classification exhibits a model-level bias spectrum

Single-stage classification accuracy on 9 VTune-validated programs:

| Model | Compute predictions | Memory predictions | Parse fail | Accuracy |
|-------|---------------------|---------------------|-----------|----------|
| GPT-4o | 5 (4 wrong) | 4 (all correct) | 0 | 5/9 (56%) |
| GPT-5.4 | 1 (wrong) | 8 (1 wrong) | 0 | 7/9 (78%) |
| GPT-5.2 | 0 | 9 (1 wrong) | 0 | 8/9 (89%) |

GPT-4o strongly biases toward "compute"; GPT-5.2 biases universally to "memory"; GPT-5.4 sits in between with a strong memory bias (8/9 memory predictions, one compute label on LULESH). Newer ≠ more accurate: GPT-5.4 scores slightly lower than GPT-5.2 because it reverts to GPT-4o's compute label on LULESH. The cascade works when S1 is compute-biased and S2 is memory-biased *and* the prompt does not actively induce agreement — all three conditions matter.

### Finding 3: GPU kernel generation strategy depends on algorithmic structure, not code size

Best achieved GPU speedup per kernel (CUDA vs serial CPU baseline):

| Kernel | Algorithm | Direct generation | Full pipeline | Stage 2 manual | Best method |
|--------|-----------|-------------------|---------------|-----------------|-------------|
| miniMD Force | embarrassingly parallel | 14.34× | 15.59× | **16.97×** (+18.3%) | Stage 2 manual (SoA + CSR neighbour list + constant memory) |
| HPCG SPMV | row-parallel | **10.30×** | 6.18× ✗ (wrong format assumed) | 7.95× | Direct generation |
| HPCG SYMGS | serial dependency | 0.02× ✗ (50× *slower* than CPU) | **5.61×** | 1.96×† | Full pipeline (multi-colouring hint essential) |

†Multi-colour GS changes the algorithm — not comparable to strict SYMGS.

**Rule of thumb:** simple parallel code → direct generation wins (cheaper, faster, more reliable). Complex dependency code → full pipeline essential. Over-analysis can be harmful (SPMV pipeline −40% due to the LLM guessing a wrong data format).

### Finding 4: LLMs are capable of architectural decisions, not just code generation

For Abinit (an ab-initio DFT code), VTune showed the real hotspots are FFT functions (`sg_ffty`, `sg_fftpx`, ~68% total). The cascaded pipeline correctly recommended **cuFFT library replacement** over writing a custom CUDA FFT kernel, and Stage 2 provided a detailed API mapping (3D batched `cufftExecZ2Z`, persistent device buffers, expected 3–10× FFT speedup / 1.8–3.5× overall). This is Case Study #8.

### Finding 5: LLM-generated CUDA most valuable where OpenMP struggles

| Kernel | Serial | OpenMP 8T | CUDA | CUDA / OpenMP |
|--------|--------|-----------|------|---------------|
| miniMD Force (100k atoms) | 41.03 ms | 6.09 ms (6.73×) | 2.86 ms (14.34×) | 2.1× |
| SPMV (100k rows, nnz=27) | 2.25 ms | 1.26 ms (1.79×) | 0.22 ms (10.30×) | **5.8×** |
| SYMGS (50k rows, 27 colours) | 3.46 ms | 0.81 ms (4.29×) | 0.62 ms (5.61×) | 1.3× |

Bandwidth-bound codes (SPMV) benefit most from LLM-generated CUDA over OpenMP. Dependency-heavy codes (SYMGS) hit an algorithmic ceiling regardless of implementation quality.

---

## Quick Start

### Requirements

- Python 3.8+ with `openai>=1.0`, `pydantic`, `pyyaml`, `tiktoken`
- OpenAI API access (GPT-4o, GPT-5.2; GPT-5.4 optional for ablations)
- **Optional:** Intel VTune Profiler 2025.9+ for ground-truth profiling
- **Optional:** CUDA Toolkit 12.6+ with `nvcc` for GPU benchmarks
- **Optional:** WSL2 Ubuntu 24.04 (the environment used throughout)

### Setup

```bash
git clone <repo-url> llm-hpc-project
cd llm-hpc-project
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="sk-..."      # Linux/macOS
$env:OPENAI_API_KEY="sk-..."        # Windows PowerShell
```

### Run a quick sanity check (no API calls)

```bash
python test_local.py
```

### Run the headline experiments

```bash
# Cascaded analysis on the original 4 programs
python test_cascaded.py

# Extended to 9 programs (adds HotSpot, SRAD, LULESH, NAS CG, Jacobi-2D)
python test_extended_cascaded.py

# Ablation A/B — does S2 need S1 context? Is GPT-5.2 as S2 essential?
python run_ablation.py

# Ablation C/D — does GPT-5.4 as S2 behave differently?
python run_ablation_cd.py

# Role-Swap — the prompt-vs-model question (Phase 3 evidence)
python run_role_swap.py                      # both V1 Neutral + V3 Biased
python run_role_swap.py --variant V1         # just V1
python run_role_swap.py --programs minimd hotspot   # subset

# Single-stage bias test across models
python test_model_bias.py

# CUDA generation + benchmarking
python test_cuda_optimization_full.py        # miniMD
python test_cuda_spmv.py                     # SPMV
python test_cuda_symgs.py                    # SYMGS direct generation (fails)
python test_cuda_symgs_v2.py                 # SYMGS with strategy hint
python test_cuda_symgs_full_pipeline.py      # SYMGS full pipeline (succeeds)

# Stage 2 manual optimisation (LLM suggestions → hand-implemented CUDA)
python stage2_manual_optimization_test.py

# Abinit → cuFFT case study
python test_abinit_cufft.py
```

Results are saved to `results/<experiment_name>/`. Full logs go to `logs/`.

---

## Repository Structure

```
llm-hpc-project/
├── src/                            # Core library (~4,000 lines Python)
│   ├── analyzer.py                 # HPCAnalyzer: single-stage LLM analysis + 5-dim evaluator
│   ├── cascaded_pipeline.py        # Fixed S1=GPT-4o, S2=GPT-5.2 pipeline (original)
│   ├── configurable_pipeline.py    # ★ Parametrised S1/S2 models for all ablations
│   ├── generalized_evaluator.py    # 5-dimension scoring against VTune ground truth
│   ├── benchmark_config.py         # Ground truth for miniMD / HPCG / Abinit
│   ├── extended_benchmark_config.py# Ground truth for HotSpot / SRAD / LULESH / NAS CG / Jacobi-2D
│   ├── vtune_integration.py        # VTune CSV parsing + hotspot registration
│   ├── analysis_pipeline.py        # End-to-end: VTune → LLM analysis → report
│   ├── converter.py                # CUDA code generation + compile/run harness
│   └── llm_client.py               # OpenAI API wrapper with cost tracking
│
├── prompts/                        # Prompt templates
│   ├── zero_shot.txt               # Baseline — no examples
│   ├── few_shot.txt / few_shot_v2.txt / few_shot_v3.txt   # With optimisation exemplars
│   └── contextual.txt              # With VTune profile data embedded
│
├── benchmarks/                     # Source code for all 9 programs
│   ├── minimd/                     # Molecular dynamics, C++
│   ├── hpcg/                       # SPMV + SYMGS, C++
│   ├── abinit/                     # DFT, Fortran (nonlop_ylm + FFT)
│   ├── hotspot/                    # Rodinia thermal simulation, C
│   ├── srad/                       # Rodinia image denoising, C
│   ├── lulesh/                     # LLNL shock hydrodynamics, C
│   ├── nas_cg/                     # NAS conjugate gradient, C
│   ├── jacobi2d/                   # PolyBench stencil, C
│   └── omp_comparison.c            # Standalone OpenMP baseline
│
├── configs/                        # YAML config files
├── docs/
│   ├── case_studies.md             # 8 detailed case studies
│   └── daily_log_*.md              # Development journal
├── gpu_conversion/                 # Generated CUDA kernels + benchmark harnesses
├── results/                        # All experiment outputs (JSON + logs)
├── logs/                           # Per-experiment run logs
│
├── run_ablation.py                 # Ablation A (5.2→5.2) + B (5.2→4o)
├── run_ablation_cd.py              # Ablation C (4o→5.4) + D (5.4→5.4)
├── run_role_swap.py                # ★ V1 Neutral + V3 Biased agreement
├── stage2_manual_optimization_test.py  # Manual CUDA from S2 suggestions
├── test_*.py                       # Per-experiment entry points
│
├── README.md                       # This file
├── requirements.txt
├── setup.py
├── Dockerfile
└── .gitignore
```

### Key design decision: `ConfigurableCascadedPipeline`

`src/configurable_pipeline.py` parametrises both S1 and S2 models. `run_role_swap.py` then subclasses it (`NeutralPipeline`, `BiasedAgreementPipeline`) and overrides only `_build_stage2_prompt` + `_get_stage2_system_prompt`. This is what made the Role-Swap experiment cheap to run — swapping one prompt template, not re-plumbing an entire pipeline.

---

## Benchmark Suite

Nine programs, all with VTune 2025.9 ground truth:

| Program | Source | Language | Domain | Top hotspot | % runtime | Bottleneck (GT) |
|---------|--------|----------|--------|-------------|-----------|-----------------|
| miniMD | Mantevo | C++ | Molecular dynamics | `ForceLJ::compute` | 75.0% | compute |
| HPCG SPMV | HPCG 3.1 | C++ | Sparse matrix-vector multiply | `ComputeSPMV_ref` | 27.2% | memory |
| HPCG SYMGS | HPCG 3.1 | C++ | Symmetric Gauss-Seidel | `ComputeSYMGS_ref` | 67.7% | memory + dependency |
| Abinit | ABINIT 9.10.4 | Fortran | DFT plane-wave | `sg_ffty` (FFT) | 40.7% | memory |
| HotSpot | Rodinia | C | Thermal simulation | `compute_tran_temp` | 100.0% | memory |
| SRAD | Rodinia | C | Image denoising (PDE) | `srad_kernel` | 98.1% | memory |
| LULESH | LLNL proxy app | C | Shock hydrodynamics | `CalcFBHourglassForceForElems` | 48.9% | memory |
| NAS CG | NAS Parallel | C | Conjugate gradient | `sparse_matvec` | 83.6% | memory |
| Jacobi-2D | PolyBench | C | 2D stencil | `jacobi_kernel` | 99.0% | memory |

**Hotspot identification accuracy of the pipeline: 9/9 = 100%** — LLMs reliably *locate* hotspots; the open question is whether they correctly *classify* why those hotspots are slow.

### Ground truth profiling

All ground truth was re-measured with Intel VTune Profiler 2025.9 (user-mode sampling, `-knob sampling-mode=sw`) on WSL2 Ubuntu 24.04. An earlier round of Callgrind-based profiling was superseded when VTune revealed that Abinit's actual hotspots are FFT functions (`sg_ffty`, `sg_fftpx`), not `nonlop_ylm` as initially assumed from code-structure analysis. See `docs/daily_log_20260305.md`.

---

## Experiments and Results

### Part I — Bottleneck Classification

#### 1. Original cascaded pipeline (S1=GPT-4o → S2=GPT-5.2, 9 programs)

| Program | Ground truth | S1 (GPT-4o) | S2 (GPT-5.2) outcome | S1 score | Cost |
|---------|--------------|-------------|----------------------|----------|------|
| miniMD | compute | compute ✓ | modified → memory/latency + sync | 84.8 | $0.064 |
| HPCG SPMV | memory | memory ✓ | confirmed | 76.3 | $0.050 |
| HPCG SYMGS | memory + dep | memory ✓ | modified → memory + dependency | 80.4 | $0.048 |
| Abinit | memory | compute ✗ | **corrected → memory + allocation** | 80.2 | $0.136 |
| HotSpot | memory | compute ✗ | **corrected → memory-bandwidth + branch** | 82.2 | $0.055 |
| SRAD | memory | compute ✗ | **corrected → memory + cache traffic** | 79.1 | $0.059 |
| LULESH | memory | compute ✗ | **corrected → memory + sync** | 88.2 | $0.074 |
| NAS CG | memory | memory ✓ | confirmed | 85.2 | $0.063 |
| Jacobi-2D | memory | memory ✓ | confirmed | 81.5 | $0.056 |

**Pattern:** GPT-4o misclassifies 4/9 programs, all in the same direction (memory → compute). S2 corrects all 4. S2 also modifies 2 programs where S1 matched GT (miniMD, SYMGS). S1+S2 combined cost: ~$0.61 for all 9 programs.

#### 2. Ablation studies — is it the pipeline or the model?

| Config | S1 | S2 | Corrections | Cost |
|--------|-----|-----|-------------|------|
| Original | GPT-4o | GPT-5.2 | 5/9 | $0.61 |
| Ablation A | GPT-5.2 | GPT-5.2 | 0/9 | $0.79 |
| Ablation B | GPT-5.2 | GPT-4o | 0/9 | $0.45 |
| Ablation C | GPT-4o | GPT-5.4 | 5/9 | $0.62 |
| Ablation D | GPT-5.4 | GPT-5.4 | 1/9 | $0.81 |

Ablation A/B show 0 corrections because GPT-5.2 as S1 already classifies everything as memory — there is nothing to correct. Ablation C matches Original (both memory-biased S2 correct the same 5 compute-biased calls). Ablation D shows one self-correction (LULESH), suggesting GPT-5.4 has slightly more self-skepticism than GPT-5.2.

#### 3. Role-Swap — the phase-3 experiment

All configurations fixed: S1=GPT-4o, S2=GPT-5.2, temperature=0, identical code inputs. Only the S2 prompt varies. All "changed" / "kept" labels below are computed by the pf taxonomy (`src/bottleneck_taxonomy.py`).

| Program | GT | S1 (4o) | Original S2 | V1 Neutral S2 | V3 Biased S2 |
|---------|-----|---------|-------------|---------------|--------------|
| miniMD | compute | compute | memory (over-corr) | memory (over-corr) | compute (kept) |
| HPCG SPMV | memory | memory | memory (kept) | memory (kept) | memory (kept) |
| HPCG SYMGS | memory | memory | memory + dep (modified, pf=kept) | memory (kept) | memory (kept) |
| Abinit | memory | compute | **memory (corrected)** | **memory (corrected)** | compute (kept) |
| HotSpot | memory | compute | **memory (corrected)** | **memory (corrected)** | **memory (corrected)** |
| SRAD | memory | compute | **memory (corrected)** | **memory (corrected)** | compute (kept) |
| LULESH | memory | compute | **memory (corrected)** | compute (kept) | compute (kept) |
| NAS CG | memory | memory | memory (kept) | memory (kept) | memory (kept) |
| Jacobi-2D | memory | memory | memory (kept) | memory (kept) | memory (kept) |
| **Changes (pf)** | | | **5/9** | **4/9** | **1/9** |

Between Original and V1 Neutral, only LULESH flips from "corrected" to "kept" — every other change survives the removal of the "validate/correct" directive. V3's single change (HotSpot) likely reflects HotSpot's low arithmetic intensity (~0.3 FLOPs/byte) overriding the agreement-inducing prompt.

**Conclusion:** removing the "validate/correct" directive is a weak intervention (5/9 → 4/9, a single program flips); adding an explicit "agree / confirm" directive is a strong intervention (5/9 → 1/9, four programs flip). S2's default is not blind compliance — independent judgment survives when the prompt is merely neutral. What collapses judgment at scale is active agreement induction. The cascade's correction capability is therefore best described as *"prompt structure preserves independent judgment when no agreement cue is present."*

### Part II — GPU Code Generation

#### Direct generation (CPU source → CUDA, single LLM call)

| Kernel | Model | CPU (ms) | GPU (ms) | Speedup | Correct? | Notes |
|--------|-------|----------|----------|---------|----------|-------|
| miniMD Force | GPT-4o | 20.79 | 1.45 | **14.34×** | ✓ (err 3.98e-13) | 100k atoms |
| miniMD Force | GPT-5.2 | 17.53 | 1.46 | 12.03× | ✓ (err 7.28e-12) | |
| HPCG SPMV | GPT-4o | — | — | 7.11× | ✓ after fix | Parameter order error (Case Study #4) |
| HPCG SPMV | GPT-5.2 | — | — | **10.30×** | ✓ | Correct first time |
| HPCG SYMGS | GPT-4o | — | — | **0.02×** ✗ | **No** | 50× *slower* than CPU — serial dependency ignored |

#### Strategy-hint generation (multi-colour hint for SYMGS)

| Kernel | Model | Speedup | Correct? |
|--------|-------|---------|----------|
| HPCG SYMGS | GPT-4o | 3.14× | ✓ |
| HPCG SYMGS | GPT-5.2 | 2.79× | ✓ |

#### Full pipeline (cascaded analysis → strategy → CUDA)

| Kernel | CPU (ms) | GPU (ms) | Speedup | Correct? |
|--------|----------|----------|---------|----------|
| miniMD Force | 27.92 | 1.79 | **15.59×** | ✓ (err 1.16e-10) |
| HPCG SPMV | 1.95 | 0.31 | 6.18× ✗ | No (wrong format assumed, CSR vs ELL-like) |
| HPCG SYMGS | — | — | **5.61×** | ✓ |

#### Stage 2 manual — LLM suggestions implemented by hand

| Kernel | Suggested technique | Speedup | vs direct gen |
|--------|---------------------|---------|---------------|
| miniMD | SoA layout + CSR neighbour list + constant memory + uint16 type | **16.97×** | +18.3% |
| SPMV | SELL-C-σ + warp-per-row (32T/row) + column-major ELL | 7.95× | −22.8% |
| SYMGS | Multi-colour GS + invDiag precompute + CUDA Graph + ELL | 1.96×† | N/A† |

†Multi-colour GS changes the algorithm; not directly comparable to strict GS direct generation.

**Interpretation:**
- miniMD Stage 2 suggestions effectively reduced memory bandwidth traffic (+18.3% beyond direct gen).
- SPMV Stage 2 suggestions provided correct coalesced access but offered no advantage at this problem size — direct gen already hits near-peak bandwidth.
- SYMGS cannot exceed ~5–6× without algorithmic change; the kernel launch overhead of multi-colouring (16 launches per SYMGS: 8 colours × forward+backward) is the limit. Berger-Vergiat et al. (2021) propose Two-Stage GS as a more GPU-friendly alternative — noted as future work.

### Part III — OpenMP vs CUDA

Same problem sizes, `gcc -O2 -fopenmp`, 8 threads.

| Kernel | Serial | OpenMP 8T | CUDA | CUDA / OpenMP |
|--------|--------|-----------|------|---------------|
| miniMD Force | 41.03 ms | 6.09 ms (6.73×) | 2.86 ms (14.34×) | 2.1× |
| SPMV | 2.25 ms | 1.26 ms (1.79×) | 0.22 ms (10.30×) | **5.8×** |
| SYMGS | 3.46 ms | 0.81 ms (4.29×) | 0.62 ms (5.61×) | 1.3× |

LLM-generated CUDA provides the most added value for bandwidth-bound codes (SPMV, 5.8×) and less for dependency-bound codes (SYMGS, 1.3×, algorithmic ceiling).

### Part IV — Abinit cuFFT case study (Case Study #8)

After VTune re-profiling revealed FFT functions as the real hotspots (~68% of runtime), the cascaded pipeline was rerun with the correct input. Stage 1 (GPT-4o) recommended `library_replacement`; Stage 2 (GPT-5.2) produced a detailed cuFFT integration plan:

| Aspect | Recommendation |
|--------|---------------|
| API mapping | `sg_fft_cc` → `cufftPlanMany` + `cufftExecZ2Z` (batched over ndat) |
| Eliminated | `sg_ffty`, `sg_fftpx` no longer needed (covered by single 3D cuFFT) |
| Data layout | `arr(2, nd1*nd2*nd3)` → `cufftDoubleComplex` (conversion kernel) |
| Memory | Persistent device buffers across SCF iterations and k-points |
| Expected FFT speedup | 3–10× (45³ grid is small; batching essential) |
| Expected overall speedup | 1.8–3.5× (FFT = 68% of runtime, transfer overhead limits) |

Demonstrates that the LLM's role in GPU acceleration extends beyond kernel generation to **architectural decision-making** — correctly choosing library replacement over hand-writing a custom FFT kernel.

---

## Case Study Summary

Full write-ups in `docs/case_studies.md`.

| # | Case | Category | Key finding |
|---|------|----------|-------------|
| 1 | miniMD → CUDA | Success | 14.34× direct gen; 16.97× with manual Stage 2 (SoA + CSR) |
| 2 | SPMV → CUDA | Success | 10.30× direct gen; SELL-C-σ offers no advantage at this problem size |
| 3 | miniMD cascaded correction | Partial | S2 modifies correct compute label to memory — contradicts GT |
| 4 | GPT-4o SPMV parameter error | Failure → fixed | LLM hallucinated argument ordering; manual fix required |
| 5 | Abinit cascaded correction | Partial | compute → memory/allocation reclassification |
| 6 | SYMGS data dependency | Key finding | Direct gen 0.02× fails; full pipeline 5.61× succeeds |
| 7 | Full pipeline comparison | Comparative | Simple code: direct > pipeline. Complex code: pipeline essential |
| 8 | Abinit FFT → cuFFT | New (Plan B) | LLM recommends library replacement over custom kernel |

---

## Evaluation Framework

Each LLM analysis is scored on a 0–100 composite across five dimensions:

| Dimension | Weight | Measures |
|-----------|--------|----------|
| Hotspot identification | 25% | Correct function/loop location (4-level matching hierarchy) |
| Bottleneck classification | 30% | compute / memory / communication (keyword-based) |
| GPU suitability | 20% | Correct Boolean + substantive reasoning (>50 chars) |
| Suggestion quality | 10% | 60% quantity + 40% completeness (target, technique, speedup, difficulty) |
| Analysis depth | 15% | Character-threshold reasoning depth across 4 sub-areas |

**Hotspot matching hierarchy:** regex (0.90) → loop keyword (0.80) → function keyword (0.75) → string similarity × 0.70.

**Bottleneck scoring:** exact match = 1.0, partial = 0.5, wrong category but identified = 0.3, unrecognisable = 0.0.

See `src/generalized_evaluator.py` for full implementation.

---

## Cost Summary

Aligned with thesis §8. All figures in USD.

| Experiment category | Programs | Cost |
|--------------------|----------|------|
| Original 4 programs cascaded analysis | 4 | $0.315 |
| Extended 5 programs cascaded analysis | 5 | $0.297 |
| Abinit cuFFT analysis | 1 | $0.079 |
| Ablation A (5.2→5.2) | 9 | $0.792 |
| Ablation B (5.2→4o) | 9 | $0.450 |
| Ablation C (4o→5.4) | 9 | $0.622 |
| Ablation D (5.4→5.4) | 9 | $0.807 |
| GPT-5.4 single-stage bias test (2026-03-18) | 9 | $0.284 |
| GPT-5.4 single-stage re-run (2026-04-17, full hotspot capture) | 9 | $0.279 |
| Role-Swap V1 Neutral | 9 | $0.560 |
| Role-Swap V3 Biased | 9 | $0.560 |
| CUDA generation + Stage 2 manual | — | included in test scripts |
| **Project total** | | **~$5.05** |

Note: the 2026-04-17 GPT-5.4 re-run was done to preserve the full S1 hotspots array (the 2026-03-18 run only saved the primary bottleneck, which made post-hoc rescoring against updated GT impossible). Both runs are listed because both cost money; only the 2026-04-17 re-run is used as canonical (see §12.2 of thesis v8).

---

## Provenance & Reproducibility

Every headline number in this README traces to one of four sources, all reproducible from the scripts in this repository:

| Source | What it covers | How to regenerate |
|--------|----------------|-------------------|
| **VTune 2025.9** | Ground-truth bottleneck type and runtime percentages for all 9 programs (§1.x of thesis v8) | Profiling results archived in `docs/vtune/` and quoted in `src/benchmark_config.py` + `src/extended_benchmark_config.py` |
| **`rescore_all.py`** | `Changes (pf)` counts for every configuration (5/9, 0/9, 0/9, 5/9, 1/9, 4/9, 1/9) | `python rescore_all.py` → writes `results/pf_summary/summary_table.md` |
| **`rescore_eval_all.py`** | Canonical S1 eval scores for every configuration under VTune-aligned percentage GT | `python rescore_eval_all.py` → writes `results/pf_summary/eval_rescore_comparison.md` |
| **`rerun_model_bias.py`** | GPT-5.4 single-stage results (7/9 accuracy, §12.2 of thesis v8) | `python rerun_model_bias.py --model gpt-5.4` (costs ~$0.28) |

### Single source of truth for bottleneck classification

All scripts import `primary()` and `classify()` from a single module, `src/bottleneck_taxonomy.py`. There are no duplicated heuristics and no local "primary category" functions anywhere else in the codebase. This eliminated the v7-era inconsistency between three slightly different counting methods (see Finding 1's "Historical note").

### Re-running the pipeline after changing ground truth

If VTune ground truth changes (new machine, different problem size), the workflow is:

1. Edit `src/benchmark_config.py` and/or `src/extended_benchmark_config.py` to reflect the new VTune values.
2. Run `python rescore_eval_all.py` — this recomputes S1 eval scores for every existing configuration without touching the API (scores are derived from the saved S1 hotspots arrays in each configuration's JSON files).
3. Run `python rescore_all.py` — this recomputes cross-configuration change counts under the pf taxonomy.
4. Optionally re-run `rerun_model_bias.py` if the underlying S1 hotspots need re-collection (not just re-scoring).

No API calls are needed for steps 2–3, so re-scoring against new GT is ~free.

---

## Experimental Environment

| Component | Details |
|-----------|---------|
| CPU | Intel Tiger Lake H, 2.304 GHz, 8 physical / 16 logical cores |
| GPU | NVIDIA RTX 3060 (6 GB), CUDA 12.6, SM 86 |
| OS | WSL2 Ubuntu 24.04 on Windows |
| Profiler | Intel VTune Profiler 2025.9 (user-mode sampling) |
| Compiler | gcc 13.2, nvcc (CUDA 12.6) |
| OpenMP | `gcc -O2 -fopenmp`, 8 threads |
| LLM models | GPT-4o, GPT-5.2, GPT-5.4 (OpenAI API, temperature=0) |
| HPC programs | miniMD, HPCG 3.1, ABINIT 9.10.4 |
| Extended benchmarks | HotSpot, SRAD (Rodinia); LULESH (LLNL); NAS CG; Jacobi-2D (PolyBench) |

---

## Narrative Evolution

This project's interpretive framing went through three phases. The thesis and all results documents reflect Phase 3 (refined); earlier framings are superseded but the underlying data is unchanged.

| Phase | Date | Framing | Replaced by |
|-------|------|---------|-------------|
| Phase 1 | Before 2026-03-18 | "Cascade works because GPT-5.2 is stronger and corrects GPT-4o" | Phase 2 |
| Phase 2 | 2026-03-18 | "Cascade works because of model × role interaction; models have mirror-image biases" | Phase 3 |
| Phase 3 | 2026-04-16 | "Cascade works because the prompt instruction forces S2 to produce independent judgment; model × bias effects are secondary" | Phase 3 (refined) |
| **Phase 3 (refined, current)** | **2026-04-17** | **"Cascade works because S2's independent judgment is preserved when the prompt is neutral-or-skeptical (4–5/9); judgment collapses to 1/9 only under explicit agreement induction. Prompt structure is a first-order variable at the same order as model-pair selection."** | — |

The Role-Swap experiment is the pivot between Phase 2 and Phase 3: identical S1 + identical S2 model + three different S2 prompts yields **5/9, 4/9, 1/9** primary-category changes respectively — a prompt-level effect of the same magnitude as any model-level effect observed in earlier ablations. (The earlier 2026-04-16 reading reported V1 Neutral as 0/9 under a buggy counting function; see Finding 1's "Historical note" for why the corrected value is 4/9.)

---

## Contributions

1. **Prompt is a first-class control variable in LLM cascades.** Moving from "model capability" to "prompt structure" as the explanatory variable is reproducible, manipulable, and engineering-actionable. The Role-Swap methodology (fix every other variable, perturb only prompt) can be applied to any cascade-style LLM system.
2. **Directional bias is a systematic property of LLM bottleneck classification.** All three tested models exhibit bias; the directions differ (4o → compute, 5.2 → memory, 5.4 → strong memory). Newer is not more accurate.
3. **GPU generation strategy depends on algorithmic structure.** Simple parallel → direct generation. Complex dependency → full pipeline. Over-analysis harms simple cases (SPMV: −40%).
4. **LLMs are capable of architectural decision-making**, e.g. correctly recommending cuFFT over custom FFT kernel (Case Study #8).
5. **Pipeline design matters more than model selection.** The cascade structure with a "validate/correct" prompt achieves 9/9 accuracy on nine programs; a single-model solution with GPT-5.2 alone achieves 8/9 but cannot catch its own errors.

---

## Limitations and Future Work

- **Problem size.** All experiments on a single consumer-grade RTX 3060 (6 GB). SELL-C-σ SPMV and other optimisations may show different relative performance at larger problem sizes.
- **Model coverage.** Only GPT-4o / 5.2 / 5.4 tested. Claude, Gemini, and open-source models (Llama, Qwen) would strengthen the generality of the prompt-vs-model finding.
- **Prompt coverage.** Role-Swap tested three S2 prompt variants. A more systematic prompt-space exploration (e.g. Chain-of-Thought, self-critique, multi-turn dialogue) is future work.
- **SYMGS algorithmic ceiling.** Two-Stage Gauss-Seidel (Berger-Vergiat et al. 2021, arXiv:2104.01196) may break the multi-colouring ceiling; evaluating whether LLMs can discover this algorithmic substitution is a natural next step.
- **Stage 2 suggestion fidelity.** Stage 2 manual showed LLM suggestions are sometimes optimistic (SPMV: −22.8% vs direct gen). Automated feedback loops (benchmark results fed back into the prompt) are unexplored here.

---

## Citation

If you use or reference this work:

```bibtex
@thesis{llm-hpc-2026,
  author = {Yusei},
  title  = {LLM-Assisted HPC Performance Analysis and GPU Code Generation:
            A Prompt-Driven View of Cascaded Pipelines},
  school = {[Your Institution]},
  year   = {2026},
  type   = {Undergraduate thesis}
}
```

---

## License

MIT License.

---

## Acknowledgements

Benchmark programs courtesy of the Mantevo project (miniMD), the HPCG project, ABINIT, the Rodinia benchmark suite, LLNL LULESH, the NAS Parallel Benchmarks, and PolyBench. Intel VTune Profiler used under the oneAPI community license.