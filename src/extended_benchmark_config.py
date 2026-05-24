"""
Extended benchmark configurations — 5 additional programs beyond miniMD/HPCG/Abinit.

2026-04-17 refactor (problem #2):
    All hotspot `time_percentage` values now match Intel VTune 2025.9
    measurements (see thesis §1.4). Previously these were static-code-analysis
    estimates, which caused evaluator.percentage_score to systematically
    penalise LLMs that reported runtime percentages closer to actual VTune
    values.

    Hotspots without VTune coverage (i.e., not appearing in the VTune
    top-hotspot report for that program) have been removed. The remaining
    hotspots are the full set that VTune actually reports, at their measured
    percentages.

    Changes summary:
        HotSpot    compute_tran_temp     85.0% → 100.0%
                   read_input            [REMOVED — not in VTune output]
        SRAD       srad_kernel           78.0% → 98.1%
                   compute_statistics     3.0% →   0.3%
        LULESH     CalcFBHourglass...    45.0% →  48.9%
                   CalcKinematics...     25.0% →  18.7%
                   CalcPressure...       [REMOVED — not in VTune output]
        NAS CG     sparse_matvec         65.0% →  83.6%
                   dot_product           [REMOVED — not in VTune output]
                   vec_axpy              [REMOVED — not in VTune output]
        Jacobi-2D  jacobi_kernel         90.0% →  99.0%
                   compute_residual       3.0% →   0.1%

    Ground truth (bottleneck_type) is unchanged — those values were already
    VTune-derived in the 2026-03-18 revision.

Usage:
    from extended_benchmark_config import EXTENDED_DEFINITIONS, register_extended_benchmarks
    register_extended_benchmarks()  # 注册到全局 registry
"""

from benchmark_config import BenchmarkDefinition, HotspotDefinition, get_registry


EXTENDED_DEFINITIONS = {

    # -------- Rodinia HotSpot --------
    # VTune (thesis §1.4, 512×512 grid, 100 iterations):
    #   compute_tran_temp  0.060s  100.0%
    "hotspot": BenchmarkDefinition(
        name="hotspot",
        full_name="HotSpot - Thermal Simulation (Rodinia Suite)",
        language="c",
        domain="thermal_simulation",
        hotspot_files=["hotspot.c"],
        hotspots=[
            HotspotDefinition(
                name="compute_tran_temp",
                location_patterns=[
                    r"compute_tran_temp", r"tran.*temp", r"thermal",
                    r"stencil", r"temperature"
                ],
                time_percentage=100.0,  # VTune measured (single dominant kernel)
                bottleneck_type="memory",  # VTune: 5-point stencil, AI ~0.3 FLOPs/byte, memory-bandwidth bound
                loop_keywords=["inner loop", "i loop", "j loop", "stencil loop", "grid loop"],
                memory_patterns=["2D grid", "5-point stencil", "regular access"]
            ),
        ],
        gpu_suitable=True,
        gpu_notes="Regular 2D grid stencil. Each cell update is independent per iteration. "
                  "Tiling can improve shared memory usage.",
        function_keywords=["compute_tran_temp", "read_input", "write_output"],
        structure_keywords=["stencil", "thermal", "temperature", "grid", "diffusion",
                            "Rx", "Ry", "Rz", "capacitance"],
    ),

    # -------- Rodinia SRAD --------
    # VTune (thesis §1.4, 2048×2048 image, 100 iterations):
    #   srad_kernel          4.317s  98.1%
    #   random_init          0.027s   0.6%   (init, not analyzed as hotspot)
    #   compute_statistics   0.012s   0.3%
    "srad": BenchmarkDefinition(
        name="srad",
        full_name="SRAD - Speckle Reducing Anisotropic Diffusion (Rodinia Suite)",
        language="c",
        domain="image_processing",
        hotspot_files=["srad.c"],
        hotspots=[
            HotspotDefinition(
                name="srad_kernel",
                location_patterns=[
                    r"srad_kernel", r"srad", r"diffusion.*kernel",
                    r"compute.*coefficient", r"anisotropic"
                ],
                time_percentage=98.1,  # VTune measured
                bottleneck_type="memory",  # VTune: memory-bandwidth bound, AI <0.3 FLOPs/byte
                loop_keywords=["pixel loop", "i loop", "j loop", "diffusion loop"],
                memory_patterns=["2D image", "neighbor access", "gradient computation"]
            ),
            HotspotDefinition(
                name="compute_statistics",
                location_patterns=[r"compute_statistics", r"statistics", r"mean.*var"],
                time_percentage=0.3,  # VTune measured
                bottleneck_type="memory",
                loop_keywords=["reduction loop"],
                memory_patterns=["streaming access", "reduction"]
            )
        ],
        gpu_suitable=True,
        gpu_notes="Regular 2D grid access. Gradient and coefficient computation are per-pixel "
                  "independent. Division and exp() calls may limit throughput on GPU.",
        function_keywords=["srad_kernel", "compute_statistics", "random_init"],
        structure_keywords=["speckle", "diffusion", "gradient", "coefficient",
                            "ICOV", "anisotropic", "image", "ultrasound"],
    ),

    # -------- LULESH --------
    # VTune (thesis §1.4, SIDE_LENGTH=150, 3.375M elements):
    #   CalcFBHourglassForceForElems  0.288s  48.9%
    #   main (includes allocation)    0.149s  25.2%  (not a pipeline hotspot)
    #   CalcKinematicsForElems        0.110s  18.7%
    #   Other                         0.043s   7.2%
    "lulesh": BenchmarkDefinition(
        name="lulesh",
        full_name="LULESH - Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics",
        language="c",
        domain="hydrodynamics",
        hotspot_files=["lulesh_simplified.c"],
        hotspots=[
            HotspotDefinition(
                name="CalcFBHourglassForceForElems",
                location_patterns=[
                    r"hourglass", r"CalcFBHourglass", r"hg.*force",
                    r"anti.*hourglass"
                ],
                time_percentage=48.9,  # VTune measured
                bottleneck_type="memory",  # VTune: memory + sync/atomics, indirect gather/scatter
                loop_keywords=["element loop", "mode loop", "node loop", "hourglass mode"],
                memory_patterns=["gather/scatter", "indirect indexing", "element connectivity"]
            ),
            HotspotDefinition(
                name="CalcKinematicsForElems",
                location_patterns=[
                    r"kinematics", r"CalcKinematics", r"volume.*derivative",
                    r"strain.*rate"
                ],
                time_percentage=18.7,  # VTune measured
                bottleneck_type="memory",  # VTune: indirect gather, memory-latency bound
                loop_keywords=["element loop", "cross product", "triple product"],
                memory_patterns=["gather from nodes", "indirect indexing"]
            ),
        ],
        gpu_suitable=True,
        gpu_notes="Element-level parallelism is straightforward. However, the scatter operation "
                  "in hourglass forces (multiple elements writing to same node) requires "
                  "atomics or coloring. Gather is efficient.",
        function_keywords=["CalcFBHourglassForceForElems", "CalcKinematicsForElems",
                           "CalcPressureForElems", "nodelist"],
        structure_keywords=["hexahedral", "hourglass", "element", "node",
                            "stress", "force", "volume", "hydrodynamics"],
    ),

    # -------- NAS CG --------
    # VTune (thesis §1.4, N=14000, NONZER=11, 15 outer iterations):
    #   sparse_matvec  0.042s  83.6%
    #   __memset       0.008s  16.4%  (libc, not application code)
    "nas_cg": BenchmarkDefinition(
        name="nas_cg",
        full_name="NAS CG - Conjugate Gradient (NAS Parallel Benchmarks)",
        language="c",
        domain="sparse_linear_algebra",
        hotspot_files=["cg.c"],
        hotspots=[
            HotspotDefinition(
                name="sparse_matvec",
                location_patterns=[
                    r"sparse_matvec", r"spmv", r"matvec", r"matrix.*vector",
                    r"SpMV"
                ],
                time_percentage=83.6,  # VTune measured
                bottleneck_type="memory",
                loop_keywords=["row loop", "inner loop", "column loop", "j loop"],
                memory_patterns=["CSR format", "indirect indexing", "irregular access",
                                 "cache miss"]
            ),
        ],
        gpu_suitable=True,
        gpu_notes="SpMV is a classic GPU kernel (cuSPARSE). However, the indirect indexing "
                  "through column indices limits memory bandwidth utilization. "
                  "Dot product requires reduction across threads.",
        function_keywords=["sparse_matvec", "dot_product", "vec_axpy", "conj_grad",
                           "vec_scale", "vec_copy"],
        structure_keywords=["sparse", "CSR", "conjugate gradient", "matrix-vector",
                            "rowptr", "colidx", "residual", "convergence"],
    ),

    # -------- Jacobi-2D --------
    # VTune (thesis §1.4, 4096×4096 grid, 500 iterations):
    #   jacobi_kernel       21.606s  99.0%
    #   init_array           0.153s   0.7%   (init, not hotspot)
    #   compute_residual     0.032s   0.1%
    #   compute_checksum     0.020s   0.1%   (verification, not hotspot)
    "jacobi2d": BenchmarkDefinition(
        name="jacobi2d",
        full_name="Jacobi-2D - Iterative Stencil (PolyBench Suite)",
        language="c",
        domain="stencil_computation",
        hotspot_files=["jacobi2d.c"],
        hotspots=[
            HotspotDefinition(
                name="jacobi_kernel",
                location_patterns=[
                    r"jacobi_kernel", r"jacobi", r"stencil",
                    r"laplace", r"iteration"
                ],
                time_percentage=99.0,  # VTune measured
                bottleneck_type="memory",
                loop_keywords=["i loop", "j loop", "stencil loop", "grid loop",
                               "forward sweep", "backward sweep"],
                memory_patterns=["5-point stencil", "ping-pong buffer", "regular access",
                                 "streaming access"]
            ),
            HotspotDefinition(
                name="compute_residual",
                location_patterns=[r"compute_residual", r"residual", r"laplacian"],
                time_percentage=0.1,  # VTune measured
                bottleneck_type="memory",
                loop_keywords=["residual loop"],
                memory_patterns=["stencil access", "reduction"]
            )
        ],
        gpu_suitable=True,
        gpu_notes="Trivially parallel within each iteration (no data dependency). "
                  "Classic GPU stencil problem. Ping-pong buffering eliminates "
                  "read-after-write hazards. Tiling with shared memory very effective.",
        function_keywords=["jacobi_kernel", "compute_residual", "init_array",
                           "compute_checksum"],
        structure_keywords=["jacobi", "stencil", "laplace", "5-point",
                            "ping-pong", "iteration", "grid", "convergence"],
    ),

    # ============ PolyBench compute-bound 扩展 (2026-05, 平衡偏置) ============
    # 全部 VTune hotspots 实测(见扩展回填表 A),GT 由稠密线代算法特性 + AI 判定为 compute。

    # -------- GEMM (PolyBench, EXTRALARGE) --------
    # VTune: kernel_gemm 98.9%
    "gemm": BenchmarkDefinition(
        name="gemm",
        full_name="GEMM - General Dense Matrix Multiply (PolyBench Suite)",
        language="c",
        domain="linear_algebra",
        hotspot_files=["gemm.c"],
        hotspots=[
            HotspotDefinition(
                name="kernel_gemm",
                location_patterns=[r"kernel_gemm", r"gemm", r"matrix.*multiply", r"C\[i\]\[j\]"],
                time_percentage=98.9,  # VTune measured
                bottleneck_type="compute",  # dense GEMM: O(N^3) FLOPs / O(N^2) data, high AI, FLOP-bound
                loop_keywords=["i loop", "j loop", "k loop", "triple loop", "inner product"],
                memory_patterns=["row-major", "blocked access", "high data reuse"]
            ),
        ],
        gpu_suitable=True,
        gpu_notes="Embarrassingly parallel dense matmul. Tiling + shared memory + register "
                  "blocking give near-peak GPU throughput. Classic compute-bound GPU workload.",
        function_keywords=["kernel_gemm", "init_array"],
        structure_keywords=["gemm", "matrix multiply", "dense", "BLAS", "alpha", "beta", "GEMM"],
    ),

    # -------- 2MM (PolyBench, LARGE) --------
    # VTune: kernel_2mm 98.9%
    "2mm": BenchmarkDefinition(
        name="2mm",
        full_name="2MM - Two Chained Dense Matrix Multiplies (PolyBench Suite)",
        language="c",
        domain="linear_algebra",
        hotspot_files=["2mm.c"],
        hotspots=[
            HotspotDefinition(
                name="kernel_2mm",
                location_patterns=[r"kernel_2mm", r"2mm", r"matrix.*multiply", r"tmp", r"D\[i\]\[j\]"],
                time_percentage=98.9,  # VTune measured
                bottleneck_type="compute",  # two chained GEMMs (D = A.B.C), FLOP-bound
                loop_keywords=["i loop", "j loop", "k loop", "two matmul", "chained"],
                memory_patterns=["row-major", "intermediate tmp matrix", "high data reuse"]
            ),
        ],
        gpu_suitable=True,
        gpu_notes="Two chained GEMMs. Each is independently tile-parallelisable; intermediate "
                  "matrix can stay in GPU memory. Compute-bound, high GPU suitability.",
        function_keywords=["kernel_2mm", "init_array"],
        structure_keywords=["2mm", "two matrix multiply", "chained", "dense", "tmp", "alpha", "beta"],
    ),

    # -------- 3MM (PolyBench, LARGE) --------
    # VTune: kernel_3mm 99.4%
    "3mm": BenchmarkDefinition(
        name="3mm",
        full_name="3MM - Three Chained Dense Matrix Multiplies (PolyBench Suite)",
        language="c",
        domain="linear_algebra",
        hotspot_files=["3mm.c"],
        hotspots=[
            HotspotDefinition(
                name="kernel_3mm",
                location_patterns=[r"kernel_3mm", r"3mm", r"matrix.*multiply", r"E\[i\]\[j\]", r"G\[i\]\[j\]"],
                time_percentage=99.4,  # VTune measured
                bottleneck_type="compute",  # three chained GEMMs (G = (A.B).(C.D)), FLOP-bound
                loop_keywords=["i loop", "j loop", "k loop", "three matmul", "chained"],
                memory_patterns=["row-major", "two intermediate matrices", "high data reuse"]
            ),
        ],
        gpu_suitable=True,
        gpu_notes="Three chained GEMMs. Same parallelisation as GEMM applied three times; "
                  "intermediates reside on device. Compute-bound, high GPU suitability.",
        function_keywords=["kernel_3mm", "init_array"],
        structure_keywords=["3mm", "three matrix multiply", "chained", "dense", "E", "F", "G"],
    ),

    # -------- SYRK (PolyBench, EXTRALARGE) --------
    # VTune: kernel_syrk 99.7%
    "syrk": BenchmarkDefinition(
        name="syrk",
        full_name="SYRK - Symmetric Rank-k Update (PolyBench Suite)",
        language="c",
        domain="linear_algebra",
        hotspot_files=["syrk.c"],
        hotspots=[
            HotspotDefinition(
                name="kernel_syrk",
                location_patterns=[r"kernel_syrk", r"syrk", r"rank.*k", r"C\[i\]\[j\]"],
                time_percentage=99.7,  # VTune measured
                bottleneck_type="compute",  # C = alpha*A*A^T + beta*C, BLAS-3, FLOP-bound
                loop_keywords=["i loop", "j loop", "k loop", "rank-k update"],
                memory_patterns=["row-major", "symmetric update", "high data reuse"]
            ),
        ],
        gpu_suitable=True,
        gpu_notes="Symmetric rank-k update (BLAS-3). Triangular output halves work but remains "
                  "compute-bound; tile-parallelisable. High GPU suitability.",
        function_keywords=["kernel_syrk", "init_array"],
        structure_keywords=["syrk", "rank-k", "symmetric", "dense", "BLAS", "alpha", "beta"],
    ),

    # -------- SYR2K (PolyBench, LARGE) --------
    # VTune: kernel_syr2k 99.3%
    "syr2k": BenchmarkDefinition(
        name="syr2k",
        full_name="SYR2K - Symmetric Rank-2k Update (PolyBench Suite)",
        language="c",
        domain="linear_algebra",
        hotspot_files=["syr2k.c"],
        hotspots=[
            HotspotDefinition(
                name="kernel_syr2k",
                location_patterns=[r"kernel_syr2k", r"syr2k", r"rank.*2k", r"C\[i\]\[j\]"],
                time_percentage=99.3,  # VTune measured
                bottleneck_type="compute",  # C = alpha*A*B^T + alpha*B*A^T + beta*C, BLAS-3, FLOP-bound
                loop_keywords=["i loop", "j loop", "k loop", "rank-2k update"],
                memory_patterns=["row-major", "symmetric update", "high data reuse"]
            ),
        ],
        gpu_suitable=True,
        gpu_notes="Symmetric rank-2k update (BLAS-3). Two coupled products; compute-bound and "
                  "tile-parallelisable. High GPU suitability.",
        function_keywords=["kernel_syr2k", "init_array"],
        structure_keywords=["syr2k", "rank-2k", "symmetric", "dense", "BLAS", "alpha", "beta"],
    ),

    # -------- DOITGEN (PolyBench, EXTRALARGE) --------
    # VTune: kernel_doitgen 97.7%
    "doitgen": BenchmarkDefinition(
        name="doitgen",
        full_name="Doitgen - Multiresolution ADI Tensor Contraction (PolyBench Suite)",
        language="c",
        domain="linear_algebra",
        hotspot_files=["doitgen.c"],
        hotspots=[
            HotspotDefinition(
                name="kernel_doitgen",
                location_patterns=[r"kernel_doitgen", r"doitgen", r"tensor", r"sum", r"A\[r\]\[q\]\[p\]"],
                time_percentage=97.7,  # VTune measured
                bottleneck_type="compute",  # 3D tensor x matrix contraction, high reuse, FLOP-bound
                loop_keywords=["r loop", "q loop", "p loop", "s loop", "tensor contraction"],
                memory_patterns=["3D tensor", "contraction over s", "high data reuse"]
            ),
        ],
        gpu_suitable=True,
        gpu_notes="Multiresolution tensor contraction (tensor x matrix). High arithmetic intensity, "
                  "parallel over outer tensor indices. Compute-bound, suitable for GPU.",
        function_keywords=["kernel_doitgen", "init_array"],
        structure_keywords=["doitgen", "tensor", "contraction", "multiresolution", "ADI", "sum"],
    ),

    # -------- GRAMSCHMIDT (PolyBench, LARGE) --------
    # VTune: kernel_gramschmidt 99.6%
    "gramschmidt": BenchmarkDefinition(
        name="gramschmidt",
        full_name="Gram-Schmidt - QR Decomposition via Orthonormalisation (PolyBench Suite)",
        language="c",
        domain="linear_algebra",
        hotspot_files=["gramschmidt.c"],
        hotspots=[
            HotspotDefinition(
                name="kernel_gramschmidt",
                location_patterns=[r"kernel_gramschmidt", r"gramschmidt", r"gram.*schmidt", r"orthogonal", r"QR"],
                time_percentage=99.6,  # VTune measured
                bottleneck_type="compute",  # QR orthonormalisation, FLOP-dense but column-carried dependency
                loop_keywords=["k loop", "i loop", "j loop", "projection", "normalisation"],
                memory_patterns=["column access", "projection", "running orthogonalisation"]
            ),
        ],
        gpu_suitable=False,
        gpu_notes="QR via classical Gram-Schmidt. Compute-dense but has column-by-column dependency "
                  "(each column orthogonalised against all previous), limiting naive parallelism. "
                  "Partial GPU suitability: within-column ops parallel, but the outer column loop is serial.",
        function_keywords=["kernel_gramschmidt", "init_array"],
        structure_keywords=["gramschmidt", "QR", "orthogonal", "orthonormal", "projection", "Q", "R"],
    ),
}


def register_extended_benchmarks():
    """注册扩展的 benchmark 配置到全局 registry"""
    registry = get_registry()
    for name, defn in EXTENDED_DEFINITIONS.items():
        registry.benchmarks[name] = defn
    print(f"Registered {len(EXTENDED_DEFINITIONS)} extended benchmarks: "
          f"{list(EXTENDED_DEFINITIONS.keys())}")


if __name__ == "__main__":
    register_extended_benchmarks()
    registry = get_registry()
    
    print(f"\nAll registered benchmarks ({len(registry.benchmarks)}):")
    for name in sorted(registry.benchmarks.keys()):
        b = registry.get(name)
        hotspot_names = [h.name for h in b.hotspots]
        print(f"  {name:12s} | {b.full_name}")
        print(f"{'':14s} | Hotspots: {hotspot_names}")
        print(f"{'':14s} | Primary bottleneck: {b.hotspots[0].bottleneck_type}")
        print(f"{'':14s} | GPU suitable: {b.gpu_suitable}")
        print()
