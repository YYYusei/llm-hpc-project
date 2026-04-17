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
