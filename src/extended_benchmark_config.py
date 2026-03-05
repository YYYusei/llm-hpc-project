"""
扩展基准程序配置 - 新增 5 个 benchmark
用于 cascaded pipeline 泛化性测试

使用方法:
    from extended_benchmark_config import EXTENDED_DEFINITIONS, register_extended_benchmarks
    register_extended_benchmarks()  # 注册到全局 registry
"""

from benchmark_config import BenchmarkDefinition, HotspotDefinition, get_registry


EXTENDED_DEFINITIONS = {

    # -------- Rodinia HotSpot --------
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
                time_percentage=85.0,
                bottleneck_type="compute",
                loop_keywords=["inner loop", "i loop", "j loop", "stencil loop", "grid loop"],
                memory_patterns=["2D grid", "5-point stencil", "regular access"]
            ),
            HotspotDefinition(
                name="read_input",
                location_patterns=[r"read_input", r"file.*read"],
                time_percentage=2.0,
                bottleneck_type="io",
                loop_keywords=["file read"],
                memory_patterns=["sequential file read"]
            )
        ],
        gpu_suitable=True,
        gpu_notes="Regular 2D grid stencil. Each cell update is independent per iteration. "
                  "Tiling can improve shared memory usage.",
        function_keywords=["compute_tran_temp", "read_input", "write_output"],
        structure_keywords=["stencil", "thermal", "temperature", "grid", "diffusion",
                            "Rx", "Ry", "Rz", "capacitance"],
    ),

    # -------- Rodinia SRAD --------
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
                time_percentage=78.0,
                bottleneck_type="compute",
                loop_keywords=["pixel loop", "i loop", "j loop", "diffusion loop"],
                memory_patterns=["2D image", "neighbor access", "gradient computation"]
            ),
            HotspotDefinition(
                name="compute_statistics",
                location_patterns=[r"compute_statistics", r"statistics", r"mean.*var"],
                time_percentage=3.0,
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
                time_percentage=45.0,
                bottleneck_type="compute",
                loop_keywords=["element loop", "mode loop", "node loop", "hourglass mode"],
                memory_patterns=["gather/scatter", "indirect indexing", "element connectivity"]
            ),
            HotspotDefinition(
                name="CalcKinematicsForElems",
                location_patterns=[
                    r"kinematics", r"CalcKinematics", r"volume.*derivative",
                    r"strain.*rate"
                ],
                time_percentage=25.0,
                bottleneck_type="compute",
                loop_keywords=["element loop", "cross product", "triple product"],
                memory_patterns=["gather from nodes", "indirect indexing"]
            ),
            HotspotDefinition(
                name="CalcPressureForElems",
                location_patterns=[r"pressure", r"CalcPressure", r"equation.*state"],
                time_percentage=10.0,
                bottleneck_type="compute",
                loop_keywords=["element loop"],
                memory_patterns=["contiguous access"]
            )
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
                time_percentage=65.0,
                bottleneck_type="memory",
                loop_keywords=["row loop", "inner loop", "column loop", "j loop"],
                memory_patterns=["CSR format", "indirect indexing", "irregular access",
                                 "cache miss"]
            ),
            HotspotDefinition(
                name="dot_product",
                location_patterns=[r"dot_product", r"dot", r"inner.*product", r"ddot"],
                time_percentage=15.0,
                bottleneck_type="memory",
                loop_keywords=["reduction loop", "vector loop"],
                memory_patterns=["streaming access", "reduction"]
            ),
            HotspotDefinition(
                name="vec_axpy",
                location_patterns=[r"vec_axpy", r"axpy", r"daxpy"],
                time_percentage=8.0,
                bottleneck_type="memory",
                loop_keywords=["vector loop"],
                memory_patterns=["streaming access"]
            )
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
                time_percentage=90.0,
                bottleneck_type="memory",
                loop_keywords=["i loop", "j loop", "stencil loop", "grid loop",
                               "forward sweep", "backward sweep"],
                memory_patterns=["5-point stencil", "ping-pong buffer", "regular access",
                                 "streaming access"]
            ),
            HotspotDefinition(
                name="compute_residual",
                location_patterns=[r"compute_residual", r"residual", r"laplacian"],
                time_percentage=3.0,
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
