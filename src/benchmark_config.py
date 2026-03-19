"""
通用基准程序配置模块
支持 miniMD, HPCG, Abinit, CP2K 等多种 HPC 基准程序

设计原则：
1. 配置驱动而非硬编码
2. 支持不同语言（C++/Fortran）
3. 支持不同瓶颈类型
4. 可扩展的关键词系统
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class HotspotDefinition:
    """热点定义"""
    name: str                          # 函数/区域名称
    location_patterns: List[str]       # 位置匹配模式（正则表达式或关键词）
    time_percentage: float             # 预期时间占比
    bottleneck_type: str               # compute / memory / communication
    loop_keywords: List[str] = field(default_factory=list)  # 循环相关关键词
    memory_patterns: List[str] = field(default_factory=list)  # 内存访问模式描述


@dataclass
class BenchmarkDefinition:
    """基准程序定义"""
    name: str                          # 程序名称
    full_name: str                     # 完整名称
    language: str                      # 编程语言 (cpp / fortran / mixed)
    domain: str                        # 应用领域 (md / linear_algebra / dft / quantum_chemistry)
    
    # 代码路径
    source_files: List[str] = field(default_factory=list)
    hotspot_files: List[str] = field(default_factory=list)
    
    # 热点定义
    hotspots: List[HotspotDefinition] = field(default_factory=list)
    
    # GPU 适用性
    gpu_suitable: bool = True
    gpu_notes: str = ""
    
    # 关键词（用于评估匹配）
    function_keywords: List[str] = field(default_factory=list)
    structure_keywords: List[str] = field(default_factory=list)
    
    # Profiling 数据模板
    profiling_template: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_keywords(self) -> set:
        """获取所有关键词"""
        keywords = set(self.function_keywords + self.structure_keywords)
        for hotspot in self.hotspots:
            keywords.update(hotspot.location_patterns)
            keywords.update(hotspot.loop_keywords)
        return keywords


# ============== 预定义的基准程序配置 ==============

BENCHMARK_DEFINITIONS: Dict[str, BenchmarkDefinition] = {
    
    # -------- miniMD --------
    "minimd": BenchmarkDefinition(
        name="minimd",
        full_name="miniMD - Molecular Dynamics Proxy App",
        language="cpp",
        domain="md",
        hotspot_files=["force_lj.cpp"],
        hotspots=[
            HotspotDefinition(
                name="ForceLJ::compute",
                location_patterns=[
                    r"forcelj", r"force_lj", r"compute",
                    r"compute_original", r"compute_halfneigh", r"compute_fullneigh"
                ],
                time_percentage=75.0,  # VTune: 3.487s / 4.650s
                bottleneck_type="compute",
                loop_keywords=["k-loop", "inner loop", "neighbor loop", "for.*numneigh"],
                memory_patterns=["neighbor list", "indirect indexing"]
            ),
            HotspotDefinition(
                name="Neighbor::build",
                location_patterns=[r"neighbor", r"neigh", r"build"],
                time_percentage=17.9,  # VTune: 0.834s / 4.650s
                bottleneck_type="memory",
                loop_keywords=["bin loop", "atom loop"],
                memory_patterns=["binning", "cell list"]
            )
        ],
        gpu_suitable=True,
        gpu_notes="Force calculation is highly parallel; each atom pair interaction is independent",
        function_keywords=["forcelj", "compute", "neighbor", "integrate"],
        structure_keywords=["atom pair", "lennard-jones", "cutoff", "neighbor list"],
        profiling_template={
            "source": "vtune",
            "metrics": ["t_force", "t_neigh", "t_integrate"]
        }
    ),
    
    # -------- HPCG --------
    "hpcg": BenchmarkDefinition(
        name="hpcg",
        full_name="HPCG - High Performance Conjugate Gradients",
        language="cpp",
        domain="linear_algebra",
        hotspot_files=["ComputeSPMV_ref.cpp", "ComputeSYMGS_ref.cpp"],
        hotspots=[
            HotspotDefinition(
                name="ComputeSYMGS_ref",
                location_patterns=[r"symgs", r"computesymgs", r"gauss.*seidel"],
                time_percentage=67.7,  # VTune: 47.480s / 70.138s
                bottleneck_type="memory",
                loop_keywords=["forward sweep", "backward sweep", "row loop"],
                memory_patterns=["sparse matrix", "indirect indexing", "sequential dependency"]
            ),
            HotspotDefinition(
                name="ComputeSPMV_ref",
                location_patterns=[r"spmv", r"computespmv", r"sparse.*matrix.*vector"],
                time_percentage=27.2,  # VTune: 19.071s / 70.138s
                bottleneck_type="memory",
                loop_keywords=["j-loop", "inner loop", "column loop"],
                memory_patterns=["CSR format", "indirect indexing", "irregular access"]
            ),
            HotspotDefinition(
                name="ComputeDotProduct_ref",
                location_patterns=[r"dot", r"ddot", r"dotproduct"],
                time_percentage=2.0,  # VTune: 1.392s / 70.138s
                bottleneck_type="memory",
                loop_keywords=["reduction loop"],
                memory_patterns=["streaming access", "reduction"]
            )
        ],
        gpu_suitable=True,
        gpu_notes="SpMV benefits from GPU but SYMGS has sequential dependencies that limit parallelism",
        function_keywords=["spmv", "symgs", "ddot", "waxpby", "mg", "cg"],
        structure_keywords=["sparse matrix", "conjugate gradient", "multigrid", "preconditioner"],
        profiling_template={
            "source": "vtune",
            "metrics": ["symgs_time", "spmv_time", "ddot_time", "waxpby_time"]
        }
    ),
    
    # -------- Abinit --------
    # Updated 2026-03-05 based on actual VTune profiling (si_big.abi, ecut=40, 8x8x8 kgrid)
    # Previous config was based on literature estimates and was incorrect:
    #   - nonlop was assumed to be 40% but does not appear in top hotspots
    #   - Actual hotspots are FFT functions (sg_ffty, sg_fftpx, fftrisc)
    "abinit": BenchmarkDefinition(
        name="abinit",
        full_name="ABINIT - First-principles DFT Code",
        language="fortran",
        domain="dft",
        hotspot_files=["m_fft.F90", "sg_fft.F90", "m_mkffkg.F90", "m_nonlop_ylm.F90"],
        hotspots=[
            HotspotDefinition(
                name="sg_ffty",
                location_patterns=[
                    r"sg_ffty", r"sg_fft", r"fft.*y",
                    r"fourier.*transform"
                ],
                time_percentage=40.7,  # VTune: 3.991s / 9.799s
                bottleneck_type="memory",  # VTune: FFT memory-bandwidth bound at this problem size
                loop_keywords=["fft loop", "butterfly", "radix", "y-direction"],
                memory_patterns=["stride access", "complex multiply", "twiddle factors"]
            ),
            HotspotDefinition(
                name="sg_fftpx",
                location_patterns=[
                    r"sg_fftpx", r"fftpx", r"fft.*px",
                    r"fft.*x"
                ],
                time_percentage=15.8,  # VTune: 1.550s / 9.799s
                bottleneck_type="memory",  # VTune: FFT memory-bandwidth bound
                loop_keywords=["fft loop", "x-direction", "plane"],
                memory_patterns=["stride access", "complex multiply"]
            ),
            HotspotDefinition(
                name="fftrisc_one_nothreadsafe",
                location_patterns=[
                    r"fftrisc", r"fft.*risc", r"fft.*dispatch",
                    r"fft.*one"
                ],
                time_percentage=11.2,  # VTune: 1.101s / 9.799s
                bottleneck_type="memory",  # VTune: FFT dispatch, memory-bound
                loop_keywords=["fft dispatch", "fft scheduler"],
                memory_patterns=["work array", "buffer management"]
            ),
            HotspotDefinition(
                name="dfpt_mkffkg",
                location_patterns=[
                    r"mkffkg", r"dfpt_mkffkg", r"ffkg",
                    r"kernel.*generation"
                ],
                time_percentage=6.4,  # VTune: 0.632s / 9.799s
                bottleneck_type="memory",  # VTune: kernel generation, memory-bound
                loop_keywords=["kernel loop", "plane wave loop", "g-vector"],
                memory_patterns=["reciprocal space", "form factors"]
            ),
            HotspotDefinition(
                name="zgemm",
                location_patterns=[r"zgemm", r"blas.*gemm", r"matrix.*multiply"],
                time_percentage=3.9,  # VTune: 0.385s / 9.799s (from libblas)
                bottleneck_type="compute",
                loop_keywords=["matrix multiply", "gemm"],
                memory_patterns=["dense matrix", "BLAS level 3"]
            )
        ],
        gpu_suitable=True,
        gpu_notes="FFT operations dominate (~68% of runtime). GPU acceleration via cuFFT "
                  "is the most effective approach. The sg_ffty and sg_fftpx functions "
                  "implement custom radix FFT kernels that could be replaced by cuFFT calls. "
                  "zgemm can be accelerated via cuBLAS.",
        function_keywords=["sg_ffty", "sg_fftpx", "fftrisc", "mkffkg", "zgemm",
                           "fourwf", "nonlop", "getghc"],
        structure_keywords=["plane wave", "FFT", "fourier", "reciprocal space",
                            "pseudopotential", "kpoint", "band", "radix"],
        profiling_template={
            "source": "vtune",
            "metrics": ["sg_ffty_time", "sg_fftpx_time", "fftrisc_time",
                        "mkffkg_time", "zgemm_time"]
        }
    ),
    
    # -------- CP2K --------
    "cp2k": BenchmarkDefinition(
        name="cp2k",
        full_name="CP2K - Quantum Chemistry and Solid State Physics",
        language="fortran",
        domain="quantum_chemistry",
        hotspot_files=["grid_integrate.F", "pw_gpu.F", "dbcsr_mm.F"],
        hotspots=[
            HotspotDefinition(
                name="grid_integrate",
                location_patterns=[r"grid.*integrate", r"collocate", r"integrate"],
                time_percentage=35.0,
                bottleneck_type="compute",
                loop_keywords=["grid loop", "gaussian loop", "atom loop"],
                memory_patterns=["grid points", "gaussian basis"]
            ),
            HotspotDefinition(
                name="pw_operations",
                location_patterns=[r"pw_", r"fft", r"plane.*wave"],
                time_percentage=25.0,
                bottleneck_type="memory",
                loop_keywords=["fft loop", "g-vector loop"],
                memory_patterns=["3D FFT", "distributed grid"]
            ),
            HotspotDefinition(
                name="dbcsr_multiply",
                location_patterns=[r"dbcsr", r"sparse.*multiply", r"mm_"],
                time_percentage=20.0,
                bottleneck_type="compute",
                loop_keywords=["block loop", "matrix multiply"],
                memory_patterns=["block sparse", "DBCSR format"]
            )
        ],
        gpu_suitable=True,
        gpu_notes="Grid operations and DBCSR multiply are GPU-accelerated in CP2K; use CUDA/HIP backend",
        function_keywords=["dbcsr", "grid", "collocate", "pw_", "qs_"],
        structure_keywords=["gaussian basis", "plane wave", "hybrid functional", "DBCSR"],
        profiling_template={
            "source": "gprof",
            "metrics": ["grid_time", "fft_time", "dbcsr_time", "diag_time"]
        }
    )
}


class BenchmarkRegistry:
    """基准程序注册表"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化注册表
        
        Args:
            config_file: 可选的YAML配置文件路径，用于覆盖或扩展默认配置
        """
        self.benchmarks: Dict[str, BenchmarkDefinition] = BENCHMARK_DEFINITIONS.copy()
        
        if config_file:
            self._load_config(config_file)
    
    def _load_config(self, config_file: str):
        """从配置文件加载/覆盖配置"""
        config_path = Path(config_file)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_file}")
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'benchmarks' in config:
            for name, bench_config in config['benchmarks'].items():
                if name in self.benchmarks:
                    # 更新现有配置
                    self._update_benchmark(name, bench_config)
                else:
                    # 添加新配置
                    logger.info(f"Adding new benchmark from config: {name}")
    
    def _update_benchmark(self, name: str, config: Dict[str, Any]):
        """更新基准程序配置"""
        bench = self.benchmarks[name]
        
        # 更新 ground_truth 相关字段
        if 'ground_truth' in config:
            gt = config['ground_truth']
            if bench.hotspots and gt.get('hotspot'):
                # 尝试匹配并更新热点
                for hotspot in bench.hotspots:
                    if hotspot.name.lower() in gt['hotspot'].lower():
                        if 'time_percentage' in gt:
                            hotspot.time_percentage = gt['time_percentage']
                        if 'bottleneck_type' in gt:
                            hotspot.bottleneck_type = gt['bottleneck_type']
            if 'gpu_suitable' in gt:
                bench.gpu_suitable = gt['gpu_suitable']
        
        # 更新 profiling_data
        if 'profiling_data' in config:
            bench.profiling_template.update(config['profiling_data'])
    
    def get(self, name: str) -> Optional[BenchmarkDefinition]:
        """获取基准程序定义"""
        return self.benchmarks.get(name.lower())
    
    def list_all(self) -> List[str]:
        """列出所有支持的基准程序"""
        return list(self.benchmarks.keys())
    
    def get_keywords_for_benchmark(self, name: str) -> set:
        """获取指定基准程序的所有关键词"""
        bench = self.get(name)
        if bench:
            return bench.get_all_keywords()
        return set()
    
    def get_all_keywords(self) -> set:
        """获取所有基准程序的关键词（用于通用匹配）"""
        all_keywords = set()
        for bench in self.benchmarks.values():
            all_keywords.update(bench.get_all_keywords())
        return all_keywords
    
    def find_matching_benchmark(self, code_content: str) -> Optional[str]:
        """
        根据代码内容自动检测是哪个基准程序
        
        Args:
            code_content: 代码文件内容
            
        Returns:
            匹配的基准程序名称，如果没有匹配则返回 None
        """
        code_lower = code_content.lower()
        
        scores = {}
        for name, bench in self.benchmarks.items():
            score = 0
            keywords = bench.get_all_keywords()
            for kw in keywords:
                if kw.lower() in code_lower:
                    score += 1
            scores[name] = score
        
        if scores:
            best_match = max(scores, key=scores.get)
            if scores[best_match] >= 2:  # 至少匹配2个关键词
                return best_match
        
        return None


# 全局注册表实例
_registry: Optional[BenchmarkRegistry] = None


def get_registry(config_file: Optional[str] = None) -> BenchmarkRegistry:
    """获取全局注册表实例"""
    global _registry
    if _registry is None:
        _registry = BenchmarkRegistry(config_file)
    return _registry


def get_benchmark(name: str) -> Optional[BenchmarkDefinition]:
    """便捷函数：获取基准程序定义"""
    return get_registry().get(name)


def get_all_keywords() -> set:
    """便捷函数：获取所有关键词"""
    return get_registry().get_all_keywords()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    registry = BenchmarkRegistry()
    
    print("Supported benchmarks:")
    for name in registry.list_all():
        bench = registry.get(name)
        print(f"  - {name}: {bench.full_name} ({bench.language})")
        print(f"    Hotspots: {[h.name for h in bench.hotspots]}")
        print(f"    Keywords: {len(bench.get_all_keywords())} total")
        print()