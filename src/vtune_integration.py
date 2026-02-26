import os
import re
import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class HotspotData:
    """热点数据"""
    function_name: str
    module: str
    cpu_time: float           # 秒
    cpu_time_percentage: float  # 百分比
    instructions_retired: Optional[int] = None
    cpi_rate: Optional[float] = None  # Cycles Per Instruction
    source_file: Optional[str] = None
    start_line: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VTuneReport:
    """VTune 分析报告"""
    timestamp: str
    analysis_type: str        # hotspots, memory-access, etc.
    total_elapsed_time: float
    total_cpu_time: float
    cpu_utilization: float
    
    # 系统信息
    cpu_model: str = ""
    cores: int = 0
    threads: int = 0
    
    # 热点列表
    hotspots: List[HotspotData] = field(default_factory=list)
    
    # 原始数据
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_top_hotspots(self, n: int = 10) -> List[HotspotData]:
        """获取 Top-N 热点"""
        sorted_hotspots = sorted(
            self.hotspots, 
            key=lambda h: h.cpu_time_percentage, 
            reverse=True
        )
        return sorted_hotspots[:n]
    
    def to_ground_truth(self) -> Dict[str, Any]:
        """转换为 ground truth 格式"""
        top_hotspots = self.get_top_hotspots(5)
        
        if not top_hotspots:
            return {}
        
        primary = top_hotspots[0]
        
        return {
            "primary_hotspot": primary.function_name,
            "hotspots": [
                {
                    "name": h.function_name,
                    "time_percentage": round(h.cpu_time_percentage, 1),
                    "bottleneck_type": self._infer_bottleneck_type(h)
                }
                for h in top_hotspots
            ],
            "gpu_suitable": True,  # 需要人工确认
            "source": "vtune",
            "timestamp": self.timestamp
        }
    
    def to_profiling_data(self) -> Dict[str, Any]:
        """转换为 profiling_data 格式（用于 contextual prompt）"""
        data = {
            "source": "vtune",
            "analysis_type": self.analysis_type,
            "total_elapsed_time": self.total_elapsed_time,
            "total_cpu_time": self.total_cpu_time,
            "cpu_utilization": round(self.cpu_utilization, 1),
            "system_info": {
                "cpu": self.cpu_model,
                "cores": self.cores,
                "threads": self.threads
            },
            "hotspots": {}
        }
        
        for h in self.get_top_hotspots(10):
            key = h.function_name.lower().replace("::", "_").replace(" ", "_")
            data["hotspots"][key] = {
                "time": round(h.cpu_time, 3),
                "percentage": round(h.cpu_time_percentage, 1)
            }
        
        return data
    
    def _infer_bottleneck_type(self, hotspot: HotspotData) -> str:
        """推断瓶颈类型（基于 CPI 和函数名）"""
        # 基于 CPI 判断
        if hotspot.cpi_rate:
            if hotspot.cpi_rate > 2.0:
                return "memory"  # 高 CPI 通常表示内存瓶颈
            elif hotspot.cpi_rate < 1.0:
                return "compute"  # 低 CPI 表示计算密集
        
        # 基于函数名关键词判断
        func_lower = hotspot.function_name.lower()
        
        memory_keywords = ["memcpy", "memset", "gather", "scatter", "load", "store", 
                          "fetch", "cache", "malloc", "free", "alloc"]
        compute_keywords = ["compute", "calc", "force", "kernel", "multiply", "add",
                           "fft", "blas", "lapack", "matmul"]
        comm_keywords = ["mpi", "send", "recv", "bcast", "reduce", "alltoall", "barrier"]
        
        for kw in memory_keywords:
            if kw in func_lower:
                return "memory"
        
        for kw in compute_keywords:
            if kw in func_lower:
                return "compute"
        
        for kw in comm_keywords:
            if kw in func_lower:
                return "communication"
        
        # 默认基于时间占比猜测
        return "compute" if hotspot.cpu_time_percentage > 50 else "memory"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "analysis_type": self.analysis_type,
            "total_elapsed_time": self.total_elapsed_time,
            "total_cpu_time": self.total_cpu_time,
            "cpu_utilization": self.cpu_utilization,
            "cpu_model": self.cpu_model,
            "cores": self.cores,
            "threads": self.threads,
            "hotspots": [h.to_dict() for h in self.hotspots],
            "raw_data": self.raw_data
        }
    
    def save(self, filepath: str):
        """保存报告到 JSON 文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"VTune report saved to {filepath}")


class VTuneParser:
    """VTune 报告解析器"""
    
    def __init__(self):
        self.supported_formats = ["csv", "txt", "json"]
    
    def parse(self, filepath: str) -> VTuneReport:
        """
        解析 VTune 报告文件
        
        支持的格式:
        - CSV: vtune -report hotspots -format csv
        - TXT: vtune -report hotspots -format text
        - JSON: 自定义 JSON 格式
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"VTune report not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        if suffix == ".csv":
            return self._parse_csv(filepath)
        elif suffix == ".txt":
            return self._parse_text(filepath)
        elif suffix == ".json":
            return self._parse_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _parse_csv(self, filepath: Path) -> VTuneReport:
        """解析 CSV 格式的 VTune 报告"""
        hotspots = []
        total_cpu_time = 0.0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            # 跳过元数据行
            lines = f.readlines()
            
            # 找到数据开始的行
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith("Function,") or "CPU Time" in line:
                    data_start = i
                    break
            
            # 解析 CSV 数据
            reader = csv.DictReader(lines[data_start:])
            
            for row in reader:
                try:
                    # VTune CSV 格式可能有不同的列名
                    func_name = row.get("Function") or row.get("Function Name") or ""
                    module = row.get("Module") or row.get("Module Name") or ""
                    
                    # CPU 时间
                    cpu_time_str = row.get("CPU Time") or row.get("CPU Time:Self") or "0"
                    cpu_time = self._parse_time(cpu_time_str)
                    
                    if cpu_time > 0 and func_name:
                        hotspots.append(HotspotData(
                            function_name=func_name.strip(),
                            module=module.strip(),
                            cpu_time=cpu_time,
                            cpu_time_percentage=0.0,  # 稍后计算
                            cpi_rate=self._safe_float(row.get("CPI Rate")),
                            instructions_retired=self._safe_int(row.get("Instructions Retired"))
                        ))
                        total_cpu_time += cpu_time
                except Exception as e:
                    logger.warning(f"Failed to parse row: {e}")
                    continue
        
        # 计算百分比
        for h in hotspots:
            if total_cpu_time > 0:
                h.cpu_time_percentage = (h.cpu_time / total_cpu_time) * 100
        
        return VTuneReport(
            timestamp=datetime.now().isoformat(),
            analysis_type="hotspots",
            total_elapsed_time=total_cpu_time,  # 近似值
            total_cpu_time=total_cpu_time,
            cpu_utilization=0.0,  # CSV 中可能没有
            hotspots=hotspots
        )
    
    def _parse_text(self, filepath: Path) -> VTuneReport:
        """解析文本格式的 VTune 报告"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        hotspots = []
        total_cpu_time = 0.0
        elapsed_time = 0.0
        cpu_utilization = 0.0
        
        # 解析总体信息
        elapsed_match = re.search(r"Elapsed Time:\s*([\d.]+)\s*s", content)
        if elapsed_match:
            elapsed_time = float(elapsed_match.group(1))
        
        cpu_util_match = re.search(r"CPU Utilization:\s*([\d.]+)\s*%", content)
        if cpu_util_match:
            cpu_utilization = float(cpu_util_match.group(1))
        
        # 解析热点表格
        # 典型格式: Function  Module  CPU Time  CPU Time:Self  ...
        hotspot_pattern = re.compile(
            r"^\s*([^\s][^\t]+?)\s{2,}([^\s]+)\s+([\d.]+)s?\s+([\d.]+)%",
            re.MULTILINE
        )
        
        for match in hotspot_pattern.finditer(content):
            func_name = match.group(1).strip()
            module = match.group(2).strip()
            cpu_time = float(match.group(3))
            percentage = float(match.group(4))
            
            # 过滤掉表头行
            if func_name.lower() in ["function", "function name"]:
                continue
            
            hotspots.append(HotspotData(
                function_name=func_name,
                module=module,
                cpu_time=cpu_time,
                cpu_time_percentage=percentage
            ))
            total_cpu_time += cpu_time
        
        return VTuneReport(
            timestamp=datetime.now().isoformat(),
            analysis_type="hotspots",
            total_elapsed_time=elapsed_time,
            total_cpu_time=total_cpu_time,
            cpu_utilization=cpu_utilization,
            hotspots=hotspots
        )
    
    def _parse_json(self, filepath: Path) -> VTuneReport:
        """解析 JSON 格式（自定义或导出）"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        hotspots = []
        for h in data.get("hotspots", []):
            hotspots.append(HotspotData(
                function_name=h.get("function_name", ""),
                module=h.get("module", ""),
                cpu_time=h.get("cpu_time", 0.0),
                cpu_time_percentage=h.get("cpu_time_percentage", 0.0),
                cpi_rate=h.get("cpi_rate"),
                instructions_retired=h.get("instructions_retired"),
                source_file=h.get("source_file"),
                start_line=h.get("start_line")
            ))
        
        return VTuneReport(
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            analysis_type=data.get("analysis_type", "hotspots"),
            total_elapsed_time=data.get("total_elapsed_time", 0.0),
            total_cpu_time=data.get("total_cpu_time", 0.0),
            cpu_utilization=data.get("cpu_utilization", 0.0),
            cpu_model=data.get("cpu_model", ""),
            cores=data.get("cores", 0),
            threads=data.get("threads", 0),
            hotspots=hotspots,
            raw_data=data
        )
    
    def _parse_time(self, time_str: str) -> float:
        """解析时间字符串"""
        if not time_str:
            return 0.0
        
        # 移除单位
        time_str = time_str.strip().lower().replace("s", "").replace(",", "")
        
        try:
            return float(time_str)
        except ValueError:
            return 0.0
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """安全转换为 float"""
        if value is None:
            return None
        try:
            return float(str(value).replace(",", ""))
        except ValueError:
            return None
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """安全转换为 int"""
        if value is None:
            return None
        try:
            return int(str(value).replace(",", ""))
        except ValueError:
            return None


class VTuneRunner:
    """VTune 运行器（自动化 profiling）"""
    
    def __init__(self, vtune_path: str = "vtune"):
        """
        初始化 VTune 运行器
        
        Args:
            vtune_path: VTune 可执行文件路径
        """
        self.vtune_path = vtune_path
        self._check_vtune()
    
    def _check_vtune(self):
        """检查 VTune 是否可用"""
        try:
            result = subprocess.run(
                [self.vtune_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"VTune found: {result.stdout.strip()}")
            else:
                logger.warning("VTune check failed, may not be available")
        except FileNotFoundError:
            logger.warning(f"VTune not found at: {self.vtune_path}")
        except Exception as e:
            logger.warning(f"VTune check error: {e}")
    
    def run_hotspot_analysis(
        self,
        command: str,
        output_dir: str,
        duration: int = 60,
        extra_args: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        运行热点分析
        
        Args:
            command: 要分析的程序命令
            output_dir: 输出目录
            duration: 分析时长（秒）
            extra_args: 额外的 VTune 参数
            
        Returns:
            报告文件路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_dir = output_dir / f"vtune_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 构建 VTune 命令
        vtune_cmd = [
            self.vtune_path,
            "-collect", "hotspots",
            "-result-dir", str(result_dir),
            "-duration", str(duration),
        ]
        
        if extra_args:
            vtune_cmd.extend(extra_args)
        
        vtune_cmd.append("--")
        vtune_cmd.extend(command.split())
        
        logger.info(f"Running VTune: {' '.join(vtune_cmd)}")
        
        try:
            result = subprocess.run(
                vtune_cmd,
                capture_output=True,
                text=True,
                timeout=duration + 120  # 额外缓冲时间
            )
            
            if result.returncode != 0:
                logger.error(f"VTune failed: {result.stderr}")
                return None
            
            # 生成报告
            report_path = output_dir / "hotspots_report.csv"
            report_cmd = [
                self.vtune_path,
                "-report", "hotspots",
                "-result-dir", str(result_dir),
                "-format", "csv",
                "-report-output", str(report_path)
            ]
            
            subprocess.run(report_cmd, capture_output=True, text=True)
            
            if report_path.exists():
                logger.info(f"VTune report saved to: {report_path}")
                return str(report_path)
            else:
                logger.error("Failed to generate VTune report")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("VTune analysis timed out")
            return None
        except Exception as e:
            logger.error(f"VTune error: {e}")
            return None


def create_profiling_data_from_vtune(
    vtune_report_path: str,
    benchmark_name: str
) -> Dict[str, Any]:
    """
    从 VTune 报告创建 profiling_data（用于 contextual prompt）
    
    Args:
        vtune_report_path: VTune 报告文件路径
        benchmark_name: 基准程序名称
        
    Returns:
        profiling_data 字典
    """
    parser = VTuneParser()
    report = parser.parse(vtune_report_path)
    
    profiling_data = report.to_profiling_data()
    profiling_data["benchmark"] = benchmark_name
    
    return profiling_data


def create_ground_truth_from_vtune(
    vtune_report_path: str,
    benchmark_name: str,
    gpu_suitable: bool = True
) -> Dict[str, Any]:
    """
    从 VTune 报告创建 ground_truth
    
    Args:
        vtune_report_path: VTune 报告文件路径
        benchmark_name: 基准程序名称
        gpu_suitable: GPU 是否适合（需要人工判断）
        
    Returns:
        ground_truth 字典
    """
    parser = VTuneParser()
    report = parser.parse(vtune_report_path)
    
    ground_truth = report.to_ground_truth()
    ground_truth["benchmark"] = benchmark_name
    ground_truth["gpu_suitable"] = gpu_suitable
    
    return ground_truth


# ============== 手动输入 VTune 数据的辅助函数 ==============

def create_manual_vtune_report(
    hotspots: List[Dict[str, Any]],
    total_time: float,
    cpu_utilization: float = 0.0,
    cpu_model: str = "",
    cores: int = 0,
    threads: int = 0
) -> VTuneReport:
    """
    手动创建 VTune 报告（当没有自动化 profiling 时）
    
    Args:
        hotspots: 热点列表，格式: [{"name": "func", "time": 1.0, "percentage": 50.0}, ...]
        total_time: 总运行时间
        cpu_utilization: CPU 利用率
        cpu_model: CPU 型号
        cores: 核心数
        threads: 线程数
        
    Returns:
        VTuneReport 对象
    """
    hotspot_list = []
    for h in hotspots:
        hotspot_list.append(HotspotData(
            function_name=h["name"],
            module=h.get("module", ""),
            cpu_time=h.get("time", 0.0),
            cpu_time_percentage=h.get("percentage", 0.0),
            cpi_rate=h.get("cpi_rate"),
            source_file=h.get("source_file"),
            start_line=h.get("start_line")
        ))
    
    return VTuneReport(
        timestamp=datetime.now().isoformat(),
        analysis_type="hotspots (manual)",
        total_elapsed_time=total_time,
        total_cpu_time=sum(h.cpu_time for h in hotspot_list),
        cpu_utilization=cpu_utilization,
        cpu_model=cpu_model,
        cores=cores,
        threads=threads,
        hotspots=hotspot_list
    )


# ============== 示例用法 ==============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 示例1: 手动创建 VTune 报告（基于你已有的数据）
    print("=== 示例1: 手动创建 miniMD VTune 报告 ===")
    
    minimd_report = create_manual_vtune_report(
        hotspots=[
            {"name": "ForceLJ::compute", "time": 3.685, "percentage": 73.7},
            {"name": "Neighbor::build", "time": 0.859, "percentage": 17.2},
            {"name": "Integrate::initialIntegrate", "time": 0.061, "percentage": 1.2},
            {"name": "Integrate::finalIntegrate", "time": 0.052, "percentage": 1.0},
        ],
        total_time=5.0,
        cpu_utilization=9.5,
        cpu_model="Intel Core i7-11800H",
        cores=8,
        threads=16
    )
    
    print(f"Top hotspots: {[h.function_name for h in minimd_report.get_top_hotspots(3)]}")
    print(f"Ground truth: {minimd_report.to_ground_truth()}")
    print()
    
    # 示例2: 手动创建 HPCG VTune 报告
    print("=== 示例2: 手动创建 HPCG VTune 报告 ===")
    
    hpcg_report = create_manual_vtune_report(
        hotspots=[
            {"name": "ComputeSYMGS_ref", "time": 45.731, "percentage": 67.3},
            {"name": "ComputeSPMV_ref", "time": 18.834, "percentage": 27.7},
            {"name": "ComputeDotProduct_ref", "time": 1.358, "percentage": 2.0},
            {"name": "ComputeWAXPBY_ref", "time": 0.865, "percentage": 1.3},
        ],
        total_time=67.977,
        cpu_utilization=11.6,
        cpu_model="Intel Core i7-11800H",
        cores=8,
        threads=16
    )
    
    print(f"Top hotspots: {[h.function_name for h in hpcg_report.get_top_hotspots(3)]}")
    print(f"Profiling data: {hpcg_report.to_profiling_data()}")
    print()
    
    # 保存报告
    minimd_report.save("results/vtune/minimd_vtune_report.json")
    hpcg_report.save("results/vtune/hpcg_vtune_report.json")
