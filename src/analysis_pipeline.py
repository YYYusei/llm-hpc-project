"""
VTune + LLM 分析集成工作流

提供端到端的分析流程:
1. 运行/解析 VTune profiling
2. 生成 ground truth
3. 运行 LLM 分析
4. 使用通用评估器评估
5. 生成报告
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from vtune_integration import (
    VTuneParser, 
    VTuneRunner, 
    VTuneReport,
    create_manual_vtune_report,
    create_profiling_data_from_vtune,
    create_ground_truth_from_vtune
)
from benchmark_config import get_benchmark, get_registry, BenchmarkDefinition
from generalized_evaluator import GeneralizedEvaluator, EvaluationResult
from analyzer import HPCAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)


class IntegratedAnalysisPipeline:
    """
    集成分析流水线
    
    支持两种模式:
    1. 自动模式: VTune 运行 → 解析 → LLM 分析 → 评估
    2. 手动模式: 提供 VTune 数据 → LLM 分析 → 评估
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        prompts_dir: str = "prompts",
        output_dir: str = "results"
    ):
        """
        初始化流水线
        
        Args:
            api_key: OpenAI API Key
            model: LLM 模型名称
            prompts_dir: Prompt 模板目录
            output_dir: 输出目录
        """
        self.analyzer = HPCAnalyzer(api_key=api_key, model=model, prompts_dir=prompts_dir)
        self.evaluator = GeneralizedEvaluator()
        self.vtune_parser = VTuneParser()
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        (self.output_dir / "vtune").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "analysis").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(parents=True, exist_ok=True)
        
        logger.info("Integrated Analysis Pipeline initialized")
    
    def run_with_vtune_report(
        self,
        code_path: str,
        vtune_report_path: str,
        benchmark_name: str,
        prompt_types: List[str] = None,
        gpu_suitable: bool = True
    ) -> Dict[str, Any]:
        """
        使用已有的 VTune 报告运行分析
        
        Args:
            code_path: 代码文件路径
            vtune_report_path: VTune 报告文件路径
            benchmark_name: 基准程序名称
            prompt_types: 要测试的 prompt 类型
            gpu_suitable: GPU 是否适合（用于 ground truth）
            
        Returns:
            完整的分析结果
        """
        prompt_types = prompt_types or ["zero_shot", "few_shot", "contextual"]
        
        # 1. 解析 VTune 报告
        logger.info(f"Parsing VTune report: {vtune_report_path}")
        vtune_report = self.vtune_parser.parse(vtune_report_path)
        
        # 2. 生成 ground truth 和 profiling data
        ground_truth = vtune_report.to_ground_truth()
        ground_truth["gpu_suitable"] = gpu_suitable
        profiling_data = vtune_report.to_profiling_data()
        
        # 3. 运行分析
        return self._run_analysis_pipeline(
            code_path=code_path,
            benchmark_name=benchmark_name,
            ground_truth=ground_truth,
            profiling_data=profiling_data,
            prompt_types=prompt_types,
            vtune_report=vtune_report
        )
    
    def run_with_manual_vtune_data(
        self,
        code_path: str,
        benchmark_name: str,
        hotspots: List[Dict[str, Any]],
        total_time: float,
        cpu_utilization: float = 0.0,
        system_info: Optional[Dict[str, Any]] = None,
        prompt_types: List[str] = None,
        gpu_suitable: bool = True
    ) -> Dict[str, Any]:
        """
        使用手动输入的 VTune 数据运行分析
        
        Args:
            code_path: 代码文件路径
            benchmark_name: 基准程序名称
            hotspots: 热点列表 [{"name": "func", "time": 1.0, "percentage": 50.0}, ...]
            total_time: 总运行时间
            cpu_utilization: CPU 利用率
            system_info: 系统信息 {"cpu": "...", "cores": 8, "threads": 16}
            prompt_types: 要测试的 prompt 类型
            gpu_suitable: GPU 是否适合
            
        Returns:
            完整的分析结果
        """
        prompt_types = prompt_types or ["zero_shot", "few_shot", "contextual"]
        system_info = system_info or {}
        
        # 1. 创建手动 VTune 报告
        vtune_report = create_manual_vtune_report(
            hotspots=hotspots,
            total_time=total_time,
            cpu_utilization=cpu_utilization,
            cpu_model=system_info.get("cpu", ""),
            cores=system_info.get("cores", 0),
            threads=system_info.get("threads", 0)
        )
        
        # 2. 生成 ground truth 和 profiling data
        ground_truth = vtune_report.to_ground_truth()
        ground_truth["gpu_suitable"] = gpu_suitable
        profiling_data = vtune_report.to_profiling_data()
        
        # 添加系统信息
        profiling_data["system_info"] = system_info
        
        # 3. 运行分析
        return self._run_analysis_pipeline(
            code_path=code_path,
            benchmark_name=benchmark_name,
            ground_truth=ground_truth,
            profiling_data=profiling_data,
            prompt_types=prompt_types,
            vtune_report=vtune_report
        )
    
    def run_with_config(
        self,
        code_path: str,
        benchmark_name: str,
        prompt_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        使用 benchmark_config 中预定义的配置运行分析
        
        Args:
            code_path: 代码文件路径
            benchmark_name: 基准程序名称（必须在 benchmark_config 中定义）
            prompt_types: 要测试的 prompt 类型
            
        Returns:
            完整的分析结果
        """
        prompt_types = prompt_types or ["zero_shot", "few_shot", "contextual"]
        
        # 1. 获取基准程序配置
        benchmark = get_benchmark(benchmark_name)
        if not benchmark:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        # 2. 从配置生成 ground truth
        ground_truth = {
            "primary_hotspot": benchmark.hotspots[0].name if benchmark.hotspots else "",
            "hotspots": [
                {
                    "name": h.name,
                    "time_percentage": h.time_percentage,
                    "bottleneck_type": h.bottleneck_type
                }
                for h in benchmark.hotspots
            ],
            "gpu_suitable": benchmark.gpu_suitable,
            "source": "benchmark_config"
        }
        
        # 3. 从配置生成 profiling data（基础版）
        profiling_data = {
            "source": benchmark.profiling_template.get("source", "config"),
            "hotspots": {
                h.name.lower().replace("::", "_"): {
                    "percentage": h.time_percentage
                }
                for h in benchmark.hotspots
            }
        }
        
        # 4. 运行分析
        return self._run_analysis_pipeline(
            code_path=code_path,
            benchmark_name=benchmark_name,
            ground_truth=ground_truth,
            profiling_data=profiling_data,
            prompt_types=prompt_types,
            vtune_report=None
        )
    
    def _run_analysis_pipeline(
        self,
        code_path: str,
        benchmark_name: str,
        ground_truth: Dict[str, Any],
        profiling_data: Dict[str, Any],
        prompt_types: List[str],
        vtune_report: Optional[VTuneReport] = None
    ) -> Dict[str, Any]:
        """
        运行分析流水线核心逻辑
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "benchmark_name": benchmark_name,
            "code_path": code_path,
            "timestamp": timestamp,
            "ground_truth": ground_truth,
            "profiling_data": profiling_data,
            "analyses": [],
            "evaluations": [],
            "summary": {}
        }
        
        # 保存 VTune 报告
        if vtune_report:
            vtune_path = self.output_dir / "vtune" / f"{benchmark_name}_{timestamp}.json"
            vtune_report.save(str(vtune_path))
            results["vtune_report_path"] = str(vtune_path)
        
        # 对每种 prompt 类型运行分析
        for prompt_type in prompt_types:
            logger.info(f"Running {prompt_type} analysis for {benchmark_name}...")
            
            try:
                # 运行 LLM 分析
                analysis = self.analyzer.analyze(
                    code_path=code_path,
                    prompt_type=prompt_type,
                    profiling_data=profiling_data if prompt_type == "contextual" else None,
                    code_name=benchmark_name
                )
                
                # 保存分析结果
                analysis_path = self.output_dir / "analysis" / f"{benchmark_name}_{prompt_type}_{timestamp}.json"
                analysis.save(str(analysis_path))
                
                # 使用通用评估器评估
                eval_result = self.evaluator.evaluate(
                    identified={
                        "hotspots": analysis.hotspots,
                        "bottleneck_type": analysis.bottleneck_type,
                        "gpu_suitability": analysis.gpu_suitability,
                        "optimization_suggestions": analysis.optimization_suggestions
                    },
                    benchmark_name=benchmark_name,
                    prompt_type=prompt_type
                )
                
                # 保存评估结果
                eval_path = self.output_dir / "evaluation" / f"{benchmark_name}_{prompt_type}_{timestamp}.json"
                self._save_evaluation(eval_result, str(eval_path))
                
                # 添加到结果
                results["analyses"].append({
                    "prompt_type": prompt_type,
                    "file_path": str(analysis_path),
                    "hotspots_count": len(analysis.hotspots),
                    "bottleneck_type": analysis.bottleneck_type.get("primary", "unknown"),
                    "gpu_suitable": analysis.gpu_suitability.get("suitable"),
                    "suggestions_count": len(analysis.optimization_suggestions),
                    "tokens_used": analysis.total_tokens,
                    "cost": analysis.cost
                })
                
                results["evaluations"].append({
                    "prompt_type": prompt_type,
                    "file_path": str(eval_path),
                    "total_score": eval_result.total_score,
                    "hotspot_score": eval_result.hotspot_score,
                    "bottleneck_score": eval_result.bottleneck_score,
                    "gpu_score": eval_result.gpu_score,
                    "suggestions_score": eval_result.suggestions_score,
                    "errors": eval_result.errors,
                    "warnings": eval_result.warnings
                })
                
                logger.info(
                    f"  {prompt_type}: score={eval_result.total_score:.2f}/100, "
                    f"cost=${analysis.cost:.4f}"
                )
                
            except Exception as e:
                logger.error(f"Failed to run {prompt_type} analysis: {e}")
                results["analyses"].append({
                    "prompt_type": prompt_type,
                    "error": str(e)
                })
        
        # 生成汇总
        results["summary"] = self._generate_summary(results)
        
        # 保存完整报告
        report_path = self.output_dir / "reports" / f"{benchmark_name}_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Full report saved to: {report_path}")
        
        return results
    
    def _save_evaluation(self, eval_result: EvaluationResult, filepath: str):
        """保存评估结果"""
        data = {
            "benchmark_name": eval_result.benchmark_name,
            "prompt_type": eval_result.prompt_type,
            "scores": {
                "total": eval_result.total_score,
                "hotspot": eval_result.hotspot_score,
                "bottleneck": eval_result.bottleneck_score,
                "gpu": eval_result.gpu_score,
                "suggestions": eval_result.suggestions_score
            },
            "hotspot_matches": [
                {
                    "identified": m.identified_location,
                    "matched": m.matched_hotspot.name if m.matched_hotspot else None,
                    "location_score": m.location_score,
                    "percentage_score": m.percentage_score,
                    "overall_score": m.overall_score
                }
                for m in eval_result.hotspot_matches
            ],
            "errors": eval_result.errors,
            "warnings": eval_result.warnings,
            "details": eval_result.details
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成汇总统计"""
        evaluations = results.get("evaluations", [])
        
        if not evaluations:
            return {"error": "No evaluations available"}
        
        valid_evals = [e for e in evaluations if "total_score" in e]
        
        if not valid_evals:
            return {"error": "No valid evaluations"}
        
        scores = [e["total_score"] for e in valid_evals]
        
        # 找出最佳 prompt 类型
        best_eval = max(valid_evals, key=lambda e: e["total_score"])
        
        # 计算总成本
        analyses = results.get("analyses", [])
        total_cost = sum(a.get("cost", 0) for a in analyses if "cost" in a)
        total_tokens = sum(a.get("tokens_used", 0) for a in analyses if "tokens_used" in a)
        
        return {
            "best_prompt_type": best_eval["prompt_type"],
            "best_score": best_eval["total_score"],
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "prompt_comparison": {
                e["prompt_type"]: e["total_score"]
                for e in valid_evals
            }
        }


# ============== 便捷函数 ==============

def run_minimd_analysis(
    code_path: str = "benchmarks/minimd/force_lj.cpp",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    运行 miniMD 分析（使用已知的 VTune 数据）
    """
    pipeline = IntegratedAnalysisPipeline(api_key=api_key)
    
    return pipeline.run_with_manual_vtune_data(
        code_path=code_path,
        benchmark_name="minimd",
        hotspots=[
            {"name": "ForceLJ::compute", "time": 3.685, "percentage": 73.7},
            {"name": "Neighbor::build", "time": 0.859, "percentage": 17.2},
            {"name": "Integrate::initialIntegrate", "time": 0.061, "percentage": 1.2},
            {"name": "Integrate::finalIntegrate", "time": 0.052, "percentage": 1.0},
        ],
        total_time=5.0,
        cpu_utilization=9.5,
        system_info={
            "cpu": "Intel Core i7-11800H",
            "cores": 8,
            "threads": 16,
            "cache_l1": "48KB",
            "cache_l2": "1.25MB",
            "cache_l3": "24MB"
        },
        gpu_suitable=True
    )


def run_hpcg_analysis(
    code_path: str = "benchmarks/hpcg/ComputeSPMV_ref.cpp",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    运行 HPCG 分析（使用已知的 VTune 数据）
    """
    pipeline = IntegratedAnalysisPipeline(api_key=api_key)
    
    return pipeline.run_with_manual_vtune_data(
        code_path=code_path,
        benchmark_name="hpcg",
        hotspots=[
            {"name": "ComputeSYMGS_ref", "time": 45.731, "percentage": 67.3},
            {"name": "ComputeSPMV_ref", "time": 18.834, "percentage": 27.7},
            {"name": "ComputeDotProduct_ref", "time": 1.358, "percentage": 2.0},
            {"name": "ComputeWAXPBY_ref", "time": 0.865, "percentage": 1.3},
        ],
        total_time=67.977,
        cpu_utilization=11.6,
        system_info={
            "cpu": "Intel Core i7-11800H",
            "cores": 8,
            "threads": 16
        },
        gpu_suitable=True
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Integrated Analysis Pipeline Demo ===")
    print()
    print("Usage examples:")
    print()
    print("1. Run with manual VTune data:")
    print("   from analysis_pipeline import run_minimd_analysis")
    print("   results = run_minimd_analysis()")
    print()
    print("2. Run with VTune report file:")
    print("   pipeline = IntegratedAnalysisPipeline()")
    print("   results = pipeline.run_with_vtune_report(")
    print("       code_path='path/to/code.cpp',")
    print("       vtune_report_path='path/to/vtune_report.csv',")
    print("       benchmark_name='myapp'")
    print("   )")
    print()
    print("3. Run with benchmark config:")
    print("   results = pipeline.run_with_config(")
    print("       code_path='path/to/code.cpp',")
    print("       benchmark_name='minimd'  # or 'hpcg', 'abinit', 'cp2k'")
    print("   )")
