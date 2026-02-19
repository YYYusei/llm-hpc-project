"""
通用评估器模块
支持配置驱动的热点评估，提高对不同HPC程序的泛化性

设计原则：
1. 基于 benchmark_config 的配置进行评估
2. 支持正则表达式匹配
3. 支持多热点评估
4. 提供详细的评估报告
"""

import re
import logging
import difflib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from benchmark_config import (
    BenchmarkDefinition, 
    HotspotDefinition, 
    get_benchmark, 
    get_registry
)

logger = logging.getLogger(__name__)


@dataclass
class HotspotMatch:
    """热点匹配结果"""
    identified_location: str
    matched_hotspot: Optional[HotspotDefinition]
    location_score: float
    percentage_score: float
    overall_score: float
    match_details: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class EvaluationResult:
    """评估结果"""
    benchmark_name: str
    prompt_type: str
    
    # 各维度分数 (0-1)
    hotspot_score: float
    bottleneck_score: float
    gpu_score: float
    suggestions_score: float
    details_score: float
    
    # 总分 (0-100)
    total_score: float
    
    # 详细匹配结果
    hotspot_matches: List[HotspotMatch] = field(default_factory=list)
    
    # 错误分析
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # 元数据
    details: Dict[str, Any] = field(default_factory=dict)


class GeneralizedEvaluator:
    """通用评估器"""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        tolerance: float = 0.3
    ):
        """
        初始化评估器
        
        Args:
            weights: 各维度权重 (hotspot, bottleneck, gpu, suggestions)
            tolerance: 百分比匹配容差
        """
        self.weights = weights or {
            "hotspot": 0.25,
            "bottleneck": 0.30,
            "gpu": 0.20,
            "suggestions": 0.10,
            "details": 0.15
        }
        self.tolerance = tolerance
        self.registry = get_registry()
    
    def evaluate(
        self,
        identified: Dict[str, Any],
        benchmark_name: str,
        prompt_type: str = "unknown"
    ) -> EvaluationResult:
        """
        评估 LLM 分析结果
        
        Args:
            identified: LLM 识别的结果 (hotspots, bottleneck_type, gpu_suitability, optimization_suggestions)
            benchmark_name: 基准程序名称
            prompt_type: Prompt 类型
            
        Returns:
            EvaluationResult 对象
        """
        benchmark = self.registry.get(benchmark_name)
        if not benchmark:
            logger.warning(f"Unknown benchmark: {benchmark_name}, using generic evaluation")
            return self._generic_evaluate(identified, benchmark_name, prompt_type)
        
        result = EvaluationResult(
            benchmark_name=benchmark_name,
            prompt_type=prompt_type,
            hotspot_score=0.0,
            bottleneck_score=0.0,
            gpu_score=0.0,
            suggestions_score=0.0,
            details_score=0.0,
            total_score=0.0
        )
        
        # 1. 评估热点识别
        result.hotspot_score, result.hotspot_matches = self._evaluate_hotspots(
            identified.get("hotspots", []),
            benchmark
        )
        
        # 2. 评估瓶颈类型
        result.bottleneck_score = self._evaluate_bottleneck(
            identified.get("bottleneck_type", {}),
            benchmark,
            result
        )
        
        # 3. 评估 GPU 适合度
        result.gpu_score = self._evaluate_gpu(
            identified.get("gpu_suitability", {}),
            benchmark,
            result
        )
        
        # 4. 评估优化建议
        result.suggestions_score = self._evaluate_suggestions(
            identified.get("optimization_suggestions", []),
            result
        )
        
        # 5. 评估详细度
        result.details_score = self._evaluate_detail(identified, result)
        
        # 6. 计算总分
        result.total_score = (
            result.hotspot_score * self.weights["hotspot"] +
            result.bottleneck_score * self.weights["bottleneck"] +
            result.gpu_score * self.weights["gpu"] +
            result.suggestions_score * self.weights["suggestions"] +
            result.details_score * self.weights["details"]
        ) * 100
        
        return result
    
    def _evaluate_hotspots(
        self,
        identified_hotspots: List[Dict[str, Any]],
        benchmark: BenchmarkDefinition
    ) -> Tuple[float, List[HotspotMatch]]:
        """
        评估热点识别
        
        使用配置驱动的匹配策略：
        1. 正则表达式模式匹配
        2. 关键词匹配
        3. 字符串相似度
        """
        if not identified_hotspots:
            return 0.0, []
        
        matches: List[HotspotMatch] = []
        expected_hotspots = benchmark.hotspots
        
        # 对每个识别出的热点，找最佳匹配
        for identified in identified_hotspots:
            location = identified.get("location", "")
            percentage = identified.get("estimated_time_percentage", "")
            
            best_match = None
            best_score = 0.0
            best_details = {}
            
            for expected in expected_hotspots:
                score, details = self._match_hotspot(
                    location, percentage, expected, benchmark
                )
                if score > best_score:
                    best_score = score
                    best_match = expected
                    best_details = details
            
            # 计算百分比分数
            percentage_score = 0.0
            if best_match:
                percentage_score = self._percentage_similarity(
                    percentage, 
                    str(best_match.time_percentage)
                )
            
            # 综合分数
            overall = best_score * 0.7 + percentage_score * 0.3
            
            matches.append(HotspotMatch(
                identified_location=location,
                matched_hotspot=best_match,
                location_score=best_score,
                percentage_score=percentage_score,
                overall_score=overall,
                match_details=best_details
            ))
        
        # 计算总体热点分数（取最高匹配分数，因为只需要识别主热点）
        if matches:
            # 优先考虑匹配到主热点（时间占比最高的）
            primary_match = max(matches, key=lambda m: m.overall_score)
            return primary_match.overall_score, matches
        
        return 0.0, matches
    
    def _match_hotspot(
        self,
        identified_location: str,
        identified_percentage: str,
        expected: HotspotDefinition,
        benchmark: BenchmarkDefinition
    ) -> Tuple[float, Dict[str, Any]]:
        """
        匹配单个热点
        
        Returns:
            (匹配分数, 匹配详情)
        """
        location_lower = identified_location.lower()
        details = {
            "pattern_matches": [],
            "keyword_matches": [],
            "string_similarity": 0.0
        }
        
        score = 0.0
        
        # 1. 正则表达式模式匹配
        for pattern in expected.location_patterns:
            try:
                if re.search(pattern, location_lower, re.IGNORECASE):
                    score = max(score, 0.9)  # 模式匹配得高分
                    details["pattern_matches"].append(pattern)
            except re.error:
                # 如果不是有效的正则表达式，作为关键词匹配
                if pattern.lower() in location_lower:
                    score = max(score, 0.85)
                    details["keyword_matches"].append(pattern)
        
        # 2. 循环关键词匹配
        for keyword in expected.loop_keywords:
            if keyword.lower() in location_lower:
                score = max(score, 0.8)
                details["keyword_matches"].append(keyword)
        
        # 3. 基准程序通用关键词匹配
        for keyword in benchmark.function_keywords:
            if keyword.lower() in location_lower:
                score = max(score, 0.75)
                details["keyword_matches"].append(keyword)
        
        # 4. 字符串相似度（兜底）
        string_sim = self._string_similarity(identified_location, expected.name)
        details["string_similarity"] = string_sim
        score = max(score, string_sim * 0.7)
        
        return score, details
    
    def _evaluate_bottleneck(
        self,
        identified: Dict[str, Any],
        benchmark: BenchmarkDefinition,
        result: EvaluationResult
    ) -> float:
        """评估瓶颈类型 - 使用关键词匹配"""
        if not identified:
            result.errors.append("未识别瓶颈类型")
            return 0.0
        
        identified_type = identified.get("primary", "").lower()
        
        # 找到主热点的预期瓶颈类型
        primary_hotspot = max(benchmark.hotspots, key=lambda h: h.time_percentage) if benchmark.hotspots else None
        expected_type = primary_hotspot.bottleneck_type.lower() if primary_hotspot else ""
        
        # 关键词匹配（更宽松）
        type_keywords = {
            "compute": ["compute", "cpu", "arithmetic", "calculation", "flop", "alu", "fp", "instruction"],
            "memory": ["memory", "bandwidth", "cache", "data", "latency", "load", "store", "fetch"],
            "mixed": ["mixed", "both", "hybrid"]
        }
        
        identified_category = None
        expected_category = None
        
        # 检查识别的类型包含哪些关键词
        for category, keywords in type_keywords.items():
            if any(kw in identified_type for kw in keywords):
                identified_category = category
                break
        
        # 检查期望的类型
        for category, keywords in type_keywords.items():
            if any(kw in expected_type for kw in keywords):
                expected_category = category
                break
        
        # 评分
        if identified_category == expected_category and identified_category is not None:
            result.details["bottleneck_match"] = True
            return 1.0
        elif identified_category is not None and expected_category is not None:
            # 部分匹配（如识别为 memory/latency，期望是 memory）
            result.warnings.append(f"瓶颈类型部分匹配: 识别为 {identified_category}, 期望 {expected_category}")
            return 0.5
        elif identified_category is not None:
            # 至少识别出了类型
            result.warnings.append(f"瓶颈类型可能不匹配: {identified_type}")
            return 0.3
        else:
            result.errors.append(f"无法确定瓶颈类型: {identified_type}")
            return 0.0
    
    def _evaluate_gpu(
        self,
        identified: Dict[str, Any],
        benchmark: BenchmarkDefinition,
        result: EvaluationResult
    ) -> float:
        """评估 GPU 适合度"""
        if not identified:
            result.errors.append("未评估 GPU 适合度")
            return 0.0
        
        identified_suitable = identified.get("suitable")
        expected_suitable = benchmark.gpu_suitable
        
        if identified_suitable == expected_suitable:
            result.details["gpu_match"] = True
            
            # 检查推理质量
            reasoning = identified.get("reasoning", "")
            if len(reasoning) > 50:
                return 1.0
            else:
                result.warnings.append("GPU 判断正确但推理较简短")
                return 0.9
        else:
            result.errors.append(f"GPU 判断错误: 识别为 {identified_suitable}, 期望 {expected_suitable}")
            return 0.0
    
    def _evaluate_suggestions(
        self,
        suggestions: List[Dict[str, Any]],
        result: EvaluationResult
    ) -> float:
        """评估优化建议"""
        if not suggestions:
            result.warnings.append("未提供优化建议")
            return 0.0
        
        count = len(suggestions)
        min_count = 2
        
        # 数量分数
        count_score = min(1.0, count / min_count)
        
        # 完整性分数
        completeness_scores = []
        for sug in suggestions:
            score = 0.0
            if sug.get("target"):
                score += 0.25
            if sug.get("suggestion") or sug.get("technique"):
                score += 0.25
            if sug.get("expected_speedup"):
                score += 0.25
            if sug.get("implementation_difficulty") or sug.get("details"):
                score += 0.25
            completeness_scores.append(score)
        
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        
        result.details["suggestions_count"] = count
        result.details["suggestions_completeness"] = avg_completeness
        
        return count_score * 0.6 + avg_completeness * 0.4

    def _evaluate_detail(
        self,
        identified: Dict[str, Any],
        result: EvaluationResult
    ) -> float:
        """评估分析详细度"""
        score = 0.0

        # 检查热点分析的详细程度
        hotspots = identified.get("hotspots", [])
        if hotspots:
            for h in hotspots:
                reason = h.get("reason", "")
                if len(reason) > 100:
                    score += 0.2
                elif len(reason) > 50:
                    score += 0.1

        # 检查瓶颈分析的详细程度
        bottleneck = identified.get("bottleneck_type", {})
        reasoning = bottleneck.get("reasoning", "")
        if len(reasoning) > 200:
            score += 0.3
        elif len(reasoning) > 100:
            score += 0.2
        elif len(reasoning) > 50:
            score += 0.1

        # 检查 GPU 分析的详细程度
        gpu = identified.get("gpu_suitability", {})
        gpu_reasoning = gpu.get("reasoning", "")
        challenges = gpu.get("challenges", [])
        if len(gpu_reasoning) > 100 or len(challenges) > 2:
            score += 0.2
        elif len(gpu_reasoning) > 50 or len(challenges) > 0:
            score += 0.1

        # 检查优化建议数量和质量
        suggestions = identified.get("optimization_suggestions", [])
        if len(suggestions) >= 5:
            score += 0.3
        elif len(suggestions) >= 3:
            score += 0.2
        elif len(suggestions) >= 1:
            score += 0.1

        result.details["detail_score"] = min(1.0, score)
        return min(1.0, score)
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def _percentage_similarity(self, actual: str, expected: str) -> float:
        """计算百分比相似度"""
        def extract(s):
            if isinstance(s, (int, float)):
                return float(s)
            match = re.search(r'(\d+\.?\d*)', str(s).replace('%', ''))
            return float(match.group(1)) if match else None
        
        actual_val = extract(actual)
        expected_val = extract(expected)
        
        if actual_val is None or expected_val is None:
            return 0.0
        if expected_val == 0:
            return 1.0 if actual_val == 0 else 0.0
        
        diff = abs(actual_val - expected_val) / expected_val
        if diff <= self.tolerance:
            return 1.0 - (diff / self.tolerance) * 0.3
        else:
            return max(0.0, 1.0 - diff)
    
    def _generic_evaluate(
        self,
        identified: Dict[str, Any],
        benchmark_name: str,
        prompt_type: str
    ) -> EvaluationResult:
        """通用评估（当没有配置时）"""
        result = EvaluationResult(
            benchmark_name=benchmark_name,
            prompt_type=prompt_type,
            hotspot_score=0.5,  # 无法验证，给中等分
            bottleneck_score=0.5,
            gpu_score=0.5,
            suggestions_score=0.0,
            total_score=0.0
        )
        
        # 只评估建议的质量
        result.suggestions_score = self._evaluate_suggestions(
            identified.get("optimization_suggestions", []),
            result
        )
        
        result.warnings.append(f"未知基准程序 '{benchmark_name}'，使用通用评估")
        
        result.total_score = (
            result.hotspot_score * self.weights["hotspot"] +
            result.bottleneck_score * self.weights["bottleneck"] +
            result.gpu_score * self.weights["gpu"] +
            result.suggestions_score * self.weights["suggestions"]
        ) * 100
        
        return result


def evaluate_analysis(
    analysis_result: Dict[str, Any],
    benchmark_name: str,
    prompt_type: str = "unknown"
) -> EvaluationResult:
    """
    便捷函数：评估分析结果
    
    Args:
        analysis_result: LLM 分析结果
        benchmark_name: 基准程序名称
        prompt_type: Prompt 类型
        
    Returns:
        EvaluationResult 对象
    """
    evaluator = GeneralizedEvaluator()
    return evaluator.evaluate(analysis_result, benchmark_name, prompt_type)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 模拟 LLM 分析结果
    mock_result = {
        "hotspots": [
            {
                "location": "ForceLJ::compute inner k-loop",
                "estimated_time_percentage": "75%",
                "reason": "Force calculation dominates",
                "confidence": "high"
            }
        ],
        "bottleneck_type": {
            "primary": "compute",
            "reasoning": "Division and multiplication operations dominate"
        },
        "gpu_suitability": {
            "suitable": True,
            "reasoning": "Parallel force calculation across atom pairs"
        },
        "optimization_suggestions": [
            {
                "target": "k-loop",
                "suggestion": "SIMD vectorization",
                "expected_speedup": "2-4x"
            }
        ]
    }
    
    # 评估
    result = evaluate_analysis(mock_result, "minimd", "zero_shot")
    
    print(f"Benchmark: {result.benchmark_name}")
    print(f"Prompt type: {result.prompt_type}")
    print(f"Total score: {result.total_score:.2f}/100")
    print(f"  - Hotspot: {result.hotspot_score:.2f}")
    print(f"  - Bottleneck: {result.bottleneck_score:.2f}")
    print(f"  - GPU: {result.gpu_score:.2f}")
    print(f"  - Suggestions: {result.suggestions_score:.2f}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
