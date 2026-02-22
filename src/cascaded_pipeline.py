"""
统一级联分析流水线
Stage 1: GPT-4o (任意 prompt) -> 初步分析
Stage 2: GPT-5.2 -> 深度优化
"""

import json
import logging
from typing import Dict, Any, Optional
from llm_client import LLMClient
from analyzer import HPCAnalyzer

logger = logging.getLogger(__name__)


class CascadedPipeline:
    """级联分析流水线"""
    
    def __init__(self):
        self.analyzer_4o = HPCAnalyzer(model="gpt-4o")
        self.client_5_2 = LLMClient(model="gpt-5.2")
        
    def analyze(
        self,
        code_path: str,
        code_name: str,
        prompt_type: str = "zero_shot",
        vtune_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        执行级联分析
        
        Args:
            code_path: 源代码路径
            code_name: 程序名称
            prompt_type: prompt 类型 (zero_shot/few_shot/contextual)
            vtune_data: VTune 分析数据
        """
        
        # ========== Stage 1: GPT-4o ==========
        logger.info(f"Stage 1: GPT-4o {prompt_type} 分析...")
        
        stage1_result = self.analyzer_4o.analyze(
            code_path=code_path,
            code_name=code_name,
            prompt_type=prompt_type
        )

        # 转换为字典方便后续处理
        stage1_dict = {
            'hotspots': stage1_result.hotspots,
            'bottleneck_type': stage1_result.bottleneck_type,
            'gpu_suitability': stage1_result.gpu_suitability,
            'optimization_suggestions': stage1_result.optimization_suggestions,
            'cost': stage1_result.cost
        }

        stage1_cost = stage1_dict['cost']
        logger.info(f"Stage 1 完成: {len(stage1_dict['hotspots'])} 热点, 花费 ${stage1_cost:.4f}")
        
        # ========== Stage 2: GPT-5.2 深度分析 ==========
        logger.info("Stage 2: GPT-5.2 深度分析...")
        
        stage2_prompt = self._build_stage2_prompt(stage1_dict, code_path)
        
        with open(code_path, 'r', encoding='utf-8', errors='ignore') as f:
            code_content = f.read()

        # 把代码内容加到 prompt 里
        full_prompt = f"""
        {stage2_prompt}

        ## 源代码:
        ```cpp
        {code_content}
        ```
        """

        stage2_response = self.client_5_2.chat(
            prompt=full_prompt,
            system_prompt=self._get_stage2_system_prompt()
        )
        
        stage2_result = self._parse_stage2_response(stage2_response)
        stage2_cost = stage2_response.cost
        
        logger.info(f"Stage 2 完成: 花费 ${stage2_cost:.4f}")
        
        # ========== 合并结果 ==========
        combined_result = {
            "code_name": code_name,
            "prompt_type": prompt_type,
            "stage1": {
                "model": "gpt-4o",
                "hotspots": stage1_dict['hotspots'],
                "bottleneck_type": stage1_dict['bottleneck_type'],
                "gpu_suitability": stage1_dict['gpu_suitability'],
                "cost": stage1_cost
            },
            "stage2": {
                "model": "gpt-5.2",
                "validation": stage2_result.get('validation', {}),
                "detailed_optimizations": stage2_result.get('optimizations', []),
                "gpu_implementation": stage2_result.get('gpu_implementation', {}),
                "performance_analysis": stage2_result.get('performance_analysis', {}),
                "cost": stage2_cost
            },
            "total_cost": stage1_cost + stage2_cost
        }
        
        return combined_result
    
    def _build_stage2_prompt(self, stage1_result: Dict, code_path: str) -> str:
        """构建 Stage 2 的 prompt"""
        
        hotspots = stage1_result.get('hotspots', [])
        bottleneck = stage1_result.get('bottleneck_type', {})
        gpu = stage1_result.get('gpu_suitability', {})
        
        hotspots_str = "\n".join([
            f"  - {h.get('location', 'unknown')}: {h.get('estimated_time_percentage', 'N/A')}"
            for h in hotspots
        ])
        
        return f"""
基于 GPT-4o 的初步分析结果，请进行深度优化分析。

## GPT-4o 初步分析结果:

### 热点识别:
{hotspots_str}

### 瓶颈类型:
- 类型: {bottleneck.get('primary', 'unknown')}
- 理由: {bottleneck.get('reasoning', 'N/A')}

### GPU 适合度:
- 适合: {gpu.get('suitable', 'unknown')}
- 理由: {gpu.get('reasoning', 'N/A')}

## 请完成以下深度分析:

1. **验证/修正**: 验证上述瓶颈判断是否正确，如有问题请修正并说明理由

2. **详细优化建议**: 针对每个热点，给出具体的代码级优化方案，包括:
   - 具体代码改动
   - 预期加速比
   - 实现难度

3. **性能分析**: 
   - 算术强度估计 (FLOPs/byte)
   - 缓存行为分析
   - 数据依赖分析

4. **GPU 实现方案**: 
   - 并行化策略
   - 数据布局建议
   - 潜在挑战和解决方案
   - 预期加速范围

请以 JSON 格式输出。
"""
    
    def _get_stage2_system_prompt(self) -> str:
        return """你是 HPC 性能优化专家。基于初步分析结果，提供深度优化建议。

输出格式 (JSON):
{
  "validation": {
    "bottleneck_correct": true/false,
    "corrected_bottleneck": "如需修正",
    "reasoning": "验证/修正理由"
  },
  "optimizations": [
    {
      "target": "优化目标",
      "technique": "优化技术",
      "code_changes": "具体代码改动",
      "expected_speedup": "预期加速",
      "difficulty": "easy/medium/hard"
    }
  ],
  "performance_analysis": {
    "arithmetic_intensity": "FLOPs/byte",
    "cache_behavior": "缓存行为分析",
    "data_dependencies": ["依赖1", "依赖2"]
  },
  "gpu_implementation": {
    "suitable": true/false,
    "strategy": "并行化策略",
    "data_layout": "数据布局建议",
    "challenges": ["挑战1", "挑战2"],
    "expected_speedup": "预期加速范围"
  }
}
"""
    
    def _parse_stage2_response(self, response) -> Dict:
        """解析 Stage 2 响应"""
        try:
            if response.parsed_json:
                return response.parsed_json
            # 尝试从文本中提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"解析 Stage 2 响应失败: {e}")
        
        return {"raw_response": response.content}


def run_cascaded_analysis(
    code_path: str,
    code_name: str,
    prompt_types: list = None
) -> Dict[str, Any]:
    """运行级联分析"""
    
    if prompt_types is None:
        prompt_types = ["zero_shot", "few_shot", "contextual"]
    
    pipeline = CascadedPipeline()
    results = {}
    
    for pt in prompt_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running cascaded analysis: {pt}")
        logger.info('='*50)
        
        result = pipeline.analyze(
            code_path=code_path,
            code_name=code_name,
            prompt_type=pt
        )
        results[pt] = result
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试
    results = run_cascaded_analysis(
        code_path="benchmarks/minimd/force_lj.cpp",
        code_name="minimd",
        prompt_types=["zero_shot"]
    )
    
    print(json.dumps(results, indent=2, default=str))