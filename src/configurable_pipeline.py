"""
可配置模型的级联分析 Pipeline（用于 Ablation 实验）
支持任意 S1/S2 模型组合

用法:
    # 原始配置 (对照组)
    pipeline = ConfigurableCascadedPipeline(s1_model="gpt-4o", s2_model="gpt-5.2")
    
    # Ablation A: 两阶段都用 GPT-5.2
    pipeline_a = ConfigurableCascadedPipeline(s1_model="gpt-5.2", s2_model="gpt-5.2")
    
    # Ablation B: S1=GPT-5.2, S2=GPT-4o
    pipeline_b = ConfigurableCascadedPipeline(s1_model="gpt-5.2", s2_model="gpt-4o")
"""

import json
import logging
from typing import Dict, Any, Optional
from llm_client import LLMClient
from analyzer import HPCAnalyzer

logger = logging.getLogger(__name__)


class ConfigurableCascadedPipeline:
    """可配置模型的级联分析 Pipeline"""
    
    def __init__(self, s1_model: str = "gpt-4o", s2_model: str = "gpt-5.2"):
        """
        Args:
            s1_model: Stage 1 使用的模型
            s2_model: Stage 2 使用的模型
        """
        self.s1_model = s1_model
        self.s2_model = s2_model
        self.analyzer_s1 = HPCAnalyzer(model=s1_model)
        self.client_s2 = LLMClient(model=s2_model)
        
        logger.info(f"Pipeline initialized: S1={s1_model}, S2={s2_model}")
    
    def analyze(
        self,
        code_path: str,
        code_name: str,
        prompt_type: str = "zero_shot",
        vtune_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        执行级联分析（与 CascadedPipeline.analyze 接口一致）
        """
        
        # ========== Stage 1 ==========
        logger.info(f"Stage 1: {self.s1_model} {prompt_type} 分析...")
        
        stage1_result = self.analyzer_s1.analyze(
            code_path=code_path,
            code_name=code_name,
            prompt_type=prompt_type
        )

        stage1_dict = {
            'hotspots': stage1_result.hotspots,
            'bottleneck_type': stage1_result.bottleneck_type,
            'gpu_suitability': stage1_result.gpu_suitability,
            'optimization_suggestions': stage1_result.optimization_suggestions,
            'cost': stage1_result.cost
        }

        stage1_cost = stage1_dict['cost']
        logger.info(f"Stage 1 完成: {len(stage1_dict['hotspots'])} 热点, 花费 ${stage1_cost:.4f}")
        
        # ========== Stage 2 ==========
        # ★ 关键：S2 prompt 与原始实验完全一致，只换了模型
        logger.info(f"Stage 2: {self.s2_model} 深度分析...")
        
        stage2_prompt = self._build_stage2_prompt(stage1_dict, code_path)
        
        with open(code_path, 'r', encoding='utf-8', errors='ignore') as f:
            code_content = f.read()

        full_prompt = f"""
        {stage2_prompt}

        ## 源代码:
        ```cpp
        {code_content}
        ```
        """

        stage2_response = self.client_s2.chat(
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
                "model": self.s1_model,
                "hotspots": stage1_dict['hotspots'],
                "bottleneck_type": stage1_dict['bottleneck_type'],
                "gpu_suitability": stage1_dict['gpu_suitability'],
                "cost": stage1_cost
            },
            "stage2": {
                "model": self.s2_model,
                "validation": stage2_result.get('validation', {}),
                "detailed_optimizations": stage2_result.get('optimizations', []),
                "gpu_implementation": stage2_result.get('gpu_implementation', {}),
                "performance_analysis": stage2_result.get('performance_analysis', {}),
                "cost": stage2_cost
            },
            "total_cost": stage1_cost + stage2_cost
        }
        
        return combined_result
    
    # ── 以下方法与原始 CascadedPipeline 完全一致 ──
    
    def _build_stage2_prompt(self, stage1_result: Dict, code_path: str) -> str:
        hotspots = stage1_result.get('hotspots', [])
        bottleneck = stage1_result.get('bottleneck_type', {})
        gpu = stage1_result.get('gpu_suitability', {})
        
        hotspots_str = "\n".join([
            f"  - {h.get('location', 'unknown')}: {h.get('estimated_time_percentage', 'N/A')}"
            for h in hotspots
        ])
        
        return f"""
基于 {self.s1_model} 的初步分析结果，请进行深度优化分析。

## {self.s1_model} 初步分析结果:

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
        try:
            if response.parsed_json:
                return response.parsed_json
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"解析 Stage 2 响应失败: {e}")
        return {"raw_response": response.content}
