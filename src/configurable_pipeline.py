"""
Configurable Cascaded Pipeline.

Parametrised S1/S2 model combination. Used for all ablation and role-swap
experiments.

2026-04-17 refactor:
    - Introduced bottleneck_taxonomy as single source of truth for "change"
      detection. The pipeline now computes classification internally and
      returns it in `result['classification']`. Callers no longer need to
      implement their own primary() logic.
    - Subclasses (NeutralPipeline, BiasedAgreementPipeline in run_role_swap.py)
      can override _extract_s2_bottleneck() to adapt to different S2 JSON
      schemas without re-implementing the classification pipeline.
"""

import json
import logging
from typing import Dict, Any, Optional

from llm_client import LLMClient
from analyzer import HPCAnalyzer
from bottleneck_taxonomy import primary, classify, GROUND_TRUTH

logger = logging.getLogger(__name__)


class ConfigurableCascadedPipeline:
    """Two-stage LLM cascade with configurable S1 and S2 models."""
    
    def __init__(self, s1_model: str = "gpt-4o", s2_model: str = "gpt-5.2"):
        self.s1_model = s1_model
        self.s2_model = s2_model
        self.analyzer_s1 = HPCAnalyzer(model=s1_model)
        self.client_s2 = LLMClient(model=s2_model)
        logger.info(f"Pipeline initialized: S1={s1_model}, S2={s2_model}")
    
    # ==================================================================
    # Public API
    # ==================================================================
    
    def analyze(
        self,
        code_path: str,
        code_name: str,
        prompt_type: str = "zero_shot",
        vtune_data: Optional[Dict] = None,
        program_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the two-stage cascade and return a structured result.
        
        Parameters:
            code_path:   path to source file
            code_name:   benchmark config name (e.g. 'hpcg')
            prompt_type: S1 prompt strategy ('zero_shot' / 'few_shot' / 'contextual')
            program_key: canonical program key for ground-truth lookup
                         (e.g. 'hpcg_spmv' vs 'hpcg_symgs'). If None, falls
                         back to code_name.
        
        Returns a dict with keys:
            code_name, prompt_type, program_key
            stage1: {model, hotspots, bottleneck_type, gpu_suitability, cost, bottleneck_raw}
            stage2: {model, validation|analysis, detailed_optimizations, gpu_implementation,
                     performance_analysis, cost, bottleneck_raw}
            classification: {s1_primary, s2_primary, ground_truth, changed,
                             s1_matches_gt, s2_matches_gt, correction_type}
            total_cost
        """
        
        # ========== Stage 1 ==========
        logger.info(f"Stage 1: {self.s1_model} {prompt_type} analysis...")
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
            'cost': stage1_result.cost,
        }
        stage1_cost = stage1_dict['cost']
        logger.info(f"Stage 1 done: {len(stage1_dict['hotspots'])} hotspots, cost=${stage1_cost:.4f}")
        
        # ========== Stage 2 ==========
        logger.info(f"Stage 2: {self.s2_model} deep analysis...")
        stage2_prompt = self._build_stage2_prompt(stage1_dict, code_path)
        
        with open(code_path, 'r', encoding='utf-8', errors='ignore') as f:
            code_content = f.read()
        
        full_prompt = f"""
{stage2_prompt}

## Source code:
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
        logger.info(f"Stage 2 done: cost=${stage2_cost:.4f}")
        
        # ========== Classification via taxonomy ==========
        s1_raw = self._extract_s1_bottleneck(stage1_dict)
        s2_raw = self._extract_s2_bottleneck(stage2_result, stage1_dict)
        
        s1_primary = primary(s1_raw)
        s2_primary = primary(s2_raw)
        
        # Ground-truth lookup via program_key (preferred) or code_name (fallback)
        gt_key = program_key or code_name
        gt = GROUND_TRUTH.get(gt_key, 'unknown')
        if gt == 'unknown':
            logger.warning(f"No ground truth for program_key={gt_key!r}; "
                           f"classification.correction_type will be 'unknown'")
        
        cls = classify(s1_primary, s2_primary, gt)
        
        # ========== Assemble result ==========
        combined_result = {
            "code_name": code_name,
            "program_key": gt_key,
            "prompt_type": prompt_type,
            "stage1": {
                "model": self.s1_model,
                "hotspots": stage1_dict['hotspots'],
                "bottleneck_type": stage1_dict['bottleneck_type'],
                "gpu_suitability": stage1_dict['gpu_suitability'],
                "cost": stage1_cost,
                "bottleneck_raw": s1_raw,
            },
            "stage2": {
                "model": self.s2_model,
                "validation": stage2_result.get('validation', {}),
                "analysis": stage2_result.get('analysis', {}),
                "detailed_optimizations": stage2_result.get('optimizations', []),
                "gpu_implementation": stage2_result.get('gpu_implementation', {}),
                "performance_analysis": stage2_result.get('performance_analysis', {}),
                "cost": stage2_cost,
                "bottleneck_raw": s2_raw,
            },
            "classification": cls.as_dict(),
            "total_cost": stage1_cost + stage2_cost,
        }
        return combined_result
    
    # ==================================================================
    # Helpers — overridable by subclasses
    # ==================================================================
    
    def _extract_s1_bottleneck(self, stage1_dict: Dict) -> str:
        """Extract the text that represents S1's primary bottleneck."""
        bt = stage1_dict.get('bottleneck_type', {})
        if isinstance(bt, dict):
            return bt.get('primary', '') or ''
        return str(bt or '')
    
    def _extract_s2_bottleneck(self, stage2_result: Dict, stage1_dict: Dict) -> str:
        """
        Extract the text that represents S2's primary bottleneck.
        
        Default (validate/correct schema):
            - If validation.bottleneck_correct is True → S2 agrees with S1,
              return S1's bottleneck text.
            - If validation.bottleneck_correct is False → return
              validation.corrected_bottleneck.
        
        Subclasses override this for alternative schemas (neutral / biased).
        """
        validation = stage2_result.get('validation', {})
        if not validation:
            return self._extract_s1_bottleneck(stage1_dict)
        
        if validation.get('bottleneck_correct', True):
            return self._extract_s1_bottleneck(stage1_dict)
        
        corrected = validation.get('corrected_bottleneck', '')
        return corrected or self._extract_s1_bottleneck(stage1_dict)
    
    # ==================================================================
    # Prompt templates — overridable by subclasses
    # ==================================================================
    
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
            m = re.search(r'\{[\s\S]*\}', response.content)
            if m:
                return json.loads(m.group())
        except Exception as e:
            logger.warning(f"Parse failed: {e}")
        return {"raw_response": response.content}
