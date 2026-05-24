"""
Role-Swap prompt experiment.

Fix S1=GPT-4o, S2=GPT-5.2, temperature=0. Vary only the S2 prompt and its
JSON schema. Three variants exist:

  Original (in run_original.py):
      Skepticism-inducing — "Validate/correct the initial analysis"
      Schema includes `validation.bottleneck_correct: true/false`

  V1 Neutral (this file, class NeutralPipeline):
      "Provide a deep performance analysis" (no validation language)
      Schema only has `analysis.primary_bottleneck` — no correctness flag

  V3 Biased agreement (this file, class BiasedAgreementPipeline):
      "Confirm the analysis and extend it with implementation details"
      Schema has `analysis.agree_with_initial: true/false`

2026-04-17 refactor:
    - The three primary/classification computations now share a single
      implementation via ConfigurableCascadedPipeline + bottleneck_taxonomy.
    - Subclasses override _extract_s2_bottleneck() to adapt to the different
      S2 JSON schemas. They do NOT fake a validation.bottleneck_correct field
      (that used to create the V1 bc=9/9 placeholder artefact).
    - "Change" detection is the taxonomy's pf rule (earliest-position match
      in the S2 text), computed inside the pipeline. No per-script primary().
    - Resume: per-program result files are re-used if present; pass --force
      to override.

Usage:
    python run_role_swap.py                     # V1 + V3 on all 9 programs
    python run_role_swap.py --variant V1
    python run_role_swap.py --programs minimd hotspot
    python run_role_swap.py --force             # re-run even if JSON exists
"""

import sys
import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from configurable_pipeline import ConfigurableCascadedPipeline
from generalized_evaluator import GeneralizedEvaluator
from extended_benchmark_config import register_extended_benchmarks
from run_ablation import BENCHMARK_SOURCES, CONFIG_NAME_MAP, run_on_program


# ==================================================================
# Prompt variants — only override prompt + response parsing + extraction
# ==================================================================

class NeutralPipeline(ConfigurableCascadedPipeline):
    """V1 Neutral: no "validate/correct" language; S2 states primary bottleneck directly."""
    
    def _build_stage2_prompt(self, stage1_result, code_path):
        hotspots = stage1_result.get('hotspots', [])
        bottleneck = stage1_result.get('bottleneck_type', {})
        gpu = stage1_result.get('gpu_suitability', {})
        hotspots_str = "\n".join([
            f"  - {h.get('location', 'unknown')}: {h.get('estimated_time_percentage', 'N/A')}"
            for h in hotspots
        ])
        
        return f"""
请对该 HPC 程序进行深度性能分析。以下是初步分析结果，供参考:

## 初步分析参考:

### 热点识别:
{hotspots_str}

### 瓶颈类型:
- 类型: {bottleneck.get('primary', 'unknown')}
- 理由: {bottleneck.get('reasoning', 'N/A')}

### GPU 适合度:
- 适合: {gpu.get('suitable', 'unknown')}
- 理由: {gpu.get('reasoning', 'N/A')}

## 请完成以下分析:

1. **瓶颈分析**: 根据源代码分析该程序的性能瓶颈类型并说明理由

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
    
    def _get_stage2_system_prompt(self):
        return """你是 HPC 性能优化专家。请提供深度性能分析。

输出格式 (JSON):
{
  "analysis": {
    "primary_bottleneck": "瓶颈类型",
    "reasoning": "分析理由"
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
    
    def _extract_s2_bottleneck(self, stage2_result, stage1_dict):
        """V1 neutral: S2 states primary bottleneck directly in `analysis.primary_bottleneck`."""
        analysis = stage2_result.get('analysis', {})
        s2_bt = analysis.get('primary_bottleneck', '')
        return s2_bt or self._extract_s1_bottleneck(stage1_dict)


class BiasedAgreementPipeline(ConfigurableCascadedPipeline):
    """V3 Biased agreement: S2 is prompted to confirm S1 and extend details."""
    
    def _build_stage2_prompt(self, stage1_result, code_path):
        hotspots = stage1_result.get('hotspots', [])
        bottleneck = stage1_result.get('bottleneck_type', {})
        gpu = stage1_result.get('gpu_suitability', {})
        hotspots_str = "\n".join([
            f"  - {h.get('location', 'unknown')}: {h.get('estimated_time_percentage', 'N/A')}"
            for h in hotspots
        ])
        
        return f"""
基于前一阶段的分析结果, 请确认分析并补充实现细节。

## 已有分析:

### 热点识别:
{hotspots_str}

### 瓶颈类型:
- 类型: {bottleneck.get('primary', 'unknown')}
- 理由: {bottleneck.get('reasoning', 'N/A')}

### GPU 适合度:
- 适合: {gpu.get('suitable', 'unknown')}
- 理由: {gpu.get('reasoning', 'N/A')}

## 请完成以下任务:

1. **确认分析**: 确认上述瓶颈判断并扩展其细节
2. **详细优化建议**: 针对每个热点，给出具体的代码级优化方案
3. **性能分析**: 算术强度 / 缓存行为 / 数据依赖
4. **GPU 实现方案**: 并行化策略 / 数据布局 / 挑战

请以 JSON 格式输出。
"""
    
    def _get_stage2_system_prompt(self):
        return """你是 HPC 性能优化专家。请基于已有分析补充实现细节。

输出格式 (JSON):
{
  "analysis": {
    "agree_with_initial": true/false,
    "primary_bottleneck": "瓶颈类型",
    "reasoning": "分析理由"
  },
  "optimizations": [
    {"target": "...", "technique": "...", "code_changes": "...", "expected_speedup": "...", "difficulty": "easy/medium/hard"}
  ],
  "performance_analysis": {
    "arithmetic_intensity": "FLOPs/byte",
    "cache_behavior": "...",
    "data_dependencies": ["..."]
  },
  "gpu_implementation": {
    "suitable": true/false,
    "strategy": "...",
    "data_layout": "...",
    "challenges": ["..."],
    "expected_speedup": "..."
  }
}
"""
    
    def _extract_s2_bottleneck(self, stage2_result, stage1_dict):
        """V3 biased: S2 states primary in `analysis.primary_bottleneck`.
        Note: `agree_with_initial` is informational only — classification uses
        primary-category comparison via the taxonomy, not this flag."""
        analysis = stage2_result.get('analysis', {})
        s2_bt = analysis.get('primary_bottleneck', '')
        return s2_bt or self._extract_s1_bottleneck(stage1_dict)


VARIANTS = {
    "V1": {"class": NeutralPipeline,          "label": "Role-Swap V1 Neutral"},
    "V3": {"class": BiasedAgreementPipeline,  "label": "Role-Swap V3 Biased agreement"},
}


# ==================================================================
# Main
# ==================================================================

def main():
    parser = argparse.ArgumentParser(description="Role-Swap prompt experiment")
    parser.add_argument("--variant", choices=["V1", "V3", "both"], default="both")
    parser.add_argument("--programs", nargs="+",
                        choices=list(BENCHMARK_SOURCES.keys()),
                        help="Specific programs (default: all 9)")
    parser.add_argument("--output-dir", default="results/role_swap")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if per-program result JSON exists")
    parser.add_argument("--s2-model", default="gpt-5.2",
                        help="S2 model (default gpt-5.2; e.g. deepseek-v3-2-251201 for cross-family)")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/role_swap.log', mode='a')
        ]
    )
    logger = logging.getLogger(__name__)
    
    programs = args.programs or list(BENCHMARK_SOURCES.keys())
    variants_to_run = ["V1", "V3"] if args.variant == "both" else [args.variant]
    
    register_extended_benchmarks()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    evaluator = GeneralizedEvaluator()
    
    for vkey in variants_to_run:
        v = VARIANTS[vkey]
        logger.info(f"\n{'#'*60}")
        logger.info(f"Running {v['label']} (S1=gpt-4o, S2={args.s2_model})")
        logger.info(f"{'#'*60}")
        
        pipeline = v["class"](s1_model="gpt-4o", s2_model=args.s2_model)
        
        variant_dir = output_dir / vkey
        variant_dir.mkdir(parents=True, exist_ok=True)
        
        variant_results = {}
        for prog in programs:
            out_file = variant_dir / f"{prog}_result.json"
            if out_file.exists() and not args.force:
                logger.info(f"[skip] {prog}: result already exists (use --force to re-run)")
                variant_results[prog] = json.load(open(out_file, encoding='utf-8'))
                continue
            
            summary = run_on_program(pipeline, evaluator, prog, v['label'], logger)
            if summary:
                variant_results[prog] = summary
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Save variant summary
        variant_summary = {
            "variant": {"key": vkey, "label": v['label']},
            "timestamp": datetime.now().isoformat(),
            "programs_tested": programs,
            "results": variant_results,
        }
        with open(variant_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(variant_summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Stats (using taxonomy classification)
        changes = sum(1 for r in variant_results.values()
                      if r.get('classification', {}).get('changed'))
        correction = sum(1 for r in variant_results.values()
                         if r.get('classification', {}).get('correction_type') == 'correction')
        overcorr = sum(1 for r in variant_results.values()
                       if r.get('classification', {}).get('correction_type') == 'over-correction')
        total = len(variant_results)
        total_cost = sum(r.get('cost', 0) for r in variant_results.values())
        
        logger.info(f"\n{v['label']} summary:")
        logger.info(f"  changes: {changes}/{total}")
        logger.info(f"  of which corrections: {correction}")
        logger.info(f"  of which over-corrections: {overcorr}")
        logger.info(f"  total cost: ${total_cost:.4f}")
    
    logger.info(f"\nAll results saved to: {output_dir}/")
    logger.info("Run rescore_all.py for canonical cross-config summary (including Original).")


if __name__ == "__main__":
    main()
