"""
Abinit FFT → cuFFT 替换方案分析
使用 cascaded pipeline 分析 Abinit 的 FFT 热点，让 LLM 建议 cuFFT 替换策略

方案 B: 不让 LLM 手写 CUDA FFT kernel，而是让它分析应该如何用 cuFFT 替换

运行方式:
    cd llm-hpc-project
    python3 test_abinit_cufft.py
"""

import sys
import os
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# VTune profiling 数据
VTUNE_DATA = """
Intel VTune Profiler 2025.9 - Hotspot Analysis for ABINIT 9.10.4
Configuration: Si bulk, 2 atoms, ecut=40 Ha, 8x8x8 k-grid, FFT grid 45x45x45
Total CPU Time: 9.799s

Top Hotspots:
  sg_ffty                     3.991s  40.7%   (FFT y-direction transform)
  sg_fftpx                    1.550s  15.8%   (FFT x-direction transform with zero-padding)
  fftrisc_one_nothreadsafe    1.101s  11.2%   (FFT dispatcher for wavefunctions)
  dfpt_mkffkg                 0.632s   6.4%   (FFT kernel generation for DFPT)
  zgemm_ (BLAS)               0.385s   3.9%   (Complex matrix multiply)
  [Others]                    2.141s  21.8%

FFT-related functions total: ~68% of runtime
"""


def run_stage1_analysis(client_4o, code_content):
    """Stage 1: GPT-4o 初步分析"""
    
    prompt = f"""
You are an HPC performance analyst. Analyze the following Fortran FFT code from ABINIT 
(a first-principles DFT code) along with its VTune profiling data.

## VTune Profiling Results:
{VTUNE_DATA}

## Source Code (key FFT functions):
```fortran
{code_content}
```

## Task:
1. Identify the performance hotspots and explain why these FFT functions dominate
2. Classify the bottleneck type (compute-bound vs memory-bound)
3. Assess GPU suitability
4. Suggest optimization strategies

Respond in JSON format:
{{
  "hotspots": [
    {{"location": "function_name", "estimated_time_percentage": "X%", "reason": "..."}}
  ],
  "bottleneck_type": {{
    "primary": "compute|memory|mixed",
    "reasoning": "..."
  }},
  "gpu_suitability": {{
    "suitable": true/false,
    "reasoning": "...",
    "recommended_approach": "custom_kernel|library_replacement|hybrid",
    "challenges": ["..."]
  }},
  "optimization_suggestions": [
    {{"target": "...", "suggestion": "...", "expected_speedup": "...", "implementation_difficulty": "..."}}
  ]
}}
"""
    
    response = client_4o.chat(
        prompt=prompt,
        system_prompt="You are an expert in HPC performance optimization, GPU computing, and scientific computing. Analyze code and profiling data to provide actionable optimization recommendations."
    )
    
    return response


def run_stage2_cufft_analysis(client_5_2, stage1_result, code_content):
    """Stage 2: GPT-5.2 深度 cuFFT 替换方案分析"""
    
    prompt = f"""
Based on the Stage 1 analysis below, provide a detailed GPU acceleration strategy 
for ABINIT's FFT hotspot functions using NVIDIA cuFFT library.

## Stage 1 Analysis (GPT-4o):
{stage1_result}

## VTune Profiling Results:
{VTUNE_DATA}

## Source Code (key FFT functions):
```fortran
{code_content}
```

## Context:
- ABINIT uses a custom Goedecker FFT implementation (sg_fft family)
- The FFT is 3D complex-to-complex, grid size 45x45x45 for this test case
- The FFT is called per k-point (29 k-points) and per SCF iteration (7 iterations)
- Data is stored as real(dp) arrays with alternating real/imaginary elements: arr(2, nd1*nd2*nd3)
- The code uses mixed-radix FFT (factors 2, 3, 5) with cache-optimized blocking

## Please provide a detailed analysis in JSON format:
{{
  "validation": {{
    "bottleneck_correct": true/false,
    "corrected_bottleneck": "if incorrect, provide correction",
    "reasoning": "detailed validation reasoning"
  }},
  "cufft_replacement_strategy": {{
    "feasibility": "high|medium|low",
    "approach": "description of overall approach",
    "data_layout_changes": {{
      "current_format": "describe current Fortran data layout",
      "required_format": "describe cuFFT-compatible format",
      "conversion_needed": true/false,
      "conversion_code_sketch": "pseudo-code for data layout conversion"
    }},
    "api_mapping": [
      {{
        "original_function": "sg_fft_cc / sg_ffty / sg_fftpx",
        "cufft_replacement": "cufftExecZ2Z / cufftPlanMany / etc",
        "notes": "specific considerations"
      }}
    ],
    "memory_management": {{
      "host_to_device": "strategy for H2D transfers",
      "device_to_host": "strategy for D2H transfers",
      "persistent_allocation": "can GPU memory be kept across calls?",
      "estimated_transfer_overhead": "estimate"
    }},
    "batch_optimization": {{
      "can_batch": true/false,
      "batch_dimension": "which dimension to batch over",
      "expected_benefit": "description"
    }},
    "expected_speedup": {{
      "fft_only": "Xx speedup for FFT portion",
      "overall": "Xx speedup for total application",
      "limiting_factors": ["list of factors limiting speedup"]
    }}
  }},
  "alternative_approaches": [
    {{
      "name": "approach name",
      "description": "brief description",
      "pros": ["..."],
      "cons": ["..."]
    }}
  ],
  "implementation_plan": [
    {{
      "step": 1,
      "task": "description",
      "estimated_effort": "hours",
      "priority": "high|medium|low"
    }}
  ],
  "risks_and_challenges": [
    {{
      "risk": "description",
      "mitigation": "how to address"
    }}
  ]
}}
"""
    
    response = client_5_2.chat(
        prompt=prompt,
        system_prompt="""You are a senior GPU computing expert specializing in cuFFT integration 
and scientific computing optimization. Provide detailed, actionable recommendations for 
replacing custom FFT implementations with cuFFT, considering data layouts, memory management, 
and performance implications. Be specific about cuFFT API calls and parameters."""
    )
    
    return response


def main():
    # 读取 FFT 源码提取
    code_path = "benchmarks/abinit/abinit_fft_extract.F90"
    if not os.path.exists(code_path):
        logger.error(f"Source file not found: {code_path}")
        return
    
    with open(code_path, 'r') as f:
        code_content = f.read()
    
    logger.info("=" * 60)
    logger.info("Abinit FFT → cuFFT Replacement Analysis (Cascaded Pipeline)")
    logger.info("=" * 60)
    
    # Stage 1: GPT-4o
    logger.info("\nStage 1: GPT-4o initial analysis...")
    client_4o = LLMClient(model="gpt-4o")
    stage1_response = run_stage1_analysis(client_4o, code_content)
    stage1_cost = stage1_response.cost
    
    logger.info(f"Stage 1 complete. Cost: ${stage1_cost:.4f}")
    
    # Parse Stage 1 result
    stage1_text = stage1_response.content
    try:
        import re
        json_match = re.search(r'\{[\s\S]*\}', stage1_text)
        if json_match:
            stage1_json = json.loads(json_match.group())
        else:
            stage1_json = {"raw": stage1_text}
    except:
        stage1_json = {"raw": stage1_text}
    
    # Print Stage 1 summary
    if "bottleneck_type" in stage1_json:
        bt = stage1_json["bottleneck_type"]
        logger.info(f"  Bottleneck: {bt.get('primary', 'unknown')}")
    if "gpu_suitability" in stage1_json:
        gs = stage1_json["gpu_suitability"]
        logger.info(f"  GPU suitable: {gs.get('suitable', 'unknown')}")
        logger.info(f"  Recommended approach: {gs.get('recommended_approach', 'unknown')}")
    
    # Stage 2: GPT-5.2
    logger.info("\nStage 2: GPT-5.2 deep cuFFT replacement analysis...")
    client_5_2 = LLMClient(model="gpt-5.2")
    stage2_response = run_stage2_cufft_analysis(client_5_2, stage1_text, code_content)
    stage2_cost = stage2_response.cost
    
    logger.info(f"Stage 2 complete. Cost: ${stage2_cost:.4f}")
    
    # Parse Stage 2 result
    stage2_text = stage2_response.content
    try:
        json_match = re.search(r'\{[\s\S]*\}', stage2_text)
        if json_match:
            stage2_json = json.loads(json_match.group())
        else:
            stage2_json = {"raw": stage2_text}
    except:
        stage2_json = {"raw": stage2_text}
    
    # Print Stage 2 summary
    if "cufft_replacement_strategy" in stage2_json:
        strat = stage2_json["cufft_replacement_strategy"]
        logger.info(f"  Feasibility: {strat.get('feasibility', 'unknown')}")
        logger.info(f"  Approach: {strat.get('approach', 'unknown')[:100]}...")
        if "expected_speedup" in strat:
            sp = strat["expected_speedup"]
            logger.info(f"  Expected FFT speedup: {sp.get('fft_only', 'unknown')}")
            logger.info(f"  Expected overall speedup: {sp.get('overall', 'unknown')}")
    
    if "validation" in stage2_json:
        val = stage2_json["validation"]
        corrected = not val.get("bottleneck_correct", True)
        logger.info(f"  Bottleneck corrected: {corrected}")
        if corrected:
            logger.info(f"  Corrected to: {val.get('corrected_bottleneck', 'N/A')}")
    
    total_cost = stage1_cost + stage2_cost
    logger.info(f"\nTotal cost: ${total_cost:.4f}")
    
    # Save results
    output_dir = "results/abinit_cufft"
    os.makedirs(output_dir, exist_ok=True)
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "Abinit FFT cuFFT replacement analysis (Plan B)",
        "stage1": {
            "model": "gpt-4o",
            "result": stage1_json,
            "raw_response": stage1_text,
            "cost": stage1_cost
        },
        "stage2": {
            "model": "gpt-5.2",
            "result": stage2_json,
            "raw_response": stage2_text,
            "cost": stage2_cost
        },
        "total_cost": total_cost,
        "vtune_data": VTUNE_DATA
    }
    
    output_file = os.path.join(output_dir, "cufft_analysis_result.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("ABINIT FFT → cuFFT REPLACEMENT ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Stage 1 (GPT-4o):  Bottleneck analysis + GPU suitability    ${stage1_cost:.4f}")
    print(f"Stage 2 (GPT-5.2): cuFFT replacement strategy               ${stage2_cost:.4f}")
    print(f"Total cost:                                                  ${total_cost:.4f}")
    print("-" * 70)
    
    if "cufft_replacement_strategy" in stage2_json:
        strat = stage2_json["cufft_replacement_strategy"]
        print(f"\nFeasibility: {strat.get('feasibility', 'N/A')}")
        print(f"Approach: {strat.get('approach', 'N/A')}")
        
        if "api_mapping" in strat:
            print("\nAPI Mapping:")
            for m in strat["api_mapping"]:
                print(f"  {m.get('original_function', '?')} → {m.get('cufft_replacement', '?')}")
        
        if "expected_speedup" in strat:
            sp = strat["expected_speedup"]
            print(f"\nExpected Speedup:")
            print(f"  FFT only:  {sp.get('fft_only', 'N/A')}")
            print(f"  Overall:   {sp.get('overall', 'N/A')}")
    
    if "implementation_plan" in stage2_json:
        print("\nImplementation Plan:")
        for step in stage2_json["implementation_plan"]:
            print(f"  Step {step.get('step', '?')}: {step.get('task', 'N/A')} "
                  f"({step.get('estimated_effort', '?')}, {step.get('priority', '?')})")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
