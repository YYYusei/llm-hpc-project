"""
Ablation Experiment: 分离 pipeline 结构 vs 模型能力的贡献

三个配置对比（只跑分类，不跑 GPU 代码生成）：
  - Original:   S1=GPT-4o,  S2=GPT-5.2  （对照组，从 summary.json 读取已有结果）
  - Ablation A: S1=GPT-5.2, S2=GPT-5.2  （测试：S2 还需要 S1 context 吗？）
  - Ablation B: S1=GPT-5.2, S2=GPT-4o   （测试：S2 的模型能力是关键吗？）

运行方式:
    cd llm-hpc-project
    python run_ablation.py                    # 跑全部 9 个程序
    python run_ablation.py --programs minimd hotspot srad  # 指定程序
    python run_ablation.py --config A         # 只跑 Ablation A
    python run_ablation.py --config B         # 只跑 Ablation B

输出:
    results/ablation/ 下生成结果 JSON + 对比表格
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/ablation.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# ── Benchmark 定义 ──
BENCHMARK_SOURCES = {
    "minimd":     "benchmarks/minimd/force_lj.cpp",
    "hpcg_spmv":  "benchmarks/hpcg/ComputeSPMV_ref.cpp",
    "hpcg_symgs": "benchmarks/hpcg/ComputeSYMGS_ref.cpp",
    "abinit":     "benchmarks/abinit/m_nonlop_ylm.F90",
    "hotspot":    "benchmarks/hotspot/hotspot.c",
    "srad":       "benchmarks/srad/srad.c",
    "lulesh":     "benchmarks/lulesh/lulesh_simplified.c",
    "nas_cg":     "benchmarks/nas_cg/cg.c",
    "jacobi2d":   "benchmarks/jacobi2d/jacobi2d.c",
}

CONFIG_NAME_MAP = {
    "minimd":     "minimd",
    "hpcg_spmv":  "hpcg",
    "hpcg_symgs": "hpcg",
    "abinit":     "abinit",
    "hotspot":    "hotspot",
    "srad":       "srad",
    "lulesh":     "lulesh",
    "nas_cg":     "nas_cg",
    "jacobi2d":   "jacobi2d",
}

# ── 原始结果（对照组）从 summary.json 读取 ──
ORIGINAL_SUMMARY_PATH = "results/extended_cascaded/summary.json"

# ── Ablation 配置 ──
ABLATION_CONFIGS = {
    "original": {"s1_model": "gpt-4o",  "s2_model": "gpt-5.2", "label": "Original (4o→5.2)"},
    "A":        {"s1_model": "gpt-5.2", "s2_model": "gpt-5.2", "label": "Ablation A (5.2→5.2)"},
    "B":        {"s1_model": "gpt-5.2", "s2_model": "gpt-4o",  "label": "Ablation B (5.2→4o)"},
}


def load_original_results() -> dict:
    """加载原始实验结果作为对照"""
    if not os.path.exists(ORIGINAL_SUMMARY_PATH):
        logger.warning(f"Original results not found: {ORIGINAL_SUMMARY_PATH}")
        return {}
    
    with open(ORIGINAL_SUMMARY_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    for r in data.get('results', []):
        prog = r['program']
        results[prog] = {
            "s1_bottleneck": r['stage1']['bottleneck_type'],
            "s1_score": r['stage1']['eval_score'],
            "s2_corrected": r['stage2']['bottleneck_corrected'],
            "s2_type": r['stage2'].get('corrected_type', 'N/A'),
            "cost": r['total_cost'],
        }
    return results


def run_ablation_on_program(pipeline, evaluator, program_name, config_label):
    """在单个程序上运行 ablation 配置"""
    
    source_path = BENCHMARK_SOURCES[program_name]
    config_name = CONFIG_NAME_MAP[program_name]
    
    if not os.path.exists(source_path):
        logger.error(f"Source file not found: {source_path}")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"[{config_label}] Program: {program_name}")
    logger.info(f"{'='*60}")
    
    try:
        result = pipeline.analyze(
            code_path=source_path,
            code_name=config_name,
            prompt_type="zero_shot"
        )
        
        # 评估 Stage 1
        stage1_eval = evaluator.evaluate(
            identified={
                "hotspots": result["stage1"]["hotspots"],
                "bottleneck_type": result["stage1"]["bottleneck_type"],
                "gpu_suitability": result["stage1"]["gpu_suitability"],
                "optimization_suggestions": [],
            },
            benchmark_name=config_name,
            prompt_type="stage1_zero_shot"
        )
        
        # Stage 2 验证结果
        stage2_validation = result["stage2"].get("validation", {})
        bottleneck_corrected = not stage2_validation.get("bottleneck_correct", True)
        corrected_type = stage2_validation.get("corrected_bottleneck", "N/A")
        
        summary = {
            "program": program_name,
            "config": config_label,
            "s1_model": result["stage1"]["model"],
            "s2_model": result["stage2"]["model"],
            "s1_bottleneck": result["stage1"].get("bottleneck_type", {}).get("primary", "unknown"),
            "s1_score": stage1_eval.total_score,
            "s1_gpu_suitable": result["stage1"].get("gpu_suitability", {}).get("suitable", "unknown"),
            "s2_corrected": bottleneck_corrected,
            "s2_type": corrected_type if bottleneck_corrected else "N/A",
            "s2_reasoning": stage2_validation.get("reasoning", "")[:300],
            "s2_gpu_suitable": result["stage2"].get("gpu_implementation", {}).get("suitable", "unknown"),
            "cost": result.get("total_cost", 0),
            "full_result": result,
        }
        
        logger.info(f"  S1 ({result['stage1']['model']}): {summary['s1_bottleneck']} (score: {summary['s1_score']:.1f})")
        logger.info(f"  S2 ({result['stage2']['model']}): corrected={summary['s2_corrected']}")
        if summary['s2_corrected']:
            logger.info(f"  → {summary['s2_type']}")
        logger.info(f"  Cost: ${summary['cost']:.4f}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error processing {program_name}: {e}", exc_info=True)
        return None


def print_comparison_table(original_results, ablation_a_results, ablation_b_results, programs):
    """打印对比表格"""
    
    print("\n" + "=" * 130)
    print("ABLATION COMPARISON TABLE")
    print("=" * 130)
    
    header = (f"{'Program':<14} | {'Original (4o→5.2)':<28} | {'Ablation A (5.2→5.2)':<28} | {'Ablation B (5.2→4o)':<28} |")
    subheader = (f"{'':14} | {'S1 BT':>8} {'S2 Corr':>8} {'Score':>8}   | {'S1 BT':>8} {'S2 Corr':>8} {'Score':>8}   | {'S1 BT':>8} {'S2 Corr':>8} {'Score':>8}   |")
    print(header)
    print(subheader)
    print("-" * 130)
    
    for prog in programs:
        orig = original_results.get(prog, {})
        ab_a = ablation_a_results.get(prog, {})
        ab_b = ablation_b_results.get(prog, {})
        
        def fmt(d):
            if not d:
                return f"{'—':>8} {'—':>8} {'—':>8}  "
            bt = d.get('s1_bottleneck', '?')[:8]
            corr = "YES" if d.get('s2_corrected', False) else "no"
            score = d.get('s1_score', 0)
            return f"{bt:>8} {corr:>8} {score:>8.1f}  "
        
        print(f"{prog:<14} | {fmt(orig)} | {fmt(ab_a)} | {fmt(ab_b)} |")
    
    print("-" * 130)
    
    # 汇总
    for label, results in [("Original", original_results), ("Ablation A", ablation_a_results), ("Ablation B", ablation_b_results)]:
        vals = [v for v in results.values() if v]
        corrections = sum(1 for v in vals if v.get('s2_corrected', False))
        total = len(vals)
        total_cost = sum(v.get('cost', 0) for v in vals)
        print(f"  {label:<20}: {corrections}/{total} corrections | Cost: ${total_cost:.4f}")
    
    print("=" * 130)
    
    # ── 解读提示 ──
    print("\nInterpretation guide:")
    print("  If A correction rate ≈ Original → S2 works on model capability alone, pipeline structure not essential")
    print("  If A correction rate < Original → S1 context (JSON) matters, pipeline structure has value")
    print("  If B correction rate < Original → S2 model capability (GPT-5.2) is essential")
    print("  If B correction rate ≈ Original → even GPT-4o as S2 can correct with the right context")
    print("  Ideal for thesis: both A and B drop → validates BOTH pipeline structure AND model pairing")


def main():
    parser = argparse.ArgumentParser(description="Ablation Experiment")
    parser.add_argument("--programs", nargs="+",
                        choices=list(BENCHMARK_SOURCES.keys()),
                        help="Specific programs to test (default: all 9)")
    parser.add_argument("--config", choices=["A", "B", "AB"],
                        default="AB",
                        help="Which ablation to run: A, B, or AB (default: both)")
    parser.add_argument("--output-dir", type=str,
                        default="results/ablation",
                        help="Output directory")
    
    args = parser.parse_args()
    
    programs = args.programs or list(BENCHMARK_SOURCES.keys())
    configs_to_run = list(args.config)  # "AB" → ["A", "B"]
    
    # 注册 benchmark 配置
    register_extended_benchmarks()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 加载原始结果
    logger.info("Loading original results as baseline...")
    original_results = load_original_results()
    logger.info(f"Loaded {len(original_results)} original results")
    
    evaluator = GeneralizedEvaluator()
    
    # ── 运行 Ablation 实验 ──
    ablation_results = {"A": {}, "B": {}}
    
    for config_key in configs_to_run:
        cfg = ABLATION_CONFIGS[config_key]
        logger.info(f"\n{'#'*60}")
        logger.info(f"Running {cfg['label']}: S1={cfg['s1_model']}, S2={cfg['s2_model']}")
        logger.info(f"{'#'*60}")
        
        pipeline = ConfigurableCascadedPipeline(
            s1_model=cfg['s1_model'],
            s2_model=cfg['s2_model']
        )
        
        config_dir = output_dir / f"ablation_{config_key}"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        for prog in programs:
            summary = run_ablation_on_program(pipeline, evaluator, prog, cfg['label'])
            if summary:
                ablation_results[config_key][prog] = summary
                
                # 保存单个结果
                with open(config_dir / f"{prog}_result.json", 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存配置汇总
        config_summary = {
            "config": cfg,
            "timestamp": datetime.now().isoformat(),
            "programs_tested": programs,
            "results": ablation_results[config_key],
        }
        with open(config_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(config_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # ── 打印对比表格 ──
    print_comparison_table(
        original_results,
        ablation_results.get("A", {}),
        ablation_results.get("B", {}),
        programs
    )
    
    # ── 保存完整对比 ──
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "programs": programs,
        "original": original_results,
        "ablation_A": {k: {kk: vv for kk, vv in v.items() if kk != 'full_result'} 
                       for k, v in ablation_results.get("A", {}).items()},
        "ablation_B": {k: {kk: vv for kk, vv in v.items() if kk != 'full_result'} 
                       for k, v in ablation_results.get("B", {}).items()},
    }
    with open(output_dir / "ablation_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
