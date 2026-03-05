"""
扩展 Cascaded Pipeline 测试
在 5 个新 benchmark 上运行级联分析，验证泛化性

运行方式:
    cd llm-hpc-project
    python test_extended_cascaded.py [--programs hotspot srad lulesh nas_cg jacobi2d]
    python test_extended_cascaded.py --all          # 运行全部 9 个 (含原有 4 个)
    python test_extended_cascaded.py --new-only     # 只运行新增 5 个

输出:
    results/extended_cascaded/ 下生成每个程序的分析结果 JSON
    最后打印汇总表格
"""

import sys
import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cascaded_pipeline import CascadedPipeline
from generalized_evaluator import GeneralizedEvaluator
from extended_benchmark_config import register_extended_benchmarks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/extended_cascaded.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Benchmark 定义: name -> source file path
BENCHMARK_SOURCES = {
    # 原有 4 个
    "minimd":     "benchmarks/minimd/force_lj.cpp",
    "hpcg_spmv":  "benchmarks/hpcg/ComputeSPMV_ref.cpp",
    "hpcg_symgs": "benchmarks/hpcg/ComputeSYMGS_ref.cpp",
    "abinit":     "benchmarks/abinit/m_nonlop_ylm.F90",
    # 新增 5 个
    "hotspot":    "benchmarks/hotspot/hotspot.c",
    "srad":       "benchmarks/srad/srad.c",
    "lulesh":     "benchmarks/lulesh/lulesh_simplified.c",
    "nas_cg":     "benchmarks/nas_cg/cg.c",
    "jacobi2d":   "benchmarks/jacobi2d/jacobi2d.c",
}

# benchmark_config 中的名称映射 (用于评估)
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

NEW_PROGRAMS = ["hotspot", "srad", "lulesh", "nas_cg", "jacobi2d"]
OLD_PROGRAMS = ["minimd", "hpcg_spmv", "hpcg_symgs", "abinit"]


def run_cascaded_on_program(pipeline, evaluator, program_name, output_dir):
    """在单个程序上运行级联分析 + 评估"""
    
    source_path = BENCHMARK_SOURCES[program_name]
    config_name = CONFIG_NAME_MAP[program_name]
    
    if not os.path.exists(source_path):
        logger.error(f"Source file not found: {source_path}")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Program: {program_name} ({source_path})")
    logger.info(f"{'='*60}")
    
    try:
        # 运行级联分析 (只用 zero_shot 以节省 API 费用)
        result = pipeline.analyze(
            code_path=source_path,
            code_name=config_name,
            prompt_type="zero_shot"
        )
        
        # 评估 Stage 1 结果
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
        
        # 提取 Stage 2 验证结果
        stage2_validation = result["stage2"].get("validation", {})
        bottleneck_corrected = not stage2_validation.get("bottleneck_correct", True)
        corrected_type = stage2_validation.get("corrected_bottleneck", "N/A")
        
        # 汇总
        summary = {
            "program": program_name,
            "source": source_path,
            "timestamp": datetime.now().isoformat(),
            "stage1": {
                "model": "gpt-4o",
                "bottleneck_type": result["stage1"].get("bottleneck_type", {}).get("primary", "unknown"),
                "hotspot_count": len(result["stage1"].get("hotspots", [])),
                "gpu_suitable": result["stage1"].get("gpu_suitability", {}).get("suitable", "unknown"),
                "eval_score": stage1_eval.total_score,
                "cost": result["stage1"].get("cost", 0),
            },
            "stage2": {
                "model": "gpt-5.2",
                "bottleneck_corrected": bottleneck_corrected,
                "corrected_type": corrected_type if bottleneck_corrected else "N/A",
                "validation_reasoning": stage2_validation.get("reasoning", ""),
                "num_optimizations": len(result["stage2"].get("detailed_optimizations", [])),
                "cost": result["stage2"].get("cost", 0),
            },
            "total_cost": result.get("total_cost", 0),
            "full_result": result,
        }
        
        # 保存结果
        output_file = output_dir / f"{program_name}_cascaded_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"  Stage 1 score: {stage1_eval.total_score:.1f}/100")
        logger.info(f"  Stage 1 bottleneck: {summary['stage1']['bottleneck_type']}")
        logger.info(f"  Stage 2 corrected: {bottleneck_corrected}")
        if bottleneck_corrected:
            logger.info(f"  Corrected to: {corrected_type}")
        logger.info(f"  Total cost: ${summary['total_cost']:.4f}")
        logger.info(f"  Saved to: {output_file}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error processing {program_name}: {e}", exc_info=True)
        return None


def print_summary_table(results):
    """打印汇总表格"""
    print("\n" + "=" * 100)
    print("CASCADED PIPELINE RESULTS SUMMARY")
    print("=" * 100)
    
    header = f"{'Program':<14} {'S1 Bottleneck':<16} {'S1 Score':<10} {'S2 Corrected':<14} {'Corrected To':<18} {'S2 Opts':<8} {'Cost':<8}"
    print(header)
    print("-" * 100)
    
    total_cost = 0
    corrections = 0
    total_programs = 0
    
    for r in results:
        if r is None:
            continue
        total_programs += 1
        s1 = r["stage1"]
        s2 = r["stage2"]
        total_cost += r["total_cost"]
        if s2["bottleneck_corrected"]:
            corrections += 1
        
        corrected_str = "YES" if s2["bottleneck_corrected"] else "no"
        corrected_to = s2["corrected_type"] if s2["bottleneck_corrected"] else "-"
        
        print(f"{r['program']:<14} {s1['bottleneck_type']:<16} {s1['eval_score']:<10.1f} "
              f"{corrected_str:<14} {corrected_to:<18} {s2['num_optimizations']:<8} "
              f"${r['total_cost']:.4f}")
    
    print("-" * 100)
    print(f"Total: {total_programs} programs | "
          f"Corrections: {corrections}/{total_programs} ({100*corrections/max(total_programs,1):.0f}%) | "
          f"Total cost: ${total_cost:.4f}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Extended Cascaded Pipeline Test")
    parser.add_argument("--programs", nargs="+", 
                        choices=list(BENCHMARK_SOURCES.keys()),
                        help="Specific programs to test")
    parser.add_argument("--all", action="store_true",
                        help="Test all 9 programs (original 4 + new 5)")
    parser.add_argument("--new-only", action="store_true", default=True,
                        help="Test only new 5 programs (default)")
    parser.add_argument("--output-dir", type=str, 
                        default="results/extended_cascaded",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # 确定要测试的程序
    if args.programs:
        programs = args.programs
    elif args.all:
        programs = list(BENCHMARK_SOURCES.keys())
    else:
        programs = NEW_PROGRAMS
    
    # 注册扩展 benchmark 配置
    register_extended_benchmarks()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 初始化 pipeline 和 evaluator
    pipeline = CascadedPipeline()
    evaluator = GeneralizedEvaluator()
    
    logger.info(f"Testing {len(programs)} programs: {programs}")
    logger.info(f"Output directory: {output_dir}")
    
    # 运行测试
    results = []
    for prog in programs:
        summary = run_cascaded_on_program(pipeline, evaluator, prog, output_dir)
        results.append(summary)
    
    # 打印汇总
    print_summary_table(results)
    
    # 保存汇总文件
    summary_file = output_dir / "summary.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "programs_tested": programs,
        "results": [r for r in results if r is not None],
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
