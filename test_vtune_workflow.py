import sys
import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, 'src')

from vtune_integration import create_manual_vtune_report, VTuneReport
from llm_client import LLMClient


def create_vtune_data() -> dict:
    """Create VTune reports for HPCG SYMGS and SPMV"""
    
    # HPCG full VTune data (based on actual profiling)
    hpcg_report = create_manual_vtune_report(
        hotspots=[
            {"name": "ComputeSYMGS_ref", "time": 45.731, "percentage": 67.3, 
             "module": "xhpcg", "cpi_rate": 2.1},
            {"name": "ComputeSPMV_ref", "time": 18.834, "percentage": 27.7,
             "module": "xhpcg", "cpi_rate": 1.9},
            {"name": "ComputeDotProduct_ref", "time": 1.358, "percentage": 2.0,
             "module": "xhpcg", "cpi_rate": 1.2},
            {"name": "ComputeWAXPBY_ref", "time": 0.865, "percentage": 1.3,
             "module": "xhpcg", "cpi_rate": 0.8},
            {"name": "ComputeMG_ref", "time": 0.521, "percentage": 0.8,
             "module": "xhpcg", "cpi_rate": 1.5},
        ],
        total_time=67.977,
        cpu_utilization=11.6,
        cpu_model="Intel Core i7-11800H @ 2.30GHz",
        cores=8,
        threads=16
    )
    
    return {
        "hpcg": hpcg_report
    }


def format_vtune_for_prompt(report: VTuneReport) -> str:
    """Format VTune data for LLM prompt"""
    
    lines = [
        "## VTune Profiling Results",
        "",
        f"- Total Elapsed Time: {report.total_elapsed_time:.2f}s",
        f"- CPU Utilization: {report.cpu_utilization:.1f}%",
        f"- CPU: {report.cpu_model}",
        f"- Cores/Threads: {report.cores}/{report.threads}",
        "",
        "### Top Hotspots:",
        "",
        "| Function | CPU Time | Percentage | CPI Rate |",
        "|----------|----------|------------|----------|",
    ]
    
    for h in report.get_top_hotspots(5):
        cpi = f"{h.cpi_rate:.2f}" if h.cpi_rate else "N/A"
        lines.append(f"| {h.function_name} | {h.cpu_time:.3f}s | {h.cpu_time_percentage:.1f}% | {cpi} |")
    
    lines.extend([
        "",
        "### Performance Indicators:",
        f"- High CPI (>1.5) indicates memory-bound behavior",
        f"- Low CPU utilization ({report.cpu_utilization:.1f}%) suggests memory bandwidth limitation",
    ])
    
    return "\n".join(lines)


def analyze_with_vtune_context(
    code_path: str,
    vtune_report: VTuneReport,
    function_name: str,
    model: str = "gpt-4o"
) -> dict:
    """Analyze code with VTune context"""
    
    # Read source code
    with open(code_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    # Format VTune data
    vtune_context = format_vtune_for_prompt(vtune_report)
    
    # Create prompt
    prompt = f'''
Analyze the following HPC code with VTune profiling data.

{vtune_context}

## Source Code ({function_name}):
```cpp
{source_code}
```

## Analysis Request:
Based on the VTune profiling data and source code, provide:

1. **Bottleneck Type**: Is this function compute-bound, memory-bound, or communication-bound? 
   Justify based on CPI rate and CPU utilization.

2. **Root Cause Analysis**: What specific code patterns cause the bottleneck?

3. **GPU Suitability**: Is this function suitable for GPU acceleration? Why?

4. **Optimization Suggestions**: Provide 3-5 specific optimization suggestions with expected speedup.

Format your response as JSON:
```json
{{
    "function": "{function_name}",
    "bottleneck_type": "memory|compute|communication",
    "bottleneck_justification": "...",
    "root_causes": ["cause1", "cause2"],
    "gpu_suitable": true|false,
    "gpu_justification": "...",
    "optimizations": [
        {{"suggestion": "...", "expected_speedup": "1.5-2x", "effort": "low|medium|high"}}
    ]
}}
```
'''
    
    # Call LLM
    client = LLMClient(model=model)
    response = client.chat(
        prompt=prompt,
        system_prompt="You are an HPC performance analysis expert. Analyze code using VTune profiling data."
    )
    
    # Parse response
    import re
    json_match = re.search(r'```json\n(.*?)```', response.content, re.DOTALL)
    if json_match:
        try:
            analysis = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            analysis = {"raw_response": response.content}
    else:
        analysis = {"raw_response": response.content}
    
    return {
        "model": model,
        "function": function_name,
        "vtune_data": {
            "cpu_time": next((h.cpu_time for h in vtune_report.hotspots if h.function_name == function_name), 0),
            "percentage": next((h.cpu_time_percentage for h in vtune_report.hotspots if h.function_name == function_name), 0),
            "cpi_rate": next((h.cpi_rate for h in vtune_report.hotspots if h.function_name == function_name), None),
        },
        "analysis": analysis,
        "cost": response.cost,
        "time": response.elapsed_time
    }


def run_vtune_workflow():
    """Run complete VTune workflow validation"""
    
    logger.info("=" * 60)
    logger.info("Q2: VTune Workflow Validation")
    logger.info("=" * 60)
    
    # Create VTune data
    vtune_data = create_vtune_data()
    hpcg_report = vtune_data["hpcg"]
    
    # Save VTune reports
    os.makedirs("results/vtune", exist_ok=True)
    hpcg_report.save("results/vtune/hpcg_vtune_report.json")
    logger.info("VTune reports saved to results/vtune/")
    
    # Analyze both functions
    results = {
        "timestamp": datetime.now().isoformat(),
        "vtune_summary": {
            "total_time": hpcg_report.total_elapsed_time,
            "cpu_utilization": hpcg_report.cpu_utilization,
            "top_hotspots": [
                {"name": h.function_name, "percentage": h.cpu_time_percentage}
                for h in hpcg_report.get_top_hotspots(3)
            ]
        },
        "analyses": {}
    }
    
    # Test functions
    test_cases = [
        ("benchmarks/hpcg/ComputeSYMGS_ref.cpp", "ComputeSYMGS_ref"),
        ("benchmarks/hpcg/ComputeSPMV_ref.cpp", "ComputeSPMV_ref"),
    ]
    
    total_cost = 0
    
    for code_path, func_name in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing: {func_name}")
        logger.info("="*60)
        
        # Analyze with GPT-4o (with VTune context)
        result = analyze_with_vtune_context(
            code_path=code_path,
            vtune_report=hpcg_report,
            function_name=func_name,
            model="gpt-4o"
        )
        
        results["analyses"][func_name] = result
        total_cost += result["cost"]
        
        # Print summary
        analysis = result.get("analysis", {})
        if isinstance(analysis, dict) and "bottleneck_type" in analysis:
            print(f"\n{func_name}:")
            print(f"  VTune: {result['vtune_data']['percentage']:.1f}% CPU time, CPI={result['vtune_data']['cpi_rate']}")
            print(f"  Bottleneck: {analysis.get('bottleneck_type', 'unknown')}")
            print(f"  GPU Suitable: {analysis.get('gpu_suitable', 'unknown')}")
            print(f"  Optimizations: {len(analysis.get('optimizations', []))} suggestions")
        else:
            print(f"\n{func_name}: Analysis completed (see JSON for details)")
        
        print(f"  Cost: ${result['cost']:.4f}, Time: {result['time']:.1f}s")
    
    results["total_cost"] = total_cost
    
    # Save results
    output_file = "results/vtune/vtune_workflow_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Print summary
    print("\n" + "=" * 70)
    print("VTUNE WORKFLOW VALIDATION SUMMARY")
    print("=" * 70)
    
    print("\n### VTune Data Integration:")
    print(f"  Total Time: {hpcg_report.total_elapsed_time:.2f}s")
    print(f"  CPU Utilization: {hpcg_report.cpu_utilization:.1f}%")
    print(f"  Top Hotspot: {hpcg_report.get_top_hotspots(1)[0].function_name} ({hpcg_report.get_top_hotspots(1)[0].cpu_time_percentage:.1f}%)")
    
    print("\n### LLM Analysis Results:")
    for func_name, result in results["analyses"].items():
        analysis = result.get("analysis", {})
        if isinstance(analysis, dict):
            print(f"\n  {func_name}:")
            print(f"    Bottleneck: {analysis.get('bottleneck_type', 'N/A')}")
            print(f"    GPU Suitable: {analysis.get('gpu_suitable', 'N/A')}")
            if "optimizations" in analysis:
                print(f"    Optimizations: {len(analysis['optimizations'])} suggestions")
    
    print(f"\n### Cost: ${total_cost:.4f}")
    print(f"### Results saved to: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_vtune_workflow()