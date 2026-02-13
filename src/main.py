#!/usr/bin/env python3
"""
LLM-HPC: LLM 辅助 HPC 性能分析与 GPU 转换
主入口程序
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import LLMClient
from analyzer import HPCAnalyzer
from converter import GPUConverter

console = Console()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/llm_hpc.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    config_path = Path(config_path)
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        console.print("Please copy configs/config.example.yaml to configs/config.yaml")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 替换环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        config['api']['openai_key'] = api_key
    
    return config


@click.group()
@click.option('--config', '-c', default='configs/config.yaml', help='配置文件路径')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
@click.pass_context
def cli(ctx, config, verbose):
    """LLM-HPC: LLM 辅助 HPC 性能分析与 GPU 转换工具"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 确保目录存在
    Path('logs').mkdir(exist_ok=True)
    Path('results/analysis').mkdir(parents=True, exist_ok=True)
    Path('results/conversion').mkdir(parents=True, exist_ok=True)


@cli.command()
@click.option('--code', '-f', required=True, help='代码文件路径或预设名称 (minimd, hpcg)')
@click.option('--prompt', '-p', default='zero_shot', 
              type=click.Choice(['zero_shot', 'few_shot', 'contextual', 'all']),
              help='Prompt 类型')
@click.option('--output', '-o', default='results/analysis', help='输出目录')
@click.pass_context
def analyze(ctx, code, prompt, output):
    """分析 HPC 代码性能瓶颈"""
    config = ctx.obj['config']
    
    console.print(Panel.fit(
        "[bold blue]LLM 性能分析[/bold blue]",
        subtitle="HPC Code Performance Analysis"
    ))
    
    # 解析代码路径
    if code in config.get('benchmarks', {}):
        benchmark = config['benchmarks'][code]
        code_path = benchmark['path']
        ground_truth = benchmark.get('ground_truth', {})
        profiling_data = benchmark.get('profiling_data', {})
        code_name = code
    else:
        code_path = code
        ground_truth = {}
        profiling_data = {}
        code_name = Path(code).stem
    
    if not Path(code_path).exists():
        console.print(f"[red]Code file not found: {code_path}[/red]")
        return
    
    # 初始化分析器
    analyzer = HPCAnalyzer(
        api_key=config['api'].get('openai_key'),
        model=config['llm'].get('model', 'gpt-4o')
    )
    
    # 运行分析
    prompt_types = ['zero_shot', 'few_shot', 'contextual'] if prompt == 'all' else [prompt]
    
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for pt in prompt_types:
            task = progress.add_task(f"Running {pt} analysis...", total=None)
            
            try:
                result = analyzer.analyze(
                    code_path=code_path,
                    prompt_type=pt,
                    profiling_data=profiling_data if pt == 'contextual' else None,
                    code_name=code_name
                )
                
                if ground_truth:
                    analyzer.evaluate(result, ground_truth)
                
                # 保存结果
                result.save(Path(output) / f"{code_name}_{pt}.json")
                results.append(result)
                
                progress.update(task, completed=True)
                
            except Exception as e:
                console.print(f"[red]Error in {pt} analysis: {e}[/red]")
                logger.exception(e)
    
    # 显示结果摘要
    _display_analysis_results(results)


def _display_analysis_results(results):
    """显示分析结果摘要"""
    if not results:
        return
    
    table = Table(title="Analysis Results")
    table.add_column("Prompt", style="cyan")
    table.add_column("Hotspot", style="green")
    table.add_column("Bottleneck", style="yellow")
    table.add_column("GPU", style="blue")
    table.add_column("Score", style="magenta")
    table.add_column("Cost", style="red")
    
    for result in results:
        hotspot = result.hotspots[0]['location'][:40] + "..." if result.hotspots else "N/A"
        bottleneck = result.bottleneck_type.get('primary', 'N/A')
        gpu = "✅" if result.gpu_suitability.get('suitable') else "❌"
        score = result.evaluation.get('score', 'N/A') if result.evaluation else 'N/A'
        
        table.add_row(
            result.prompt_type,
            hotspot,
            bottleneck,
            gpu,
            str(score),
            f"${result.cost:.4f}"
        )
    
    console.print(table)


@cli.command()
@click.option('--code', '-f', required=True, help='代码文件路径或预设名称')
@click.option('--function', '-fn', required=True, help='要转换的函数名')
@click.option('--output', '-o', default='results/conversion', help='输出目录')
@click.option('--optimize', is_flag=True, help='是否进行二次优化')
@click.pass_context
def convert(ctx, code, function, output, optimize):
    """将 CPU 代码转换为 CUDA"""
    config = ctx.obj['config']
    
    console.print(Panel.fit(
        "[bold green]GPU 代码转换[/bold green]",
        subtitle="CPU to CUDA Conversion"
    ))
    
    # 解析代码路径
    if code in config.get('benchmarks', {}):
        code_path = config['benchmarks'][code]['path']
    else:
        code_path = code
    
    if not Path(code_path).exists():
        console.print(f"[red]Code file not found: {code_path}[/red]")
        return
    
    # 初始化转换器
    converter = GPUConverter(
        api_key=config['api'].get('openai_key'),
        model=config['llm'].get('model', 'gpt-4o')
    )
    
    # 转换
    with console.status(f"Converting {function} to CUDA..."):
        result = converter.convert(code_path, function)
    
    console.print(f"[green]✅ Conversion complete![/green]")
    console.print(f"Generated {len(result.full_code)} characters of CUDA code")
    console.print(f"Cost: ${result.cost:.4f}")
    
    # 验证编译
    with console.status("Verifying CUDA compilation..."):
        compile_ok = converter.verify_compilation(result, output)
    
    if compile_ok:
        console.print("[green]✅ CUDA compilation successful![/green]")
    elif compile_ok is None:
        console.print("[yellow]⚠️ nvcc not found, skipping compilation check[/yellow]")
    else:
        console.print(f"[red]❌ Compilation failed: {result.compile_error}[/red]")
    
    # 优化（可选）
    if optimize:
        with console.status("Optimizing CUDA code..."):
            with open(code_path, 'r') as f:
                original_code = f.read()
            optimized = converter.optimize(result, original_code)
        
        optimized.save_code(Path(output) / f"{function}_optimized.cu")
        console.print("[green]✅ Optimization complete![/green]")
    
    # 保存结果
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result.save_code(output_dir / f"{function}.cu")
    result.save_result(output_dir / f"{function}.json")
    
    console.print(f"\nOutput saved to: {output_dir}")


@cli.command()
@click.option('--all', 'run_all', is_flag=True, help='运行所有实验')
@click.option('--analysis-only', is_flag=True, help='只运行分析实验')
@click.option('--conversion-only', is_flag=True, help='只运行转换实验')
@click.pass_context
def experiment(ctx, run_all, analysis_only, conversion_only):
    """运行完整实验"""
    config = ctx.obj['config']
    
    console.print(Panel.fit(
        "[bold magenta]完整实验[/bold magenta]",
        subtitle="Full Experiment Suite"
    ))
    
    if run_all or analysis_only:
        console.print("\n[bold]Phase 1: Performance Analysis[/bold]")
        
        for name, benchmark in config.get('benchmarks', {}).items():
            console.print(f"\n[cyan]Analyzing {name}...[/cyan]")
            
            analyzer = HPCAnalyzer(
                api_key=config['api'].get('openai_key'),
                model=config['llm'].get('model', 'gpt-4o')
            )
            
            try:
                results = analyzer.run_experiment(
                    code_path=benchmark['path'],
                    ground_truth=benchmark.get('ground_truth', {}),
                    profiling_data=benchmark.get('profiling_data', {}),
                    code_name=name
                )
                _display_analysis_results(results)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    if run_all or conversion_only:
        console.print("\n[bold]Phase 2: GPU Conversion[/bold]")
        
        for name, benchmark in config.get('benchmarks', {}).items():
            functions = config.get('conversion', {}).get('target_functions', {}).get(name, [])
            
            for func in functions:
                console.print(f"\n[cyan]Converting {name}::{func}...[/cyan]")
                
                converter = GPUConverter(
                    api_key=config['api'].get('openai_key'),
                    model=config['llm'].get('model', 'gpt-4o')
                )
                
                try:
                    result = converter.convert(benchmark['path'], func)
                    result.save_code(f"results/conversion/{name}_{func}.cu")
                    console.print(f"[green]✅ {func} converted successfully[/green]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
    
    console.print("\n[bold green]Experiment complete![/bold green]")


@cli.command()
@click.pass_context
def status(ctx):
    """显示项目状态"""
    config = ctx.obj['config']
    
    console.print(Panel.fit(
        "[bold]项目状态[/bold]",
        subtitle="Project Status"
    ))
    
    # 检查 API Key
    api_key = config['api'].get('openai_key')
    if api_key and api_key != "${OPENAI_API_KEY}":
        console.print("✅ API Key configured")
    else:
        console.print("❌ API Key not configured")
    
    # 检查基准代码
    console.print("\n[bold]Benchmarks:[/bold]")
    for name, benchmark in config.get('benchmarks', {}).items():
        path = Path(benchmark['path'])
        status = "✅" if path.exists() else "❌"
        console.print(f"  {status} {name}: {path}")
    
    # 检查 Prompt 文件
    console.print("\n[bold]Prompts:[/bold]")
    for prompt_type in ['zero_shot', 'few_shot', 'contextual']:
        path = Path(f"prompts/{prompt_type}.txt")
        status = "✅" if path.exists() else "❌"
        console.print(f"  {status} {prompt_type}")
    
    # 检查结果
    console.print("\n[bold]Results:[/bold]")
    analysis_count = len(list(Path("results/analysis").glob("*.json"))) if Path("results/analysis").exists() else 0
    conversion_count = len(list(Path("results/conversion").glob("*.cu"))) if Path("results/conversion").exists() else 0
    console.print(f"  Analysis results: {analysis_count}")
    console.print(f"  Conversion results: {conversion_count}")


if __name__ == "__main__":
    cli()
