"""
GPU 代码转换模块
使用 LLM 将 CPU 代码转换为 CUDA
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

from llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """转换结果"""
    # 基本信息
    source_file: str
    function_name: str
    timestamp: str
    model: str
    
    # 生成的代码
    cuda_kernel: str
    host_code: str
    full_code: str
    
    # 元数据
    elapsed_time: float
    total_tokens: int
    cost: float
    
    # 编译/测试结果
    compile_success: Optional[bool] = None
    compile_error: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save_code(self, filepath: str):
        """保存生成的 CUDA 代码"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.full_code)
        logger.info(f"CUDA code saved to {filepath}")
    
    def save_result(self, filepath: str):
        """保存完整结果"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Result saved to {filepath}")


class GPUConverter:
    """GPU 代码转换器"""
    
    # CUDA 转换 Prompt 模板
    CONVERSION_PROMPT = """You are an expert in CUDA programming and HPC optimization. Convert the following C++ function to an efficient CUDA implementation.

## Source Function
Function name: {function_name}

```cpp
{code}
```

## System Information
- Target GPU: NVIDIA RTX 3060 (Compute Capability 8.6)
- CUDA Cores: 3840
- Memory: 6 GB GDDR6
- Memory Bandwidth: 336 GB/s

## Requirements
1. Create a CUDA kernel that parallelizes the main computation
2. Use one thread per primary work unit (e.g., one thread per atom/row)
3. Ensure coalesced memory access where possible
4. Use shared memory for frequently accessed data if beneficial
5. Handle boundary conditions correctly
6. Minimize atomic operations
7. Include necessary memory transfers (host to device, device to host)

## Output Format
Provide the complete CUDA code including:
1. Necessary includes and defines
2. CUDA kernel function(s)
3. Host wrapper function that:
   - Allocates device memory
   - Copies data to device
   - Launches kernel with appropriate grid/block dimensions
   - Copies results back to host
   - Frees device memory
4. Comments explaining key optimizations

Output ONLY the code, no additional explanation.
"""

    OPTIMIZATION_PROMPT = """Review and optimize the following CUDA code for better performance on RTX 3060.

Current code:
```cuda
{cuda_code}
```

Original CPU function context:
```cpp
{original_code}
```

Suggest optimizations for:
1. Memory access patterns (coalescing, bank conflicts)
2. Occupancy and register usage
3. Shared memory utilization
4. Warp divergence reduction
5. Instruction-level optimizations

Provide the optimized code with comments explaining each optimization.
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o"
    ):
        """
        初始化转换器
        
        Args:
            api_key: OpenAI API Key
            model: 模型名称
        """
        self.client = LLMClient(api_key=api_key, model=model)
        logger.info("GPU Converter initialized")
    
    def convert(
        self,
        code_path: str,
        function_name: str,
        context: Optional[str] = None
    ) -> ConversionResult:
        """
        将 CPU 函数转换为 CUDA
        
        Args:
            code_path: 源代码文件路径
            function_name: 要转换的函数名
            context: 额外上下文信息
            
        Returns:
            ConversionResult 对象
        """
        # 读取源代码
        code_path = Path(code_path)
        with open(code_path, 'r', encoding='utf-8') as f:
            full_code = f.read()
        
        # 提取目标函数（简单实现，可能需要更复杂的解析）
        function_code = self._extract_function(full_code, function_name)
        if not function_code:
            logger.warning(f"Could not extract function {function_name}, using full file")
            function_code = full_code
        
        # 构建 prompt
        prompt = self.CONVERSION_PROMPT.format(
            function_name=function_name,
            code=function_code
        )
        
        if context:
            prompt += f"\n\nAdditional context:\n{context}"
        
        # 调用 LLM
        logger.info(f"Converting {function_name} to CUDA...")
        response = self.client.chat(prompt, parse_json=False)
        
        # 提取 CUDA 代码
        cuda_code = self._extract_cuda_code(response.content)
        kernel_code, host_code = self._split_kernel_and_host(cuda_code)
        
        result = ConversionResult(
            source_file=str(code_path),
            function_name=function_name,
            timestamp=datetime.now().isoformat(),
            model=response.model,
            cuda_kernel=kernel_code,
            host_code=host_code,
            full_code=cuda_code,
            elapsed_time=response.elapsed_time,
            total_tokens=response.total_tokens,
            cost=response.cost
        )
        
        logger.info(f"Conversion complete, generated {len(cuda_code)} characters of CUDA code")
        
        return result
    
    def _extract_function(self, code: str, function_name: str) -> Optional[str]:
        """从代码中提取指定函数"""
        # 简单的函数提取（基于模板匹配）
        # 实际使用可能需要更复杂的 C++ 解析器
        
        # 尝试匹配模板函数
        patterns = [
            # 模板函数
            rf'template\s*<[^>]+>\s*\n?\s*\w+\s+\w+::{function_name}\s*\([^)]*\)\s*\{{',
            # 普通成员函数
            rf'\w+\s+\w+::{function_name}\s*\([^)]*\)\s*\{{',
            # 普通函数
            rf'\w+\s+{function_name}\s*\([^)]*\)\s*\{{'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, code)
            if match:
                start = match.start()
                # 找到匹配的闭合括号
                brace_count = 0
                end = start
                in_function = False
                
                for i, char in enumerate(code[start:], start):
                    if char == '{':
                        brace_count += 1
                        in_function = True
                    elif char == '}':
                        brace_count -= 1
                        if in_function and brace_count == 0:
                            end = i + 1
                            break
                
                if end > start:
                    return code[start:end]
        
        return None
    
    def _extract_cuda_code(self, response: str) -> str:
        """从 LLM 响应中提取 CUDA 代码"""
        # 尝试提取代码块
        code_block_pattern = r'```(?:cuda|cpp|c\+\+)?\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return '\n\n'.join(matches)
        
        # 如果没有代码块，返回整个响应
        return response
    
    def _split_kernel_and_host(self, cuda_code: str) -> tuple:
        """分离 kernel 和 host 代码"""
        kernel_pattern = r'__global__\s+void\s+\w+\s*\([^)]*\)\s*\{[^}]*\}'
        kernel_matches = re.findall(kernel_pattern, cuda_code, re.DOTALL)
        
        kernel_code = '\n\n'.join(kernel_matches) if kernel_matches else ""
        
        # Host code is everything else (简化处理)
        host_code = cuda_code
        for kernel in kernel_matches:
            host_code = host_code.replace(kernel, '')
        
        return kernel_code.strip(), host_code.strip()
    
    def optimize(
        self,
        conversion_result: ConversionResult,
        original_code: str
    ) -> ConversionResult:
        """
        优化已转换的 CUDA 代码
        
        Args:
            conversion_result: 初始转换结果
            original_code: 原始 CPU 代码
            
        Returns:
            优化后的 ConversionResult
        """
        prompt = self.OPTIMIZATION_PROMPT.format(
            cuda_code=conversion_result.full_code,
            original_code=original_code
        )
        
        logger.info("Optimizing CUDA code...")
        response = self.client.chat(prompt, parse_json=False)
        
        optimized_code = self._extract_cuda_code(response.content)
        kernel_code, host_code = self._split_kernel_and_host(optimized_code)
        
        optimized_result = ConversionResult(
            source_file=conversion_result.source_file,
            function_name=conversion_result.function_name + "_optimized",
            timestamp=datetime.now().isoformat(),
            model=response.model,
            cuda_kernel=kernel_code,
            host_code=host_code,
            full_code=optimized_code,
            elapsed_time=response.elapsed_time,
            total_tokens=response.total_tokens,
            cost=response.cost
        )
        
        return optimized_result
    
    def verify_compilation(
        self,
        result: ConversionResult,
        output_dir: str = "results/conversion"
    ) -> bool:
        """
        验证 CUDA 代码是否可以编译
        
        Args:
            result: 转换结果
            output_dir: 输出目录
            
        Returns:
            是否编译成功
        """
        import subprocess
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存代码到临时文件
        cuda_file = output_dir / f"{result.function_name}.cu"
        result.save_code(cuda_file)
        
        # 尝试编译
        try:
            compile_result = subprocess.run(
                ["nvcc", "-c", str(cuda_file), "-o", str(cuda_file.with_suffix('.o'))],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if compile_result.returncode == 0:
                result.compile_success = True
                logger.info("CUDA compilation successful")
                return True
            else:
                result.compile_success = False
                result.compile_error = compile_result.stderr
                logger.error(f"CUDA compilation failed: {compile_result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.warning("nvcc not found, skipping compilation check")
            return None
        except subprocess.TimeoutExpired:
            result.compile_success = False
            result.compile_error = "Compilation timeout"
            return False


class ConversionExperiment:
    """GPU 转换实验管理器"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        output_dir: str = "results/conversion"
    ):
        self.converter = GPUConverter(api_key=api_key, model=model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_conversion_experiment(
        self,
        code_path: str,
        function_name: str,
        iterations: int = 1
    ) -> List[ConversionResult]:
        """
        运行转换实验
        
        Args:
            code_path: 源代码路径
            function_name: 函数名
            iterations: 重复次数（测试一致性）
            
        Returns:
            转换结果列表
        """
        results = []
        
        for i in range(iterations):
            logger.info(f"Conversion iteration {i+1}/{iterations}")
            
            result = self.converter.convert(code_path, function_name)
            
            # 验证编译
            self.converter.verify_compilation(result, str(self.output_dir))
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result.save_result(
                self.output_dir / f"{function_name}_{timestamp}.json"
            )
            result.save_code(
                self.output_dir / f"{function_name}_{timestamp}.cu"
            )
            
            results.append(result)
        
        return results


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    converter = GPUConverter()
    
    # 示例转换
    result = converter.convert(
        code_path="benchmarks/minimd/force_lj.cpp",
        function_name="compute_fullneigh"
    )
    
    print(f"Generated {len(result.full_code)} characters of CUDA code")
    print(f"Cost: ${result.cost:.4f}")
