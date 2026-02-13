"""
LLM API 客户端模块
支持 OpenAI GPT 系列模型
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 响应数据类"""
    content: str
    parsed_json: Optional[Dict[str, Any]]
    model: str
    elapsed_time: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    @property
    def cost(self) -> float:
        """估算 API 调用成本 (USD)"""
        # GPT-4o 价格: $2.5/1M input, $10/1M output
        input_cost = self.prompt_tokens * 2.5 / 1_000_000
        output_cost = self.completion_tokens * 10 / 1_000_000
        return input_cost + output_cost


class LLMClient:
    """LLM API 客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0,
        max_tokens: int = 4096,
        timeout: int = 120
    ):
        """
        初始化 LLM 客户端
        
        Args:
            api_key: OpenAI API Key，默认从环境变量读取
            model: 模型名称
            temperature: 温度参数，0 表示确定性输出
            max_tokens: 最大输出 token 数
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API Key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        
        logger.info(f"LLM Client initialized with model: {model}")
    
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        parse_json: bool = True
    ) -> LLMResponse:
        """
        发送聊天请求
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            parse_json: 是否尝试解析 JSON 响应
            
        Returns:
            LLMResponse 对象
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        logger.debug(f"Sending request to {self.model}...")
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
        
        elapsed_time = time.time() - start_time
        
        content = response.choices[0].message.content
        
        # 尝试解析 JSON
        parsed_json = None
        if parse_json:
            parsed_json = self._try_parse_json(content)
        
        result = LLMResponse(
            content=content,
            parsed_json=parsed_json,
            model=self.model,
            elapsed_time=elapsed_time,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        logger.info(
            f"Response received in {elapsed_time:.2f}s, "
            f"tokens: {result.total_tokens}, "
            f"cost: ${result.cost:.4f}"
        )
        
        return result
    
    def _try_parse_json(self, content: str) -> Optional[Dict[str, Any]]:
        """尝试从响应中解析 JSON"""
        # 清理 markdown 代码块
        content_clean = content.strip()
        
        if content_clean.startswith("```"):
            lines = content_clean.split("\n")
            # 移除第一行和最后一行的 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content_clean = "\n".join(lines)
        
        try:
            return json.loads(content_clean)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None
    
    def analyze_code(
        self,
        code: str,
        prompt_template: str,
        **kwargs
    ) -> LLMResponse:
        """
        分析代码的便捷方法
        
        Args:
            code: 要分析的代码
            prompt_template: Prompt 模板，使用 {code} 占位符
            **kwargs: 其他模板变量
            
        Returns:
            LLMResponse 对象
        """
        prompt = prompt_template.format(code=code, **kwargs)
        return self.chat(prompt)
    
    def convert_to_cuda(
        self,
        code: str,
        function_name: str,
        context: Optional[str] = None
    ) -> LLMResponse:
        """
        将 CPU 代码转换为 CUDA
        
        Args:
            code: CPU 代码
            function_name: 要转换的函数名
            context: 额外上下文信息
            
        Returns:
            LLMResponse 对象
        """
        prompt = f"""Convert the following C++ function to CUDA.

Function to convert: {function_name}

Requirements:
1. Use one thread per primary loop iteration where possible
2. Ensure coalesced memory access
3. Use shared memory for frequently accessed data
4. Handle boundary conditions correctly
5. Provide both the kernel and the host launch code

{f"Additional context: {context}" if context else ""}

Code:
```cpp
{code}
```

Output the CUDA code with comments explaining key optimizations.
"""
        return self.chat(prompt, parse_json=False)


# 便捷函数
def create_client(**kwargs) -> LLMClient:
    """创建 LLM 客户端的便捷函数"""
    return LLMClient(**kwargs)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    client = create_client()
    
    response = client.chat("Say 'Hello, HPC!' in one line.")
    print(f"Response: {response.content}")
    print(f"Tokens: {response.total_tokens}")
    print(f"Cost: ${response.cost:.4f}")
