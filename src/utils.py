"""
工具函数模块
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import yaml

logger = logging.getLogger(__name__)


def load_yaml(filepath: str) -> Dict[str, Any]:
    """加载 YAML 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], filepath: str):
    """保存 YAML 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_json(filepath: str) -> Dict[str, Any]:
    """加载 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """保存 JSON 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_code(filepath: str) -> str:
    """读取代码文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def ensure_dir(path: str) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """获取时间戳字符串"""
    return datetime.now().strftime(format)


def count_tokens(text: str) -> int:
    """估算 token 数量（简单实现）"""
    # 简单估算：每 4 个字符约 1 个 token
    return len(text) // 4


def format_cost(cost: float) -> str:
    """格式化成本显示"""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
):
    """配置日志"""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level_map.get(level.upper(), logging.INFO),
        format=format_str,
        handlers=handlers
    )


class ExperimentTracker:
    """实验追踪器"""
    
    def __init__(self, experiment_name: str, output_dir: str = "results"):
        self.experiment_name = experiment_name
        self.timestamp = get_timestamp()
        self.output_dir = ensure_dir(f"{output_dir}/{experiment_name}_{self.timestamp}")
        
        self.results = []
        self.metadata = {
            "name": experiment_name,
            "timestamp": self.timestamp,
            "start_time": datetime.now().isoformat()
        }
        
        logger.info(f"Experiment tracker initialized: {self.output_dir}")
    
    def log_result(self, name: str, data: Dict[str, Any]):
        """记录结果"""
        result = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.results.append(result)
        
        # 保存单个结果
        save_json(result, self.output_dir / f"{name}.json")
        
        logger.info(f"Result logged: {name}")
    
    def finish(self):
        """完成实验，保存汇总"""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["results_count"] = len(self.results)
        
        summary = {
            "metadata": self.metadata,
            "results": self.results
        }
        
        save_json(summary, self.output_dir / "experiment_summary.json")
        
        logger.info(f"Experiment finished: {len(self.results)} results saved")
        
        return self.output_dir


if __name__ == "__main__":
    # 测试
    setup_logging("DEBUG")
    
    tracker = ExperimentTracker("test_experiment")
    tracker.log_result("test1", {"score": 95})
    tracker.log_result("test2", {"score": 100})
    output = tracker.finish()
    
    print(f"Output saved to: {output}")
