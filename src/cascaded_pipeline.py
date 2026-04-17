"""
Cascaded Pipeline (Original configuration: S1 = GPT-4o, S2 = GPT-5.2).

2026-04-17 refactor:
    This module used to duplicate ConfigurableCascadedPipeline's implementation.
    It now subclasses ConfigurableCascadedPipeline with fixed model pairing,
    to keep a single source of truth. All behaviour (including the
    bottleneck_taxonomy-based classification) is inherited automatically.
"""

import logging
from typing import Dict, Any, List
from configurable_pipeline import ConfigurableCascadedPipeline

logger = logging.getLogger(__name__)


class CascadedPipeline(ConfigurableCascadedPipeline):
    """Fixed S1=GPT-4o, S2=GPT-5.2 cascade. Preserved for backward compatibility."""
    
    def __init__(self):
        super().__init__(s1_model="gpt-4o", s2_model="gpt-5.2")


def run_cascaded_analysis(
    code_path: str,
    code_name: str,
    prompt_types: List[str] = None,
    program_key: str = None,
) -> Dict[str, Any]:
    """Run cascaded analysis under one or more prompt types (original entry point)."""
    if prompt_types is None:
        prompt_types = ["zero_shot", "few_shot", "contextual"]
    
    pipeline = CascadedPipeline()
    results = {}
    
    for pt in prompt_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running cascaded analysis: {pt}")
        logger.info('='*50)
        result = pipeline.analyze(
            code_path=code_path,
            code_name=code_name,
            prompt_type=pt,
            program_key=program_key,
        )
        results[pt] = result
    
    return results


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    results = run_cascaded_analysis(
        code_path="benchmarks/minimd/force_lj.cpp",
        code_name="minimd",
        prompt_types=["zero_shot"],
        program_key="minimd",
    )
    print(json.dumps(results, indent=2, default=str))
