"""
output 模块 - 结构化输出

包含：
- StructuredOutputFormatter: 结构化输出格式化器（JSON Schema约束）
- HallucinationDetector: 幻觉检测器
"""

from .structured_output import StructuredOutputFormatter
from .hallucination_detector import HallucinationDetector

__all__ = ["StructuredOutputFormatter", "HallucinationDetector"]
