"""量化格式定义 - GPTQ/AWQ/GGUF 格式配置"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any


class QuantFormat(Enum):
    """量化格式"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ_INT4 = "gptq_int4"
    GPTQ_INT8 = "gptq_int8"
    AWQ_INT4 = "awq_int4"
    GGUF_Q4_0 = "gguf_q4_0"
    GGUF_Q4_K_M = "gguf_q4_k_m"
    GGUF_Q5_K_M = "gguf_q5_k_m"
    GGUF_Q8_0 = "gguf_q8_0"


@dataclass
class FormatSpec:
    """格式规格"""
    name: str
    bits_per_weight: float
    group_size: int
    has_zero_point: bool
    description: str


# 各格式规格
FORMAT_SPECS: Dict[QuantFormat, FormatSpec] = {
    QuantFormat.FP32: FormatSpec("FP32", 32, 0, False, "32-bit floating point, full precision"),
    QuantFormat.FP16: FormatSpec("FP16", 16, 0, False, "16-bit floating point, half precision"),
    QuantFormat.BF16: FormatSpec("BF16", 16, 0, False, "Brain floating point 16-bit"),
    QuantFormat.INT8: FormatSpec("INT8", 8, 32, True, "8-bit integer quantization"),
    QuantFormat.INT4: FormatSpec("INT4", 4, 32, True, "4-bit integer quantization"),
    QuantFormat.GPTQ_INT4: FormatSpec("GPTQ-INT4", 4, 128, True, "GPTQ 4-bit, group size 128"),
    QuantFormat.GPTQ_INT8: FormatSpec("GPTQ-INT8", 8, 128, True, "GPTQ 8-bit, group size 128"),
    QuantFormat.AWQ_INT4: FormatSpec("AWQ-INT4", 4, 128, True, "AWQ 4-bit, activation-aware"),
    QuantFormat.GGUF_Q4_0: FormatSpec("GGUF-Q4_0", 4, 32, True, "GGUF 4-bit quantization, block size 32"),
    QuantFormat.GGUF_Q4_K_M: FormatSpec("GGUF-Q4_K_M", 4.5, 256, True, "GGUF 4-bit K-quant medium"),
    QuantFormat.GGUF_Q5_K_M: FormatSpec("GGUF-Q5_K_M", 5.5, 256, True, "GGUF 5-bit K-quant medium"),
    QuantFormat.GGUF_Q8_0: FormatSpec("GGUF-Q8_0", 8, 32, True, "GGUF 8-bit quantization, block size 32"),
}


@dataclass
class GPTQConfig:
    """GPTQ 量化配置"""
    bits: int = 4
    group_size: int = 128
    desc_act: bool = True
    damp_percent: float = 0.01
    use_cuda_fp16: bool = True

    @property
    def format_enum(self) -> QuantFormat:
        return QuantFormat.GPTQ_INT4 if self.bits == 4 else QuantFormat.GPTQ_INT8


@dataclass
class AWQConfig:
    """AWQ 量化配置"""
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    version: str = "GEMM"

    @property
    def format_enum(self) -> QuantFormat:
        return QuantFormat.AWQ_INT4


@dataclass
class GGUFConfig:
    """GGUF 量化配置"""
    quant_type: str = "q4_k_m"
    thread_count: int = 8
    ftype: int = 1  # 0=fp32, 1=fp16

    # 类型映射
    TYPE_MAP: Dict = field(default_factory=lambda: {
        "q4_0": QuantFormat.GGUF_Q4_0,
        "q4_k_m": QuantFormat.GGUF_Q4_K_M,
        "q5_k_m": QuantFormat.GGUF_Q5_K_M,
        "q8_0": QuantFormat.GGUF_Q8_0,
    })

    @property
    def format_enum(self) -> QuantFormat:
        return self.TYPE_MAP.get(self.quant_type, QuantFormat.GGUF_Q4_K_M)


def get_format_spec(fmt: QuantFormat) -> FormatSpec:
    """获取格式规格"""
    return FORMAT_SPECS[fmt]


def list_formats() -> Dict[str, Dict[str, Any]]:
    """列出所有支持的格式"""
    result = {}
    for fmt, spec in FORMAT_SPECS.items():
        result[fmt.value] = {
            "bits_per_weight": spec.bits_per_weight,
            "group_size": spec.group_size,
            "has_zero_point": spec.has_zero_point,
            "description": spec.description,
        }
    return result
