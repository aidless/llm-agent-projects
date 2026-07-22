"""量化模拟器 - 通过数学模型计算量化对延迟/内存的影响"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .formats import QuantFormat, FORMAT_SPECS, FormatSpec


@dataclass
class QuantizationConfig:
    """量化配置"""
    format: QuantFormat = QuantFormat.FP16
    num_parameters: int = 7_000_000_000
    num_layers: int = 32
    hidden_size: int = 4096

    # 模拟参数
    memory_bandwidth_gb_s: float = 2039.0  # A100 HBM bandwidth
    compute_tflops: float = 312.0  # A100 FP16 TFLOPS
    # 量化精度损失因子 (模拟值，非真实值)
    quality_factor: Dict[str, float] = field(default_factory=lambda: {
        "fp32": 1.0,
        "fp16": 0.999,
        "bf16": 0.998,
        "int8": 0.985,
        "int4": 0.950,
        "gptq_int4": 0.960,
        "gptq_int8": 0.988,
        "awq_int4": 0.965,
        "gguf_q4_0": 0.940,
        "gguf_q4_k_m": 0.955,
        "gguf_q5_k_m": 0.970,
        "gguf_q8_0": 0.986,
    })


class Quantizer:
    """量化模拟器"""

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

    def calculate_memory(self, fmt: Optional[QuantFormat] = None) -> Dict[str, float]:
        """计算指定量化格式下的内存占用 (GB)"""
        fmt = fmt or self.config.format
        spec = FORMAT_SPECS[fmt]
        num_params = self.config.num_parameters

        # 权重内存
        weight_bytes = num_params * spec.bits_per_weight / 8

        # KV Cache 和激活值 (不受量化影响，始终为 FP16)
        kv_cache_bytes = self._estimate_kv_cache_size()
        activation_bytes = num_params * 2  # FP16 activations (rough estimate)

        total_bytes = weight_bytes + kv_cache_bytes + activation_bytes

        return {
            "weight_memory_gb": weight_bytes / (1024 ** 3),
            "kv_cache_memory_gb": kv_cache_bytes / (1024 ** 3),
            "activation_memory_gb": activation_bytes / (1024 ** 3),
            "total_memory_gb": total_bytes / (1024 ** 3),
            "compression_ratio": (num_params * 4) / weight_bytes if weight_bytes > 0 else 1.0,
        }

    def _estimate_kv_cache_size(self) -> int:
        """估算 KV Cache 大小 (bytes)"""
        # 假设: seq_len=2048, batch=32, layers=32, hidden=4096, FP16
        seq_len = 2048
        batch_size = 32
        num_kv_heads = 32
        head_dim = self.config.hidden_size // num_kv_heads

        # KV Cache per token per layer: 2 (K+V) * num_kv_heads * head_dim * 2 bytes
        bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * 2
        total = seq_len * batch_size * self.config.num_layers * bytes_per_token_per_layer
        return total

    def estimate_latency(
        self,
        fmt: Optional[QuantFormat] = None,
        batch_size: int = 1,
        seq_len: int = 512,
        num_output_tokens: int = 128,
    ) -> Dict[str, float]:
        """估算推理延迟 (ms)"""
        fmt = fmt or self.config.format
        spec = FORMAT_SPECS[fmt]
        num_params = self.config.num_parameters

        # 权重读取时间 (memory-bound)
        weight_bytes = num_params * spec.bits_per_weight / 8
        weight_read_time = weight_bytes / (self.config.memory_bandwidth_gb_s * 1e9) * 1000  # ms

        # Prefill 阶段 (compute-bound for short sequences, memory-bound for long ones)
        prefill_flops = 2 * num_params * seq_len  # approximate FLOPs for prefill
        prefill_compute_time = prefill_flops / (self.config.compute_tflops * 1e12) * 1000  # ms
        prefill_time = max(weight_read_time * batch_size, prefill_compute_time)

        # Decode 阶段 (memory-bound)
        decode_weight_read = weight_bytes / (self.config.memory_bandwidth_gb_s * 1e9) * 1000
        decode_time_per_token = decode_weight_read
        total_decode_time = decode_time_per_token * num_output_tokens

        # 量化额外开销 (dequantize kernels)
        if spec.bits_per_weight < 16:
            dequant_overhead = (16 - spec.bits_per_weight) / 16 * 0.1  # ms per token
        else:
            dequant_overhead = 0.0
        total_dequant = dequant_overhead * num_output_tokens * batch_size

        total_latency = prefill_time + total_decode_time + total_dequant

        return {
            "prefill_latency_ms": prefill_time,
            "decode_latency_ms": total_decode_time,
            "dequantization_overhead_ms": total_dequant,
            "total_latency_ms": total_latency,
            "time_per_output_token_ms": total_latency / max(1, num_output_tokens),
        }

    def estimate_quality(self, fmt: Optional[QuantFormat] = None) -> Dict[str, float]:
        """估算量化精度损失"""
        fmt = fmt or self.config.format
        factor = self.config.quality_factor.get(fmt.value, 0.9)

        return {
            "format": fmt.value,
            "quality_factor": factor,
            "quality_percentage": factor * 100,
            "perplexity_increase_pct": (1.0 / factor - 1.0) * 100,
            "estimated_perplexity": 10.0 / factor,  # baseline PPL=10
        }

    def compare_formats(self) -> List[Dict[str, any]]:
        """对比所有量化格式"""
        results = []
        for fmt in QuantFormat:
            mem = self.calculate_memory(fmt)
            lat = self.estimate_latency(fmt)
            qual = self.estimate_quality(fmt)

            results.append({
                "format": fmt.value,
                "memory_gb": round(mem["total_memory_gb"], 2),
                "compression_ratio": round(mem["compression_ratio"], 2),
                "prefill_ms": round(lat["prefill_latency_ms"], 2),
                "decode_ms": round(lat["decode_latency_ms"], 2),
                "total_ms": round(lat["total_latency_ms"], 2),
                "quality_pct": round(qual["quality_percentage"], 1),
            })

        return sorted(results, key=lambda x: x["memory_gb"])
