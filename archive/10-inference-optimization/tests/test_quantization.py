"""Quantization 模块测试"""

import pytest

from quantization.quantizer import Quantizer, QuantizationConfig
from quantization.formats import (
    QuantFormat, FORMAT_SPECS, FormatSpec,
    GPTQConfig, AWQConfig, GGUFConfig,
    list_formats, get_format_spec,
)
from quantization.evaluator import QuantEvaluator, AccuracyReport


class TestQuantizationFormats:
    """测试量化格式定义"""

    def test_all_formats_have_specs(self):
        for fmt in QuantFormat:
            assert fmt in FORMAT_SPECS
            spec = FORMAT_SPECS[fmt]
            assert spec.bits_per_weight > 0
            assert spec.name

    def test_get_format_spec(self):
        spec = get_format_spec(QuantFormat.FP32)
        assert spec.bits_per_weight == 32

    def test_list_formats(self):
        formats = list_formats()
        assert len(formats) > 0
        assert "fp32" in formats
        assert "int4" in formats

    def test_gptq_config(self):
        config = GPTQConfig(bits=4, group_size=128)
        assert config.format_enum == QuantFormat.GPTQ_INT4

    def test_gptq_config_int8(self):
        config = GPTQConfig(bits=8)
        assert config.format_enum == QuantFormat.GPTQ_INT8

    def test_awq_config(self):
        config = AWQConfig(bits=4)
        assert config.format_enum == QuantFormat.AWQ_INT4

    def test_gguf_config(self):
        config = GGUFConfig(quant_type="q4_k_m")
        assert config.format_enum == QuantFormat.GGUF_Q4_K_M

    def test_gguf_config_q8(self):
        config = GGUFConfig(quant_type="q8_0")
        assert config.format_enum == QuantFormat.GGUF_Q8_0

    def test_format_ordering(self):
        """精度从高到低"""
        assert FORMAT_SPECS[QuantFormat.FP32].bits_per_weight > FORMAT_SPECS[QuantFormat.FP16].bits_per_weight
        assert FORMAT_SPECS[QuantFormat.FP16].bits_per_weight > FORMAT_SPECS[QuantFormat.INT8].bits_per_weight
        assert FORMAT_SPECS[QuantFormat.INT8].bits_per_weight > FORMAT_SPECS[QuantFormat.INT4].bits_per_weight


class TestQuantizer:
    """测试量化模拟器"""

    def test_memory_calculation_fp32(self):
        quantizer = Quantizer()
        mem = quantizer.calculate_memory(QuantFormat.FP32)
        assert mem["weight_memory_gb"] > 0
        assert mem["compression_ratio"] == 1.0

    def test_memory_calculation_int4(self):
        quantizer = Quantizer()
        mem_fp32 = quantizer.calculate_memory(QuantFormat.FP32)
        mem_int4 = quantizer.calculate_memory(QuantFormat.INT4)
        assert mem_int4["weight_memory_gb"] < mem_fp32["weight_memory_gb"]
        assert mem_int4["compression_ratio"] > mem_fp32["compression_ratio"]

    def test_latency_estimation(self):
        quantizer = Quantizer()
        lat = quantizer.estimate_latency(QuantFormat.FP16)
        assert "prefill_latency_ms" in lat
        assert "decode_latency_ms" in lat
        assert "total_latency_ms" in lat
        assert lat["total_latency_ms"] > 0

    def test_latency_lower_precision(self):
        quantizer = Quantizer()
        lat_fp16 = quantizer.estimate_latency(QuantFormat.FP16)
        lat_int4 = quantizer.estimate_latency(QuantFormat.INT4)
        # INT4 应该有 dequantization overhead
        assert lat_int4["dequantization_overhead_ms"] > lat_fp16["dequantization_overhead_ms"]

    def test_quality_estimation(self):
        quantizer = Quantizer()
        qual_fp32 = quantizer.estimate_quality(QuantFormat.FP32)
        qual_int4 = quantizer.estimate_quality(QuantFormat.INT4)
        assert qual_fp32["quality_factor"] > qual_int4["quality_factor"]
        assert qual_fp32["quality_percentage"] == 100.0

    def test_compare_formats(self):
        quantizer = Quantizer()
        results = quantizer.compare_formats()
        assert len(results) > 0
        # 应按内存排序
        for i in range(len(results) - 1):
            assert results[i]["memory_gb"] <= results[i + 1]["memory_gb"]

    def test_custom_config(self):
        config = QuantizationConfig(
            num_parameters=1_000_000_000,
            num_layers=16,
            hidden_size=2048,
        )
        quantizer = Quantizer(config)
        mem = quantizer.calculate_memory(QuantFormat.FP16)
        assert mem["weight_memory_gb"] > 0


class TestQuantEvaluator:
    """测试量化精度评估"""

    def test_evaluate_fp32(self):
        evaluator = QuantEvaluator()
        report = evaluator.evaluate(QuantFormat.FP32)
        assert isinstance(report, AccuracyReport)
        assert report.format == "fp32"
        assert report.accuracy_percentage == 100.0

    def test_evaluate_int4(self):
        evaluator = QuantEvaluator()
        report = evaluator.evaluate(QuantFormat.INT4)
        assert report.accuracy_percentage < 100.0
        assert report.accuracy_percentage > 80.0
        assert report.perplexity > 0

    def test_compare_formats(self):
        evaluator = QuantEvaluator()
        reports = evaluator.compare_formats()
        assert len(reports) > 0

    def test_recommendation(self):
        evaluator = QuantEvaluator()
        rec = evaluator.get_recommendation(max_memory_gb=24.0, min_accuracy_pct=90.0)
        assert "recommendation" in rec
        assert "constraints" in rec

    def test_recommendation_strict(self):
        evaluator = QuantEvaluator()
        rec = evaluator.get_recommendation(max_memory_gb=0.001, min_accuracy_pct=99.0)
        # 内存限制极低，可能没有候选
        assert "recommendation" in rec
