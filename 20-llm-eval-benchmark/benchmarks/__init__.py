"""
benchmarks - 评测基准模块

提供 MMLU、GSM8K、HumanEval、MT-Bench 以及自定义基准的实现。
"""

from .base import BaseBenchmark, BenchmarkQuestion, BenchmarkResult
from .mmlu import MMLUBenchmark
from .gsm8k import GSM8KBenchmark
from .humaneval import HumanEvalBenchmark
from .mt_bench import MTBenchBenchmark
from .custom import CustomBenchmark

__all__ = [
    "BaseBenchmark",
    "BenchmarkQuestion",
    "BenchmarkResult",
    "MMLUBenchmark",
    "GSM8KBenchmark",
    "HumanEvalBenchmark",
    "MTBenchBenchmark",
    "CustomBenchmark",
]

# 基准注册表: name -> class
BENCHMARK_REGISTRY: dict = {
    "mmlu": MMLUBenchmark,
    "gsm8k": GSM8KBenchmark,
    "humaneval": HumanEvalBenchmark,
    "mt_bench": MTBenchBenchmark,
    "custom": CustomBenchmark,
}


def get_benchmark(name: str, data_path: str | None = None, **kwargs) -> BaseBenchmark:
    """根据名称获取基准实例."""
    if name not in BENCHMARK_REGISTRY:
        available = ", ".join(BENCHMARK_REGISTRY.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    cls = BENCHMARK_REGISTRY[name]
    if data_path:
        return cls(data_path=data_path, **kwargs)
    return cls(**kwargs)
