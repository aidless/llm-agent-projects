"""FastAPI 主入口"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from .api.inference import router as inference_router, init_inference
from .api.monitor import router as monitor_router, init_monitoring
from .api.config import router as config_router, init_config
from .models import HealthResponse, ErrorResponse

from ..engine.simulator import InferenceSimulator, SimulatorConfig
from ..engine.tokenizer_mock import MockTokenizer, TokenizerConfig
from ..engine.batcher import DynamicBatcher
from ..engine.scheduler import RequestScheduler

from ..quantization.quantizer import Quantizer, QuantizationConfig
from ..quantization.evaluator import QuantEvaluator

from ..cache.kv_cache import KVCacheManager
from ..cache.paged_cache import PagedKVCache
from ..cache.memory_manager import MemoryManager

from ..monitoring.metrics import MetricsCollector
from ..monitoring.prometheus_exporter import PrometheusExporter
from ..monitoring.dashboard import DashboardAPI

from ..proxy.router import RequestRouter, RouterConfig, Backend
from ..proxy.rate_limiter import TokenBucketLimiter, LimiterConfig
from ..proxy.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# 全局组件
_start_time = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _start_time
    _start_time = time.time()

    # 初始化引擎
    sim_config = SimulatorConfig(model_name="mock-llm-7b")
    simulator = InferenceSimulator(sim_config)

    # 初始化量化
    quant_config = QuantizationConfig()
    quantizer = Quantizer(quant_config)
    evaluator = QuantEvaluator(quant_config)

    # 初始化缓存
    kv_manager = KVCacheManager(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        total_memory_bytes=4 * 1024 * 1024 * 1024,
    )
    paged_kv = PagedKVCache(
        block_size=16,
        num_blocks=1024,
        num_layers=32,
    )
    mem_manager = MemoryManager(
        total_memory_bytes=24 * 1024 * 1024 * 1024,
    )

    # 初始化监控
    metrics_collector = MetricsCollector()
    prom_exporter = PrometheusExporter(metrics_collector)
    dash_api = DashboardAPI(metrics_collector)

    # 初始化代理
    router = RequestRouter(RouterConfig())
    # 添加默认后端
    router.add_backend(Backend(name="engine-0", url="http://localhost:8000", weight=1))
    router.add_backend(Backend(name="engine-1", url="http://localhost:8001", weight=1))

    rate_limiter = TokenBucketLimiter(LimiterConfig(
        max_requests=100,
        window_seconds=1.0,
        burst_size=10,
        refill_rate=50.0,
    ))

    circuit_breaker = CircuitBreaker("default", CircuitBreakerConfig())

    # 注入到 API 模块
    init_inference(simulator, metrics_collector)
    init_monitoring(metrics_collector, prom_exporter, dash_api)
    init_config(
        quantizer, evaluator, kv_manager, paged_kv, mem_manager,
        router, rate_limiter, circuit_breaker,
    )

    # 初始化 GPU 统计
    metrics_collector.set_gpu_stats(
        utilization=0.85,
        memory_bytes=16 * 1024 * 1024 * 1024,
    )
    metrics_collector.set_cache_utilization(0.6)

    yield

    # 清理 (如有需要)


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="LLM Inference Optimization Service",
        description="模拟 vLLM/SGLang 核心优化策略的推理服务",
        version="1.0.0",
        lifespan=lifespan,
    )

    # 注册路由
    app.include_router(inference_router)
    app.include_router(monitor_router)
    app.include_router(config_router)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health():
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=time.time() - _start_time,
        )

    @app.get("/", tags=["system"])
    async def root():
        return {
            "service": "LLM Inference Optimization",
            "version": "1.0.0",
            "docs": "/docs",
            "metrics": "/v1/metrics/prometheus",
        }

    return app


app = create_app()
