"""配置管理 API - 量化、缓存、路由配置"""

from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, HTTPException

from ...quantization.quantizer import Quantizer, QuantizationConfig
from ...quantization.formats import QuantFormat, list_formats, get_format_spec
from ...quantization.evaluator import QuantEvaluator
from ...cache.kv_cache import KVCacheManager
from ...cache.paged_cache import PagedKVCache
from ...cache.memory_manager import MemoryManager, MemoryPool
from ...proxy.router import RequestRouter, Backend, RoutingStrategy, RouterConfig
from ...proxy.rate_limiter import (
    RateLimiter, TokenBucketLimiter, SlidingWindowLimiter,
    LimiterConfig, create_limiter, LimiterType,
)
from ...proxy.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

router = APIRouter(prefix="/v1", tags=["config"])

# 全局实例
quantizer: Quantizer = None  # type: ignore
evaluator: QuantEvaluator = None  # type: ignore
kv_cache_manager: KVCacheManager = None  # type: ignore
paged_cache: PagedKVCache = None  # type: ignore
memory_manager: MemoryManager = None  # type: ignore
request_router: RequestRouter = None  # type: ignore
rate_limiter: RateLimiter = None  # type: ignore
circuit_breaker: CircuitBreaker = None  # type: ignore


def init_config(
    q: Quantizer, e: QuantEvaluator,
    kv: KVCacheManager, pc: PagedKVCache, mm: MemoryManager,
    rr: RequestRouter, rl: RateLimiter, cb: CircuitBreaker,
) -> None:
    global quantizer, evaluator, kv_cache_manager, paged_cache, memory_manager
    global request_router, rate_limiter, circuit_breaker
    quantizer = q
    evaluator = e
    kv_cache_manager = kv
    paged_cache = pc
    memory_manager = mm
    request_router = rr
    rate_limiter = rl
    circuit_breaker = cb


# ========== 量化 ==========

@router.get("/quantization/formats")
async def list_quantization_formats():
    """列出所有量化格式"""
    return list_formats()


@router.get("/quantization/compare")
async def compare_quantization():
    """对比所有量化格式"""
    if quantizer is None:
        raise HTTPException(status_code=503, detail="Quantizer not initialized")
    return quantizer.compare_formats()


@router.get("/quantization/memory/{fmt}")
async def get_quantization_memory(fmt: str):
    """获取指定格式的内存占用"""
    if quantizer is None:
        raise HTTPException(status_code=503, detail="Quantizer not initialized")
    try:
        format_enum = QuantFormat(fmt)
        return quantizer.calculate_memory(format_enum)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown format: {fmt}")


@router.get("/quantization/latency/{fmt}")
async def get_quantization_latency(fmt: str):
    """获取指定格式的延迟估算"""
    if quantizer is None:
        raise HTTPException(status_code=503, detail="Quantizer not initialized")
    try:
        format_enum = QuantFormat(fmt)
        return quantizer.estimate_latency(format_enum)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown format: {fmt}")


@router.get("/quantization/evaluate/{fmt}")
async def evaluate_quantization(fmt: str):
    """评估量化精度"""
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    try:
        format_enum = QuantFormat(fmt)
        report = evaluator.evaluate(format_enum)
        return {
            "format": report.format,
            "perplexity": round(report.perplexity, 4),
            "accuracy_percentage": round(report.accuracy_percentage, 2),
            "token_match_rate": round(report.token_match_rate, 4),
            "semantic_similarity": round(report.semantic_similarity, 4),
            "details": report.details,
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown format: {fmt}")


@router.get("/quantization/recommend")
async def recommend_quantization(
    max_memory_gb: float = 24.0,
    min_accuracy_pct: float = 95.0,
):
    """推荐量化格式"""
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    return evaluator.get_recommendation(max_memory_gb, min_accuracy_pct)


# ========== 缓存 ==========

@router.get("/cache/kv/stats")
async def get_kv_cache_stats():
    """获取 KV Cache 统计"""
    if kv_cache_manager is None:
        raise HTTPException(status_code=503, detail="KV Cache not initialized")
    stats = kv_cache_manager.get_stats()
    return {
        "total_allocations": stats.total_allocations,
        "total_evictions": stats.total_evictions,
        "total_hits": stats.total_hits,
        "total_misses": stats.total_misses,
        "hit_rate": round(stats.hit_rate, 4),
        "current_usage_bytes": stats.current_usage_bytes,
        "peak_usage_bytes": stats.peak_usage_bytes,
    }


@router.get("/cache/kv/layers")
async def get_kv_cache_layers():
    """获取每层 KV Cache 使用情况"""
    if kv_cache_manager is None:
        raise HTTPException(status_code=503, detail="KV Cache not initialized")
    return kv_cache_manager.get_per_layer_usage()


@router.get("/cache/paged/usage")
async def get_paged_cache_usage():
    """获取分页缓存使用情况"""
    if paged_cache is None:
        raise HTTPException(status_code=503, detail="Paged Cache not initialized")
    return paged_cache.get_usage()


@router.get("/cache/memory/overview")
async def get_memory_overview():
    """获取内存概览"""
    if memory_manager is None:
        raise HTTPException(status_code=503, detail="Memory Manager not initialized")
    return memory_manager.get_overview()


@router.get("/cache/memory/pools")
async def get_memory_pools():
    """获取各内存池统计"""
    if memory_manager is None:
        raise HTTPException(status_code=503, detail="Memory Manager not initialized")
    return memory_manager.get_pool_stats()


# ========== 路由 ==========

@router.get("/proxy/backends")
async def get_backends():
    """获取后端列表"""
    if request_router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    return request_router.get_backends_status()


@router.get("/proxy/stats")
async def get_router_stats():
    """获取路由统计"""
    if request_router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    return request_router.get_stats()


@router.post("/proxy/backends")
async def add_backend(name: str, url: str, weight: int = 1):
    """添加后端"""
    if request_router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    backend = Backend(name=name, url=url, weight=weight)
    request_router.add_backend(backend)
    return {"status": "ok", "backend": name}


@router.delete("/proxy/backends/{name}")
async def remove_backend(name: str):
    """移除后端"""
    if request_router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    removed = request_router.remove_backend(name)
    return {"status": "ok" if removed else "not_found", "backend": name}


@router.get("/proxy/health")
async def health_check():
    """健康检查"""
    if request_router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    return request_router.health_check()


# ========== 限流 ==========

@router.get("/proxy/rate_limiter/stats")
async def get_rate_limiter_stats():
    """获取限流器统计"""
    if rate_limiter is None:
        raise HTTPException(status_code=503, detail="Rate limiter not initialized")
    return rate_limiter.get_stats()


@router.post("/proxy/rate_limiter/reset")
async def reset_rate_limiter():
    """重置限流器"""
    if rate_limiter is None:
        raise HTTPException(status_code=503, detail="Rate limiter not initialized")
    rate_limiter.reset()
    return {"status": "ok"}


# ========== 熔断 ==========

@router.get("/proxy/circuit_breaker/status")
async def get_circuit_breaker_status():
    """获取熔断器状态"""
    if circuit_breaker is None:
        raise HTTPException(status_code=503, detail="Circuit breaker not initialized")
    return circuit_breaker.get_stats()


@router.post("/proxy/circuit_breaker/reset")
async def reset_circuit_breaker():
    """重置熔断器"""
    if circuit_breaker is None:
        raise HTTPException(status_code=503, detail="Circuit breaker not initialized")
    circuit_breaker.reset()
    return {"status": "ok"}
