"""
FastAPI 主应用 - 多模型智能路由网关
提供统一的 OpenAI 兼容 API 接口，集成智能路由、并发控制和监控
"""
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth import init_auth, is_auth_enabled, verify_token
from app.config import get_config, AppConfig
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelErrorResponse,
)
from monitor.stats_collector import stats_collector, RequestRecord
from routers.manager import get_adapter_manager, reset_adapter_manager
from strategy.router import get_smart_router, reset_smart_router
from reqqueue.request_queue import get_request_queue, get_failover_manager, reset_queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 全局配置和组件引用（在 lifespan 中初始化）
_config: Optional[AppConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时初始化各组件，关闭时清理资源
    """
    global _config
    logger.info("=" * 60)
    logger.info("  多模型智能路由网关 - 启动中")
    logger.info("=" * 60)

    # 加载配置
    _config = get_config()
    logger.info(f"  最大并发数: {_config.max_concurrent_requests}")
    logger.info(f"  请求超时: {_config.request_timeout}s")

    # ⚠️ 2026-07-22: 初始化 Bearer Token 鉴权（fail-safe 启动检查）
    init_auth()

    # 初始化请求队列
    get_request_queue(
        max_concurrent=_config.max_concurrent_requests,
        timeout=float(_config.request_timeout),
    )

    # 初始化适配器管理器（触发适配器注册）
    manager = get_adapter_manager()
    available = manager.get_available_models()
    logger.info(f"  可用模型: {available}")

    # 初始化智能路由器
    router = get_smart_router()
    logger.info(f"  可用策略: {router.get_available_strategies()}")

    logger.info("  网关启动完成!")
    logger.info("=" * 60)

    yield

    # 关闭清理
    logger.info("网关关闭中...")
    reset_adapter_manager()
    reset_smart_router()
    reset_queue()
    logger.info("网关已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="多模型智能路由网关",
    description=(
        "统一的 LLM API 网关，支持 OpenAI GPT-4o / DeepSeek V4 / Qwen / Claude "
        "多模型智能路由、成本优化、失败降级和监控统计"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ⚠️ 2026-07-22: CORS 收紧。生产环境必须显式配置允许来源；之前默认 allow_origins=* 是 P0 风险。
import os as _os
_allowed_origins = _os.getenv("GATEWAY_CORS_ORIGINS", "").split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()]
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    logger.info(f"  CORS allowed origins: {_allowed_origins}")
else:
    logger.warning("  GATEWAY_CORS_ORIGINS 未配置，CORS 中间件未启用（仅服务端调用）")


async def _handle_chat_request(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    处理聊天补全请求的核心逻辑
    包含: 路由决策 -> 队列控制 -> 模型调用 -> 失败降级 -> 统计记录
    """
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    # 1. 路由决策
    smart_router = get_smart_router()
    route_result = smart_router.route(request)
    logger.info(f"[{request_id}] 路由结果: {route_result}")

    # 2. 获取并发控制和降级管理器
    req_queue = get_request_queue()
    failover = get_failover_manager()
    adapter_manager = get_adapter_manager()

    # 3. 定义模型调用函数（用于降级）
    async def call_model(model_key: str) -> ChatCompletionResponse:
        """调用指定模型"""
        return await adapter_manager.call_model(
            model_key=model_key,
            request=request,
            timeout=float(_config.request_timeout) if _config else 60.0,
        )

    # 4. 在并发控制下执行，带失败降级
    try:
        result, actual_model = await req_queue.execute(
            failover.execute_with_fallback,
            call_func=call_model,
            primary_model=route_result.model_key,
            fallback_models=route_result.fallback_models,
        )
    except asyncio.TimeoutError:
        latency = (time.time() - start_time) * 1000
        _record_stats(
            request_id=request_id,
            model=route_result.model_key,
            strategy=route_result.strategy,
            task_type=route_result.task_type,
            latency_ms=latency,
            success=False,
            error_msg="请求超时",
        )
        raise HTTPException(status_code=504, detail="请求排队或执行超时")
    except RuntimeError as e:
        latency = (time.time() - start_time) * 1000
        _record_stats(
            request_id=request_id,
            model=route_result.model_key,
            strategy=route_result.strategy,
            task_type=route_result.task_type,
            latency_ms=latency,
            success=False,
            error_msg=str(e),
        )
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        _record_stats(
            request_id=request_id,
            model=route_result.model_key,
            strategy=route_result.strategy,
            task_type=route_result.task_type,
            latency_ms=latency,
            success=False,
            error_msg=str(e),
        )
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

    # 5. 计算延迟和附加路由信息
    latency = (time.time() - start_time) * 1000
    result.latency_ms = latency
    result.routed_model = actual_model
    result.strategy_used = route_result.strategy

    # 6. 记录统计数据
    _record_stats(
        request_id=request_id,
        model=actual_model,
        strategy=route_result.strategy,
        task_type=route_result.task_type,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        total_tokens=result.usage.total_tokens,
        latency_ms=latency,
        cost_usd=result.cost_usd,
        success=True,
    )

    logger.info(
        f"[{request_id}] 完成: 模型={actual_model}, "
        f"tokens={result.usage.total_tokens}, "
        f"延迟={latency:.0f}ms, "
        f"成本=${result.cost_usd:.6f}"
    )

    return result


def _record_stats(
    request_id: str = "",
    model: str = "",
    strategy: str = "",
    task_type: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    success: bool = False,
    error_msg: str = "",
):
    """记录请求统计数据"""
    record = RequestRecord(
        request_id=request_id,
        model=model,
        strategy=strategy,
        task_type=task_type,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens or (prompt_tokens + completion_tokens),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        success=success,
        error_msg=error_msg,
    )
    stats_collector.record_request(record)


# ==================== API 路由 ====================


@app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
async def chat_completions(request: Request):
    """
    聊天补全 API - 兼容 OpenAI API 格式

    请求体（JSON）:
        model: 可选，指定模型名称（不指定则智能路由）
        messages: 消息列表 [{role, content}]
        temperature: 温度参数
        max_tokens: 最大生成 token 数
        stream: 是否流式（当前版本不支持流式，会返回非流式结果）
        strategy: 路由策略 (task_type / cost / latency / load_balance)
        user_hint: 任务类型提示 (code / creative / analysis / translation / summary)

    额外响应字段:
        routed_model: 实际路由到的模型名称
        strategy_used: 使用的路由策略
        latency_ms: 请求延迟（毫秒）
        cost_usd: 本次请求成本（美元）
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体 JSON 解析失败")

    # 解析请求
    chat_request = ChatCompletionRequest.from_openai_dict(body)

    if not chat_request.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    # 调用核心处理逻辑
    result = await _handle_chat_request(chat_request)

    return JSONResponse(content=result.to_openai_dict())


@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    """
    列出所有可用的模型
    返回 OpenAI 兼容格式的模型列表
    """
    manager = get_adapter_manager()
    config = get_config()

    models = []
    for key in manager.get_available_models():
        model_config = config.models.get(key)
        if model_config:
            models.append({
                "id": key,
                "object": "model",
                "created": 1700000000,
                "owned_by": model_config.provider,
                "name": model_config.name,
                "model_id": model_config.model_id,
                "input_price_per_1k": model_config.input_price_per_1k,
                "output_price_per_1k": model_config.output_price_per_1k,
            })

    return {"object": "list", "data": models}


# ==================== 监控 API ====================


@app.get("/health")
async def health_check():
    """
    健康检查接口（公开，无需认证 — LB / K8s 探针必须可访问）
    返回网关运行状态和各组件健康情况
    """
    manager = get_adapter_manager()
    req_queue = get_request_queue()

    return {
        "status": "healthy",
        "available_models": manager.get_available_models(),
        "queue": req_queue.get_stats(),
        "auth_enabled": is_auth_enabled(),  # 2026-07-22: 让运维一眼看出认证是否启用
    }


@app.get("/stats")
async def get_stats():
    """
    获取详细的统计数据
    包括: 总请求量、各模型指标、Token 用量、成本、延迟分布
    """
    all_stats = stats_collector.get_all_stats()
    req_queue = get_request_queue()

    return {
        "stats": all_stats,
        "queue": req_queue.get_stats(),
    }


@app.get("/stats/recent")
async def get_recent_requests(limit: int = 20):
    """
    获取最近的请求记录
    参数:
        limit: 返回记录数量，默认 20，最大 100
    """
    limit = min(max(1, limit), 100)
    records = stats_collector.get_recent_records(limit)
    return {"recent_requests": records, "count": len(records)}


@app.get("/stats/{model_name}")
async def get_model_stats(model_name: str):
    """
    获取指定模型的统计数据
    """
    stats = stats_collector.get_model_stats(model_name)
    return {"model": model_name, "stats": stats.to_dict()}


@app.delete("/stats")
async def reset_stats():
    """重置所有统计数据"""
    stats_collector.reset()
    return {"message": "统计数据已重置"}


@app.get("/strategies")
async def list_strategies():
    """
    列出所有可用的路由策略
    """
    router = get_smart_router()
    strategies = router.get_available_strategies()
    strategy_descriptions = {
        "task_type": "按任务类型路由（代码生成→DeepSeek，创意写作→Claude，通用→GPT-4o）",
        "cost": "按成本优化路由（简单任务用便宜模型，复杂任务用强模型）",
        "latency": "按延迟优化路由（选择历史延迟最低的模型）",
        "load_balance": "负载均衡路由（多模型轮询分配请求）",
    }
    return {
        "strategies": [
            {
                "name": s,
                "description": strategy_descriptions.get(s, ""),
            }
            for s in strategies
        ]
    }


