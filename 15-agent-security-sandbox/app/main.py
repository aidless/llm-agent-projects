"""AI Agent Security Sandbox - FastAPI 入口"""

import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.execute import router as execute_router
from app.api.policy import router as policy_router
from app.api.audit import router as audit_router
from app.api.execute import audit_logger, event_tracker, policy_engine
from app.models import HealthResponse

app = FastAPI(
    title="AI Agent Security Sandbox",
    description="为 Agent 代码执行提供安全隔离环境，支持系统调用拦截、资源限制和行为审计",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(execute_router)
app.include_router(policy_router)
app.include_router(audit_router)


@app.get("/", tags=["root"])
async def root():
    return {
        "name": "AI Agent Security Sandbox",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["monitor"])
async def health_check():
    """健康检查"""
    log_stats = audit_logger.get_stats()
    event_stats = event_tracker.get_stats()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        total_executions=log_stats.get("total_executions", 0),
        active_policies=len(policy_engine.list_policies()),
        log_entries=log_stats.get("total_entries", 0),
        alerts=event_stats.get("total_alerts", 0),
    )


@app.get("/ready", tags=["monitor"])
async def readiness_check():
    """就绪检查"""
    return {"status": "ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)