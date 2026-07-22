"""FastAPI 入口"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from storage.workflow_store import WorkflowStore
from storage.execution_log import ExecutionLog
from engine.executor import WorkflowExecutor
from nodes import register_all_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# ── 全局单例 ──────────────────────────────────────────────

workflow_store = WorkflowStore()
execution_log = ExecutionLog()
executor = WorkflowExecutor(max_workers=4, default_retry=0)

# 注册所有内置节点处理器
register_all_handlers(executor)


# ── 应用生命周期 ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI 工作流自动化平台启动")
    yield
    executor.shutdown()
    logger.info("AI 工作流自动化平台关闭")


# ── FastAPI 应用 ─────────────────────────────────────────

app = FastAPI(
    title="AI Workflow Automation Platform",
    description="可视化工作流编排引擎 - 支持 DAG 工作流定义、节点拖拽、条件分支、循环等",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 注册路由 ──────────────────────────────────────────────

from app.api.workflow import router as workflow_router
from app.api.execution import router as execution_router
from app.api.template import router as template_router

app.include_router(workflow_router)
app.include_router(execution_router)
app.include_router(template_router)


# ── 健康检查 ──────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ai-workflow-platform"}


@app.get("/")
def root():
    return {
        "name": "AI Workflow Automation Platform",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
