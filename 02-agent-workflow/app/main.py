# ============================================
# FastAPI 主应用入口
# 提供 REST API 接口供外部调用 Agent 工作流
# ============================================

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import get_settings
from workflows.research_workflow import ResearchWorkflow

# ==========================================
# 应用生命周期管理
# ==========================================

# 全局工作流实例
_workflow: Optional[ResearchWorkflow] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化，关闭时清理"""
    global _workflow

    # 配置日志
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
               "<level>{message}</level>",
    )
    logger.add(
        os.path.join(settings.log_dir, "app_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="30 days",
        level=settings.log_level,
        encoding="utf-8",
    )

    # 确保日志目录存在
    os.makedirs(settings.log_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)

    # 初始化工作流
    logger.info("正在初始化 ResearchWorkflow...")
    _workflow = ResearchWorkflow(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        memory_max_turns=settings.memory_max_turns,
        max_retries=settings.max_retries,
    )
    logger.info("ResearchWorkflow 初始化完成")

    yield

    # 清理资源
    logger.info("应用关闭，清理资源...")
    _workflow = None


# ==========================================
# FastAPI 应用实例
# ==========================================

app = FastAPI(
    title="AI Agent 自动化工作流平台",
    description="基于 LangGraph + FastAPI 的多 Agent 协作自动化工作流平台",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 导入路由模块
# ==========================================
from app.routes import workflow_routes, tool_routes, memory_routes, health_routes

# 注册路由
app.include_router(health_routes.router, tags=["健康检查"])
app.include_router(workflow_routes.router, prefix="/api/v1", tags=["工作流"])
app.include_router(tool_routes.router, prefix="/api/v1", tags=["工具"])
app.include_router(memory_routes.router, prefix="/api/v1", tags=["记忆"])


# ==========================================
# 启动命令
# ==========================================
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )
