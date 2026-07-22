"""
FastAPI 入口 - Agent 长期记忆系统。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from manager.memory_manager import MemoryManager


# 全局记忆管理器
manager = MemoryManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    # 启动时初始化
    from app.api.memory import set_manager as set_memory_manager
    from app.api.search import set_manager as set_search_manager
    from app.api.agent import set_manager as set_agent_manager

    set_memory_manager(manager)
    set_search_manager(manager)
    set_agent_manager(manager)
    yield
    # 关闭时清理（如有需要）


app = FastAPI(
    title="Agent Long-Term Memory System",
    description="为 AI Agent 提供持久化的长期记忆能力，支持记忆的存储/检索/遗忘/整合。",
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

# 注册路由
from app.api.memory import router as memory_router
from app.api.search import router as search_router
from app.api.agent import router as agent_router

app.include_router(memory_router)
app.include_router(search_router)
app.include_router(agent_router)


@app.get("/")
async def root():
    return {
        "name": "Agent Long-Term Memory System",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}