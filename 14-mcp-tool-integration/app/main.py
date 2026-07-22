"""FastAPI 应用入口"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from registry.tool_registry import ToolRegistry
from client.mcp_client import MCPClient
from agent.tool_agent import ToolAgent
from server.builtin_servers import FileSystemServer, CalculatorServer, DatabaseServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局实例
registry = ToolRegistry()
client = MCPClient()
agent = ToolAgent(client=client, registry=registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting MCP Tool Integration System...")

    # 注册内置服务器
    builtin_servers = [
        FileSystemServer(),
        CalculatorServer(),
        DatabaseServer(),
    ]

    for server in builtin_servers:
        # 注册到注册中心
        await registry.register_server(server)
        # 连接客户端
        try:
            await client.connect(server)
        except Exception as e:
            logger.warning(f"Failed to connect to {server.name}: {e}")

    logger.info(f"Registered {registry.server_count} servers, {registry.tool_count} tools")

    yield

    # 清理
    logger.info("Shutting down...")
    await registry.shutdown()
    await client.close()
    logger.info("Shutdown complete")


# 创建 FastAPI 应用
app = FastAPI(
    title="MCP Tool Integration System",
    description="MCP (Model Context Protocol) 工具集成系统",
    version="1.0.0",
    lifespan=lifespan,
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
from app.api.registry import router as registry_router
from app.api.tools import router as tools_router
from app.api.servers import router as servers_router

app.include_router(registry_router)
app.include_router(tools_router)
app.include_router(servers_router)


@app.get("/", tags=["root"])
async def root():
    return {
        "name": "MCP Tool Integration System",
        "version": "1.0.0",
        "servers": registry.server_count,
        "tools": registry.tool_count,
    }


@app.get("/health", tags=["root"])
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)