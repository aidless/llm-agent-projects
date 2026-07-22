"""
RAG 知识库问答系统 - FastAPI 主入口

启动方式:
    python main.py

访问:
    API 文档: http://localhost:8000/docs
    健康检查: http://localhost:8000/api/v1/health
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import settings, setup_logging
from app.services.rag_service import RAGService

# 全局 RAG 服务
_rag_service: RAGService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    启动时初始化 RAG 服务，关闭时清理资源
    """
    global _rag_service

    # 配置日志
    setup_logging()
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name} v{settings.app_version}")
    logger.info("=" * 60)

    # 确保数据目录存在
    settings.ensure_directories()

    # 初始化 RAG 服务
    logger.info("正在初始化 RAG 服务（首次启动可能需要加载模型，请耐心等待）...")
    try:
        _rag_service = RAGService()
        _rag_service.init_components()
        logger.info("RAG 服务初始化成功!")

        # 注入到各 API 路由
        from app.api import query, documents, system
        query.set_rag_service(_rag_service)
        documents.set_rag_service(_rag_service)
        system.set_rag_service(_rag_service)

    except Exception as e:
        logger.error(f"RAG 服务初始化失败: {e}")
        logger.warning("服务将以降级模式运行（仅文档管理可用，查询不可用）")

    logger.info(f"服务启动完成，访问 http://{settings.app_host}:{settings.app_port}/docs 查看API文档")

    yield

    # 关闭时清理
    logger.info("正在关闭服务...")
    _rag_service = None
    logger.info("服务已关闭")


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        ## RAG 知识库问答系统

        企业级 RAG（Retrieval-Augmented Generation）知识库问答系统。

        ### 核心功能
        - 多格式文档解析（PDF/Word/Markdown/TXT）
        - 语义分块（固定长度/按段落/按语义）
        - 多种 Embedding 模型支持（BGE/OpenAI/本地）
        - 混合检索（BM25 + 向量检索）
        - Reranker 二次排序
        - 流式输出（SSE）
        - 引用溯源

        ### 使用流程
        1. 上传文档到知识库
        2. 通过查询接口提问
        3. 系统检索相关文档段落并生成回答
        """,
        lifespan=lifespan,
    )

    # 配置 CORS（允许前端跨域访问）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    from app.api import query, documents, system
    app.include_router(query.router)
    app.include_router(documents.router)
    app.include_router(system.router)

    # 根路径重定向到文档
    @app.get("/", tags=["root"])
    async def root():
        """根路径，返回系统信息"""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": f"http://{settings.app_host}:{settings.app_port}/docs",
        }

    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="info",
    )