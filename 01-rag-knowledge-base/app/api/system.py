"""
系统管理 API 路由
提供健康检查、统计信息、知识库重置等接口
"""
from fastapi import APIRouter, HTTPException

from app.models.schemas import HealthResponse, StatsResponse
from app.services.rag_service import RAGService
from core.config import settings
from loguru import logger

router = APIRouter(prefix="/api/v1", tags=["system"])

# 全局 RAG 服务实例
rag_service: RAGService = None


def set_rag_service(service: RAGService) -> None:
    """设置全局 RAG 服务实例"""
    global rag_service
    rag_service = service


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check() -> HealthResponse:
    """检查系统各组件运行状态"""
    components = {
        "app": {"status": "ok", "version": settings.app_version},
    }

    if rag_service is not None:
        try:
            stats = rag_service.get_stats()
            components["rag_service"] = {"status": "ok", "stats": stats}
        except Exception as e:
            components["rag_service"] = {"status": "error", "message": str(e)}
    else:
        components["rag_service"] = {"status": "not_initialized"}

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        components=components,
    )


@router.get("/stats", response_model=StatsResponse, summary="系统统计信息")
async def get_stats() -> StatsResponse:
    """获取知识库统计信息"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪")

    try:
        stats = rag_service.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.post("/reset", summary="重置知识库")
async def reset_knowledge_base() -> dict:
    """
    重置知识库

    清空所有文档和向量数据，此操作不可撤销
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪")

    try:
        rag_service.reset()
        logger.warning("知识库已被重置")
        return {"status": "success", "message": "知识库已重置"}
    except Exception as e:
        logger.error(f"重置知识库失败: {e}")
        raise HTTPException(status_code=500, detail=f"重置失败: {str(e)}")