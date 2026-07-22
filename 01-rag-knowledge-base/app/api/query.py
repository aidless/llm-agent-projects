"""
知识库查询 API 路由
提供查询、流式查询接口
"""
import json
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_service import RAGService
from loguru import logger

router = APIRouter(prefix="/api/v1", tags=["query"])

# 全局 RAG 服务实例（由 main.py 初始化时注入）
rag_service: RAGService = None


def set_rag_service(service: RAGService) -> None:
    """设置全局 RAG 服务实例"""
    global rag_service
    rag_service = service


@router.post("/query", response_model=QueryResponse, summary="知识库查询")
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """
    知识库问答

    基于已上传的文档回答用户问题，返回完整的回答和引用来源
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪，请稍后重试")

    try:
        result = rag_service.query(
            question=request.question,
            top_k=request.top_k,
            stream=False,
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@router.post("/query/stream", summary="知识库流式查询")
async def query_knowledge_base_stream(request: QueryRequest):
    """
    知识库流式问答

    使用 Server-Sent Events (SSE) 流式返回回答
    事件类型:
    - citations: 引用来源信息
    - content: 文本片段
    - done: 完成信号（包含完整回答）
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪，请稍后重试")

    async def event_generator():
        try:
            for chunk in rag_service.query_stream(
                question=request.question,
                top_k=request.top_k,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            error_data = json.dumps(
                {"type": "error", "message": str(e)},
                ensure_ascii=False,
            )
            yield f"data: {error_data}\n\n"

    return EventSourceResponse(event_generator())