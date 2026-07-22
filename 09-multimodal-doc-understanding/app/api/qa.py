"""文档问答 API。"""

from fastapi import APIRouter, HTTPException

from app.models import APIResponse, QARequest, QAResult
from rag.document_indexer import DocumentIndexer, SimpleVectorStore
from rag.qa_chain import QAChain
from rag.retriever import HybridRetriever

router = APIRouter(prefix="/qa", tags=["文档问答"])

# 全局实例 (生产环境应使用依赖注入)
_store = SimpleVectorStore()
_indexer = DocumentIndexer(vector_store=_store)
_qa_chain = QAChain(indexer=_indexer)


@router.post("/ask", response_model=APIResponse)
async def ask_question(req: QARequest):
    """文档问答。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        result = _qa_chain.ask(
            question=req.question,
            top_k=req.top_k,
            document_ids=req.document_ids,
        )
        return APIResponse(
            success=True,
            message="问答完成",
            data=result.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=APIResponse)
async def list_documents():
    """列出已索引的文档。"""
    docs = _indexer.get_indexed_documents()
    return APIResponse(
        success=True,
        message=f"共 {len(docs)} 个已索引文档",
        data={"documents": docs, "total_chunks": _store.count()},
    )


@router.delete("/documents/{document_id}", response_model=APIResponse)
async def delete_document(document_id: str):
    """删除已索引的文档。"""
    success = _indexer.delete_document(document_id)
    if success:
        return APIResponse(success=True, message=f"已删除文档: {document_id}")
    raise HTTPException(status_code=404, detail=f"文档不存在: {document_id}")


@router.get("/health", response_model=APIResponse)
async def health_check():
    """健康检查。"""
    return APIResponse(success=True, message="服务正常", data={"status": "healthy"})