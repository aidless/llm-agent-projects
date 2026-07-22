"""知识库管理 API"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models import KnowledgeAddRequest, KnowledgeBaseStats, KnowledgeSearchRequest, KnowledgeSearchResult

router = APIRouter()


def _get_components():
    from app.main import knowledge_base, retriever
    return {"knowledge_base": knowledge_base, "retriever": retriever}


@router.get("/knowledge/stats", response_model=KnowledgeBaseStats)
async def get_knowledge_stats():
    """获取知识库统计"""
    kb = _get_components()["knowledge_base"]
    return kb.get_stats()


@router.post("/knowledge/search", response_model=list[KnowledgeSearchResult])
async def search_knowledge(request: KnowledgeSearchRequest):
    """搜索知识库"""
    r = _get_components()["retriever"]
    return r.retrieve(request.query, top_k=request.top_k, category=request.category)


@router.post("/knowledge/add")
async def add_knowledge(request: KnowledgeAddRequest):
    """添加知识条目"""
    kb = _get_components()["knowledge_base"]
    item = kb.add_item(
        question=request.question,
        answer=request.answer,
        category=request.category,
        sub_category=request.sub_category,
        keywords=request.keywords,
    )
    return {"id": item.id, "status": "ok"}


@router.delete("/knowledge/{item_id}")
async def delete_knowledge(item_id: str):
    """删除知识条目"""
    kb = _get_components()["knowledge_base"]
    success = kb.remove_item(item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Knowledge item not found")
    return {"status": "ok"}


@router.get("/knowledge/item/{item_id}")
async def get_knowledge_item(item_id: str):
    """获取知识条目"""
    kb = _get_components()["knowledge_base"]
    item = kb.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Knowledge item not found")
    return item.model_dump()
