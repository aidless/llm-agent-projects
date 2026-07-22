# ============================================
# 记忆路由
# 提供记忆管理的 API
# ============================================

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models import (
    MemoryAddRequest,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryItem,
)
from config import get_settings

router = APIRouter()


def _get_long_term_memory():
    """获取长期记忆实例"""
    settings = get_settings()
    from memory.long_term import LongTermMemory
    return LongTermMemory(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )


@router.get("/memory", summary="获取记忆统计")
async def get_memory_stats():
    """
    获取长期记忆的统计信息
    """
    memory = _get_long_term_memory()
    try:
        all_memories = memory.get_all_memories(limit=1000)
        return {
            "total_count": memory.count(),
            "sample_memories": [
                {"id": m["id"], "content": m["content"][:100], "metadata": m.get("metadata", {})}
                for m in all_memories[:10]
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取记忆失败: {str(e)}")


@router.post("/memory/add", summary="添加记忆")
async def add_memory(request: MemoryAddRequest):
    """
    添加一条长期记忆

    Args:
        request: 包含记忆内容和可选元数据
    """
    memory = _get_long_term_memory()
    try:
        memory_id = memory.add_memory(
            content=request.content,
            metadata=request.metadata,
        )
        return {"memory_id": memory_id, "status": "success"}
    except Exception as e:
        logger.error(f"[API] 添加记忆失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加记忆失败: {str(e)}")


@router.post("/memory/search", response_model=MemorySearchResponse, summary="搜索记忆")
async def search_memory(request: MemorySearchRequest):
    """
    语义搜索长期记忆

    Args:
        request: 包含搜索查询和返回数量
    """
    memory = _get_long_term_memory()
    try:
        results = memory.search(query=request.query, top_k=request.top_k)

        memory_items = [
            MemoryItem(
                id=r["id"],
                content=r["content"],
                metadata=r.get("metadata", {}),
                distance=r.get("distance"),
            )
            for r in results
        ]

        return MemorySearchResponse(
            query=request.query,
            results=memory_items,
            total=len(memory_items),
        )
    except Exception as e:
        logger.error(f"[API] 搜索记忆失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索记忆失败: {str(e)}")


@router.delete("/memory/{memory_id}", summary="删除记忆")
async def delete_memory(memory_id: str):
    """
    删除指定的长期记忆

    Args:
        memory_id: 记忆 ID
    """
    memory = _get_long_term_memory()
    success = memory.delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"记忆 '{memory_id}' 不存在或删除失败")
    return {"status": "success", "message": f"已删除记忆 '{memory_id}'"}


@router.delete("/memory", summary="清空所有记忆")
async def clear_all_memory():
    """
    清空所有长期记忆（慎用！）
    """
    memory = _get_long_term_memory()
    memory.clear()
    return {"status": "success", "message": "已清空所有长期记忆"}
