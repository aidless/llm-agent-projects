# ============================================
# 健康检查路由
# ============================================

from fastapi import APIRouter

from config import get_settings
from app.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """
    检查服务运行状态

    返回服务状态、LLM 配置、记忆数量等信息
    """
    settings = get_settings()

    # 检查长期记忆状态
    memory_count = 0
    try:
        from memory.long_term import LongTermMemory
        mem = LongTermMemory(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        memory_count = mem.count()
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_provider=settings.llm_provider,
        llm_model=settings.current_model,
        memory_count=memory_count,
    )
