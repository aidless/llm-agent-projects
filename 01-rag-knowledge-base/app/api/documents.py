"""
文档管理 API 路由
提供文档上传、列表、删除等接口
"""
import os
import shutil
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.models.schemas import DocumentUploadResponse, DocumentInfo, DocumentDeleteRequest
from app.services.rag_service import RAGService
from core.config import settings
from core.parsers.document_parser import DocumentParser
from loguru import logger

router = APIRouter(prefix="/api/v1", tags=["documents"])

# 全局 RAG 服务实例（由 main.py 初始化时注入）
rag_service: RAGService = None


def set_rag_service(service: RAGService) -> None:
    """设置全局 RAG 服务实例"""
    global rag_service
    rag_service = service


@router.post("/documents/upload", response_model=DocumentUploadResponse, summary="上传文档")
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    上传文档到知识库

    支持 PDF、Word、Markdown、TXT 格式
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪")

    # 检查文件类型
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.supported_file_types_list:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}，支持: {settings.supported_file_types}",
        )

    # 检查文件大小
    file.file.seek(0, 2)  # 移动到文件末尾
    file_size = file.file.tell()
    file.file.seek(0)  # 移回文件开头

    if file_size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大: {file_size / 1024 / 1024:.1f}MB，最大: {settings.max_file_size_mb}MB",
        )

    # 保存上传文件
    upload_path = os.path.join(settings.upload_dir, file.filename)
    os.makedirs(settings.upload_dir, exist_ok=True)

    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"文件已保存: {upload_path}")
    except Exception as e:
        logger.error(f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 处理文档
    try:
        result = rag_service.upload_document(upload_path)
        return DocumentUploadResponse(**result)
    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        # 清理已上传的文件
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.get("/documents", response_model=List[DocumentInfo], summary="获取文档列表")
async def list_documents() -> List[DocumentInfo]:
    """获取知识库中所有已上传的文档列表"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪")

    try:
        docs = rag_service.list_documents()
        return [DocumentInfo(**d) for d in docs]
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.delete("/documents", summary="删除文档")
async def delete_document(request: DocumentDeleteRequest) -> dict:
    """删除指定文档及其所有文本块"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务未就绪")

    try:
        result = rag_service.delete_document(request.filename)
        return result
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


@router.get("/documents/supported-types", summary="获取支持的文件类型")
async def get_supported_types() -> dict:
    """获取系统支持的文档类型列表"""
    parser = DocumentParser()
    return {
        "supported_extensions": parser.get_supported_extensions(),
        "supported_types": settings.supported_file_types,
        "max_file_size_mb": settings.max_file_size_mb,
    }