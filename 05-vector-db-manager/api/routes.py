# -*- coding: utf-8 -*-
"""
向量数据库管理平台 - REST API 路由
提供所有后端接口：集合管理、文档导入、相似度搜索、模型管理等
"""
import logging
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field

from services.vector_store import VectorStoreService
from services.file_parser import FileParser
from services.chunker import DocumentChunker, ChunkConfig
from embeddings.manager import embedding_manager
from app.config import settings

logger = logging.getLogger(__name__)

# 创建 API 路由
router = APIRouter(prefix="/api")

# 初始化向量存储服务
vector_store = VectorStoreService()


# ========== Pydantic 请求/响应模型 ==========

class CreateCollectionRequest(BaseModel):
    """创建 Collection 请求"""
    name: str = Field(..., description="Collection 名称")
    description: str = Field(default="", description="描述信息")


class ImportFilesRequest(BaseModel):
    """导入文件请求"""
    collection_name: str = Field(..., description="目标 Collection 名称")
    file_paths: List[str] = Field(..., description="文件路径列表")
    chunk_size: int = Field(default=500, description="分块大小")
    chunk_overlap: int = Field(default=50, description="分块重叠")
    chunk_strategy: str = Field(default="fixed", description="分块策略")


class ImportDirectoryRequest(BaseModel):
    """导入目录请求"""
    collection_name: str = Field(..., description="目标 Collection 名称")
    dir_path: str = Field(..., description="目录路径")
    chunk_size: int = Field(default=500, description="分块大小")
    chunk_overlap: int = Field(default=50, description="分块重叠")
    chunk_strategy: str = Field(default="fixed", description="分块策略")


class SearchRequest(BaseModel):
    """搜索请求"""
    collection_name: str = Field(..., description="Collection 名称")
    query: str = Field(..., description="查询文本")
    top_k: int = Field(default=5, description="返回结果数量")
    threshold: float = Field(default=0.0, description="相似度阈值")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="元数据过滤")


class CompareSearchRequest(BaseModel):
    """多索引对比搜索请求"""
    collection_names: List[str] = Field(..., description="Collection 名称列表")
    query: str = Field(..., description="查询文本")
    top_k: int = Field(default=5, description="每个集合返回结果数量")


class LoadModelRequest(BaseModel):
    """加载模型请求"""
    model_name: str = Field(..., description="模型名称")
    provider_type: str = Field(default="local", description="提供者类型: local / openai")
    api_key: Optional[str] = Field(default=None, description="OpenAI API 密钥")
    api_base: Optional[str] = Field(default=None, description="OpenAI API 基础地址")


class ChunkTestRequest(BaseModel):
    """分块测试请求"""
    text: str = Field(..., description="待分块文本")
    chunk_size: int = Field(default=500, description="分块大小")
    chunk_overlap: int = Field(default=50, description="分块重叠")
    chunk_strategy: str = Field(default="fixed", description="分块策略")


# ========== 首页 / 健康检查 ==========

@router.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "message": "向量数据库管理平台运行中"}


# ========== Embedding 模型管理 ==========

@router.get("/models")
async def list_models():
    """
    列出所有已加载的 Embedding 模型

    返回当前已加载模型列表及其状态信息
    """
    models = embedding_manager.list_loaded_models()
    current = embedding_manager.get_current_model_name()
    return {
        "current_model": current,
        "models": models,
    }


@router.post("/models/load")
async def load_model(req: LoadModelRequest):
    """
    加载或切换 Embedding 模型

    支持 BGE/OpenAI/本地模型。首次加载本地模型时需下载，可能耗时较长。
    """
    try:
        provider = embedding_manager.load_model(
            model_name=req.model_name,
            provider_type=req.provider_type,
            api_key=req.api_key,
            api_base=req.api_base,
        )
        return {
            "message": f"模型加载成功: {req.model_name}",
            "model_name": provider.name,
            "dimension": provider.dimension,
            "provider_type": provider.provider_type,
        }
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Collection 管理 ==========

@router.get("/collections")
async def list_collections():
    """列出所有 Collection"""
    try:
        collections = vector_store.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections")
async def create_collection(req: CreateCollectionRequest):
    """创建新的 Collection"""
    try:
        # 检查是否已加载模型
        embedding_manager.get_current_provider()
        result = vector_store.create_collection(
            name=req.name,
            description=req.description,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=f"请先加载 Embedding 模型: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{name}")
async def get_collection_detail(name: str):
    """获取 Collection 详细信息"""
    try:
        return vector_store.get_collection_metadata(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """删除 Collection"""
    try:
        return vector_store.delete_collection(name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{name}/clear")
async def clear_collection(name: str):
    """清空 Collection 中的所有文档"""
    try:
        return vector_store.clear_collection(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 文档导入 ==========

@router.post("/import/files")
async def import_files(req: ImportFilesRequest):
    """
    批量导入文件到指定 Collection

    支持的格式：TXT, MD, JSON, CSV
    """
    try:
        result = vector_store.import_files(
            collection_name=req.collection_name,
            file_paths=req.file_paths,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            chunk_strategy=req.chunk_strategy,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/directory")
async def import_directory(req: ImportDirectoryRequest):
    """
    导入整个目录的文件到指定 Collection

    自动递归扫描目录中的所有支持格式文件
    """
    try:
        result = vector_store.import_directory(
            collection_name=req.collection_name,
            dir_path=req.dir_path,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            chunk_strategy=req.chunk_strategy,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/upload")
async def upload_files(
    collection_name: str = Form(...),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
    chunk_strategy: str = Form(default="fixed"),
    files: List[UploadFile] = File(...),
):
    """
    通过 HTTP 上传文件并导入到指定 Collection

    支持多文件上传
    """
    try:
        upload_path = settings.get_upload_path()

        # 保存上传文件
        saved_paths = []
        for file in files:
            file_path = upload_path / file.filename
            content = await file.read()
            file_path.write_bytes(content)
            saved_paths.append(str(file_path))
            logger.info(f"文件已保存: {file_path}")

        # 导入到向量数据库
        result = vector_store.import_files(
            collection_name=collection_name,
            file_paths=saved_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 相似度搜索 ==========

@router.post("/search")
async def search(req: SearchRequest):
    """
    在指定 Collection 中进行相似度搜索

    支持 top-k 搜索、相似度阈值过滤、元数据过滤
    """
    try:
        result = vector_store.search(
            collection_name=req.collection_name,
            query=req.query,
            top_k=req.top_k,
            threshold=req.threshold,
            metadata_filter=req.metadata_filter,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 多索引对比 ==========

@router.post("/compare")
async def compare_search(req: CompareSearchRequest):
    """
    在多个 Collection 上执行相同查询，对比搜索效果

    用于比较不同分块策略和嵌入模型的检索效果
    """
    try:
        results = vector_store.compare_search(
            collection_names=req.collection_names,
            query=req.query,
            top_k=req.top_k,
        )
        return {"query": req.query, "comparisons": results}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 分块测试 ==========

@router.post("/chunk/test")
async def test_chunking(req: ChunkTestRequest):
    """
    测试文档分块效果

    对输入文本应用指定分块策略，返回分块结果
    """
    try:
        chunker = DocumentChunker(ChunkConfig(
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            chunk_strategy=req.chunk_strategy,
        ))
        from services.file_parser import ParsedDocument
        doc = ParsedDocument(content=req.text, metadata={"source": "test"})
        chunks = chunker.chunk_document(doc)
        return {
            "total_chunks": len(chunks),
            "strategy": req.chunk_strategy,
            "chunk_size": req.chunk_size,
            "chunk_overlap": req.chunk_overlap,
            "chunks": [
                {
                    "index": c.chunk_index,
                    "text": c.text,
                    "length": len(c.text),
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                }
                for c in chunks
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chunk/strategies")
async def list_chunk_strategies():
    """列出所有可用的分块策略及其描述"""
    return {
        "strategies": DocumentChunker.get_strategy_descriptions(),
        "supported": DocumentChunker.SUPPORTED_STRATEGIES,
    }


# ========== 统计面板 ==========

@router.get("/statistics")
async def get_statistics():
    """
    获取系统统计信息

    包括 Collection 数量、文档数量、向量维度、存储占用等
    """
    try:
        return vector_store.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
