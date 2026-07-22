"""
数据模型定义
定义 API 请求和响应的数据结构
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# ========== 枚举类型 ==========

class ChunkStrategy(str, Enum):
    """分块策略枚举"""
    FIXED = "fixed"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


class EmbeddingProvider(str, Enum):
    """Embedding 提供商枚举"""
    BGE = "bge"
    OPENAI = "openai"
    LOCAL = "local"


# ========== 请求模型 ==========

class QueryRequest(BaseModel):
    """知识库查询请求"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="检索结果数量")
    stream: Optional[bool] = Field(default=False, description="是否使用流式输出")


class DocumentDeleteRequest(BaseModel):
    """文档删除请求"""
    filename: str = Field(..., description="要删除的文件名")


# ========== 响应模型 ==========

class CitationSource(BaseModel):
    """引用来源"""
    source_file: str = Field(default="", description="来源文件名")
    doc_title: str = Field(default="", description="文档标题")
    chunk_id: int = Field(default=0, description="文本块编号")
    content: str = Field(default="", description="引用的原始内容")
    score: float = Field(default=0.0, description="相关性得分")
    start_index: int = Field(default=0, description="在原文中的起始位置")
    end_index: int = Field(default=0, description="在原文中的结束位置")


class QueryResponse(BaseModel):
    """知识库查询响应"""
    question: str = Field(default="", description="用户问题")
    answer: str = Field(default="", description="系统回答")
    citations: List[CitationSource] = Field(default_factory=list, description="引用来源列表")
    retrieval_count: int = Field(default=0, description="检索到的文档数量")
    elapsed_seconds: float = Field(default=0.0, description="处理耗时（秒）")


class QueryStreamChunk(BaseModel):
    """流式查询响应块"""
    type: str = Field(default="content", description="事件类型: citations/content/done")
    text: Optional[str] = Field(default=None, description="文本内容（type=content 时）")
    citations: Optional[List[CitationSource]] = Field(default=None, description="引用来源（type=citations 时）")
    full_answer: Optional[str] = Field(default=None, description="完整回答（type=done 时）")
    elapsed_seconds: Optional[float] = Field(default=None, description="总耗时（type=done 时）")


class DocumentInfo(BaseModel):
    """文档信息"""
    filename: str = Field(default="", description="文件名")
    title: str = Field(default="", description="文档标题")
    file_type: str = Field(default="", description="文件类型")
    chunk_count: int = Field(default=0, description="文本块数量")


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    filename: str = Field(default="", description="文件名")
    file_type: str = Field(default="", description="文件类型")
    title: str = Field(default="", description="文档标题")
    chunk_count: int = Field(default=0, description="生成的文本块数量")
    total_chars: int = Field(default=0, description="文档总字符数")
    doc_count: int = Field(default=0, description="知识库中总文档块数")
    elapsed_seconds: float = Field(default=0.0, description="处理耗时")
    status: str = Field(default="success", description="处理状态")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="ok", description="服务状态")
    version: str = Field(default="", description="版本号")
    components: Dict[str, Any] = Field(default_factory=dict, description="组件状态")


class StatsResponse(BaseModel):
    """系统统计信息"""
    total_documents: int = Field(default=0, description="文档总数")
    total_chunks: int = Field(default=0, description="文本块总数")
    embedding_provider: str = Field(default="", description="Embedding 提供商")
    embedding_dimension: int = Field(default=0, description="向量维度")
    llm_provider: str = Field(default="", description="LLM 提供商")
    llm_model: str = Field(default="", description="LLM 模型名称")
    reranker_enabled: bool = Field(default=False, description="Reranker 是否启用")
    chunk_strategy: str = Field(default="", description="分块策略")
    chunk_size: int = Field(default=0, description="分块大小")