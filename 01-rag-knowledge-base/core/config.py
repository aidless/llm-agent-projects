"""
全局配置模块
从 .env 文件加载配置，提供统一的配置访问接口
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from loguru import logger


class Settings(BaseSettings):
    """应用配置类，自动从 .env 加载环境变量"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- 应用基础配置 ---
    app_name: str = Field(default="RAG-Knowledge-Base", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- 数据目录配置 ---
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")
    processed_dir: str = Field(default="./data/processed", alias="PROCESSED_DIR")
    chroma_persist_dir: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIR")

    # --- Embedding 模型配置 ---
    embedding_provider: str = Field(default="bge", alias="EMBEDDING_PROVIDER")
    bge_model_name: str = Field(default="BAAI/bge-large-zh-v1.5", alias="BGE_MODEL_NAME")
    bge_embedding_dimension: int = Field(default=1024, alias="BGE_EMBEDDING_DIMENSION")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_api_base: str = Field(default="https://api.openai.com/v1", alias="OPENAI_API_BASE")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", alias="OPENAI_EMBEDDING_MODEL")
    local_embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="LOCAL_EMBEDDING_MODEL")
    local_embedding_dimension: int = Field(default=384, alias="LOCAL_EMBEDDING_DIMENSION")

    # --- LLM 配置 ---
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-3.5-turbo", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    llm_top_p: float = Field(default=0.9, alias="LLM_TOP_P")

    # 智谱AI
    zhipu_api_key: Optional[str] = Field(default=None, alias="ZHIPU_API_KEY")
    zhipu_model: str = Field(default="glm-4", alias="ZHIPU_MODEL")

    # 阿里云百炼
    dashscope_api_key: Optional[str] = Field(default=None, alias="DASHSCOPE_API_KEY")
    dashscope_model: str = Field(default="qwen-turbo", alias="DASHSCOPE_MODEL")

    # --- 分块配置 ---
    chunk_strategy: str = Field(default="semantic", alias="CHUNK_STRATEGY")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    chunk_separator: str = Field(default="\n", alias="CHUNK_SEPARATOR")

    # --- 检索配置 ---
    retrieval_top_k: int = Field(default=10, alias="RETRIEVAL_TOP_K")
    hybrid_search_weight_vector: float = Field(default=0.7, alias="HYBRID_SEARCH_WEIGHT_VECTOR")
    hybrid_search_weight_bm25: float = Field(default=0.3, alias="HYBRID_SEARCH_WEIGHT_BM25")
    bm25_k1: float = Field(default=1.5, alias="BM25_K1")
    bm25_b: float = Field(default=0.75, alias="BM25_B")

    # --- Reranker 配置 ---
    reranker_enabled: bool = Field(default=True, alias="RERANKER_ENABLED")
    reranker_model: str = Field(default="BAAI/bge-reranker-large", alias="RERANKER_MODEL")
    reranker_top_k: int = Field(default=5, alias="RERANKER_TOP_K")

    # --- 引用溯源 ---
    citation_enabled: bool = Field(default=True, alias="CITATION_ENABLED")
    citation_max_sources: int = Field(default=3, alias="CITATION_MAX_SOURCES")

    # --- 文档处理 ---
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    supported_file_types: str = Field(default="pdf,docx,doc,md,txt", alias="SUPPORTED_FILE_TYPES")

    @property
    def supported_file_types_list(self) -> List[str]:
        """返回支持的文件类型列表"""
        return [t.strip() for t in self.supported_file_types.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        """返回最大文件大小（字节）"""
        return self.max_file_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """确保所有必要的目录存在"""
        dirs = [
            self.data_dir,
            self.upload_dir,
            self.processed_dir,
            self.chroma_persist_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"数据目录初始化完成: {dirs}")


# 全局配置单例
settings = Settings()


def setup_logging() -> None:
    """配置日志系统"""
    logger.remove()  # 移除默认处理器
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    # 同时输出到文件
    logger.add(
        sink="rag_system.log",
        level=settings.log_level,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
    )
