# -*- coding: utf-8 -*-
"""
向量数据库管理平台 - 配置管理模块
从 .env 文件加载配置，支持环境变量覆盖
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field


# 加载项目根目录下的 .env 文件
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    """应用配置类，自动从环境变量中读取"""

    # ===== 应用配置 =====
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=True, alias="DEBUG")

    # ===== ChromaDB 配置 =====
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8001, alias="CHROMA_PORT")
    chroma_persist_dir: str = Field(default="./chroma_data", alias="CHROMA_PERSIST_DIR")

    # ===== Embedding 模型配置 =====
    default_embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        alias="DEFAULT_EMBEDDING_MODEL"
    )
    embedding_model_cache_dir: str = Field(
        default="./model_cache",
        alias="EMBEDDING_MODEL_CACHE_DIR"
    )

    # ===== OpenAI API 配置 =====
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_API_BASE"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-ada-002",
        alias="OPENAI_EMBEDDING_MODEL"
    )

    # ===== 文档处理配置 =====
    default_chunk_size: int = Field(default=500, alias="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=50, alias="DEFAULT_CHUNK_OVERLAP")
    chunk_strategy: str = Field(default="sentence", alias="CHUNK_STRATEGY")

    # ===== 数据目录 =====
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")

    class Config:
        env_file = str(BASE_DIR / ".env")
        case_sensitive = False

    def get_chroma_persist_path(self) -> Path:
        """获取 ChromaDB 持久化存储路径，确保目录存在"""
        path = Path(self.chroma_persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_model_cache_path(self) -> Path:
        """获取模型缓存目录路径，确保目录存在"""
        path = Path(self.embedding_model_cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_upload_path(self) -> Path:
        """获取文件上传目录路径，确保目录存在"""
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_data_path(self) -> Path:
        """获取数据目录路径，确保目录存在"""
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


# 全局配置单例
settings = Settings()
