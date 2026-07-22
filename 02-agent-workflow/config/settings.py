# ============================================
# 配置模块 - 从 .env 加载所有配置
# ============================================

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """全局配置类，自动从 .env 文件加载"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ---- LLM API 配置 ----
    openai_api_key: str = Field(default="sk-placeholder", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    deepseek_api_key: str = Field(default="sk-placeholder", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")

    # ---- FastAPI 服务配置 ----
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")

    # ---- 记忆模块配置 ----
    chroma_persist_dir: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="agent_memory", alias="CHROMA_COLLECTION_NAME")
    memory_max_turns: int = Field(default=20, alias="MEMORY_MAX_TURNS")
    memory_similarity_top_k: int = Field(default=5, alias="MEMORY_SIMILARITY_TOP_K")

    # ---- 任务追踪配置 ----
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_delay: int = Field(default=2, alias="RETRY_DELAY")
    task_timeout: int = Field(default=300, alias="TASK_TIMEOUT")

    # ---- 搜索工具配置 ----
    search_api_url: str = Field(default="", alias="SEARCH_API_URL")
    search_api_key: str = Field(default="", alias="SEARCH_API_KEY")

    # ---- 日志配置 ----
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: str = Field(default="./logs", alias="LOG_DIR")

    @property
    def current_api_key(self) -> str:
        """返回当前 LLM 提供商对应的 API Key"""
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key
        return self.openai_api_key

    @property
    def current_base_url(self) -> str:
        """返回当前 LLM 提供商对应的 Base URL"""
        if self.llm_provider == "deepseek":
            return self.deepseek_base_url
        return self.openai_base_url

    @property
    def current_model(self) -> str:
        """返回当前 LLM 提供商对应的模型名称"""
        if self.llm_provider == "deepseek":
            return self.deepseek_model
        return self.openai_model


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置单例（带缓存）"""
    return Settings()
