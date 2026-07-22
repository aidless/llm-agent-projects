"""
配置管理模块 - 使用 .env 文件和环境变量管理配置

支持从 .env 文件加载配置，并可通过环境变量覆盖。
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM API 配置"""
    api_key: str = ""                           # API Key（从环境变量读取）
    api_base: str = "https://api.openai.com/v1"  # API 基础 URL
    model: str = "gpt-3.5-turbo"                # 模型名称
    temperature: float = 0.7                    # 生成温度
    max_tokens: int = 2048                      # 最大生成 token 数
    timeout: int = 30                           # 请求超时时间（秒）


@dataclass
class GuardConfig:
    """Guardrails 配置"""
    max_input_length: int = 4000                 # 最大输入长度
    max_output_length: int = 8000               # 最大输出长度
    enable_input_guard: bool = True              # 是否启用输入守卫
    enable_output_guard: bool = True             # 是否启用输出守卫
    enable_hallucination_detect: bool = True    # 是否启用幻觉检测
    enable_audit_log: bool = True                # 是否启用审计日志
    safety_threshold: float = 60.0              # 安全评分阈值（低于此值将被阻止）


@dataclass
class AppConfig:
    """应用总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    host: str = "0.0.0.0"                       # 服务监听地址
    port: int = 8000                             # 服务端口
    debug: bool = False                          # 调试模式
    log_level: str = "INFO"                      # 日志级别
    persist_dir: str = "./data/sessions"        # 会话持久化目录
    max_history_per_session: int = 50             # 每个会话最大历史轮数
    data_dir: str = "./data"                     # 数据目录（敏感词库等）

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        从环境变量和 .env 文件加载配置

        优先级：环境变量 > .env 文件 > 默认值
        """
        config = cls()

        # 加载 .env 文件
        config._load_env_file()

        # LLM 配置
        config.llm.api_key = os.getenv("LLM_API_KEY", config.llm.api_key)
        config.llm.api_base = os.getenv("LLM_API_BASE", config.llm.api_base)
        config.llm.model = os.getenv("LLM_MODEL", config.llm.model)
        config.llm.temperature = float(os.getenv("LLM_TEMPERATURE", str(config.llm.temperature)))
        config.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(config.llm.max_tokens)))
        config.llm.timeout = int(os.getenv("LLM_TIMEOUT", str(config.llm.timeout)))

        # Guardrails 配置
        config.guard.max_input_length = int(os.getenv("MAX_INPUT_LENGTH", str(config.guard.max_input_length)))
        config.guard.max_output_length = int(os.getenv("MAX_OUTPUT_LENGTH", str(config.guard.max_output_length)))
        config.guard.enable_input_guard = os.getenv("ENABLE_INPUT_GUARD", str(config.guard.enable_input_guard)).lower() in ("true", "1", "yes")
        config.guard.enable_output_guard = os.getenv("ENABLE_OUTPUT_GUARD", str(config.guard.enable_output_guard)).lower() in ("true", "1", "yes")
        config.guard.enable_hallucination_detect = os.getenv("ENABLE_HALLUCINATION", str(config.guard.enable_hallucination_detect)).lower() in ("true", "1", "yes")
        config.guard.enable_audit_log = os.getenv("ENABLE_AUDIT_LOG", str(config.guard.enable_audit_log)).lower() in ("true", "1", "yes")
        config.guard.safety_threshold = float(os.getenv("SAFETY_THRESHOLD", str(config.guard.safety_threshold)))

        # 应用配置
        config.host = os.getenv("APP_HOST", config.host)
        config.port = int(os.getenv("APP_PORT", str(config.port)))
        config.debug = os.getenv("APP_DEBUG", str(config.debug)).lower() in ("true", "1", "yes")
        config.log_level = os.getenv("LOG_LEVEL", config.log_level)
        config.persist_dir = os.getenv("PERSIST_DIR", config.persist_dir)
        config.max_history_per_session = int(os.getenv("MAX_HISTORY", str(config.max_history_per_session)))
        config.data_dir = os.getenv("DATA_DIR", config.data_dir)

        return config

    @staticmethod
    def _load_env_file() -> None:
        """
        加载 .env 文件中的环境变量

        简单实现：逐行解析 KEY=VALUE 格式
        """
        env_paths = [".env", "../.env"]
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_paths = [os.path.join(base_dir, p) for p in env_paths]

        for env_path in env_paths:
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # 跳过空行和注释
                        if not line or line.startswith("#"):
                            continue
                        # 解析 KEY=VALUE
                        if "=" in line:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            # 不覆盖已有的环境变量（环境变量优先级更高）
                            if key not in os.environ:
                                os.environ[key] = value
                break  # 只加载第一个找到的 .env 文件
