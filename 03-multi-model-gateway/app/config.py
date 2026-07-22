"""
全局配置模块 - 从 .env 文件加载所有配置项
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """单个模型的配置信息"""
    name: str                           # 模型显示名称
    provider: str                       # 服务商标识: openai / deepseek / qwen / claude
    api_key: str                        # API 密钥
    base_url: str                       # API 基础地址
    model_id: str                       # 实际模型标识符
    input_price_per_1k: float           # 输入价格（美元/千 token）
    output_price_per_1k: float          # 输出价格（美元/千 token）
    extra_keys: list[str] = field(default_factory=list)  # 用于负载均衡的额外 API Key

    @property
    def all_keys(self) -> list[str]:
        """返回所有可用的 API Key（主 Key + 额外 Key）"""
        keys = [self.api_key]
        keys.extend(k for k in self.extra_keys if k)
        return keys


@dataclass
class AppConfig:
    """应用全局配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 10
    request_timeout: int = 60
    models: dict[str, ModelConfig] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量加载配置"""
        from dotenv import load_dotenv
        load_dotenv()

        config = cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
        )

        # 注册 OpenAI GPT-4o
        config.models["gpt-4o"] = ModelConfig(
            name="GPT-4o",
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model_id=os.getenv("OPENAI_MODEL", "gpt-4o"),
            input_price_per_1k=float(os.getenv("GPT4O_INPUT_PRICE", "0.0025")),
            output_price_per_1k=float(os.getenv("GPT4O_OUTPUT_PRICE", "0.01")),
            extra_keys=_parse_extra_keys(os.getenv("OPENAI_API_KEYS", "")),
        )

        # 注册 DeepSeek V4
        config.models["deepseek"] = ModelConfig(
            name="DeepSeek V4",
            provider="deepseek",
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            model_id=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            input_price_per_1k=float(os.getenv("DEEPSEEK_INPUT_PRICE", "0.00027")),
            output_price_per_1k=float(os.getenv("DEEPSEEK_OUTPUT_PRICE", "0.0011")),
            extra_keys=_parse_extra_keys(os.getenv("DEEPSEEK_API_KEYS", "")),
        )

        # 注册 Qwen（通义千问）
        config.models["qwen"] = ModelConfig(
            name="Qwen Plus",
            provider="qwen",
            api_key=os.getenv("QWEN_API_KEY", ""),
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model_id=os.getenv("QWEN_MODEL", "qwen-plus"),
            input_price_per_1k=float(os.getenv("QWEN_INPUT_PRICE", "0.0004")),
            output_price_per_1k=float(os.getenv("QWEN_OUTPUT_PRICE", "0.0012")),
            extra_keys=_parse_extra_keys(os.getenv("QWEN_API_KEYS", "")),
        )

        # 注册 Claude
        config.models["claude"] = ModelConfig(
            name="Claude Sonnet 4",
            provider="claude",
            api_key=os.getenv("CLAUDE_API_KEY", ""),
            base_url=os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1"),
            model_id=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            input_price_per_1k=float(os.getenv("CLAUDE_INPUT_PRICE", "0.003")),
            output_price_per_1k=float(os.getenv("CLAUDE_OUTPUT_PRICE", "0.015")),
            extra_keys=_parse_extra_keys(os.getenv("CLAUDE_API_KEYS", "")),
        )

        return config


def _parse_extra_keys(keys_str: str) -> list[str]:
    """解析逗号分隔的额外 API Key 列表"""
    if not keys_str or not keys_str.strip():
        return []
    return [k.strip() for k in keys_str.split(",") if k.strip()]


# 全局配置单例，延迟初始化
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """获取全局配置实例（单例模式）"""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config