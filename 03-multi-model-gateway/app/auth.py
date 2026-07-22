"""
API 认证模块 - Bearer Token

⚠️ **2026-07-22 安全加固**：原网关 `/v1/*` 无任何认证，任意人可调用你的
OpenAI/DeepSeek/Qwen 凭据，账单直接爆炸。本模块加 Bearer Token 保护。

设计：
- Token 来源：环境变量 `GATEWAY_API_KEY`（推荐 32+ 随机字符）
- 比较方式：`hmac.compare_digest` 时间恒定比较（防计时攻击）
- 调试旁路：`GATEWAY_AUTH_DISABLED=true` 跳过认证（启动日志会有 WARNING）
- 错误：未配置 token → 启动失败（fail-safe）；请求缺/错 token → 401
- 公开端点：`/health`、`/stats*`、`/strategies`（LB 探针 + 内部监控）
- 受保护端点：`/v1/chat/completions`、`/v1/models`

用法：
    @app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
    async def chat(...): ...

生成 token：
    python -c "import secrets; print(secrets.token_urlsafe(32))"
"""

import hmac
import logging
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# auto_error=False 让 FastAPI 不会自动抛 403；我们自己抛 401 更标准
_bearer = HTTPBearer(auto_error=False)

_API_KEY: str | None = None
_AUTH_DISABLED: bool = False


def init_auth() -> None:
    """在 FastAPI 启动时初始化认证。

    - 未配置 GATEWAY_API_KEY 且未显式禁用 → 抛 RuntimeError（启动失败）
    - GATEWAY_AUTH_DISABLED=true → 跳过认证（dev 模式，启动 WARNING）
    """
    global _API_KEY, _AUTH_DISABLED

    _API_KEY = os.getenv("GATEWAY_API_KEY", "").strip() or None
    _AUTH_DISABLED = os.getenv("GATEWAY_AUTH_DISABLED", "").lower() in ("true", "1", "yes")

    if _AUTH_DISABLED:
        logger.warning("=" * 60)
        logger.warning("  ⚠️  GATEWAY_AUTH_DISABLED=true")
        logger.warning("  所有 /v1/* 路由无需认证可访问")
        logger.warning("  仅用于本地开发！生产环境严禁开启！")
        logger.warning("=" * 60)
        return

    if not _API_KEY:
        raise RuntimeError(
            "GATEWAY_API_KEY 未配置。\n"
            "请生成 token 并设置：\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
            "  export GATEWAY_API_KEY=<token>\n"
            "或临时关闭认证（仅开发）：\n"
            "  export GATEWAY_AUTH_DISABLED=true"
        )

    # 不在日志中打印 token 本身
    logger.info(f"GATEWAY_API_KEY 已配置 (长度={len(_API_KEY)})")


def is_auth_enabled() -> bool:
    """供 /health 等端点报告认证状态。"""
    return not _AUTH_DISABLED and bool(_API_KEY)


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """FastAPI Depends：校验 Bearer Token。

    返回值：当前请求的 token 字符串（可被子函数用于 audit log）。
    """
    # 开发旁路
    if _AUTH_DISABLED:
        return "dev-bypass"

    # 未配置 → init_auth 应已失败；这里再防一次
    if not _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="server auth not configured",
        )

    # 请求未带 Authorization header
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 时间恒定比较，防止计时攻击
    if not hmac.compare_digest(credentials.credentials, _API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials