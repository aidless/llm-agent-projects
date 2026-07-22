"""
API 认证模块 - Bearer Token

⚠️ **2026-07-22 安全加固**：原 guardrails-chat `/chat`、`/safety/check` 等
核心防护路由无任何认证，攻击者可滥用 LLM 凭据 + 绕过守卫。本模块加 Bearer
Token 保护。

设计：
- Token 来源：环境变量 `GUARDRAILS_API_KEY`（推荐 32+ 随机字符）
- 比较方式：`hmac.compare_digest` 时间恒定比较（防计时攻击）
- 调试旁路：`GUARDRAILS_AUTH_DISABLED=true` 跳过认证（启动 WARNING）
- 错误：未配置 token → 启动失败（fail-safe）；请求缺/错 token → 401
- 公开端点：`/health`（LB/K8s 探针）
- 受保护端点：`/chat`、`/chat/stream`、`/sessions*`、`/audit/*`、`/safety/check`

用法：
    @app.post("/chat", dependencies=[Depends(verify_token)])
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

_bearer = HTTPBearer(auto_error=False)

_API_KEY: str | None = None
_AUTH_DISABLED: bool = False


def init_auth() -> None:
    """在 FastAPI 启动时初始化认证。"""
    global _API_KEY, _AUTH_DISABLED

    _API_KEY = os.getenv("GUARDRAILS_API_KEY", "").strip() or None
    _AUTH_DISABLED = os.getenv("GUARDRAILS_AUTH_DISABLED", "").lower() in ("true", "1", "yes")

    if _AUTH_DISABLED:
        logger.warning("=" * 60)
        logger.warning("  ⚠️  GUARDRAILS_AUTH_DISABLED=true")
        logger.warning("  所有受保护路由（/chat、/safety/check 等）无需认证可访问")
        logger.warning("  仅用于本地开发！生产环境严禁开启！")
        logger.warning("=" * 60)
        return

    if not _API_KEY:
        raise RuntimeError(
            "GUARDRAILS_API_KEY 未配置。\n"
            "请生成 token 并设置：\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
            "  export GUARDRAILS_API_KEY=<token>\n"
            "或临时关闭认证（仅开发）：\n"
            "  export GUARDRAILS_AUTH_DISABLED=true"
        )

    logger.info(f"GUARDRAILS_API_KEY 已配置 (长度={len(_API_KEY)})")


def is_auth_enabled() -> bool:
    return not _AUTH_DISABLED and bool(_API_KEY)


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """FastAPI Depends：校验 Bearer Token。"""
    if _AUTH_DISABLED:
        return "dev-bypass"

    if not _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="server auth not configured",
        )

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(credentials.credentials, _API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials