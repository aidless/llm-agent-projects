"""
Pytest conftest for 04-guardrails-chat.

⚠️ **2026-07-22**：服务加了 Bearer Token 鉴权，测试模式下自动启用 dev
旁路（GUARDRAILS_AUTH_DISABLED=true）。
"""

import os

os.environ.setdefault("GUARDRAILS_AUTH_DISABLED", "true")