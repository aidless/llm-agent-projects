"""
Pytest conftest for 03-multi-model-gateway.

⚠️ **2026-07-22**：服务加了 Bearer Token 鉴权（fail-safe，未配 token 会
启动失败）。本 conftest 在 import app 后立即启用 dev 旁路。

⚠️ **重要**：必须在 `from app.main import app` 之后再调用 init_auth()，
否则 app.auth 模块级的 _API_KEY 已被读取。
"""

import os

# 第一步：设置 dev 旁路（必须在 import 任何 app.* 之前）
os.environ.setdefault("GATEWAY_AUTH_DISABLED", "true")

# 第二步：import app
from app import auth as _auth  # noqa: E402

# 第三步：显式调用 init_auth（必须在 TestClient 进入 lifespan 之前）
_auth.init_auth()

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient，会话级复用。"""
    from app.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def _dev_auth_headers():
    """所有测试都注入 dev bypass token header（即使是 unit 测试也安全）。"""
    return {"Authorization": "Bearer dev-bypass"}