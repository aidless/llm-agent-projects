"""Authentication tests for the sandbox.

These tests verify the production auth path. They run in a fresh process
state where SANDBOX_AUTH_DISABLED is NOT set and SANDBOX_AUTH_TOKEN IS set,
then restore the test-suite default at teardown.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def auth_client(monkeypatch):
    """Build a TestClient with production auth enabled.

    The dependency-injected app.main.app reads env at request time, so
    setting these before any request hits works.
    """
    monkeypatch.delenv("SANDBOX_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("SANDBOX_AUTH_TOKEN", "test-token-abc123")

    # Reload app to ensure env changes propagate cleanly
    from app.main import app
    return TestClient(app)


@pytest.fixture
def no_auth_client(monkeypatch):
    """Build a TestClient with NO token configured and dev-bypass OFF.

    The auth dependency should refuse (503) rather than silently allow.
    """
    monkeypatch.delenv("SANDBOX_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("SANDBOX_AUTH_TOKEN", raising=False)

    from app.main import app
    return TestClient(app)


class TestProductionAuth:
    """Verify the auth path when SANDBOX_AUTH_TOKEN is configured."""

    def test_no_header_returns_401(self, auth_client):
        r = auth_client.post(
            "/api/v1/execute",
            json={"code": "print(1)", "policy_name": "medium"},
        )
        assert r.status_code == 401
        assert "Bearer" in r.json()["detail"]

    def test_wrong_token_returns_401(self, auth_client):
        r = auth_client.post(
            "/api/v1/execute",
            json={"code": "print(1)", "policy_name": "medium"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert r.status_code == 401

    def test_correct_token_passes(self, auth_client):
        r = auth_client.post(
            "/api/v1/execute",
            json={"code": "print('hello')", "policy_name": "medium"},
            headers={"Authorization": "Bearer test-token-abc123"},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_get_execution_requires_auth(self, auth_client):
        r = auth_client.get("/api/v1/execute/some-fake-id")
        assert r.status_code == 401

    def test_health_endpoints_do_not_require_auth(self, auth_client):
        """Health checks should remain public for load balancers / k8s probes."""
        for path in ("/", "/health", "/ready"):
            r = auth_client.get(path)
            assert r.status_code == 200, f"{path} should be public, got {r.status_code}"


class TestNoTokenConfigured:
    """Verify fail-safe behavior when no token is set at all."""

    def test_unconfigured_refuses_with_503(self, no_auth_client):
        r = no_auth_client.post(
            "/api/v1/execute",
            json={"code": "print(1)", "policy_name": "medium"},
        )
        assert r.status_code == 503
        # Should explain what's needed
        detail = r.json()["detail"].lower()
        assert "sandbox_auth_token" in detail or "sanbox_auth_disabled" in detail

    def test_root_reports_auth_state(self, no_auth_client):
        r = no_auth_client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["auth"]["token_configured"] is False
        assert body["auth"]["dev_bypass"] is False