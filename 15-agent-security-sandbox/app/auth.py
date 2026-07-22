"""Authentication & authorization helpers.

The sandbox provides bearer-token authentication for sensitive endpoints
(`/api/v1/execute` and `/api/v1/execute/{execution_id}`).

Configuration:
- Set `SANDBOX_AUTH_TOKEN` in the environment to enable auth.
- If unset, the sandbox falls back to **DENY BY DEFAULT** for execute endpoints
  (the previous open behavior was a critical security vulnerability).
- For local development only, set `SANDBOX_AUTH_DISABLED=true` to bypass —
  this triggers a loud warning on every request.
"""

import hmac
import os
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Reusable bearer scheme — auto_error=False so we can return our own 401 message
_bearer_scheme = HTTPBearer(auto_error=False)


def _auth_disabled_for_dev() -> bool:
    """Return True only if dev-mode bypass is explicitly enabled."""
    return os.environ.get("SANDBOX_AUTH_DISABLED", "").lower() == "true"


def _expected_token() -> Optional[str]:
    """Return the configured bearer token, or None if not set."""
    return os.environ.get("SANDBOX_AUTH_TOKEN") or None


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> str:
    """FastAPI dependency that enforces bearer-token authentication.

    Returns the authenticated subject (the token) on success.
    Raises 401 if authentication is required and the token is missing/wrong.

    Three-state behavior:
    - `SANDBOX_AUTH_TOKEN` set & token matches → 200 (subject = token)
    - `SANDBOX_AUTH_TOKEN` set & token missing/wrong → 401
    - `SANDBOX_AUTH_TOKEN` unset:
        - `SANDBOX_AUTH_DISABLED=true` → 200 (with WARNING in logs)
        - else → 503 (refuse to start serving sensitive ops without auth)
    """
    import logging
    logger = logging.getLogger("sandbox.auth")

    expected = _expected_token()

    # State 1: no token configured
    if expected is None:
        if _auth_disabled_for_dev():
            logger.warning(
                "SANDBOX_AUTH_DISABLED=true — sandbox is accepting execute "
                "requests WITHOUT authentication. This MUST NOT be used in "
                "production or any internet-exposed environment."
            )
            return "dev-bypass"
        # Refuse by default — this is the security-critical fix
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Sandbox authentication is not configured. Set "
                "SANDBOX_AUTH_TOKEN (recommended) or SANDBOX_AUTH_DISABLED=true "
                "(development only)."
            ),
        )

    # State 2: token configured — must verify
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials


def generate_dev_token() -> str:
    """Generate a cryptographically secure token suitable for SANDBOX_AUTH_TOKEN."""
    return secrets.token_urlsafe(32)