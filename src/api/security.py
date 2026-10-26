"""API-key authentication for scoring and state-changing routes."""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

from src.config import get_settings


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject requests without the configured key; no key configured means dev mode."""
    settings = get_settings()
    if not settings.auth_required:
        return
    if not secrets.compare_digest(x_api_key or "", settings.api_key or ""):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
