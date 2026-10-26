"""Auth, CORS configuration and rate-limit tests."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.ratelimit import RateLimitMiddleware, SlidingWindowLimiter
from src.api.security import require_api_key
from src.config import get_settings

PRODUCTION = {"APP_ENV": "production", "API_KEY": "s3cret"}


def _settings(monkeypatch, **env):
    for key in ("APP_ENV", "API_KEY", "ALLOWED_ORIGINS", "RATE_LIMIT_PER_MINUTE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def test_production_refuses_to_start_without_api_key(monkeypatch):
    with pytest.raises(RuntimeError, match="API_KEY"):
        _settings(monkeypatch, APP_ENV="production")


def test_production_rejects_wildcard_cors(monkeypatch):
    with pytest.raises(RuntimeError, match="Wildcard"):
        _settings(monkeypatch, **PRODUCTION, ALLOWED_ORIGINS="*")


def test_settings_parse_origins_and_rate_limit(monkeypatch):
    settings = _settings(
        monkeypatch,
        **PRODUCTION,
        ALLOWED_ORIGINS="https://a.example, https://b.example/",
        RATE_LIMIT_PER_MINUTE="30",
    )

    assert settings.allowed_origins == ("https://a.example", "https://b.example")
    assert settings.rate_limit_per_minute == 30
    assert settings.auth_required


def test_api_key_is_enforced_only_when_configured(monkeypatch):
    _settings(monkeypatch)
    require_api_key(None)  # development: open

    _settings(monkeypatch, API_KEY="s3cret")
    require_api_key("s3cret")
    for presented in (None, "wrong", ""):
        with pytest.raises(HTTPException) as error:
            require_api_key(presented)
        assert error.value.status_code == 401


def test_protected_route_rejects_missing_key(monkeypatch):
    _settings(monkeypatch, **PRODUCTION, ALLOWED_ORIGINS="http://localhost:8000")
    from src.api.main import app

    client = TestClient(app)
    payload = {"card1": 9500, "addr1": 299, "TransactionAmt": 120.0}

    assert client.post("/predict", json=payload).status_code == 401
    allowed = client.post("/predict", json=payload, headers={"X-API-Key": "s3cret"})
    assert allowed.status_code == 200
    assert client.get("/metrics").status_code == 200  # read-only routes stay public


def test_limiter_allows_then_blocks_and_recovers():
    now = [0.0]
    limiter = SlidingWindowLimiter(limit=2, window_seconds=60, clock=lambda: now[0])

    assert limiter.allow("a")
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")

    now[0] = 61.0
    assert limiter.allow("a")


def test_rate_limit_middleware_returns_429():
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limit=2)

    @app.get("/ping")
    def ping() -> dict:
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    blocked = client.get("/ping")

    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "60"
