"""Single source of runtime settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

DEFAULT_ORIGINS = ("http://localhost:8000", "http://127.0.0.1:8000")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Process configuration; env vars are the only inputs."""

    environment: str = "development"
    api_key: Optional[str] = None
    allowed_origins: Tuple[str, ...] = DEFAULT_ORIGINS
    rate_limit_per_minute: int = 600
    log_level: str = "INFO"
    metadata_path: str = "models/metadata.json"
    model_path: str = "models/fraud_detector_lgbm.pkl"
    pipeline_path: str = "models/feature_pipeline.pkl"
    explainer_path: str = "models/shap_explainer.pkl"
    predictor_cache_size: int = 10_000
    max_accounts: int = 50_000
    max_events_per_account: int = 2_000
    # Comma-separated list of trusted proxy IPs/CIDRs; "*" trusts all (use only behind
    # a known ingress that sets X-Forwarded-For). Empty disables proxy header parsing.
    trusted_proxies: Tuple[str, ...] = ()

    @property
    def auth_required(self) -> bool:
        """A configured key turns auth on; production refuses to run without one."""
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        environment = os.getenv("APP_ENV", "development").strip().lower()
        api_key = os.getenv("API_KEY", "").strip() or None
        origins = tuple(
            origin.strip().rstrip("/")
            for origin in os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",")
            if origin.strip()
        )
        try:
            limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "600"))
        except ValueError as exc:
            raise RuntimeError("RATE_LIMIT_PER_MINUTE must be an integer") from exc

        max_accounts = _env_int("MAX_ACCOUNTS", 50_000)
        max_events = _env_int("MAX_EVENTS_PER_ACCOUNT", 2_000)
        cache_size = _env_int("PREDICTOR_CACHE_SIZE", 10_000)
        trusted_proxies = tuple(_env_list("TRUSTED_PROXIES", []))

        if limit < 1:
            raise RuntimeError("RATE_LIMIT_PER_MINUTE must be at least 1")
        if max_accounts < 1 or max_events < 1 or cache_size < 1:
            raise RuntimeError(
                "MAX_ACCOUNTS, MAX_EVENTS_PER_ACCOUNT and PREDICTOR_CACHE_SIZE must be at least 1"
            )
        if not origins:
            raise RuntimeError("ALLOWED_ORIGINS must list at least one origin")

        if environment == "production":
            if not api_key:
                raise RuntimeError("API_KEY must be set when APP_ENV=production")
            if "*" in origins:
                raise RuntimeError("Wildcard CORS is not allowed when APP_ENV=production")

        return cls(
            environment=environment,
            api_key=api_key,
            allowed_origins=origins,
            rate_limit_per_minute=limit,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            max_accounts=max_accounts,
            max_events_per_account=max_events,
            predictor_cache_size=cache_size,
            trusted_proxies=trusted_proxies,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings; tests call ``get_settings.cache_clear()`` after changing env."""
    return Settings.from_env()
