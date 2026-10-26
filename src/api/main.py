"""FastAPI main application entrypoint with lifespan preloading, CORS and rate limits."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.ratelimit import RateLimitMiddleware
from src.api.routes.metrics import router as metrics_router
from src.api.routes.predict import get_predictor, router as predict_router
from src.api.routes.stream import router as stream_router
from src.config import get_settings
from src.observability import (
    ObservabilityMiddleware,
    configure_logging,
    render_metrics,
    set_gauge,
)

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model artifacts into memory on startup to eliminate cold-start latency."""
    print("[API] Starting up: pre-warming LightGBM booster & SHAP explainer in memory...")
    try:
        predictor = get_predictor()
        if predictor.is_ready:
            print("[API] Preload success: FraudPredictor ready for scoring.")
    except Exception as e:
        print(f"[API] Warning during preload: {e}")
    if settings.auth_required:
        print("[API] X-API-Key authentication enabled for scoring and mutation routes.")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="Real-Time Transaction Fraud Detection API",
    description=(
        "Low-latency payment fraud scoring service with causal account features, "
        "cost-based thresholds, opt-in TreeSHAP explanations and a live operator dashboard."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

# Rate limit first so the CORS middleware wraps it and 429s stay readable by browsers.
app.add_middleware(RateLimitMiddleware, limit=settings.rate_limit_per_minute)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)
# Outermost: times and logs every response, including rate-limited ones.
app.add_middleware(ObservabilityMiddleware)

# Register route modules
app.include_router(predict_router)
app.include_router(metrics_router)
app.include_router(stream_router)


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Service health and readiness probe."""
    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded.",
        )
    return {
        "status": "HEALTHY",
        "service": "Real-Time Transaction Fraud Detection",
        "version": "1.2.0",
        "model_loaded": predictor.is_ready,
        "champion_algorithm": _champion_algorithm(),
        "auth_required": settings.auth_required,
        "sla_target_p95_ms": 25.0,
    }


def _champion_algorithm() -> str:
    """Read the champion label from metadata instead of hardcoding the model name."""
    try:
        with open(settings.metadata_path, encoding="utf-8") as f:
            return str(json.load(f).get("champion_algorithm") or "unknown")
    except (OSError, ValueError, TypeError):
        return "unknown"


@app.get("/prometheus", tags=["Observability"], include_in_schema=False)
def prometheus_metrics() -> Response:
    """Prometheus text exposition; scraped unauthenticated like /health."""
    predictor = get_predictor()
    set_gauge("fraud_model_loaded", 1.0 if predictor.is_ready else 0.0)
    set_gauge("fraud_dedupe_entries", predictor.cache_entries)
    if predictor.pipeline is not None:
        state = predictor.pipeline.aggregator.state_size()
        set_gauge("fraud_accounts_tracked", state["accounts"])
        set_gauge("fraud_state_events", state["events"])
    return Response(render_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8")


# Mount static frontend for instant dashboard presentation
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
