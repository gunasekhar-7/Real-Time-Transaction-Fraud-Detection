"""API Route handlers for predictions, metrics, and streaming."""

from .predict import router as predict_router
from .metrics import router as metrics_router
from .stream import router as stream_router

__all__ = ["predict_router", "metrics_router", "stream_router"]
