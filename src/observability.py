"""Dependency-free request logging and Prometheus metrics.

The API already exposes JSON model metadata at ``/metrics``; Prometheus text
lives at ``/prometheus`` so the two do not collide. Labels use route templates,
never raw URLs, so cardinality stays bounded.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

_lock = threading.Lock()
_counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
_gauges: Dict[str, float] = {}
# name+labels -> [non-cumulative bucket counts, sum, count]
_histograms: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], List] = {}


def _key(name: str, labels: Optional[Dict[str, str]]):
    return name, tuple(sorted((labels or {}).items()))


def increment(name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
    key = _key(name, labels)
    with _lock:
        _counters[key] = _counters.get(key, 0.0) + value


def set_gauge(name: str, value: float) -> None:
    with _lock:
        _gauges[name] = float(value)


def observe(
    name: str,
    value: float,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    key = _key(name, labels)
    with _lock:
        entry = _histograms.get(key)
        if entry is None:
            entry = _histograms[key] = [[0] * len(REQUEST_BUCKETS), 0.0, 0]
        for index, edge in enumerate(REQUEST_BUCKETS):
            if value <= edge:
                entry[0][index] += 1
                break
        entry[1] += value
        entry[2] += 1


def reset_metrics() -> None:
    with _lock:
        _counters.clear()
        _gauges.clear()
        _histograms.clear()


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(items: Tuple[Tuple[str, str], ...], extra: Optional[Tuple[str, str]] = None) -> str:
    pairs = list(items) + ([extra] if extra else [])
    if not pairs:
        return ""
    return "{" + ",".join(f'{key}="{_escape(value)}"' for key, value in pairs) + "}"


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def render_metrics() -> str:
    """Prometheus text exposition of the in-process registry."""
    with _lock:
        counters = dict(_counters)
        gauges = dict(_gauges)
        histograms = {
            key: [list(buckets), total, count]
            for key, (buckets, total, count) in _histograms.items()
        }

    lines: List[str] = []
    seen: set[str] = set()

    for (name, labels), value in sorted(counters.items()):
        if name not in seen:
            lines.append(f"# TYPE {name} counter")
            seen.add(name)
        lines.append(f"{name}{_labels(labels)} {_number(value)}")

    for name, value in sorted(gauges.items()):
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {_number(value)}")

    for (name, labels), (buckets, total, count) in sorted(histograms.items()):
        if name not in seen:
            lines.append(f"# TYPE {name} histogram")
            seen.add(name)
        cumulative = 0
        for edge, bucket_count in zip(REQUEST_BUCKETS, buckets):
            cumulative += bucket_count
            lines.append(f"{name}_bucket{_labels(labels, ('le', str(edge)))} {cumulative}")
        lines.append(f"{name}_bucket{_labels(labels, ('le', '+Inf'))} {count}")
        lines.append(f"{name}_sum{_labels(labels)} {repr(float(total))}")
        lines.append(f"{name}_count{_labels(labels)} {count}")

    return "\n".join(lines) + ("\n" if lines else "")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, so log shippers do not need a regex."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event:
            payload.update(event)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Install the JSON handler once and return the application logger."""
    logger = logging.getLogger("fraud")
    if any(isinstance(handler.formatter, JsonFormatter) for handler in logger.handlers):
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


request_logger = logging.getLogger("fraud.api")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Log every request and record latency and status counters."""

    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - started
            route = request.scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            labels = {"route": path}
            increment("fraud_requests_total", {"route": path, "status": str(status_code)})
            observe("fraud_request_duration_seconds", elapsed, labels)
            request_logger.info(
                "request",
                extra={
                    "event": {
                        "method": request.method,
                        "path": path,
                        "status": status_code,
                        "duration_ms": round(elapsed * 1000.0, 2),
                        "client": request.client.host if request.client else "unknown",
                    }
                },
            )
