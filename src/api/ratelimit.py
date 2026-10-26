"""Per-client sliding-window rate limiting with no external dependency."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class SlidingWindowLimiter:
    """Allow at most ``limit`` hits per client in a rolling window."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self.clock = clock
        self._hits: Dict[str, Deque[float]] = {}

    def allow(self, client: str) -> bool:
        now = self.clock()
        cutoff = now - self.window_seconds
        hits = self._hits.setdefault(client, deque())
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        # shortcut: client keys live for the process; move to Redis behind a load balancer.
        if len(self._hits) > self.max_clients:
            self._hits.pop(next(iter(self._hits)))
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply the limiter to every HTTP request, keyed by client address."""

    def __init__(self, app, limit: int, window_seconds: float = 60.0) -> None:
        super().__init__(app)
        self.limiter = SlidingWindowLimiter(limit, window_seconds)
        self.window_seconds = window_seconds

    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        if not self.limiter.allow(client):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(int(self.window_seconds))},
            )
        return await call_next(request)
