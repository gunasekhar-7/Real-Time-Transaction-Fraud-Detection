"""Per-client sliding-window rate limiting with no external dependency."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


def _get_client_ip(request: Request, trusted_proxies: Optional[list[str]] = None) -> str:
    """Extract the real client IP, respecting proxy headers when configured.

    When behind a trusted proxy (nginx, Traefik, ALB, etc.):
    - X-Real-IP is set by the proxy to the immediate peer (preferred, cannot be
      spoofed by the client if the proxy overwrites it).
    - X-Forwarded-For contains a chain: client, proxy1, proxy2, ... The proxy
      appends the true remote address to the RIGHT. A malicious client can
      prepend fake IPs to the LEFT. We must take the RIGHTMOST IP that is NOT
      in our trusted_proxies list.
    """
    if not request.client:
        return "unknown"

    client_host = request.client.host

    if not trusted_proxies:
        return client_host

    # "*" means trust all proxies (e.g., when the app only ever runs behind
    # a single known ingress that sets X-Forwarded-For)
    if trusted_proxies == ["*"] or client_host in trusted_proxies:
        # 1. X-Real-IP is set by the reverse proxy to the immediate peer IP.
        #    This is the most reliable if the proxy is configured to overwrite it.
        xri = request.headers.get("X-Real-IP")
        if xri:
            return xri.strip()

        # 2. Parse X-Forwarded-For from RIGHT to LEFT, skipping trusted proxies.
        #    The rightmost non-trusted IP is the genuine client.
        xff = request.headers.get("X-Forwarded-For")
        if xff:
            ips = [ip.strip() for ip in xff.split(",")]
            # Iterate from right to left
            for ip in reversed(ips):
                if ip and ip not in trusted_proxies:
                    return ip
            # If all IPs in the chain are trusted (unusual), fall back to leftmost
            if ips:
                return ips[0].strip()

    return client_host


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

    def __init__(
        self,
        app,
        limit: int,
        window_seconds: float = 60.0,
        trusted_proxies: Optional[list[str]] = None,
    ) -> None:
        super().__init__(app)
        self.limiter = SlidingWindowLimiter(limit, window_seconds)
        self.window_seconds = window_seconds
        self.trusted_proxies = trusted_proxies

    async def dispatch(self, request: Request, call_next):
        client = _get_client_ip(request, self.trusted_proxies)
        if not self.limiter.allow(client):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(int(self.window_seconds))},
            )
        return await call_next(request)
