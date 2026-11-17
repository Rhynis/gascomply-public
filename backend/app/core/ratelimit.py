"""In-process rate limiting for the public endpoints (build plan A6 Phase 3.7, issue #90).

No external dependency: the project pulls in nothing beyond the standard library
and what ``pyproject.toml`` already lists (C12), and a single FastAPI process is the
whole deployment today -- Phase 5.2 decides what a distributed limit needs; this is
a per-process guard against one client's own excess, not a shared quota across
workers.

A fixed window: a client (keyed by IP) may make up to ``limit`` requests inside each
``window_seconds``-second window; a request beyond that gets ``429`` with
``Retry-After``. Not sliding, not token-bucket -- a window boundary can let a short
burst through right at the edge, an accepted trade-off for one counter and one
timestamp per key, needing no lock: a dependency runs to completion with no
``await`` between reading and writing that state, so two concurrent requests on
asyncio's single-threaded event loop can never interleave inside one call.

Each router owns its own ``RateLimiter`` instance, constructed fresh per app
(``app.main.create_app``) rather than as a module-level singleton -- so a test that
builds its own app (``tests/conftest.py``'s ``client`` fixture, one per test) starts
with an empty counter, never carrying over count from an earlier test.
"""

import time

from fastapi import HTTPException, Request, status


class RateLimiter:
    """A fixed-window limiter for one endpoint, keyed by client IP."""

    def __init__(self, *, limit: int, window_seconds: float) -> None:
        """Initialize the limiter.

        Args:
            limit: Maximum requests allowed per key inside one window.
            window_seconds: Window length, in seconds.
        """
        self._limit = limit
        self._window_seconds = window_seconds
        self._counts: dict[str, tuple[int, float]] = {}

    def __call__(self, request: Request) -> None:
        """Count this request against its key's current window; raise once over.

        Args:
            request: The incoming request, used only for the caller's address.

        Raises:
            HTTPException: 429, with a ``Retry-After`` header naming how many
                seconds remain in the current window, once the key's count for it
                reaches ``limit``.
        """
        key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        count, window_start = self._counts.get(key, (0, now))
        if now - window_start >= self._window_seconds:
            count, window_start = 0, now
        if count >= self._limit:
            retry_after = max(0.0, self._window_seconds - (now - window_start))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests.",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
        self._counts[key] = (count + 1, window_start)
