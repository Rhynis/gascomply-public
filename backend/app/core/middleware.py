"""HTTP middleware applied to every response (Part C3 - security headers).

The ASGI server's own ``Server`` banner is suppressed by the entrypoint
(``app/__main__.py``); this middleware only strips one set by the application.
"""

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}
"""Headers added to every response; HSTS is terminated by the platform proxy."""


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach the baseline security headers and drop the ``Server`` banner."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Run the downstream handler and harden its response headers.

        Args:
            request: Incoming request.
            call_next: Downstream ASGI handler.

        Returns:
            The downstream response with security headers applied.
        """
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if "server" in response.headers:
            del response.headers["server"]
        return response
