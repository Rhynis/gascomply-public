"""Console entrypoint: ``python -m app``.

Uvicorn appends its own ``Server`` banner after the application has returned, so
it is disabled here rather than in the middleware (Part C3).
"""

import uvicorn

from app.core.config import AppEnv, get_settings


def main() -> None:
    """Run the ASGI server with the hardened defaults."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        # Bind all interfaces: the container platform terminates external traffic.
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env is AppEnv.LOCAL,
        server_header=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
