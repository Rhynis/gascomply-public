"""FastAPI application factory and ASGI entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import ask, documents, health
from app.core.config import Settings, get_settings
from app.core.db import create_engine, create_session_factory
from app.core.logging import configure_logging
from app.core.middleware import SecurityHeadersMiddleware
from app.core.ratelimit import RateLimiter
from app.embedders.factory import create_embedder
from app.llm.factory import create_llm_provider
from app.rerankers.factory import create_reranker
from app.services.generator import GENERATION_RETRIEVAL_CONFIG

APP_TITLE = "GasComply API"
APP_DESCRIPTION = (
    "Citation-grounded lookup of Vietnamese LPG-business and fire-safety (PCCC) regulations."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI application.

    Args:
        settings: Settings to run with; the process settings are used when omitted.

    Returns:
        The configured application. ``app.state`` carries the engine (when a database
        is configured), its session factory, the configured embedder, LLM provider and
        reranker (``None`` when reranking is off), the retrieval config ``/ask`` runs with,
        and one ``RateLimiter`` per rate-limited router -- all built once
        here, per app, so every request shares them (``app/api/deps.py``) and every
        app -- so every test -- gets its own counters, never another app's.
    """
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open the database engine (if configured) on startup; dispose it on shutdown.

        No database configured is not a startup failure (``app/core/config.py``'s
        own rule for ``database_url``): ``/ask`` and ``/doc/{id}`` fail per request
        with 503 instead (``app/api/deps.py``'s ``require_session_factory``), so the
        process still starts and ``/health`` still answers.
        """
        engine = create_engine(resolved.database_url) if resolved.database_url else None
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine) if engine else None
        app.state.embedder = create_embedder(resolved)
        app.state.llm_provider = create_llm_provider(resolved)
        app.state.reranker = create_reranker(resolved)
        app.state.retrieval_config = GENERATION_RETRIEVAL_CONFIG.model_copy(
            update={"rerank_top_n": resolved.rerank_top_n}
        )
        app.state.ask_rate_limiter = RateLimiter(
            limit=ask.RATE_LIMIT_PER_MINUTE, window_seconds=60.0
        )
        app.state.doc_rate_limiter = RateLimiter(
            limit=documents.RATE_LIMIT_PER_MINUTE, window_seconds=60.0
        )
        try:
            yield
        finally:
            if engine is not None:
                await engine.dispose()

    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=__version__,
        docs_url="/docs" if resolved.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if resolved.docs_enabled else None,
        lifespan=lifespan,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.include_router(health.router)
    app.include_router(ask.router)
    app.include_router(documents.router)
    return app


app = create_app()
