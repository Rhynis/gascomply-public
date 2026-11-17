"""Shared FastAPI dependencies for the API routers (build plan A6 Phase 3.7, issue #90).

The engine, session factory, embedder and LLM provider are built once, in
``app.main.create_app``, and stashed on ``app.state`` -- the same lifetime a
process-wide singleton would have, but owned by the app object so a test that
builds its own app (``tests/conftest.py``) gets its own, never one left over from
another test.
"""

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.embedders.base import Embedder
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def require_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    """Return the process's session factory, set on ``app.state`` when a database is configured.

    A plain function, not itself a ``Depends`` target: ``/ask`` (issue #90) calls it directly,
    *after* the emergency check has already returned, so a database connection is never
    opened for a question that never needed one. ``/doc/{id}`` uses it through ``get_session``
    below instead, since a document lookup has no such ordering to protect.

    Raises:
        HTTPException: 503, if ``Settings.database_url`` was never given.
    """
    session_factory: async_sessionmaker[AsyncSession] | None = getattr(
        request.app.state, "session_factory", None
    )
    if session_factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No database is configured."
        )
    return session_factory


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a database session for one request, its own transaction, committed on success."""
    session_factory = require_session_factory(request)
    async with session_factory() as session, session.begin():
        yield session


def get_embedder(request: Request) -> Embedder:
    """Return the process's configured ``Embedder`` (mock or real, by ``Settings.embedder``)."""
    return request.app.state.embedder  # type: ignore[no-any-return]


def get_llm_provider(request: Request) -> LLMProvider:
    """Return the process's configured ``LLMProvider`` (mock or real, ``Settings.llm_provider``)."""
    return request.app.state.llm_provider  # type: ignore[no-any-return]


SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
LLMProviderDep = Annotated[LLMProvider, Depends(get_llm_provider)]
