"""Async database engine and session factory (build plan A2, A6 Phase 1.5b).

A plain Postgres image with the pgvector extension is the whole database
requirement for development and CI -- ``pgvector/pgvector:pg16``, run as a
GitHub Actions service container in CI and via ``docker-compose.yml`` locally.
Supabase is a Phase 5.2 production concern only: at that point it is nothing
more than a connection string assigned to ``Settings.database_url``, and
nothing in this module or its callers needs to change.

The engine and session factory are created explicitly by whoever needs them
(``ingest/run.py``, ``app/core/migrate.py``) rather than as a process-wide
singleton, since a short-lived CLI invocation should own and dispose of its
own connections instead of leaking a pool past the run that created it.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)


def create_engine(database_url: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine for ``database_url``.

    Args:
        database_url: A ``postgresql+asyncpg://`` connection string.

    Returns:
        The engine. The caller owns it and must call ``await engine.dispose()``
        when done.
    """
    return create_async_engine(database_url, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to ``engine``.

    Args:
        engine: The engine sessions from this factory will use.

    Returns:
        A callable that creates a new ``AsyncSession`` each time it is invoked.
    """
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Run a block of work in one transaction: commit on success, roll back on error.

    Args:
        session_factory: Factory to create the session from, e.g. from
            ``create_session_factory``.

    Yields:
        A session whose transaction is committed when the block exits
        normally, or rolled back if it raises.
    """
    async with session_factory() as session, session.begin():
        yield session
