"""Apply pending SQL migrations (build plan Part E, A6 Phase 1.5b).

``python -m app.core.migrate [--database-url URL]``

Applies every ``.sql`` file in ``backend/migrations/``, in filename order,
that ``schema_migrations`` does not already record as applied.

**Never called from application startup.** BUILD_PLAN_FOUNDATION.md Part E is
explicit: migrations are the owner's deliberate action, not something a
process runs on boot -- a schema change should never happen as a side effect
of an app instance starting.

Raw SQL files, not a migration framework: one initial migration does not earn
Alembic, its config file, and its dependency tree (C12). Alembic is the
correct upgrade path the moment a *second* migration needs ordering against
existing data or a rollback path -- not before.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import get_settings
from app.core.db import create_engine
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"

_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


async def apply_pending_migrations(
    engine: AsyncEngine, migrations_dir: Path = MIGRATIONS_DIR
) -> list[str]:
    """Apply every ``.sql`` file in ``migrations_dir`` not yet recorded.

    Args:
        engine: Engine to run the migrations against.
        migrations_dir: Directory of ``.sql`` files, applied in filename sort
            order -- hence the ``NNN_name.sql`` naming convention.

    Returns:
        Filenames applied during this call, in the order applied. An empty
        list means every migration was already applied.
    """
    async with engine.begin() as conn:
        await conn.execute(text(_TRACKING_TABLE_SQL))
        rows = await conn.execute(text("SELECT filename FROM schema_migrations"))
        already_applied = {row[0] for row in rows}

    applied: list[str] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        if path.name in already_applied:
            continue
        await _apply_one(engine, path)
        applied.append(path.name)
        logger.info("Applied migration %s", path.name)
    return applied


async def _apply_one(engine: AsyncEngine, path: Path) -> None:
    """Apply one migration file and record it, in a single transaction."""
    statements = _split_statements(path.read_text(encoding="utf-8"))
    async with engine.begin() as conn:
        for statement in statements:
            await conn.execute(text(statement))
        await conn.execute(
            text("INSERT INTO schema_migrations (filename) VALUES (:filename)"),
            {"filename": path.name},
        )


def _split_statements(sql: str) -> list[str]:
    """Split a migration file into individual statements.

    Some drivers -- asyncpg included, via SQLAlchemy's async engine -- do not
    reliably execute multiple statements passed in one call, and Part E calls
    for one statement per executed block when that is the case. Full-line
    ``--`` comments are dropped first (an ordinary sentence in a comment can
    contain a semicolon of its own, which a naive split would mistake for a
    statement terminator), then what remains is split on ``;``. This is
    enough for this project's migrations, which contain no semicolons inside
    a string literal, a dollar-quoted block, or a trailing same-line comment.
    """
    without_comments = "\n".join(
        line for line in sql.splitlines() if not line.strip().startswith("--")
    )
    return [statement.strip() for statement in without_comments.split(";") if statement.strip()]


async def _amain(database_url: str) -> list[str]:
    """Create an engine, apply pending migrations, and dispose of the engine."""
    engine = create_engine(database_url)
    try:
        return await apply_pending_migrations(engine)
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: apply pending migrations against the configured database.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        ``0`` on success, ``1`` if no database is configured.
    """
    configure_logging()
    args = _parse_args(argv)
    if not args.database_url:
        logger.error("No database configured: set DATABASE_URL or pass --database-url.")
        return 1

    applied = asyncio.run(_amain(args.database_url))
    if applied:
        logger.info("Applied %d migration(s): %s", len(applied), ", ".join(applied))
    else:
        logger.info("No pending migrations.")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m app.core.migrate",
        description="Apply pending SQL migrations. Never run this from application startup.",
    )
    parser.add_argument(
        "--database-url",
        default=get_settings().database_url,
        help="Postgres URL to migrate (default: the DATABASE_URL setting/env var).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
