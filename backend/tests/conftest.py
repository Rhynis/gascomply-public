"""Shared pytest fixtures."""

import asyncio
import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url

from app.core.config import AppEnv, Settings, get_settings
from app.core.db import create_engine
from app.core.migrate import apply_pending_migrations
from app.main import create_app


@pytest.fixture(autouse=True)
def _no_ambient_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test's ``Settings`` blind to a real ``backend/.env`` file (issue #204).

    The clean-clone proof for issue #204 found two tests (``test_ingest_cli.py``,
    ``test_ingest_persistence.py``) failing only because a developer's own ``.env`` -- copied
    from ``.env.example`` exactly as the README's "Local setup" asks -- set
    ``EMBEDDER=bge-m3``/``LLM_PROVIDER=ollama``, and ``ingest.run.main()``'s call to the
    process-wide ``get_settings()`` picked that up instead of the ``mock`` defaults those tests
    assume (see https://github.com/Rhynis/gascomply/issues/204#issuecomment-6945533510). It is
    the same class of bug issue #64 fixed for ``DATABASE_URL``/``TEST_DATABASE_URL`` -- a real
    ``.env`` silently arming behaviour a test never asked for -- just not limited to one field:
    any ``Settings`` field a local ``.env`` happens to set can do this to any test that reads
    ``get_settings()`` directly rather than injecting a ``Settings`` instance.

    Monkeypatching ``Settings.model_config["env_file"]`` to ``None`` makes every plain
    ``Settings()`` call made during a test -- including ``get_settings()``'s own, and any test's
    own ``Settings(...)`` construction that does not pass ``_env_file`` itself -- ignore
    ``backend/.env`` entirely, for every field, not just the one a narrower fixture happens to
    ``delenv``. A test that explicitly passes ``_env_file=...`` is unaffected: that is a
    call-site override, and pydantic-settings lets an explicit constructor argument win over
    ``model_config`` either way. Environment variables set with ``monkeypatch.setenv`` still
    apply -- only the dotenv *file* is turned off, the same thing a real process without a
    ``.env`` already looks like.

    ``get_settings()`` is cached with ``lru_cache``, so the cache is cleared both before and
    after every test: before, so a value some earlier test (or this fixture's own module import)
    already cached never leaks in; after, so the next test file -- and a real process outside
    pytest -- is unaffected by anything this test looked up.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Return deterministic settings for tests, ignoring any local ``.env`` file."""
    return Settings(app_env=AppEnv.CI, log_level="WARNING", _env_file=None)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Return a test client bound to an app built from the test settings."""
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_settings(migrated_db: str) -> Settings:
    """Settings pointed at the migrated test database (issue #90's API tests).

    ``embedder``/``llm_provider`` are left at their own defaults (``mock``), so a
    test using this gets a real database with the same offline-safe mock stack
    every other test runs with, unless it overrides ``app.state`` itself.
    """
    return Settings(
        app_env=AppEnv.CI, log_level="WARNING", database_url=migrated_db, _env_file=None
    )


@pytest.fixture
def db_client(db_settings: Settings) -> Iterator[TestClient]:
    """A test client whose app is backed by the migrated test database.

    Entering this fixture runs the app's ``lifespan``, which builds the engine,
    session factory, embedder and LLM provider onto ``app.state`` -- a test that
    needs a specific double for one of those (e.g. an ``LLMProvider`` stub) sets
    ``db_client.app.state.llm_provider = ...`` after requesting this fixture, and
    before making a request.
    """
    app = create_app(db_settings)
    app.dependency_overrides[get_settings] = lambda: db_settings
    with TestClient(app) as test_client:
        yield test_client


# --- database (build plan A6 Phase 1.5b) ------------------------------------------
#
# Every database-backed test depends on `migrated_db` (directly, or via a
# fixture that itself depends on it), so TEST_DATABASE_URL being unset skips
# them all with one clear, loud reason -- never a silent pass. Each test
# creates and disposes its own engine inside its own `asyncio.run()` call
# rather than sharing one across tests: an asyncpg connection is bound to the
# event loop that created it, and `asyncio.run()` opens a fresh loop every
# call, so a pooled connection from an earlier test's loop cannot be reused
# safely.
#
# TEST_DATABASE_URL, never DATABASE_URL (issue #64): this fixture truncates
# every persistence table on every database-backed test. DATABASE_URL is the
# application's own setting -- configuring the app to point at a real
# database must never be able to arm a fixture that empties it, which is
# exactly what happened to the shared dev database when a hand-off exported
# DATABASE_URL for a one-off script and then ran pytest in the same shell.
# A dedicated variable makes that class of mistake impossible: the app never
# reads TEST_DATABASE_URL, and this fixture never reads DATABASE_URL.

_TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


def _require_test_database(database_url: str) -> None:
    """Raise unless ``database_url`` names a database ending in ``_test``.

    Defence in depth, independent of the ``TEST_DATABASE_URL``/``DATABASE_URL``
    split above: even a correctly-named environment variable could still
    point at the wrong database by mistake, and this is the last check
    before anything destructive runs. Deliberately raises rather than
    skipping -- a skip is silent, and a test run aimed at a real database is
    an error someone must see, not a quietly-empty test report. Parses with
    SQLAlchemy's own ``make_url`` rather than hand-rolling URL parsing, so
    it handles every driver/query-string shape ``make_url`` does.

    Args:
        database_url: The URL a caller is about to run destructive fixtures
            against.

    Raises:
        ValueError: If the database name does not end in ``_test``, naming
            it in the message.
    """
    name = make_url(database_url).database
    if name is None or not name.endswith("_test"):
        msg = (
            f"refusing to run destructive test fixtures against database {name!r} -- "
            "its name must end in '_test' (issue #64)"
        )
        raise ValueError(msg)


@pytest.fixture(scope="session")
def db_url() -> str:
    """The test database URL, or skip every dependent test with a clear reason."""
    if not _TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set -- skipping tests that need a live Postgres")
    return _TEST_DATABASE_URL


@pytest.fixture
def migrated_db(db_url: str) -> str:
    """Apply pending migrations, truncate the persistence tables, return the URL.

    Every database-backed test starts from the same empty, already-migrated
    database, regardless of what an earlier test left behind.
    """
    asyncio.run(_prepare_database(db_url))
    return db_url


async def _prepare_database(database_url: str) -> None:
    """Apply migrations, then truncate every table for test isolation."""
    _require_test_database(database_url)
    engine = create_engine(database_url)
    try:
        await apply_pending_migrations(engine)
        async with engine.begin() as conn:
            await conn.execute(
                text("TRUNCATE documents, units, chunks, entities, edges RESTART IDENTITY CASCADE")
            )
    finally:
        await engine.dispose()
