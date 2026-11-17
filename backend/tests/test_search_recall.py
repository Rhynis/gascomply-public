"""Recall tests for vector search over ``chunks.embedding`` (build plan A6 issue #49).

This defect -- an approximate ivfflat index silently returning the wrong
nearest neighbor at this corpus's size -- was invisible for six phases and
was found only because someone typed a question against the real corpus by
hand. These tests are what stops that from happening silently again.

Needs a live Postgres with pgvector: set ``TEST_DATABASE_URL``. Every test here
depends, directly or transitively, on the ``migrated_db`` fixture, so a
missing ``TEST_DATABASE_URL`` skips them all with one clear reason.

No Ollama here, and none should ever be needed: CI has no Ollama and fails on
any skipped test (issue #26), so a ``skipif(no ollama)`` guard would turn CI
red. Every vector below is hand-constructed with known geometry -- a query
vector and several candidates whose exact cosine distance from it is computed
by hand, not by a model -- which proves the search *query* returns true
nearest neighbors without needing a single real embedding.

No ``pytest-asyncio`` (C12): each test drives its own async body with a
single ``asyncio.run()`` call, matching ``tests/test_repositories.py``.
"""

import asyncio
import math
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.repositories.chunks import CHUNKS_EMBEDDING_DIMENSION, ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.units import UnitRepository
from app.schemas.chunk import Chunk
from app.schemas.document import DocStatus, DocType, Document
from app.schemas.unit import Unit, UnitType

DOC_ID = "gas-decree-87-2018"

_SEARCH_SQL = text(
    "SELECT unit_id, embedding <=> :query_vector AS distance FROM chunks "
    "ORDER BY embedding <=> :query_vector"
).bindparams(bindparam("query_vector", type_=Vector(CHUNKS_EMBEDDING_DIMENSION)))


def _document(**overrides: Any) -> Document:
    fields: dict[str, Any] = {
        "id": DOC_ID,
        "number": "87/2018/NĐ-CP",
        "title_vi": "Nghị định về kinh doanh khí",
        "doc_type": DocType.DECREE,
        "issued": date(2018, 6, 15),
        "effective": date(2018, 8, 1),
        "status": DocStatus.IN_FORCE,
        "path": f"corpus/{DOC_ID}.html",
    }
    fields.update(overrides)
    return Document(**fields)


def _unit(dieu_no: int, **overrides: Any) -> Unit:
    fields: dict[str, Any] = {
        "id": f"{DOC_ID}#dieu-{dieu_no}",
        "document_id": DOC_ID,
        "unit_type": UnitType.ARTICLE,
        "dieu_no": dieu_no,
        "body": f"Điều {dieu_no}.",
        "ordinal": dieu_no,
    }
    fields.update(overrides)
    return Unit(**fields)


def _chunk(dieu_no: int, **overrides: Any) -> Chunk:
    fields: dict[str, Any] = {
        "unit_id": f"{DOC_ID}#dieu-{dieu_no}",
        "document_id": DOC_ID,
        "dieu_no": dieu_no,
        "effective": date(2018, 8, 1),
        "status": DocStatus.IN_FORCE,
        "body": f"Điều {dieu_no}.",
    }
    fields.update(overrides)
    return Chunk(**fields)


def _unit_vector(hot_index: int, dimension: int = CHUNKS_EMBEDDING_DIMENSION) -> list[float]:
    """Return a one-hot vector: 1.0 at ``hot_index``, 0.0 everywhere else.

    Cosine distance between two one-hot vectors is exactly computable by
    hand: identical index -> distance 0 (identical direction); any other
    pair of distinct indices -> distance 1 (orthogonal, similarity 0). This
    is what makes the expected ranking below provable, not just plausible.
    """
    vector = [0.0] * dimension
    vector[hot_index] = 1.0
    return vector


def _scaled_vector(hot_index: int, magnitude: float) -> list[float]:
    """A one-hot vector scaled by ``magnitude`` -- same direction, same cosine distance.

    Proves the ranking is by angle (cosine), not by raw magnitude or
    Euclidean distance -- a vector 100x the query's length but pointing the
    same way must still rank as the closest match.
    """
    vector = _unit_vector(hot_index)
    return [component * magnitude for component in vector]


def _opposite_vector(hot_index: int) -> list[float]:
    """A vector pointing the exact opposite direction from ``_unit_vector(hot_index)``."""
    vector = [0.0] * CHUNKS_EMBEDDING_DIMENSION
    vector[hot_index] = -1.0
    return vector


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    """Create an engine, run ``body(engine)`` to completion, then dispose it."""

    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


async def _seed(engine: AsyncEngine, chunks_and_vectors: list[tuple[Chunk, list[float]]]) -> None:
    """Persist one document, one unit and one chunk per (chunk, vector) pair."""
    session_factory = create_session_factory(engine)
    async with session_scope(session_factory) as session:
        await DocumentRepository(session).upsert(_document())
        units = [_unit(chunk.dieu_no) for chunk, _ in chunks_and_vectors if chunk.dieu_no]
        await UnitRepository(session).upsert_many(units)
        chunks = [chunk for chunk, _ in chunks_and_vectors]
        vectors = [vector for _, vector in chunks_and_vectors]
        await ChunkRepository(session).upsert_many(chunks, vectors)


# --- exact search returns the true nearest neighbor, by hand-computed geometry ---


def test_exact_search_ranks_the_identical_direction_first(migrated_db: str):
    # Query direction: hot at index 0. Candidates, by increasing exact cosine
    # distance from the query: identical (0), orthogonal (1), opposite (2).
    async def body(engine: AsyncEngine) -> None:
        await _seed(
            engine,
            [
                (_chunk(1, body="opposite"), _opposite_vector(0)),
                (_chunk(2, body="orthogonal"), _unit_vector(1)),
                (_chunk(3, body="identical"), _unit_vector(0)),
            ],
        )
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            rows = (await session.execute(_SEARCH_SQL, {"query_vector": _unit_vector(0)})).all()

        ranked_ids = [row.unit_id for row in rows]
        assert ranked_ids == [
            f"{DOC_ID}#dieu-3",  # identical direction -- distance 0
            f"{DOC_ID}#dieu-2",  # orthogonal -- distance 1
            f"{DOC_ID}#dieu-1",  # opposite -- distance 2
        ]
        assert math.isclose(rows[0].distance, 0.0, abs_tol=1e-6)
        assert math.isclose(rows[1].distance, 1.0, abs_tol=1e-6)
        assert math.isclose(rows[2].distance, 2.0, abs_tol=1e-6)

    _run(migrated_db, body)


def test_exact_search_ranks_by_direction_not_by_magnitude(migrated_db: str):
    # A vector 50x the query's length, same direction, must still win over a
    # unit-length vector pointing a different way -- cosine distance is
    # angle-only, unlike a raw Euclidean or dot-product ranking.
    async def body(engine: AsyncEngine) -> None:
        await _seed(
            engine,
            [
                (_chunk(1, body="different direction"), _unit_vector(1)),
                (_chunk(2, body="same direction, scaled up 50x"), _scaled_vector(0, 50.0)),
            ],
        )
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            [top, *_] = (
                await session.execute(_SEARCH_SQL, {"query_vector": _unit_vector(0)})
            ).all()
        assert top.unit_id == f"{DOC_ID}#dieu-2"
        assert math.isclose(top.distance, 0.0, abs_tol=1e-6)

    _run(migrated_db, body)


def test_exact_search_finds_the_true_top_1_among_many_candidates(migrated_db: str):
    # A larger candidate set, one correct nearest neighbor buried among many
    # distractors -- the shape of the real failure this issue fixes: the
    # true answer must win even when it is not the only, or the first,
    # plausible-looking candidate.
    async def body(engine: AsyncEngine) -> None:
        distractors = [
            (_chunk(index, body=f"distractor {index}"), _unit_vector(index))
            for index in range(1, 21)
        ]
        true_match = (_chunk(99, body="the true nearest neighbor"), _unit_vector(0))
        await _seed(engine, [*distractors, true_match])

        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            [top, *_] = (
                await session.execute(_SEARCH_SQL, {"query_vector": _unit_vector(0)})
            ).all()
        assert top.unit_id == f"{DOC_ID}#dieu-99"

    _run(migrated_db, body)


# --- the index itself must stay gone -------------------------------------------


def test_no_approximate_index_exists_on_chunks_embedding(migrated_db: str):
    # The standing guard against silently reintroducing this defect: if a
    # future migration adds an ivfflat/hnsw index back without also
    # reintroducing a recall proof (ADR 0001), this fails loudly.
    async def body(engine: AsyncEngine) -> None:
        async with engine.begin() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT indexdef FROM pg_indexes "
                        "WHERE tablename = 'chunks' AND indexdef ILIKE '%embedding%'"
                    )
                )
            ).all()
        approximate = [
            row.indexdef for row in rows if "ivfflat" in row.indexdef or "hnsw" in row.indexdef
        ]
        assert approximate == [], (
            f"an approximate index exists on chunks.embedding: {approximate} -- "
            "see docs/adr/0001-drop-ivfflat-index.md before reintroducing one"
        )

    _run(migrated_db, body)
