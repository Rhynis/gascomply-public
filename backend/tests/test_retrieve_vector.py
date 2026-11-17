"""Tests for vector retrieval (build plan A6 Phase 3.1, issue #89).

``ChunkRepository.search`` is exact cosine top-k (no ANN index, per ADR 0001 -- migration 003
dropped it after it silently returned the wrong top hit at this corpus's size). As in
``tests/test_search_recall.py``, every vector here is hand-constructed with known geometry (a
one-hot vector's cosine distance from another is computable by hand: 0 for the same index, 1 for
any other), so what is under test is that the query returns the true ranking -- proven, not
assumed from a real embedding, which the mock test database has no semantics for.

Needs a live Postgres with pgvector: set ``TEST_DATABASE_URL``. No Ollama: every test constructs
its own vectors, and ``Retriever`` is exercised with ``MockEmbedder`` only for the plumbing
(question -> embedding -> repository call), never for real similarity.

No ``pytest-asyncio`` (C12): each test drives its own ``asyncio.run()``, matching
``tests/test_traversal.py`` and ``tests/test_search_recall.py``.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.embedders.mock import MockEmbedder
from app.repositories.chunks import CHUNKS_EMBEDDING_DIMENSION, ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.units import UnitRepository
from app.schemas.chunk import Chunk
from app.schemas.document import DocStatus, DocType, Document
from app.schemas.retrieval import RetrievalConfig
from app.schemas.unit import Unit, UnitType
from app.services.retriever import Retriever

DOC_ID = "gas-decree-87-2018"


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
    vector = [0.0] * dimension
    vector[hot_index] = 1.0
    return vector


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


async def _seed(engine: AsyncEngine, chunks_and_vectors: list[tuple[Chunk, list[float]]]) -> None:
    session_factory = create_session_factory(engine)
    async with session_scope(session_factory) as session:
        await DocumentRepository(session).upsert(_document())
        units = [_unit(chunk.dieu_no) for chunk, _ in chunks_and_vectors if chunk.dieu_no]
        await UnitRepository(session).upsert_many(units)
        chunks = [chunk for chunk, _ in chunks_and_vectors]
        vectors = [vector for _, vector in chunks_and_vectors]
        await ChunkRepository(session).upsert_many(chunks, vectors)


def _search(database_url: str, query_vector: list[float], **kwargs: Any) -> list[tuple[str, float]]:
    """Run one ``ChunkRepository.search`` and return ``[(unit_id, similarity)]``, ranked order."""
    results: list[tuple[str, float]] = []

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            matches = await ChunkRepository(session).search(query_vector, **kwargs)
        results.extend((match.unit_id, match.similarity) for match in matches)

    _run(database_url, body)
    return results


# --- exact top-k ranking, by hand-computed geometry ------------------------------------------


def test_search_returns_the_true_top_1(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(
            engine,
            [
                (_chunk(1, body="orthogonal"), _unit_vector(1)),
                (_chunk(2, body="identical"), _unit_vector(0)),
                (_chunk(3, body="opposite"), [-x for x in _unit_vector(0)]),
            ],
        )

    _run(migrated_db, body)
    ranked = _search(migrated_db, _unit_vector(0), limit=10)

    assert [unit_id for unit_id, _ in ranked] == [
        f"{DOC_ID}#dieu-2",
        f"{DOC_ID}#dieu-1",
        f"{DOC_ID}#dieu-3",
    ]


def test_search_similarity_matches_hand_computed_cosine(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(engine, [(_chunk(1, body="identical"), _unit_vector(0))])

    _run(migrated_db, body)
    [(_, similarity)] = _search(migrated_db, _unit_vector(0), limit=10)

    assert similarity == pytest.approx(1.0, abs=1e-6)


def test_search_finds_the_true_top_1_among_many_distractors(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        distractors = [
            (_chunk(index, body=f"distractor {index}"), _unit_vector(index))
            for index in range(1, 21)
        ]
        true_match = (_chunk(99, body="the true nearest neighbor"), _unit_vector(0))
        await _seed(engine, [*distractors, true_match])

    _run(migrated_db, body)
    [(top, _), *_] = _search(migrated_db, _unit_vector(0), limit=1)

    assert top == f"{DOC_ID}#dieu-99"


# --- limit ------------------------------------------------------------------------------------


def test_search_respects_limit(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        chunks = [(_chunk(i, body=f"c{i}"), _unit_vector(i)) for i in range(1, 6)]
        await _seed(engine, chunks)

    _run(migrated_db, body)

    assert len(_search(migrated_db, _unit_vector(1), limit=3)) == 3
    assert len(_search(migrated_db, _unit_vector(1), limit=100)) == 5


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_search_with_a_non_positive_limit_returns_nothing_and_does_not_query(
    migrated_db: str, limit: int
):
    async def body(engine: AsyncEngine) -> None:
        await _seed(engine, [(_chunk(1), _unit_vector(0))])

    _run(migrated_db, body)

    assert _search(migrated_db, _unit_vector(0), limit=limit) == []


def test_search_on_an_empty_table_returns_nothing(migrated_db: str):
    assert _search(migrated_db, _unit_vector(0), limit=5) == []


# --- dimension validation -----------------------------------------------------------------------


def test_search_rejects_a_query_vector_of_the_wrong_dimension(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            with pytest.raises(ValueError, match="dimension"):
                await ChunkRepository(session).search([0.1, 0.2], limit=5)

    _run(migrated_db, body)


# --- document_ids scoping (used by the supersession successor search) -------------------------


def test_search_scoped_to_document_ids_ignores_other_documents(migrated_db: str):
    other_doc = "pccc-decree-105-2025"

    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await DocumentRepository(session).upsert(_document(id=other_doc, number="105/2025"))
            await UnitRepository(session).upsert_many(
                [_unit(1), _unit(1, id=f"{other_doc}#dieu-1", document_id=other_doc)]
            )
            await ChunkRepository(session).upsert_many(
                [
                    _chunk(1, body="in gas-decree-87-2018"),
                    _chunk(
                        1,
                        body="in pccc-decree-105-2025",
                        unit_id=f"{other_doc}#dieu-1",
                        document_id=other_doc,
                    ),
                ],
                [_unit_vector(0), _unit_vector(0)],
            )

    _run(migrated_db, body)
    ranked = _search(migrated_db, _unit_vector(0), limit=10, document_ids=[other_doc])

    assert [unit_id for unit_id, _ in ranked] == [f"{other_doc}#dieu-1"]


def test_search_with_no_document_ids_searches_the_whole_corpus(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(engine, [(_chunk(1), _unit_vector(0))])

    _run(migrated_db, body)

    assert len(_search(migrated_db, _unit_vector(0), limit=10, document_ids=[])) == 1


# --- Retriever.vector_search: the thin plumbing above the repository ----------------------------


def test_retriever_vector_search_embeds_the_question_and_forwards_k(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(engine, [(_chunk(i, body=f"c{i}"), [0.0] * 1024) for i in range(1, 4)])
        async with session_scope(create_session_factory(engine)) as session:
            retriever = Retriever(session, MockEmbedder(), RetrievalConfig(k=2))
            matches = await retriever.vector_search("diện tích tối thiểu cửa hàng LPG")
        assert len(matches) == 2

    _run(migrated_db, body)


def test_retriever_vector_search_rejects_a_blank_question(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            retriever = Retriever(session, MockEmbedder())
            with pytest.raises(ValueError, match="blank"):
                await retriever.vector_search("   ")

    _run(migrated_db, body)
