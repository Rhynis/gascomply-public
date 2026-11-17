"""Tests for the effective-date filter (build plan A5.2, A6 Phase 3.3, issue #89).

A5.2's in-force rule, applied in ``ChunkRepository`` (``_IS_CURRENT``, shared by ``search`` and
``score`` so the two cannot drift apart): a chunk is superseded when its own ``status`` is not
``in_force``, or its document is the source of an *approved* ``REPLACED_BY`` edge. Both queries
drop a superseded chunk unless ``include_history=True``, which keeps it and reports
``replaced_by``. A *pending* ``REPLACED_BY`` does neither -- it only sets
``possibly_superseded`` -- the same rule ``GraphTraversalService`` already applies to units.

This file also proves the wrinkle the issue calls out explicitly: a stub (``status='unknown'``,
inserted by ``DocumentRepository.insert_stubs``, no ``units`` or ``chunks`` rows ever produced for
it) cannot leak text through this filter, because it never has a chunk to filter in the first
place -- asserted here, and again on the real corpus in the PR body, rather than assumed.

Needs a live Postgres with pgvector (``TEST_DATABASE_URL``). No Ollama: vectors are hand-built
one-hot vectors, as in ``tests/test_retrieve_vector.py``; only the filter, not real similarity,
is under test here.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.repositories.chunks import CHUNKS_EMBEDDING_DIMENSION, ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.edges import EdgeRepository
from app.repositories.units import UnitRepository
from app.schemas.chunk import Chunk
from app.schemas.document import DocStatus, DocType, Document, DocumentStub
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from app.schemas.unit import Unit, UnitType

OLD_DOC = "doc-old"
NEW_DOC = "doc-new"


def _document(doc_id: str, **overrides: Any) -> Document:
    fields: dict[str, Any] = {
        "id": doc_id,
        "number": "87/2018/NĐ-CP",
        "title_vi": "Nghị định về kinh doanh khí",
        "doc_type": DocType.DECREE,
        "issued": date(2018, 6, 15),
        "effective": date(2018, 8, 1),
        "status": DocStatus.IN_FORCE,
        "path": f"corpus/{doc_id}.html",
    }
    fields.update(overrides)
    return Document(**fields)


def _unit(doc_id: str, dieu_no: int) -> Unit:
    return Unit(
        id=f"{doc_id}#dieu-{dieu_no}",
        document_id=doc_id,
        unit_type=UnitType.ARTICLE,
        dieu_no=dieu_no,
        body=f"Điều {dieu_no}.",
        ordinal=dieu_no,
    )


def _chunk(doc_id: str, dieu_no: int, status: DocStatus = DocStatus.IN_FORCE) -> Chunk:
    return Chunk(
        unit_id=f"{doc_id}#dieu-{dieu_no}",
        document_id=doc_id,
        dieu_no=dieu_no,
        effective=date(2018, 8, 1),
        status=status,
        body=f"Điều {dieu_no}.",
    )


def _unit_vector(hot_index: int, dimension: int = CHUNKS_EMBEDDING_DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    vector[hot_index] = 1.0
    return vector


def _edge(src_id: str, dst_id: str, review_status: ReviewStatus) -> Edge:
    return Edge(
        src_id=src_id,
        dst_id=dst_id,
        edge_type=EdgeType.REPLACED_BY,
        review_status=review_status,
        source=EdgeSource.CURATED,
    )


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


def _search(database_url: str, **kwargs: Any) -> list[tuple[str, tuple[str, ...], bool]]:
    """Run ``search`` and return ``[(unit_id, replaced_by, possibly_superseded)]``."""
    results: list[tuple[str, tuple[str, ...], bool]] = []

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            matches = await ChunkRepository(session).search(_unit_vector(0), limit=50, **kwargs)
        results.extend((m.unit_id, m.replaced_by, m.possibly_superseded) for m in matches)

    _run(database_url, body)
    return results


# --- status != in_force -----------------------------------------------------------------------


def test_search_drops_a_chunk_of_an_expired_document(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document(OLD_DOC, status=DocStatus.EXPIRED))
            await UnitRepository(session).upsert_many([_unit(OLD_DOC, 1)])
            await ChunkRepository(session).upsert_many(
                [_chunk(OLD_DOC, 1, status=DocStatus.EXPIRED)], [_unit_vector(0)]
            )

    _run(migrated_db, body)

    assert _search(migrated_db) == []
    [(unit_id, *_rest)] = _search(migrated_db, include_history=True)
    assert unit_id == f"{OLD_DOC}#dieu-1"


def test_search_keeps_a_chunk_of_an_in_force_document(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document(NEW_DOC))
            await UnitRepository(session).upsert_many([_unit(NEW_DOC, 1)])
            await ChunkRepository(session).upsert_many([_chunk(NEW_DOC, 1)], [_unit_vector(0)])

    _run(migrated_db, body)

    [(unit_id, replaced_by, possibly_superseded)] = _search(migrated_db)
    assert unit_id == f"{NEW_DOC}#dieu-1"
    assert replaced_by == ()
    assert possibly_superseded is False


# --- approved REPLACED_BY ----------------------------------------------------------------------


def test_search_drops_a_chunk_of_a_document_with_an_approved_replaced_by_edge(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document(OLD_DOC))
            await DocumentRepository(session).upsert(_document(NEW_DOC))
            await UnitRepository(session).upsert_many([_unit(OLD_DOC, 1)])
            await ChunkRepository(session).upsert_many([_chunk(OLD_DOC, 1)], [_unit_vector(0)])
            await EdgeRepository(session).upsert_many(
                [_edge(OLD_DOC, NEW_DOC, ReviewStatus.APPROVED)]
            )

    _run(migrated_db, body)

    assert _search(migrated_db) == []
    [(unit_id, replaced_by, _)] = _search(migrated_db, include_history=True)
    assert unit_id == f"{OLD_DOC}#dieu-1"
    assert replaced_by == (NEW_DOC,)


def test_search_keeps_but_flags_a_chunk_with_a_pending_replaced_by_edge(migrated_db: str):
    # A pending edge is a machine's guess, not a human decision: it must not silently hide law
    # that a person has not confirmed is retired (the same rule graph_traversal applies to units).
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document(OLD_DOC))
            await DocumentRepository(session).upsert(_document(NEW_DOC))
            await UnitRepository(session).upsert_many([_unit(OLD_DOC, 1)])
            await ChunkRepository(session).upsert_many([_chunk(OLD_DOC, 1)], [_unit_vector(0)])
            await EdgeRepository(session).upsert_many(
                [_edge(OLD_DOC, NEW_DOC, ReviewStatus.PENDING)]
            )

    _run(migrated_db, body)

    [(unit_id, replaced_by, possibly_superseded)] = _search(migrated_db)
    assert unit_id == f"{OLD_DOC}#dieu-1"
    assert replaced_by == ()  # not excluded: a pending edge never counts as "replaced"
    assert possibly_superseded is True


def test_search_drops_a_chunk_replaced_by_more_than_one_document_and_lists_them_all(
    migrated_db: str,
):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document(OLD_DOC))
            await DocumentRepository(session).upsert(_document("doc-new-a"))
            await DocumentRepository(session).upsert(_document("doc-new-b"))
            await UnitRepository(session).upsert_many([_unit(OLD_DOC, 1)])
            await ChunkRepository(session).upsert_many([_chunk(OLD_DOC, 1)], [_unit_vector(0)])
            await EdgeRepository(session).upsert_many(
                [
                    _edge(OLD_DOC, "doc-new-a", ReviewStatus.APPROVED),
                    _edge(OLD_DOC, "doc-new-b", ReviewStatus.APPROVED),
                ]
            )

    _run(migrated_db, body)

    [(_, replaced_by, _superseded)] = _search(migrated_db, include_history=True)
    assert replaced_by == ("doc-new-a", "doc-new-b")


def test_a_rejected_replaced_by_edge_does_not_supersede_a_chunk(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document(OLD_DOC))
            await DocumentRepository(session).upsert(_document(NEW_DOC))
            await UnitRepository(session).upsert_many([_unit(OLD_DOC, 1)])
            await ChunkRepository(session).upsert_many([_chunk(OLD_DOC, 1)], [_unit_vector(0)])
            await EdgeRepository(session).upsert_many(
                [_edge(OLD_DOC, NEW_DOC, ReviewStatus.REJECTED)]
            )

    _run(migrated_db, body)

    [(unit_id, replaced_by, possibly_superseded)] = _search(migrated_db)
    assert unit_id == f"{OLD_DOC}#dieu-1"
    assert replaced_by == ()
    assert possibly_superseded is False


# --- ChunkRepository.score applies the same filter ----------------------------------------------


def test_score_also_drops_a_superseded_chunk(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document(OLD_DOC, status=DocStatus.EXPIRED))
            await UnitRepository(session).upsert_many([_unit(OLD_DOC, 1)])
            await ChunkRepository(session).upsert_many(
                [_chunk(OLD_DOC, 1, status=DocStatus.EXPIRED)], [_unit_vector(0)]
            )
            matches = await ChunkRepository(session).score(_unit_vector(0), [f"{OLD_DOC}#dieu-1"])
            assert matches == []
            history_matches = await ChunkRepository(session).score(
                _unit_vector(0), [f"{OLD_DOC}#dieu-1"], include_history=True
            )
        assert [m.unit_id for m in history_matches] == [f"{OLD_DOC}#dieu-1"]

    _run(migrated_db, body)


# --- the stub wrinkle: a retired document holds no chunk to filter in the first place -----------


def test_a_stub_never_has_a_chunk_by_construction(migrated_db: str):
    # cited-136-2020-nd-cp on the real corpus: status='unknown', no units, no chunks. Asserted
    # here on a seeded stub, and again on the real corpus in the PR body (a live SQL count).
    stub = DocumentStub(number="136/2020/NĐ-CP")

    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).insert_stubs([stub])
            row = (
                await session.execute(
                    text(
                        "SELECT (SELECT count(*) FROM units WHERE document_id = :id) AS units, "
                        "(SELECT count(*) FROM chunks WHERE document_id = :id) AS chunks, "
                        "(SELECT status FROM documents WHERE id = :id) AS status"
                    ),
                    {"id": stub.id},
                )
            ).one()
        assert (row.units, row.chunks, row.status) == (0, 0, "unknown")

    _run(migrated_db, body)


def test_a_stub_status_is_excluded_by_the_same_in_force_rule_a_chunk_would_be(migrated_db: str):
    # The rule (status != 'in_force' supersedes) would exclude a stub's chunk if it had one; a
    # unit of the same status, seeded directly, proves the rule covers 'unknown' too.
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document(OLD_DOC, status=DocStatus.UNKNOWN))
            await UnitRepository(session).upsert_many([_unit(OLD_DOC, 1)])
            await ChunkRepository(session).upsert_many(
                [_chunk(OLD_DOC, 1, status=DocStatus.UNKNOWN)], [_unit_vector(0)]
            )

    _run(migrated_db, body)

    assert _search(migrated_db) == []
