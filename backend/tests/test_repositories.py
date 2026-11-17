"""Tests for the async persistence repositories (build plan A6 Phase 1.5b, 2.1).

Needs a live Postgres with pgvector: set ``TEST_DATABASE_URL`` (``docker-compose.yml``
starts one locally on port 5433; CI runs a service container). Every test here
depends, directly or transitively, on the ``migrated_db`` fixture, so a missing
``TEST_DATABASE_URL`` skips them all with one clear reason rather than a silent pass.

No ``pytest-asyncio`` (C12): each test drives its own async body with a single
``asyncio.run()`` call, since an asyncpg connection is bound to the event loop
that opened it and ``asyncio.run()`` opens a fresh one every call.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.core.migrate import apply_pending_migrations
from app.repositories.chunks import CHUNKS_EMBEDDING_DIMENSION, ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.edges import EdgeRepository
from app.repositories.units import UnitRepository
from app.schemas.chunk import Chunk
from app.schemas.document import DocStatus, DocType, Document
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from app.schemas.unit import Unit, UnitType

DOC_ID = "gas-decree-87-2018"

_COUNT_DOCUMENTS = text("SELECT count(*) FROM documents")
_COUNT_UNITS = text("SELECT count(*) FROM units")
_COUNT_CHUNKS = text("SELECT count(*) FROM chunks")
_COUNT_EDGES = text("SELECT count(*) FROM edges")


def _document(**overrides: Any) -> Document:
    fields: dict[str, Any] = {
        "id": DOC_ID,
        "number": "87/2018/NĐ-CP",
        "title_vi": "Nghị định về kinh doanh khí",
        "doc_type": DocType.DECREE,
        "issued": date(2018, 6, 15),
        "effective": date(2018, 8, 1),
        "status": DocStatus.IN_FORCE,
        "path": "corpus/gas-decree-87-2018.html",
        "meta": {"note": "test"},
    }
    fields.update(overrides)
    return Document(**fields)


def _article(**overrides: Any) -> Unit:
    fields: dict[str, Any] = {
        "id": f"{DOC_ID}#dieu-1",
        "document_id": DOC_ID,
        "unit_type": UnitType.ARTICLE,
        "dieu_no": 1,
        "heading": "Phạm vi điều chỉnh",
        "body": "Nội dung điều 1.",
        "ordinal": 0,
    }
    fields.update(overrides)
    return Unit(**fields)


def _clause(**overrides: Any) -> Unit:
    fields: dict[str, Any] = {
        "id": f"{DOC_ID}#dieu-1#khoan-1",
        "document_id": DOC_ID,
        "unit_type": UnitType.CLAUSE,
        "dieu_no": 1,
        "khoan_no": 1,
        "parent_id": f"{DOC_ID}#dieu-1",
        "body": "Nội dung khoản 1.",
        "ordinal": 1,
    }
    fields.update(overrides)
    return Unit(**fields)


def _chunk(**overrides: Any) -> Chunk:
    fields: dict[str, Any] = {
        "unit_id": f"{DOC_ID}#dieu-1",
        "document_id": DOC_ID,
        "dieu_no": 1,
        "effective": date(2018, 8, 1),
        "status": DocStatus.IN_FORCE,
        "body": "Điều 1. Phạm vi điều chỉnh\nNội dung điều 1.",
    }
    fields.update(overrides)
    return Chunk(**fields)


def _embedding(seed: float = 0.1) -> list[float]:
    return [seed] * CHUNKS_EMBEDDING_DIMENSION


def _edge(**overrides: Any) -> Edge:
    fields: dict[str, Any] = {
        "src_id": f"{DOC_ID}#dieu-1#khoan-1",
        "dst_id": f"{DOC_ID}#dieu-1",
        "edge_type": EdgeType.BELONGS_TO,
        "review_status": ReviewStatus.APPROVED,
        "source": EdgeSource.STRUCTURAL,
    }
    fields.update(overrides)
    return Edge(**fields)


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    """Create an engine, run ``body(engine)`` to completion, then dispose it."""

    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


# --- documents -----------------------------------------------------------------


def test_document_upsert_replaces_rather_than_duplicates(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document(status=DocStatus.EXPIRED))

        async with engine.begin() as conn:
            count = (await conn.execute(_COUNT_DOCUMENTS)).scalar_one()
            status = (
                await conn.execute(
                    text("SELECT status FROM documents WHERE id = :id"), {"id": DOC_ID}
                )
            ).scalar_one()
        assert count == 1
        assert status == "expired"

    _run(migrated_db, body)


def test_document_meta_round_trips_as_jsonb(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document(meta={"note": "roundtrip", "n": 1}))

        async with engine.begin() as conn:
            meta = (
                await conn.execute(
                    text("SELECT meta FROM documents WHERE id = :id"), {"id": DOC_ID}
                )
            ).scalar_one()
        assert meta == {"note": "roundtrip", "n": 1}

    _run(migrated_db, body)


# --- units -----------------------------------------------------------------------


def test_unit_upsert_many_replaces_rather_than_duplicates(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await UnitRepository(session).upsert_many([_article(), _clause()])
        async with session_scope(session_factory) as session:
            await UnitRepository(session).upsert_many(
                [_article(heading="Updated heading"), _clause()]
            )

        async with engine.begin() as conn:
            count = (await conn.execute(_COUNT_UNITS)).scalar_one()
            heading = (
                await conn.execute(
                    text("SELECT heading FROM units WHERE id = :id"), {"id": f"{DOC_ID}#dieu-1"}
                )
            ).scalar_one()
        assert count == 2
        assert heading == "Updated heading"

    _run(migrated_db, body)


def test_unit_upsert_many_respects_parent_before_child_order(migrated_db: str):
    # The clause's parent_id references the article; inserting in reading
    # order (article, then clause) must not violate the foreign key.
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await UnitRepository(session).upsert_many([_article(), _clause()])

        async with engine.begin() as conn:
            count = (await conn.execute(_COUNT_UNITS)).scalar_one()
        assert count == 2

    _run(migrated_db, body)


def test_list_ids_by_documents_groups_by_document_and_skips_the_rest(migrated_db: str):
    # Used by Retriever._expand_focus (issue #106) to find every real unit nested under a
    # returned chunk: grouped by document, and a document not asked for is left out entirely
    # even though it has units of its own.
    other_doc = "other-decree-2020"

    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await DocumentRepository(session).upsert(_document(id=other_doc))
            await UnitRepository(session).upsert_many([_article(), _clause()])
            await UnitRepository(session).upsert_many(
                [_article(id=f"{other_doc}#dieu-1", document_id=other_doc)]
            )

            found = await UnitRepository(session).list_ids_by_documents([DOC_ID])

        assert set(found[DOC_ID]) == {f"{DOC_ID}#dieu-1", f"{DOC_ID}#dieu-1#khoan-1"}
        assert other_doc not in found

    _run(migrated_db, body)


def test_list_ids_by_documents_with_no_document_ids_makes_no_query(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            found = await UnitRepository(session).list_ids_by_documents([])

        assert found == {}

    _run(migrated_db, body)


# --- unit contexts for grounding (issue #118's own follow-up review) --------------------


def test_get_unit_contexts_includes_every_descendant_when_the_own_body_is_empty(
    migrated_db: str,
):
    """An article with an empty own body must still hand the grounding check its real content.

    Its real content lives entirely in its clauses, the common case on this corpus, so it
    must still hand the grounding check that content, exactly as the generator's own
    retrieved passage already carries it (the PR review: without this, the verifier was
    shown less text for a cited id than the generator was).
    """

    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await UnitRepository(session).upsert_many(
                [
                    _article(body="", heading="Điều kiện"),
                    _clause(body="Nội dung khoản một."),
                    _clause(
                        id=f"{DOC_ID}#dieu-1#khoan-2",
                        khoan_no=2,
                        body="Nội dung khoản hai.",
                        ordinal=2,
                    ),
                ]
            )

            contexts = await DocumentRepository(session).get_unit_contexts((f"{DOC_ID}#dieu-1",))

        context = contexts[f"{DOC_ID}#dieu-1"]
        assert "Nội dung khoản một." in context
        assert "Nội dung khoản hai." in context
        assert "Điều kiện" in context  # the article's own heading, still present too

    _run(migrated_db, body)


def test_get_unit_contexts_descendants_come_in_ordinal_order(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await UnitRepository(session).upsert_many(
                [
                    _article(body=""),
                    _clause(
                        id=f"{DOC_ID}#dieu-1#khoan-2",
                        khoan_no=2,
                        body="hai",
                        ordinal=2,
                    ),
                    _clause(body="một", ordinal=1),
                ]
            )

            contexts = await DocumentRepository(session).get_unit_contexts((f"{DOC_ID}#dieu-1",))

        context = contexts[f"{DOC_ID}#dieu-1"]
        assert context.index("một") < context.index("hai")

    _run(migrated_db, body)


def test_get_appendix_items_returns_only_items_and_subitems(migrated_db: str):
    """A form, an article, and a clause never come back, even if their id is asked for.

    Issue #146's own defense: ``get_appendix_items``'s SQL filters to
    ``unit_type IN ('appendix_item', 'appendix_subitem')`` itself, so the generator's quote
    step can never be handed a form or an article/clause id under the pretence it is an
    appendix item -- regardless of what a caller asks for.
    """

    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await UnitRepository(session).upsert_many(
                [
                    Unit(
                        id=f"{DOC_ID}#phu-luc-7",
                        document_id=DOC_ID,
                        unit_type=UnitType.APPENDIX,
                        appendix_no=7,
                        ordinal=10,
                    ),
                    Unit(
                        id=f"{DOC_ID}#phu-luc-7#muc-15",
                        document_id=DOC_ID,
                        unit_type=UnitType.APPENDIX_ITEM,
                        appendix_no=7,
                        item_no=15,
                        parent_id=f"{DOC_ID}#phu-luc-7",
                        body=(
                            "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg."
                        ),
                        ordinal=11,
                    ),
                    Unit(
                        id=f"{DOC_ID}#phu-luc-7#muc-15#muc-1",
                        document_id=DOC_ID,
                        unit_type=UnitType.APPENDIX_SUBITEM,
                        appendix_no=7,
                        item_no=15,
                        subitem_no=1,
                        parent_id=f"{DOC_ID}#phu-luc-7#muc-15",
                        body="Nội dung mục con.",
                        ordinal=12,
                    ),
                    Unit(
                        id=f"{DOC_ID}#phu-luc-7#mau-pc01",
                        document_id=DOC_ID,
                        unit_type=UnitType.APPENDIX_FORM,
                        appendix_no=7,
                        form_code="pc01",
                        parent_id=f"{DOC_ID}#phu-luc-7",
                        body="Mẫu PC01 nội dung.",
                        ordinal=13,
                    ),
                    _article(),
                    _clause(),
                ]
            )

            rows = await DocumentRepository(session).get_appendix_items(
                (
                    f"{DOC_ID}#phu-luc-7#muc-15",
                    f"{DOC_ID}#phu-luc-7#muc-15#muc-1",
                    f"{DOC_ID}#phu-luc-7#mau-pc01",
                    f"{DOC_ID}#dieu-1",
                    f"{DOC_ID}#dieu-1#khoan-1",
                )
            )

        assert set(rows) == {
            f"{DOC_ID}#phu-luc-7#muc-15",
            f"{DOC_ID}#phu-luc-7#muc-15#muc-1",
        }
        assert rows[f"{DOC_ID}#phu-luc-7#muc-15"].body.startswith("Cơ sở kinh doanh khí đốt")
        assert rows[f"{DOC_ID}#phu-luc-7#muc-15"].appendix_no == 7
        assert rows[f"{DOC_ID}#phu-luc-7#muc-15"].item_no == 15
        # The sub-item's own number ("1" in "15.1") is not a column on `units` at all --
        # `item_no` holds the *parent* item's number for this unit type (issue #146's own
        # finding); the generator reads the sub-item's own number from the id itself.
        assert rows[f"{DOC_ID}#phu-luc-7#muc-15#muc-1"].item_no == 15

    _run(migrated_db, body)


def test_get_appendix_items_with_no_ids_makes_no_query(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            rows = await DocumentRepository(session).get_appendix_items([])

        assert rows == {}

    _run(migrated_db, body)


# --- chunks ------------------------------------------------------------------------


def test_chunk_upsert_many_replaces_rather_than_duplicates(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await UnitRepository(session).upsert_many([_article()])
            await ChunkRepository(session).upsert_many([_chunk()], [_embedding()])
        async with session_scope(session_factory) as session:
            await ChunkRepository(session).upsert_many(
                [_chunk(body="Updated body")], [_embedding(seed=0.2)]
            )

        async with engine.begin() as conn:
            count = (await conn.execute(_COUNT_CHUNKS)).scalar_one()
            body_text = (
                await conn.execute(
                    text("SELECT body FROM chunks WHERE unit_id = :id"), {"id": f"{DOC_ID}#dieu-1"}
                )
            ).scalar_one()
        assert count == 1
        assert body_text == "Updated body"

    _run(migrated_db, body)


def test_chunk_embedding_round_trips_at_the_declared_dimension(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await UnitRepository(session).upsert_many([_article()])
            await ChunkRepository(session).upsert_many([_chunk()], [_embedding()])

        async with engine.begin() as conn:
            dims = (
                await conn.execute(
                    text("SELECT vector_dims(embedding) FROM chunks WHERE unit_id = :id"),
                    {"id": f"{DOC_ID}#dieu-1"},
                )
            ).scalar_one()
        assert dims == CHUNKS_EMBEDDING_DIMENSION

    _run(migrated_db, body)


def test_chunk_upsert_many_rejects_a_dimension_mismatch(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        with pytest.raises(ValueError, match="dimension"):
            async with session_scope(session_factory) as session:
                await DocumentRepository(session).upsert(_document())
                await UnitRepository(session).upsert_many([_article()])
                await ChunkRepository(session).upsert_many([_chunk()], [[0.1] * 768])

        async with engine.begin() as conn:
            count = (await conn.execute(_COUNT_CHUNKS)).scalar_one()
        assert count == 0, "a rejected embedding must not leave a partial row"

    _run(migrated_db, body)


def test_chunk_upsert_many_rejects_a_length_mismatch(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await UnitRepository(session).upsert_many([_article()])
            with pytest.raises(ValueError, match="chunk"):
                await ChunkRepository(session).upsert_many([_chunk()], [])

    _run(migrated_db, body)


# --- edges (build plan A6 Phase 2.1) ---------------------------------------------


def test_edge_upsert_many_replaces_rather_than_duplicates(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await UnitRepository(session).upsert_many([_article(), _clause()])
            await EdgeRepository(session).upsert_many([_edge()])
        async with session_scope(session_factory) as session:
            await EdgeRepository(session).upsert_many([_edge(evidence="re-ingested")])

        async with engine.begin() as conn:
            count = (await conn.execute(_COUNT_EDGES)).scalar_one()
            evidence = (
                await conn.execute(
                    text("SELECT evidence FROM edges WHERE src_id = :src"),
                    {"src": f"{DOC_ID}#dieu-1#khoan-1"},
                )
            ).scalar_one()
        assert count == 1
        assert evidence == "re-ingested"

    _run(migrated_db, body)


def test_edge_upsert_many_treats_edge_type_as_part_of_the_conflict_key(migrated_db: str):
    # Same (src_id, dst_id) but a different edge_type must not collide --
    # UNIQUE (src_id, dst_id, edge_type) is a three-column key, not two.
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document())
            await UnitRepository(session).upsert_many([_article(), _clause()])
            await EdgeRepository(session).upsert_many(
                [_edge(), _edge(edge_type=EdgeType.REFERENCES)]
            )

        async with engine.begin() as conn:
            count = (await conn.execute(_COUNT_EDGES)).scalar_one()
        assert count == 2

    _run(migrated_db, body)


def test_edge_upsert_many_with_no_edges_is_a_no_op(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await EdgeRepository(session).upsert_many([])

        async with engine.begin() as conn:
            count = (await conn.execute(_COUNT_EDGES)).scalar_one()
        assert count == 0

    _run(migrated_db, body)


def test_edges_src_and_dst_indexes_actually_exist(migrated_db: str):
    """Queries pg_indexes rather than trusting the DDL ran (the ivfflat trap, again)."""

    async def body(engine: AsyncEngine) -> None:
        async with engine.begin() as conn:
            index_names = {
                row[0]
                for row in (
                    await conn.execute(
                        text("SELECT indexname FROM pg_indexes WHERE tablename = 'edges'")
                    )
                )
            }
        assert "edges_src_type" in index_names
        assert "edges_dst_type" in index_names

    _run(migrated_db, body)


# --- migration -----------------------------------------------------------------


def test_the_ivfflat_index_was_actually_dropped(migrated_db: str):
    """Queries pg_indexes rather than trusting the DDL ran (a stated trap).

    Migration 003 drops the index 001 created (build plan A6 issue #49 --
    see docs/adr/0001-drop-ivfflat-index.md); this proves the DROP INDEX
    statement in that file actually executed, the same "don't trust, verify"
    principle this test originally applied to the CREATE INDEX it now
    reverses. ``tests/test_search_recall.py`` carries the fuller recall
    proof this issue's fix exists for.
    """

    async def body(engine: AsyncEngine) -> None:
        async with engine.begin() as conn:
            index_names = {
                row[0]
                for row in (
                    await conn.execute(
                        text("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'")
                    )
                )
            }
        assert "chunks_embedding_ivfflat" not in index_names

    _run(migrated_db, body)


def test_applying_migrations_twice_is_a_no_op(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        applied_again = await apply_pending_migrations(engine)
        assert applied_again == []

    _run(migrated_db, body)
