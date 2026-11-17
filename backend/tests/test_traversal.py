"""Tests for the recursive-CTE graph traversal service (build plan A6 Phase 2.5).

Needs a live Postgres with pgvector: set ``TEST_DATABASE_URL`` (``docker-compose.yml``
starts one locally on port 5433; CI runs a service container). Every test here
depends, directly or transitively, on the ``migrated_db`` fixture, so a missing
``TEST_DATABASE_URL`` skips them all with one clear reason rather than a silent pass.

No ``pytest-asyncio`` (C12): each test drives its own async body with a single
``asyncio.run()`` call, matching ``tests/test_repositories.py``.

Traversal only ever reads ``documents`` and ``edges`` -- neither has a foreign
key onto ``units`` (see ``migrations/002_graph.sql``) -- so every fixture here
seeds those two tables directly and never needs a real ``units`` row.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.repositories.documents import DocumentRepository
from app.repositories.edges import EdgeRepository
from app.schemas.document import DocStatus, DocType, Document
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from app.services.graph_traversal import GraphTraversalService


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


def _edge(src_id: str, dst_id: str, **overrides: Any) -> Edge:
    fields: dict[str, Any] = {
        "src_id": src_id,
        "dst_id": dst_id,
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


# --- basic walk, hops, and the max_hops boundary ---------------------------------


def test_traverse_walks_a_chain_up_to_max_hops(migrated_db: str):
    # doc-a -> doc-b -> doc-c -> doc-d, all BELONGS_TO, all approved.
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            for doc_id in ("doc-a", "doc-b", "doc-c", "doc-d"):
                await DocumentRepository(session).upsert(_document(doc_id))
            await EdgeRepository(session).upsert_many(
                [
                    _edge("doc-a", "doc-b"),
                    _edge("doc-b", "doc-c"),
                    _edge("doc-c", "doc-d"),
                ]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["doc-a"], max_hops=2, follow_types=[EdgeType.BELONGS_TO]
            )
        ids_and_hops = [(hit.unit_id, hit.hop) for hit in hits]
        assert ids_and_hops == [("doc-b", 1), ("doc-c", 2)], hits
        assert "doc-d" not in {hit.unit_id for hit in hits}, "max_hops=2 must not reach a 3rd hop"

    _run(migrated_db, body)


def test_traverse_never_returns_a_seed_id_itself(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            for doc_id in ("doc-a", "doc-b"):
                await DocumentRepository(session).upsert(_document(doc_id))
            await EdgeRepository(session).upsert_many([_edge("doc-a", "doc-b")])
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["doc-a"], max_hops=5, follow_types=[EdgeType.BELONGS_TO]
            )
        assert "doc-a" not in {hit.unit_id for hit in hits}

    _run(migrated_db, body)


def test_traverse_with_no_seed_ids_or_no_follow_types_returns_empty_without_querying(
    migrated_db: str,
):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document("doc-a"))
            await DocumentRepository(session).upsert(_document("doc-b"))
            await EdgeRepository(session).upsert_many([_edge("doc-a", "doc-b")])
        async with session_scope(session_factory) as session:
            service = GraphTraversalService(session)
            assert await service.traverse([], max_hops=5, follow_types=[EdgeType.BELONGS_TO]) == ()
            assert await service.traverse(["doc-a"], max_hops=5, follow_types=[]) == ()

    _run(migrated_db, body)


# --- the cycle guard: seeded deliberately, never trusted from the tree's shape ---


def test_traverse_terminates_on_a_deliberately_seeded_cycle(migrated_db: str):
    # doc-a REFERENCES doc-b REFERENCES doc-c REFERENCES doc-a -- a real,
    # ordinary shape for reference edges (a decree citing a circular that
    # cites it back), which BELONGS_TO's tree shape never exercises.
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            for doc_id in ("doc-a", "doc-b", "doc-c"):
                await DocumentRepository(session).upsert(_document(doc_id))
            await EdgeRepository(session).upsert_many(
                [
                    _edge("doc-a", "doc-b", edge_type=EdgeType.REFERENCES, source=EdgeSource.REGEX),
                    _edge("doc-b", "doc-c", edge_type=EdgeType.REFERENCES, source=EdgeSource.REGEX),
                    _edge("doc-c", "doc-a", edge_type=EdgeType.REFERENCES, source=EdgeSource.REGEX),
                ]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["doc-a"], max_hops=10, follow_types=[EdgeType.REFERENCES]
            )
        # Terminates (the test itself would hang/error on a runaway recursive
        # query otherwise) and never re-enters a node already on the path.
        ids_and_hops = sorted((hit.unit_id, hit.hop) for hit in hits)
        assert ids_and_hops == [("doc-b", 1), ("doc-c", 2)], hits

    _run(migrated_db, body)


# --- only approved edges are ever walked -----------------------------------------


def test_traverse_never_follows_a_pending_or_rejected_edge(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            for doc_id in ("doc-a", "doc-b", "doc-c"):
                await DocumentRepository(session).upsert(_document(doc_id))
            await EdgeRepository(session).upsert_many(
                [
                    _edge(
                        "doc-a",
                        "doc-b",
                        edge_type=EdgeType.REFERENCES,
                        review_status=ReviewStatus.PENDING,
                        source=EdgeSource.REGEX,
                    ),
                    _edge(
                        "doc-a",
                        "doc-c",
                        edge_type=EdgeType.REFERENCES,
                        review_status=ReviewStatus.REJECTED,
                        source=EdgeSource.REGEX,
                    ),
                ]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["doc-a"], max_hops=5, follow_types=[EdgeType.REFERENCES]
            )
        assert hits == ()

    _run(migrated_db, body)


# --- cross-document multi-hop, once a reference edge is approved -----------------


def test_traverse_crosses_documents_once_a_reference_edge_is_approved(migrated_db: str):
    # This is the behaviour Phase 2.5 exists for: today's corpus only carries
    # PENDING reference edges (Phase 2.3 hasn't approved any yet), but the
    # machinery must already work the moment one is. Seeded directly here
    # rather than waiting for that data to exist.
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            for doc_id in ("gas-decree-17-2020", "gas-decree-87-2018"):
                await DocumentRepository(session).upsert(_document(doc_id))
            await EdgeRepository(session).upsert_many(
                [
                    _edge(
                        "gas-decree-17-2020#dieu-15#khoan-1",
                        "gas-decree-87-2018#dieu-38",
                        edge_type=EdgeType.REFERENCES,
                        evidence="sửa đổi Điều 38 Nghị định số 87/2018/NĐ-CP",
                        source=EdgeSource.REGEX,
                    ),
                    _edge(
                        "gas-decree-87-2018#dieu-38",
                        "gas-decree-87-2018",
                        edge_type=EdgeType.BELONGS_TO,
                    ),
                ]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["gas-decree-17-2020#dieu-15#khoan-1"],
                max_hops=2,
                follow_types=[EdgeType.REFERENCES, EdgeType.BELONGS_TO],
            )
        ids_and_hops = {(hit.unit_id, hit.hop) for hit in hits}
        assert ("gas-decree-87-2018#dieu-38", 1) in ids_and_hops, hits
        assert ("gas-decree-87-2018", 2) in ids_and_hops, hits

    _run(migrated_db, body)


def test_follow_types_selects_only_the_requested_edge_type(migrated_db: str):
    # A hand-written raw string could never reach this function at all
    # (mypy rejects it at the call site -- follow_types is typed
    # Sequence[EdgeType]); this proves the runtime side of the same
    # guarantee: asking for BELONGS_TO does not also walk a REFERENCES edge
    # sitting on the very same src_id.
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            for doc_id in ("doc-a", "doc-b", "doc-c"):
                await DocumentRepository(session).upsert(_document(doc_id))
            await EdgeRepository(session).upsert_many(
                [
                    _edge("doc-a", "doc-b", edge_type=EdgeType.BELONGS_TO),
                    _edge("doc-a", "doc-c", edge_type=EdgeType.REFERENCES, source=EdgeSource.REGEX),
                ]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["doc-a"], max_hops=5, follow_types=[EdgeType.BELONGS_TO]
            )
        assert {hit.unit_id for hit in hits} == {"doc-b"}

    _run(migrated_db, body)


# --- the in-force filter (A5.2) ---------------------------------------------------


def test_traverse_excludes_a_hit_whose_document_status_is_not_in_force(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document("doc-a"))
            await DocumentRepository(session).upsert(_document("doc-b", status=DocStatus.EXPIRED))
            await EdgeRepository(session).upsert_many([_edge("doc-a", "doc-b")])
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["doc-a"], max_hops=5, follow_types=[EdgeType.BELONGS_TO]
            )
        assert hits == ()

    _run(migrated_db, body)


def test_traverse_excludes_a_hit_with_an_approved_replaced_by_edge(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            for doc_id in ("doc-a", "doc-b", "doc-b-new"):
                await DocumentRepository(session).upsert(_document(doc_id))
            await EdgeRepository(session).upsert_many(
                [
                    _edge("doc-a", "doc-b"),
                    _edge("doc-b", "doc-b-new", edge_type=EdgeType.REPLACED_BY),
                ]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["doc-a"], max_hops=1, follow_types=[EdgeType.BELONGS_TO]
            )
        assert hits == ()

    _run(migrated_db, body)


def test_traverse_flags_but_does_not_exclude_a_pending_replaced_by_edge(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            for doc_id in ("doc-a", "doc-b", "doc-b-new"):
                await DocumentRepository(session).upsert(_document(doc_id))
            await EdgeRepository(session).upsert_many(
                [
                    _edge("doc-a", "doc-b"),
                    _edge(
                        "doc-b",
                        "doc-b-new",
                        edge_type=EdgeType.REPLACED_BY,
                        review_status=ReviewStatus.PENDING,
                        source=EdgeSource.REGEX,
                    ),
                ]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["doc-a"], max_hops=1, follow_types=[EdgeType.BELONGS_TO]
            )
        assert len(hits) == 1
        assert hits[0].unit_id == "doc-b"
        assert hits[0].possibly_superseded is True

    _run(migrated_db, body)


def test_include_history_bypasses_the_in_force_filter_entirely(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            for doc_id in ("doc-a", "doc-b", "doc-b-new"):
                await DocumentRepository(session).upsert(_document(doc_id))
            await EdgeRepository(session).upsert_many(
                [
                    _edge("doc-a", "doc-b"),
                    _edge("doc-b", "doc-b-new", edge_type=EdgeType.REPLACED_BY),
                ]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["doc-a"], max_hops=1, follow_types=[EdgeType.BELONGS_TO], include_history=True
            )
        assert len(hits) == 1
        assert hits[0].unit_id == "doc-b"
        assert hits[0].possibly_superseded is False

    _run(migrated_db, body)


# --- exactly one walk query, one in-force query -- never per-hop, never per-hit -


def test_traverse_runs_exactly_two_queries_regardless_of_hit_count(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            docs = ["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"]
            for doc_id in docs:
                await DocumentRepository(session).upsert(_document(doc_id))
            await EdgeRepository(session).upsert_many([_edge("doc-a", dst) for dst in docs[1:]])

        statements: list[str] = []

        def _record(*args: object) -> None:
            statements.append(str(args[2]))

        sync_engine = engine.sync_engine
        async with session_scope(session_factory) as session:
            event.listen(sync_engine, "before_cursor_execute", _record)
            try:
                hits = await GraphTraversalService(session).traverse(
                    ["doc-a"], max_hops=3, follow_types=[EdgeType.BELONGS_TO]
                )
            finally:
                event.remove(sync_engine, "before_cursor_execute", _record)
        assert len(hits) == 4
        select_statements = [
            s for s in statements if s.strip().upper().startswith(("WITH", "SELECT"))
        ]
        assert len(select_statements) == 2, select_statements

    _run(migrated_db, body)


# --- entity hits (req:..., concept:...) pass the in-force filter unchanged (#58) -


def test_an_entity_hit_does_not_raise_and_passes_through_unchanged(migrated_db: str):
    # An entity id names no document at all -- the walk must reach it without
    # the in-force filter's document lookup raising, and since it has no
    # status or effective date of its own, it is returned unchanged rather
    # than dropped (issue #58; this KeyError'd on main before the fix).
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await EdgeRepository(session).upsert_many(
                [
                    _edge(
                        "req:minimum-area",
                        "concept:lpg-retail-store",
                        edge_type=EdgeType.APPLIES_TO,
                    )
                ]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["req:minimum-area"], max_hops=1, follow_types=[EdgeType.APPLIES_TO]
            )
        assert len(hits) == 1
        assert hits[0].unit_id == "concept:lpg-retail-store"
        assert hits[0].hop == 1
        assert hits[0].possibly_superseded is False

    _run(migrated_db, body)


def test_a_mixed_walk_returns_both_a_unit_hit_and_an_entity_hit(migrated_db: str):
    # The shape Phase 3 will actually issue: one traverse() call whose hits
    # span both an ordinary document/unit id (in-force checked as usual) and
    # an entity id (passed through unchanged).
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            for doc_id in ("doc-a", "doc-b"):
                await DocumentRepository(session).upsert(_document(doc_id))
            await EdgeRepository(session).upsert_many(
                [
                    _edge("doc-a", "doc-b"),
                    _edge(
                        "req:minimum-area",
                        "concept:lpg-retail-store",
                        edge_type=EdgeType.APPLIES_TO,
                    ),
                ]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["doc-a", "req:minimum-area"],
                max_hops=1,
                follow_types=[EdgeType.BELONGS_TO, EdgeType.APPLIES_TO],
            )
        ids = {hit.unit_id for hit in hits}
        assert ids == {"doc-b", "concept:lpg-retail-store"}

    _run(migrated_db, body)


def test_a_hit_naming_a_nonexistent_document_is_excluded_not_returned(migrated_db: str):
    # `edges` carries no foreign key on either endpoint, so a dangling edge
    # can exist: its target names a document with no row in `documents` at
    # all, so that document's status is unknown. Not an entity id (no
    # `req:`/`concept:` prefix), so it must be excluded as a dangling edge,
    # never returned as current law (issue #58's review -- the first #58 fix
    # inferred "entity" from a failed lookup and returned this unchanged).
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document("doc-a"))
            await EdgeRepository(session).upsert_many([_edge("doc-a", "ghost-decree-2099#dieu-5")])
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["doc-a"], max_hops=1, follow_types=[EdgeType.BELONGS_TO]
            )
        assert hits == ()

    _run(migrated_db, body)


# --- the real lpg-qcvn-08-2012 / -2019 REPLACED_BY edge (issue #63) --------------


def test_the_approved_replaced_by_edge_reaches_qcvn_08_2019_in_history_mode(migrated_db: str):
    # The exact case issue #63 exists for: lpg-qcvn-08-2012 is expired and
    # acquired=false, kept only so this edge gives it a history meaning
    # once a human approves the curated seed.
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(
                _document("lpg-qcvn-08-2012", status=DocStatus.EXPIRED)
            )
            await DocumentRepository(session).upsert(
                _document("lpg-qcvn-08-2019", status=DocStatus.IN_FORCE)
            )
            await EdgeRepository(session).upsert_many(
                [
                    _edge(
                        "lpg-qcvn-08-2012",
                        "lpg-qcvn-08-2019",
                        edge_type=EdgeType.REPLACED_BY,
                        source=EdgeSource.CURATED,
                    )
                ]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["lpg-qcvn-08-2012"],
                max_hops=1,
                follow_types=[EdgeType.REPLACED_BY],
                include_history=True,
            )
        assert [hit.unit_id for hit in hits] == ["lpg-qcvn-08-2019"]

    _run(migrated_db, body)


def test_lpg_qcvn_08_2012_itself_is_excluded_by_the_default_in_force_filter(migrated_db: str):
    # The other half of the same acceptance case: reached as an ordinary
    # hit (not a seed), the expired QCVN 08:2012 node must not surface
    # under the default filter -- only include_history=True should ever
    # show it.
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await DocumentRepository(session).upsert(_document("gas-decree-87-2018"))
            await DocumentRepository(session).upsert(
                _document("lpg-qcvn-08-2012", status=DocStatus.EXPIRED)
            )
            await EdgeRepository(session).upsert_many(
                [_edge("gas-decree-87-2018", "lpg-qcvn-08-2012")]
            )
        async with session_scope(session_factory) as session:
            hits = await GraphTraversalService(session).traverse(
                ["gas-decree-87-2018"], max_hops=1, follow_types=[EdgeType.BELONGS_TO]
            )
        assert hits == ()

    _run(migrated_db, body)
