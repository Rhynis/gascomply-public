"""Tests for walking an edge type in reverse (issue #86).

Needs a live Postgres with pgvector (see ``tests/test_traversal.py``'s module docstring).
This is a separate file so the forward-traversal tests stay byte-for-byte what they were:
that is how "forward traversal is unchanged" is demonstrated, not asserted.

What is under test is the reverse path, and every safety property of the forward one holds
in it identically: only ``approved`` edges, the in-force filter, entity ids (#58/#61), and the
cycle guard. Each has a test here, and the queries themselves are checked structurally.

The graph the walks run over has the shape of the real requirement cluster: a requirement
``REFERENCES`` the clause that states it and ``APPLIES_TO`` a concept, so from a clause the
obligations are upstream and from a concept the requirements are.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from typing import Any

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.repositories.documents import DocumentRepository
from app.repositories.edges import EdgeRepository
from app.schemas.document import DocStatus, DocType, Document, DocumentStub
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from app.services.graph_traversal import (
    _DIRECTED_WALK_SQL,
    _WALK_SQL,
    GraphTraversalService,
)

REFERENCES = EdgeType.REFERENCES
APPLIES_TO = EdgeType.APPLIES_TO
BELONGS_TO = EdgeType.BELONGS_TO
REPLACED_BY = EdgeType.REPLACED_BY

CLAUSE_1 = "doc-a#dieu-1#khoan-1"
CLAUSE_2 = "doc-a#dieu-1#khoan-2"
CONCEPT = "concept:store"


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


def _edge(src_id: str, dst_id: str, edge_type: EdgeType = REFERENCES, **overrides: Any) -> Edge:
    fields: dict[str, Any] = {
        "src_id": src_id,
        "dst_id": dst_id,
        "edge_type": edge_type,
        "review_status": ReviewStatus.APPROVED,
        "source": EdgeSource.CURATED,
    }
    fields.update(overrides)
    return Edge(**fields)


def _cluster() -> list[Edge]:
    """The requirement cluster's shape: three requirements, two clauses, one concept."""
    return [
        _edge("req:min-area", CLAUSE_1),
        _edge("req:distance", CLAUSE_2),
        _edge("req:fire", CLAUSE_2),
        _edge("req:min-area", CONCEPT, APPLIES_TO),
        _edge("req:distance", CONCEPT, APPLIES_TO),
        _edge("req:fire", CONCEPT, APPLIES_TO),
    ]


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


def _seed(
    database_url: str,
    edges: Sequence[Edge],
    documents: Sequence[Document] = (),
    stubs: Sequence[DocumentStub] = (),
) -> None:
    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            for document in documents:
                await DocumentRepository(session).upsert(document)
            await DocumentRepository(session).insert_stubs(stubs)
            await EdgeRepository(session).upsert_many(edges)

    _run(database_url, body)


def _walk(database_url: str, seeds: Sequence[str], **kwargs: Any) -> list[tuple[str, int]]:
    """Run one traversal and return ``[(id, hop)]`` in the service's own order."""
    hits: list[tuple[str, int]] = []

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            # Without its cycle guard a walk over a cycle grows exponentially: a regression
            # must fail here in seconds, not hang the suite.
            await session.execute(text("SET LOCAL statement_timeout = '10s'"))
            result = await GraphTraversalService(session).traverse(seeds, **kwargs)
        hits.extend((hit.unit_id, hit.hop) for hit in result)

    _run(database_url, body)
    return hits


# --- the two shapes of the product's central question ------------------------------------


def test_a_forward_walk_cannot_reach_what_points_at_the_clause(migrated_db: str):
    # The defect issue #86 measured on the real database: nothing leaves the clause.
    _seed(migrated_db, _cluster(), [_document("doc-a")])

    assert _walk(migrated_db, [CLAUSE_1], max_hops=3, follow_types=[REFERENCES, APPLIES_TO]) == []


def test_a_reverse_references_walk_reaches_the_requirement_that_points_at_the_clause(
    migrated_db: str,
):
    _seed(migrated_db, _cluster(), [_document("doc-a")])

    hits = _walk(migrated_db, [CLAUSE_2], max_hops=1, follow_types=[], reverse_types=[REFERENCES])

    assert hits == [("req:distance", 1), ("req:fire", 1)]  # not req:min-area: it names clause 1


def test_a_reverse_applies_to_walk_reaches_every_requirement_of_a_concept(migrated_db: str):
    _seed(migrated_db, _cluster())

    hits = _walk(migrated_db, [CONCEPT], max_hops=1, follow_types=[], reverse_types=[APPLIES_TO])

    assert hits == [("req:distance", 1), ("req:fire", 1), ("req:min-area", 1)]


def test_clause_to_requirement_to_concept_takes_two_hops_in_one_call(migrated_db: str):
    # REFERENCES backwards, then APPLIES_TO forwards: a direction per edge type is what lets
    # one walk express this, and a single flag for the whole call could not.
    _seed(migrated_db, _cluster(), [_document("doc-a")])

    hits = _walk(
        migrated_db,
        [CLAUSE_1],
        max_hops=2,
        follow_types=[APPLIES_TO],
        reverse_types=[REFERENCES],
    )

    assert hits == [("req:min-area", 1), (CONCEPT, 2)]


def test_max_hops_bounds_a_reverse_walk_too(migrated_db: str):
    _seed(migrated_db, _cluster(), [_document("doc-a")])

    hits = _walk(
        migrated_db,
        [CLAUSE_1],
        max_hops=1,
        follow_types=[APPLIES_TO],
        reverse_types=[REFERENCES],
    )

    assert hits == [("req:min-area", 1)]


def test_a_reverse_chain_is_walked_up_to_max_hops(migrated_db: str):
    # d -> c -> b -> a, all REFERENCES: from a, the reverse walk climbs b, c, d.
    _seed(
        migrated_db,
        [_edge("doc-d", "doc-c"), _edge("doc-c", "doc-b"), _edge("doc-b", "doc-a")],
        [_document(name) for name in ("doc-a", "doc-b", "doc-c", "doc-d")],
    )

    hits = _walk(migrated_db, ["doc-a"], max_hops=2, follow_types=[], reverse_types=[REFERENCES])

    assert hits == [("doc-b", 1), ("doc-c", 2)]  # doc-d would be a third hop


# --- direction is deliberate: per call, per edge type -------------------------------------


def test_a_call_that_names_no_reverse_types_never_walks_backwards(migrated_db: str):
    _seed(migrated_db, [_edge("doc-b", "doc-a")], [_document("doc-a"), _document("doc-b")])

    assert _walk(migrated_db, ["doc-a"], max_hops=3, follow_types=[REFERENCES]) == []
    assert (
        _walk(migrated_db, ["doc-a"], max_hops=3, follow_types=[REFERENCES], reverse_types=[]) == []
    )


def test_only_the_edge_types_named_are_followed_in_reverse(migrated_db: str):
    child = "doc-a#dieu-1#khoan-1#diem-a"
    _seed(
        migrated_db,
        [_edge("req:min-area", CLAUSE_1), _edge(child, CLAUSE_1, BELONGS_TO)],
        [_document("doc-a")],
    )

    by_references = _walk(
        migrated_db, [CLAUSE_1], max_hops=1, follow_types=[], reverse_types=[REFERENCES]
    )
    by_belongs_to = _walk(
        migrated_db, [CLAUSE_1], max_hops=1, follow_types=[], reverse_types=[BELONGS_TO]
    )

    # A reverse BELONGS_TO is the clause's own children, and only when asked for.
    assert by_references == [("req:min-area", 1)]
    assert by_belongs_to == [(child, 1)]


def test_a_type_named_in_both_lists_is_followed_both_ways(migrated_db: str):
    _seed(
        migrated_db,
        [_edge("doc-a", "doc-b"), _edge("doc-b", "doc-c")],
        [_document(name) for name in ("doc-a", "doc-b", "doc-c")],
    )

    hits = _walk(
        migrated_db, ["doc-b"], max_hops=1, follow_types=[REFERENCES], reverse_types=[REFERENCES]
    )

    assert hits == [("doc-a", 1), ("doc-c", 1)]


def test_a_call_with_neither_kind_of_type_returns_empty_without_querying(migrated_db: str):
    _seed(migrated_db, _cluster(), [_document("doc-a")])
    statements: list[str] = []

    async def body(engine: AsyncEngine) -> None:
        def _record(*args: object) -> None:
            statements.append(str(args[2]))

        event.listen(engine.sync_engine, "before_cursor_execute", _record)
        try:
            async with session_scope(create_session_factory(engine)) as session:
                service = GraphTraversalService(session)
                assert await service.traverse([CLAUSE_1], max_hops=2, follow_types=[]) == ()
                assert (
                    await service.traverse(
                        [CLAUSE_1], max_hops=2, follow_types=[], reverse_types=[]
                    )
                    == ()
                )
                assert (
                    await service.traverse(
                        [], max_hops=2, follow_types=[], reverse_types=[REFERENCES]
                    )
                    == ()
                )
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", _record)

    _run(migrated_db, body)

    assert [s for s in statements if s.strip().upper().startswith(("WITH", "SELECT"))] == []


def test_a_raw_string_in_reverse_types_raises_like_one_in_follow_types(migrated_db: str):
    # Silently matching zero rows would hide a typo; the enum's ``.value`` is what raises.
    _seed(migrated_db, _cluster())

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            with pytest.raises(AttributeError):
                await GraphTraversalService(session).traverse(
                    [CLAUSE_1],
                    max_hops=1,
                    follow_types=[],
                    reverse_types=["references"],  # type: ignore[list-item]
                )

    _run(migrated_db, body)


# --- only approved edges, in both directions --------------------------------------------------


def test_a_reverse_walk_follows_only_approved_edges(migrated_db: str):
    _seed(
        migrated_db,
        [
            _edge("req:approved", CLAUSE_1),
            _edge("req:pending", CLAUSE_1, review_status=ReviewStatus.PENDING),
            _edge("req:rejected", CLAUSE_1, review_status=ReviewStatus.REJECTED),
        ],
    )

    hits = _walk(migrated_db, [CLAUSE_1], max_hops=1, follow_types=[], reverse_types=[REFERENCES])

    assert hits == [("req:approved", 1)]


def test_a_pending_edge_on_the_second_step_is_not_followed_either(migrated_db: str):
    _seed(
        migrated_db,
        [
            _edge("req:min-area", CLAUSE_1),
            _edge("req:min-area", CONCEPT, APPLIES_TO, review_status=ReviewStatus.PENDING),
        ],
    )

    hits = _walk(
        migrated_db,
        [CLAUSE_1],
        max_hops=2,
        follow_types=[APPLIES_TO],
        reverse_types=[REFERENCES],
    )

    assert hits == [("req:min-area", 1)]


def test_approving_the_edge_is_the_only_thing_that_makes_a_reverse_walk_follow_it(
    migrated_db: str,
):
    _seed(migrated_db, [_edge("req:x", CLAUSE_1, review_status=ReviewStatus.PENDING)])
    walk = {"max_hops": 1, "follow_types": [], "reverse_types": [REFERENCES]}
    assert _walk(migrated_db, [CLAUSE_1], **walk) == []

    async def approve(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            await session.execute(text("UPDATE edges SET review_status = 'approved'"))

    _run(migrated_db, approve)

    assert _walk(migrated_db, [CLAUSE_1], **walk) == [("req:x", 1)]


# --- the in-force filter judges what a reverse step reaches ---------------------------------


def test_a_reverse_walk_returns_a_unit_of_an_in_force_document(migrated_db: str):
    _seed(
        migrated_db,
        [_edge("doc-b#dieu-2", CLAUSE_1)],
        [_document("doc-a"), _document("doc-b")],
    )

    hits = _walk(migrated_db, [CLAUSE_1], max_hops=1, follow_types=[], reverse_types=[REFERENCES])

    assert hits == [("doc-b#dieu-2", 1)]


def test_a_reverse_walk_excludes_a_unit_of_an_expired_document(migrated_db: str):
    _seed(
        migrated_db,
        [_edge("doc-old#dieu-9", CLAUSE_1)],
        [_document("doc-a"), _document("doc-old", status=DocStatus.EXPIRED)],
    )
    walk = {"max_hops": 1, "follow_types": [], "reverse_types": [REFERENCES]}

    assert _walk(migrated_db, [CLAUSE_1], **walk) == []
    assert _walk(migrated_db, [CLAUSE_1], include_history=True, **walk) == [("doc-old#dieu-9", 1)]


def test_a_reverse_walk_excludes_a_document_with_an_approved_replaced_by_edge(migrated_db: str):
    _seed(
        migrated_db,
        [_edge("doc-b#dieu-2", CLAUSE_1), _edge("doc-b", "doc-new", REPLACED_BY)],
        [_document("doc-a"), _document("doc-b"), _document("doc-new")],
    )
    walk = {"max_hops": 1, "follow_types": [], "reverse_types": [REFERENCES]}

    assert _walk(migrated_db, [CLAUSE_1], **walk) == []
    assert _walk(migrated_db, [CLAUSE_1], include_history=True, **walk) == [("doc-b#dieu-2", 1)]


def test_a_reverse_walk_flags_but_does_not_exclude_a_pending_replaced_by_edge(migrated_db: str):
    _seed(
        migrated_db,
        [
            _edge("doc-b#dieu-2", CLAUSE_1),
            _edge("doc-b", "doc-new", REPLACED_BY, review_status=ReviewStatus.PENDING),
        ],
        [_document("doc-a"), _document("doc-b"), _document("doc-new")],
    )
    flags: dict[str, bool] = {}

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            hits = await GraphTraversalService(session).traverse(
                [CLAUSE_1], max_hops=1, follow_types=[], reverse_types=[REFERENCES]
            )
        flags.update({hit.unit_id: hit.possibly_superseded for hit in hits})

    _run(migrated_db, body)

    assert flags == {"doc-b#dieu-2": True}


def test_a_stub_is_never_returned_as_current_law_by_a_reverse_walk(migrated_db: str):
    # A stub's status is `unknown` -- not in force -- and reverse REPLACED_BY is exactly how a
    # walk from the repealing document reaches the documents it retired.
    stub = DocumentStub(number="136/2020/NĐ-CP")
    _seed(
        migrated_db,
        [_edge(stub.id, "doc-new", REPLACED_BY)],
        [_document("doc-new")],
        [stub],
    )
    walk = {"max_hops": 1, "follow_types": [], "reverse_types": [REPLACED_BY]}

    assert _walk(migrated_db, ["doc-new"], **walk) == []
    assert _walk(migrated_db, ["doc-new"], include_history=True, **walk) == [(stub.id, 1)]


# --- entity ids and dangling nodes (#58, #61), reached in reverse ------------------------------


def test_an_entity_reached_in_reverse_passes_the_filter_unchanged(migrated_db: str):
    # `req:` and `concept:` name no document, so the in-force lookup has nothing to say about
    # them; they must come back, not be dropped, and not raise.
    _seed(migrated_db, _cluster())
    flags: dict[str, bool] = {}

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            hits = await GraphTraversalService(session).traverse(
                [CLAUSE_1],
                max_hops=2,
                follow_types=[APPLIES_TO],
                reverse_types=[REFERENCES],
            )
        flags.update({hit.unit_id: hit.possibly_superseded for hit in hits})

    _run(migrated_db, body)

    assert flags == {"req:min-area": False, CONCEPT: False}


def test_a_dangling_node_reached_in_reverse_is_excluded_and_logged(
    migrated_db: str, caplog: pytest.LogCaptureFixture
):
    # `edges` has no foreign key on either endpoint, so an approved edge can start at a document
    # that has no row. Its status is unknown, so it must never come back as current law.
    _seed(migrated_db, [_edge("ghost-decree-2099#dieu-5", CLAUSE_1)], [_document("doc-a")])
    walk = {"max_hops": 1, "follow_types": [], "reverse_types": [REFERENCES]}

    with caplog.at_level(logging.WARNING, logger="app.services.graph_traversal"):
        hits = _walk(migrated_db, [CLAUSE_1], **walk)

    assert hits == []
    assert any("ghost-decree-2099#dieu-5" in record.getMessage() for record in caplog.records)
    assert _walk(migrated_db, [CLAUSE_1], include_history=True, **walk) == [
        ("ghost-decree-2099#dieu-5", 1)
    ]


# --- the cycle guard covers the reverse path ----------------------------------------------------


def test_a_reverse_walk_terminates_on_a_cycle_and_never_returns_the_seed(migrated_db: str):
    # a -> b -> c -> a. Backwards from a: c (who references a), then b, then a again -- which is
    # on the path already, so the guard stops it.
    _seed(
        migrated_db,
        [_edge("doc-a", "doc-b"), _edge("doc-b", "doc-c"), _edge("doc-c", "doc-a")],
        [_document(name) for name in ("doc-a", "doc-b", "doc-c")],
    )

    hits = _walk(migrated_db, ["doc-a"], max_hops=50, follow_types=[], reverse_types=[REFERENCES])

    assert hits == [("doc-c", 1), ("doc-b", 2)]


def test_a_reverse_walk_terminates_on_a_mutual_reference(migrated_db: str):
    _seed(
        migrated_db,
        [_edge("doc-a", "doc-b"), _edge("doc-b", "doc-a")],
        [_document("doc-a"), _document("doc-b")],
    )

    hits = _walk(migrated_db, ["doc-a"], max_hops=50, follow_types=[], reverse_types=[REFERENCES])

    assert hits == [("doc-b", 1)]


def test_a_self_loop_does_not_return_the_seed_in_reverse(migrated_db: str):
    _seed(migrated_db, [_edge("doc-a", "doc-a")], [_document("doc-a")])

    assert (
        _walk(migrated_db, ["doc-a"], max_hops=5, follow_types=[], reverse_types=[REFERENCES]) == []
    )


def test_a_walk_in_both_directions_terminates_on_a_cycle(migrated_db: str):
    _seed(
        migrated_db,
        [_edge("doc-a", "doc-b"), _edge("doc-b", "doc-c"), _edge("doc-c", "doc-a")],
        [_document(name) for name in ("doc-a", "doc-b", "doc-c")],
    )

    hits = _walk(
        migrated_db, ["doc-a"], max_hops=50, follow_types=[REFERENCES], reverse_types=[REFERENCES]
    )

    assert hits == [("doc-b", 1), ("doc-c", 1)]


# --- cost: the same two queries, and no scan --------------------------------------------------


def test_a_reverse_traversal_runs_exactly_two_queries(migrated_db: str):
    _seed(migrated_db, _cluster(), [_document("doc-a")])
    statements: list[str] = []

    async def body(engine: AsyncEngine) -> None:
        def _record(*args: object) -> None:
            statements.append(str(args[2]))

        event.listen(engine.sync_engine, "before_cursor_execute", _record)
        try:
            async with session_scope(create_session_factory(engine)) as session:
                await GraphTraversalService(session).traverse(
                    [CLAUSE_2], max_hops=2, follow_types=[APPLIES_TO], reverse_types=[REFERENCES]
                )
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", _record)

    _run(migrated_db, body)

    selects = [s for s in statements if s.strip().upper().startswith(("WITH", "SELECT"))]
    assert len(selects) == 2, selects  # the walk, and one batched in-force lookup


# --- the queries themselves: structural, and equivalence --------------------------------------


def test_both_walk_queries_hard_code_the_approved_only_rule():
    # `review_status='approved'` is a literal, never a bind parameter, so no argument can reach it.
    for query in (_WALK_SQL, _DIRECTED_WALK_SQL):
        assert ":review_status" not in query.text
        assert "review_status" in query.text
    assert _WALK_SQL.text.count("review_status='approved'") == 2  # anchor and recursive step
    # One literal per direction of the step: forwards and backwards.
    assert _DIRECTED_WALK_SQL.text.count("review_status='approved'") == 2


def test_both_walk_queries_carry_the_cycle_guard():
    for query in (_WALK_SQL, _DIRECTED_WALK_SQL):
        assert "ANY(w.path)" in query.text


def _random_edges(rng: random.Random, nodes: int, count: int) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    while len(pairs) < count:
        src, dst = rng.randrange(nodes), rng.randrange(nodes)
        if src != dst:  # a self-loop is the one input where the two queries differ, by design
            pairs.add((f"n{src}", f"n{dst}"))
    return sorted(pairs)


async def _rows(session: Any, query: Any, params: dict[str, object]) -> list[tuple[str, int]]:
    return sorted((row.unit_id, row.hop) for row in await session.execute(query, params))


def test_the_directed_query_with_no_reverse_types_equals_the_original_forward_query(
    migrated_db: str,
):
    # The guarantee that forward traversal is unchanged does not rest on the forward path
    # being untouched alone: the reverse-capable query must agree with it on a cyclic graph.
    pairs = _random_edges(random.Random(86), nodes=14, count=40)
    _seed(migrated_db, [_edge(src, dst) for src, dst in pairs])

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            for seeds in (["n0"], ["n3", "n7"], ["n1", "n2", "n9"]):
                for max_hops in range(1, 7):
                    forward = {
                        "seed_ids": seeds,
                        "follow_types": ["references"],
                        "max_hops": max_hops,
                    }
                    original = await _rows(session, _WALK_SQL, forward)
                    directed = await _rows(
                        session, _DIRECTED_WALK_SQL, {**forward, "reverse_types": []}
                    )
                    assert directed == original, (seeds, max_hops)

    _run(migrated_db, body)


def test_a_reverse_walk_equals_a_forward_walk_over_the_transposed_graph(migrated_db: str):
    # Store every edge once as REFERENCES (src -> dst) and once flipped as APPLIES_TO
    # (dst -> src). Walking REFERENCES backwards must then reach exactly what walking
    # APPLIES_TO forwards does, at the same hop counts, cycles and diamonds included.
    pairs = _random_edges(random.Random(861), nodes=14, count=40)
    edges = [_edge(src, dst) for src, dst in pairs]
    edges += [_edge(dst, src, APPLIES_TO) for src, dst in pairs]
    _seed(migrated_db, edges)

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            for seeds in (["n0"], ["n5"], ["n2", "n11"]):
                for max_hops in range(1, 7):
                    reverse = await _rows(
                        session,
                        _DIRECTED_WALK_SQL,
                        {
                            "seed_ids": seeds,
                            "follow_types": [],
                            "reverse_types": ["references"],
                            "max_hops": max_hops,
                        },
                    )
                    transposed = await _rows(
                        session,
                        _WALK_SQL,
                        {"seed_ids": seeds, "follow_types": ["applies_to"], "max_hops": max_hops},
                    )
                    assert reverse == transposed, (seeds, max_hops)

    _run(migrated_db, body)
