"""Tests for the curated document-level edge seed (build plan A6, issue #63).

Covers seed loading, endpoint validation against what a run actually
ingested (not the manifest -- issue #63's review, ``--limit`` narrows what
gets ingested without narrowing the manifest), the regex-vs-curated
collision, the review-status-survives-re-ingest guarantee, and the
new-versus-updated edge count the stats table reports.

DB-backed tests need a live Postgres: set ``TEST_DATABASE_URL``. Every test
that needs one depends, directly or transitively, on the ``migrated_db``
fixture, so a missing ``TEST_DATABASE_URL`` skips them with one clear reason
rather than a silent pass. No ``pytest-asyncio`` (C12): each DB-backed test
drives its own async body with a single ``asyncio.run()`` call, matching
``tests/test_repositories.py``.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.repositories.edges import EdgeRepository
from app.schemas.document import DocStatus, DocType, Document
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from ingest.graph import (
    DocumentEdgeSeed,
    build_document_edges,
    document_edge_target_ids,
    load_document_edges,
)
from ingest.manifest import Manifest
from ingest.run import DOCUMENT_EDGES_PATH, _ingest_document_edges

# --- load_document_edges -----------------------------------------------------------


def test_the_real_seed_file_loads_and_has_five_edges():
    # A schema-drift guard on the actual committed data/document_edges.json,
    # not just the loader logic -- a human editing this file by hand should
    # have something catch a malformed edit immediately.
    seeds = load_document_edges(DOCUMENT_EDGES_PATH)
    assert len(seeds) == 5
    triples = {(seed.src_id, seed.dst_id, seed.edge_type) for seed in seeds}
    assert triples == {
        ("pccc-decree-105-2025", "pccc-law-2024", EdgeType.GUIDED_BY),
        ("pccc-circular-36-2025", "pccc-law-2024", EdgeType.GUIDED_BY),
        ("pccc-circular-36-2025", "pccc-decree-105-2025", EdgeType.GUIDED_BY),
        ("gas-decree-87-2018", "gas-decree-17-2020", EdgeType.AMENDED_BY),
        ("lpg-qcvn-08-2012", "lpg-qcvn-08-2019", EdgeType.REPLACED_BY),
    }


def test_load_document_edges_parses_a_minimal_file(tmp_path: Path):
    path = tmp_path / "document_edges.json"
    path.write_text(
        json.dumps(
            {
                "edges": [
                    {"src_id": "doc-a", "dst_id": "doc-b", "edge_type": "guided_by"},
                    {
                        "src_id": "doc-c",
                        "dst_id": "doc-d",
                        "edge_type": "replaced_by",
                        "note": "a note",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    seeds = load_document_edges(path)
    assert seeds == (
        DocumentEdgeSeed(src_id="doc-a", dst_id="doc-b", edge_type=EdgeType.GUIDED_BY),
        DocumentEdgeSeed(
            src_id="doc-c", dst_id="doc-d", edge_type=EdgeType.REPLACED_BY, note="a note"
        ),
    )


# --- document_edge_target_ids -------------------------------------------------------


def test_document_edge_target_ids_extracts_every_src_and_dst():
    seeds = (
        DocumentEdgeSeed(src_id="doc-a", dst_id="doc-b", edge_type=EdgeType.GUIDED_BY),
        DocumentEdgeSeed(src_id="doc-c", dst_id="doc-a", edge_type=EdgeType.AMENDED_BY),
    )
    assert document_edge_target_ids(seeds) == {"doc-a", "doc-b", "doc-c"}


# --- build_document_edges ------------------------------------------------------------


def test_build_document_edges_raises_when_an_endpoint_is_unknown_listing_every_offender():
    seeds = (
        DocumentEdgeSeed(src_id="doc-a", dst_id="ghost-1", edge_type=EdgeType.GUIDED_BY),
        DocumentEdgeSeed(src_id="ghost-2", dst_id="doc-a", edge_type=EdgeType.AMENDED_BY),
    )
    with pytest.raises(ValueError, match="ghost-1") as error:
        build_document_edges(seeds, {"doc-a"})
    assert "ghost-2" in str(error.value)


def test_every_curated_document_edge_is_pending_review_and_sourced_as_curated():
    seeds = (DocumentEdgeSeed(src_id="doc-a", dst_id="doc-b", edge_type=EdgeType.GUIDED_BY),)
    (edge,) = build_document_edges(seeds, {"doc-a", "doc-b"})
    assert edge.review_status is ReviewStatus.PENDING
    assert edge.source is EdgeSource.CURATED


def test_evidence_comes_from_the_seed_note():
    seeds = (
        DocumentEdgeSeed(
            src_id="doc-a", dst_id="doc-b", edge_type=EdgeType.GUIDED_BY, note="why this edge"
        ),
    )
    (edge,) = build_document_edges(seeds, {"doc-a", "doc-b"})
    assert edge.evidence == "why this edge"


def test_the_real_seed_file_validates_against_the_ids_it_names():
    # No manifest or corpus needed -- build_document_edges only checks
    # membership in the id set its caller passes.
    seeds = load_document_edges(DOCUMENT_EDGES_PATH)
    edges = build_document_edges(seeds, document_edge_target_ids(seeds))
    assert len(edges) == 5


# --- _ingest_document_edges (ingest/run.py orchestration) ---------------------------


def _document(doc_id: str, **overrides: Any) -> Document:
    fields: dict[str, Any] = {
        "id": doc_id,
        "number": "87/2018/NĐ-CP",
        "title_vi": "Nghị định thử nghiệm",
        "doc_type": DocType.DECREE,
        "status": DocStatus.IN_FORCE,
        "path": f"corpus/{doc_id}.html",
    }
    fields.update(overrides)
    return Document(**fields)


def _manifest(*documents: Document) -> Manifest:
    return Manifest(root=Path("."), documents=documents)


def _real_document_ids() -> set[str]:
    return document_edge_target_ids(load_document_edges(DOCUMENT_EDGES_PATH))


def _real_manifest() -> Manifest:
    return _manifest(*(_document(doc_id) for doc_id in _real_document_ids()))


def test_ingest_document_edges_skips_when_the_manifest_is_unrelated():
    manifest = _manifest(_document("unrelated-doc"))
    stats = asyncio.run(_ingest_document_edges(manifest, set(), session_factory=None))
    assert stats.document_id == "document_edges"
    assert stats.edges == 0
    assert stats.error is None


def test_ingest_document_edges_fails_when_limit_narrows_what_was_actually_ingested():
    # The exact --limit shape from the review: the manifest declares every
    # document the five seeds need, but this run only actually ingested
    # two of them (as --limit 2 would) -- must fail, not silently write
    # edges whose other three endpoints have no node anywhere.
    manifest = _real_manifest()
    ingested = {"pccc-decree-105-2025", "pccc-law-2024"}
    stats = asyncio.run(_ingest_document_edges(manifest, ingested, session_factory=None))
    assert stats.error is not None
    assert "pccc-circular-36-2025" in stats.error
    assert "gas-decree-87-2018" in stats.error


# --- persistence, the regex/curated collision, and review-status survival (DB) -----


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    """Create an engine, run ``body(engine)`` to completion, then dispose it."""

    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


def test_persisting_the_five_curated_edges_lands_pending_and_curated(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        stats = await _ingest_document_edges(
            _real_manifest(), _real_document_ids(), session_factory
        )
        assert stats.error is None
        assert stats.edges == 5
        assert stats.edges_updated == 0
        async with engine.begin() as conn:
            rows = (await conn.execute(text("SELECT review_status, source FROM edges"))).all()
        assert len(rows) == 5
        assert all(row.review_status == "pending" and row.source == "curated" for row in rows)

    _run(migrated_db, body)


def test_approving_a_curated_edge_survives_a_second_ingest(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        manifest = _real_manifest()
        document_ids = _real_document_ids()
        await _ingest_document_edges(manifest, document_ids, session_factory)
        async with engine.begin() as conn:
            edge_id = (
                await conn.execute(
                    text(
                        "SELECT id FROM edges WHERE src_id = 'lpg-qcvn-08-2012' "
                        "AND dst_id = 'lpg-qcvn-08-2019' AND edge_type = 'replaced_by'"
                    )
                )
            ).scalar_one()
            await conn.execute(
                text("UPDATE edges SET review_status = 'approved' WHERE id = :id"),
                {"id": edge_id},
            )

        # Re-ingest the same seed data.
        await _ingest_document_edges(manifest, document_ids, session_factory)

        async with engine.begin() as conn:
            status = (
                await conn.execute(
                    text("SELECT review_status FROM edges WHERE id = :id"), {"id": edge_id}
                )
            ).scalar_one()
        assert status == "approved"

    _run(migrated_db, body)


def test_a_colliding_regex_edge_is_overwritten_to_curated_and_counted_as_updated_not_new(
    migrated_db: str,
):
    # pccc-decree-105-2025 GUIDED_BY pccc-law-2024 can also come from the
    # regex pass. The curated pass runs after the per-document loop
    # (ingest/run.py's own call order), so its upsert must be the one that
    # decides `source` -- verified directly here, independent of the whole
    # pipeline's ordering, by upserting a regex edge first and then this
    # module's own curated edges. Also pins the stats-table invariant issue
    # #75 settled on: `edges` reports every seed edge this pass built
    # (uniform across every row, issue #75), and `edges_updated`, a subset
    # of it, says how many were an update rather than a new row -- so the
    # database's actual net gain is `edges - edges_updated`, not `edges`
    # itself. The TOTAL row no longer sums this column at all; it queries
    # `edges` directly (`EdgeRepository.count`), which is what actually
    # protects the table's total from exceeding the real row count now.
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await EdgeRepository(session).upsert_many(
                [
                    Edge(
                        src_id="pccc-decree-105-2025",
                        dst_id="pccc-law-2024",
                        edge_type=EdgeType.GUIDED_BY,
                        evidence="regex-extracted citation",
                        review_status=ReviewStatus.PENDING,
                        source=EdgeSource.REGEX,
                    )
                ]
            )
        async with engine.begin() as conn:
            before = (await conn.execute(text("SELECT count(*) FROM edges"))).scalar_one()

        stats = await _ingest_document_edges(
            _real_manifest(), _real_document_ids(), session_factory
        )

        async with engine.begin() as conn:
            after = (await conn.execute(text("SELECT count(*) FROM edges"))).scalar_one()
            row = (
                await conn.execute(
                    text(
                        "SELECT source FROM edges WHERE src_id = 'pccc-decree-105-2025' "
                        "AND dst_id = 'pccc-law-2024' AND edge_type = 'guided_by'"
                    )
                )
            ).one()

        assert row.source == "curated"
        assert stats.edges == 5  # every seed this pass built, updated or not
        assert stats.edges_updated == 1
        assert stats.edges - stats.edges_updated == after - before
        assert after - before == 4  # 5 seeds, 1 pre-existed -> 4 new rows

    _run(migrated_db, body)
