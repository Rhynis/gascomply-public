"""Persistence tests for repeal edges (issue #68): reconciliation, re-ingest, traversal.

Needs a live Postgres (``TEST_DATABASE_URL``, see ``tests/test_repositories.py``).

A repeal edge runs OLD -> NEW, so it starts at the repealed document while the
sentence that repeals it lives in the repealing one. ``reconcile_document_edges``
(issue #75) used to attribute an edge to a document by ``src_id``, which got that
shape wrong twice: an edge starting at a stub belonged to nobody, so a stale one was
never pruned; and a repealed document that was itself in the manifest saw the edge
as "not produced by my own parse" and deleted it on every run (or reported a false
conflict, once a human had approved it) while the repealing document re-created it.
The fix scopes reconciliation by ``origin_id``, the unit whose text produced the
edge. Every test here runs the real pipeline, twice where the point is churn.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from app.core.db import create_engine, create_session_factory, session_scope
from app.core.ids import stub_document_id
from app.core.migrate import apply_pending_migrations
from app.repositories.edges import EdgeRepository
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from app.services.graph_traversal import GraphTraversalService
from ingest.manifest import MANIFEST_FILENAME
from ingest.run import main

_REPEALING = "repealing-decree-200-2025"
_OLD = "old-decree-300-2015"
_STUB_83 = stub_document_id("83/2017/NĐ-CP")
_POINT_A = f"{_REPEALING}#dieu-2#khoan-2#diem-a"
_POINT_B = f"{_REPEALING}#dieu-2#khoan-2#diem-b"

_REPEALING_HTML = """<html><body>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Phạm vi điều chỉnh</p>
<p>Nghị định này quy định về điều kiện kinh doanh khí dầu mỏ hóa lỏng tại thị trường
Việt Nam, bao gồm các cửa hàng bán lẻ và trạm chiết nạp khí.</p>
<p>Điều 2. Hiệu lực thi hành</p>
<p>1. Nghị định này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2025.</p>
<p>2. Các Nghị định sau đây hết hiệu lực kể từ ngày Nghị định này có hiệu lực thi hành:</p>
<p>a) Nghị định số 83/2017/NĐ-CP ngày 18 tháng 7 năm 2017 quy định về công tác cứu nạn,
cứu hộ của lực lượng phòng cháy và chữa cháy;</p>
<p>b) Nghị định số 300/2015/NĐ-CP ngày 01 tháng 1 năm 2015 quy định về kinh doanh khí
dầu mỏ hóa lỏng.</p>
<p>3. Bãi bỏ khoản 1 và 2 Điều 30 của Nghị định số 400/2016/NĐ-CP.</p>
</body></html>"""
"""Real drafting shapes: a "hết hiệu lực" list whose point (b) names a document that
is *in the manifest*, so its edge starts at a document that is itself ingested and
reconciled -- the shape #75's scope got wrong -- and whose point (a) names one that is
not, so its edge starts at a stub. Khoản 3 is a partial repeal: it must build nothing."""

_REPEALING_HTML_WITHOUT_THE_REPEAL = """<html><body>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Phạm vi điều chỉnh</p>
<p>Nghị định này quy định về điều kiện kinh doanh khí dầu mỏ hóa lỏng tại thị trường
Việt Nam, bao gồm các cửa hàng bán lẻ và trạm chiết nạp khí.</p>
<p>Điều 2. Hiệu lực thi hành</p>
<p>1. Nghị định này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2025.</p>
</body></html>"""

_OLD_HTML = """<html><body>
<p>Điều 1. Nội dung nghị định.</p>
<p>Nghị định này quy định về kinh doanh khí dầu mỏ hóa lỏng trên toàn quốc, bao gồm
trách nhiệm của tổ chức, cá nhân trong hoạt động sản xuất, kinh doanh, sinh hoạt hằng
ngày, cũng như trách nhiệm của các cơ quan quản lý nhà nước có liên quan.</p>
</body></html>"""

_UNREADABLE_HTML = """<html><body>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Hiệu lực thi hành</p>
<p>1. Nghị định số 12/2019/NĐ-CP hết hiệu lực từ ngày 01 tháng 1 năm 2026.</p>
<p>2. Nghị định này quy định về điều kiện kinh doanh khí dầu mỏ hóa lỏng tại thị trường
Việt Nam, bao gồm các cửa hàng bán lẻ và trạm chiết nạp khí.</p>
</body></html>"""


def _entry(doc_id: str, filename: str, number: str) -> dict[str, Any]:
    return {
        "id": doc_id,
        "number": number,
        "title_vi": f"Văn bản {number}",
        "doc_type": "decree",
        "status": "in_force",
        "effective": "2015-01-01",
        "path": f"corpus/{filename}",
    }


def _write_corpus(
    root: Path, *, repealing_first: bool = True, repealing: str | None = None
) -> Path:
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    entries = [
        _entry(_REPEALING, "repealing.html", "200/2025/NĐ-CP"),
        _entry(_OLD, "old.html", "300/2015/NĐ-CP"),
    ]
    if not repealing_first:
        entries.reverse()
    (corpus / MANIFEST_FILENAME).write_text(json.dumps({"documents": entries}), encoding="utf-8")
    (corpus / "repealing.html").write_text(repealing or _REPEALING_HTML, encoding="utf-8")
    (corpus / "old.html").write_text(_OLD_HTML, encoding="utf-8")
    return corpus


def _args(corpus: Path, database_url: str, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--corpus",
        str(corpus),
        "--database-url",
        database_url,
        "--references",
        str(tmp_path / "references.csv"),
        *extra,
    ]


def _query(database_url: str, sql: str, params: dict[str, Any] | None = None) -> list[Any]:
    async def _run() -> list[Any]:
        engine = create_engine(database_url)
        try:
            async with engine.begin() as conn:
                return list((await conn.execute(text(sql), params or {})).all())
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def _execute(database_url: str, sql: str, params: dict[str, Any] | None = None) -> None:
    async def _run() -> None:
        engine = create_engine(database_url)
        try:
            async with engine.begin() as conn:
                await conn.execute(text(sql), params or {})
        finally:
            await engine.dispose()

    asyncio.run(_run())


def _repeal_rows(database_url: str) -> list[Any]:
    return _query(
        database_url,
        "SELECT id, src_id, dst_id, review_status, source, origin_id, evidence "
        "FROM edges WHERE edge_type = 'replaced_by' ORDER BY src_id",
    )


def _approve(database_url: str, where: str) -> None:
    _execute(database_url, f"UPDATE edges SET review_status = 'approved' WHERE {where}")


# --- what the pipeline persists ---------------------------------------------------------


def test_the_pipeline_persists_one_pending_repeal_edge_per_whole_document_and_none_for_a_partial(
    migrated_db: str, tmp_path: Path
):
    assert main(_args(_write_corpus(tmp_path), migrated_db, tmp_path)) == 0

    rows = _repeal_rows(migrated_db)

    assert [(row.src_id, row.dst_id) for row in rows] == [
        (_STUB_83, _REPEALING),  # a stub: 83/2017 is not in the manifest
        (_OLD, _REPEALING),  # a manifest document: 300/2015
    ]
    assert {(row.review_status, row.source) for row in rows} == {("pending", "regex")}
    assert [row.origin_id for row in rows] == [_POINT_A, _POINT_B]
    assert all(row.evidence.startswith("Các Nghị định sau đây hết hiệu lực") for row in rows)
    # Khoản 3 ("Bãi bỏ khoản 1 và 2 Điều 30 của Nghị định số 400/2016/NĐ-CP") built nothing.
    assert not any("400-2016" in row.src_id for row in rows)


def test_the_stub_a_repeal_edge_starts_at_exists_before_the_edge_does(
    migrated_db: str, tmp_path: Path
):
    main(_args(_write_corpus(tmp_path), migrated_db, tmp_path))

    stub = _query(
        migrated_db, "SELECT status, doc_type, path FROM documents WHERE id = :id", {"id": _STUB_83}
    )
    assert [(row.status, row.doc_type, row.path) for row in stub] == [("unknown", "unknown", None)]


# --- (a) the repeal edges survive a re-ingest unchanged, whatever order documents run in -----


@pytest.mark.parametrize("repealing_first", [True, False])
def test_repeal_edges_survive_a_reingest_with_the_same_rows_not_deleted_and_recreated(
    migrated_db: str, tmp_path: Path, capsys: Any, repealing_first: bool
):
    corpus = _write_corpus(tmp_path, repealing_first=repealing_first)
    args = _args(corpus, migrated_db, tmp_path)
    assert main(args) == 0
    capsys.readouterr()
    before = _repeal_rows(migrated_db)
    assert len(before) == 2

    assert main(args) == 0
    out = capsys.readouterr().out

    # A bigserial id changes if a row is deleted and inserted again: equal ids across the
    # two runs is churn-free, and it must hold for both the stub-sourced edge and the one
    # that starts at a manifest document reconciled in the very same run.
    assert _repeal_rows(migrated_db) == before
    assert "stale edge(s) removed" not in out
    assert "review needed" not in out


# --- (b) an approved repeal edge is never reported as a #75 conflict ------------------------


@pytest.mark.parametrize("repealing_first", [True, False])
def test_an_approved_repeal_edge_is_never_reported_as_a_stale_conflict(
    migrated_db: str, tmp_path: Path, capsys: Any, repealing_first: bool
):
    corpus = _write_corpus(tmp_path, repealing_first=repealing_first)
    args = _args(corpus, migrated_db, tmp_path)
    main(args)
    _approve(migrated_db, "edge_type = 'replaced_by'")
    capsys.readouterr()

    assert main(args) == 0
    out = capsys.readouterr().out

    assert "is approved but this parse no longer produces it" not in out
    assert "review needed" not in out
    rows = _repeal_rows(migrated_db)
    assert {row.review_status for row in rows} == {"approved"}
    assert len(rows) == 2


# --- (c) a repeal edge disappears when the clause that produced it does ---------------------


def test_a_stale_pending_repeal_edge_is_pruned_when_the_repealing_document_stops_stating_it(
    migrated_db: str, tmp_path: Path
):
    corpus = _write_corpus(tmp_path)
    args = _args(corpus, migrated_db, tmp_path)
    main(args)
    assert len(_repeal_rows(migrated_db)) == 2

    (corpus / "repealing.html").write_text(_REPEALING_HTML_WITHOUT_THE_REPEAL, encoding="utf-8")
    assert main(args) == 0

    # Both go: the one that starts at a stub -- which no ingested document owns by src_id --
    # and the one that starts at a manifest document.
    assert _repeal_rows(migrated_db) == []


def test_a_stale_approved_repeal_edge_is_reported_as_a_conflict_not_deleted(
    migrated_db: str, tmp_path: Path, capsys: Any
):
    corpus = _write_corpus(tmp_path)
    args = _args(corpus, migrated_db, tmp_path)
    main(args)
    _approve(migrated_db, "edge_type = 'replaced_by'")
    capsys.readouterr()
    ids = [row.id for row in _repeal_rows(migrated_db)]

    (corpus / "repealing.html").write_text(_REPEALING_HTML_WITHOUT_THE_REPEAL, encoding="utf-8")
    assert main(args) == 0
    out = capsys.readouterr().out

    assert {row.review_status for row in _repeal_rows(migrated_db)} == {"approved"}
    for edge_id in ids:
        assert f"edge id={edge_id}" in out
    assert "is approved but this parse no longer produces it" in out


def test_ingesting_only_the_repealed_document_never_touches_a_repeal_edge_it_did_not_produce(
    migrated_db: str, tmp_path: Path
):
    corpus = _write_corpus(tmp_path, repealing_first=False)  # manifest order: old, repealing
    args = _args(corpus, migrated_db, tmp_path)
    main(args)
    before = _repeal_rows(migrated_db)

    assert main([*args, "--limit", "1"]) == 0  # only the repealed document runs

    assert _repeal_rows(migrated_db) == before


# --- reconcile, at the repository: the mechanism, without the pipeline -----------------------


def _repeal_edge(src: str, dst: str, origin: str, *, source: EdgeSource = EdgeSource.REGEX) -> Edge:
    return Edge(
        src_id=src,
        dst_id=dst,
        edge_type=EdgeType.REPLACED_BY,
        evidence="Các Nghị định sau đây hết hiệu lực ...",
        review_status=ReviewStatus.PENDING,
        source=source,
        origin_id=origin,
    )


def _reconcile(database_url: str, document_id: str, produced: list[Edge]) -> Any:
    async def _run() -> Any:
        engine = create_engine(database_url)
        try:
            async with session_scope(create_session_factory(engine)) as session:
                return await EdgeRepository(session).reconcile_document_edges(document_id, produced)
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def _seed(database_url: str, *edges: Edge) -> None:
    async def _run() -> None:
        engine = create_engine(database_url)
        try:
            async with session_scope(create_session_factory(engine)) as session:
                await EdgeRepository(session).upsert_many(edges)
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_an_edge_that_starts_at_a_stub_is_reconciled_by_the_document_that_produced_it(
    migrated_db: str,
):
    _seed(migrated_db, _repeal_edge(_STUB_83, _REPEALING, _POINT_A))

    result = _reconcile(migrated_db, _REPEALING, produced=[])

    assert result.removed == 1  # by src_id this edge belonged to nobody, and was never pruned
    assert _repeal_rows(migrated_db) == []


def test_an_edge_another_document_produced_is_never_stale_for_the_document_it_starts_at(
    migrated_db: str,
):
    _seed(migrated_db, _repeal_edge(_OLD, _REPEALING, _POINT_B))
    _approve(migrated_db, "edge_type = 'replaced_by'")

    result = _reconcile(migrated_db, _OLD, produced=[])

    assert result.removed == 0
    assert result.conflicts == ()
    assert len(_repeal_rows(migrated_db)) == 1


def test_the_producing_document_keeps_an_edge_it_still_produces_and_reports_one_it_does_not(
    migrated_db: str,
):
    kept = _repeal_edge(_OLD, _REPEALING, _POINT_B)
    dropped = _repeal_edge(_STUB_83, _REPEALING, _POINT_A)
    _seed(migrated_db, kept, dropped)
    _approve(migrated_db, f"src_id = '{_STUB_83}'")

    result = _reconcile(migrated_db, _REPEALING, produced=[kept])

    assert result.removed == 0
    assert [(c.src_id, c.review_status) for c in result.conflicts] == [(_STUB_83, "approved")]


def test_a_curated_edge_and_an_edge_with_no_origin_are_in_no_documents_scope(migrated_db: str):
    curated = _repeal_edge(_OLD, _REPEALING, _POINT_B, source=EdgeSource.CURATED)
    unattributed = Edge(
        src_id=_STUB_83,
        dst_id=_REPEALING,
        edge_type=EdgeType.REPLACED_BY,
        review_status=ReviewStatus.PENDING,
        source=EdgeSource.REGEX,
    )
    _seed(migrated_db, curated, unattributed)

    result = _reconcile(migrated_db, _REPEALING, produced=[])

    assert (result.removed, result.conflicts) == (0, ())
    assert len(_repeal_rows(migrated_db)) == 2


def test_reconcile_does_not_mistake_a_document_whose_id_prefixes_another_for_it(
    migrated_db: str,
):
    longer = f"{_REPEALING}-bis"
    _seed(migrated_db, _repeal_edge(_STUB_83, longer, f"{longer}#dieu-2"))

    result = _reconcile(migrated_db, _REPEALING, produced=[])

    assert result.removed == 0
    assert len(_repeal_rows(migrated_db)) == 1


# --- migration 004: origin_id is backfilled from src_id, the rule it replaces ----------------


def test_the_migration_backfills_origin_from_the_source_so_no_existing_row_changes_scope(
    migrated_db: str,
):
    async def _rewind_and_reapply() -> list[str]:
        engine = create_engine(migrated_db)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("ALTER TABLE edges DROP COLUMN origin_id"))
                await conn.execute(
                    text("DELETE FROM schema_migrations WHERE filename = '004_edge_origin.sql'")
                )
                await conn.execute(
                    text(
                        "INSERT INTO edges (src_id, dst_id, edge_type, review_status, source) "
                        "VALUES ('doc-a#dieu-1', 'doc-a', 'belongs_to', 'approved', 'structural'), "
                        "('doc-a#dieu-1', 'cited-x', 'references', 'pending', 'regex'), "
                        "('doc-a', 'doc-b', 'guided_by', 'pending', 'curated')"
                    )
                )
            return await apply_pending_migrations(engine)
        finally:
            await engine.dispose()

    applied = asyncio.run(_rewind_and_reapply())

    assert applied == ["004_edge_origin.sql"]
    rows = _query(migrated_db, "SELECT src_id, source, origin_id FROM edges ORDER BY id")
    assert [(row.source, row.origin_id) for row in rows] == [
        ("structural", "doc-a#dieu-1"),
        ("regex", "doc-a#dieu-1"),
        ("curated", None),  # a curated edge is produced by no document's text
    ]


# --- traversal: the two mechanisms, told apart ----------------------------------------------


def _traverse(
    database_url: str, seed: str, follow: list[EdgeType], *, include_history: bool
) -> dict[str, bool]:
    async def _run() -> dict[str, bool]:
        engine = create_engine(database_url)
        try:
            async with session_scope(create_session_factory(engine)) as session:
                hits = await GraphTraversalService(session).traverse(
                    [seed], max_hops=1, follow_types=follow, include_history=include_history
                )
            return {hit.unit_id: hit.possibly_superseded for hit in hits}
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def test_an_approved_repeal_edge_reaches_the_repealing_document_and_removes_the_repealed_one(
    migrated_db: str, tmp_path: Path
):
    main(_args(_write_corpus(tmp_path), migrated_db, tmp_path))
    # Only the citation that lets a walk reach the old decree is approved first.
    _approve(migrated_db, f"edge_type = 'references' AND src_id = '{_POINT_B}'")
    reach_old = [EdgeType.REFERENCES]

    # Control: nothing has approved the repeal yet. The old decree is in force by its own
    # status, so the walk returns it -- flagged, because a *pending* replacement exists.
    assert _traverse(migrated_db, _POINT_B, reach_old, include_history=False) == {_OLD: True}

    _approve(migrated_db, f"edge_type = 'replaced_by' AND src_id = '{_OLD}'")

    # Mechanism 1, the EDGE: the old decree's own status is still in_force. It is the
    # approved REPLACED_BY that removes it from an ordinary answer...
    assert (
        _query(migrated_db, "SELECT status FROM documents WHERE id = :id", {"id": _OLD})[0].status
        == "in_force"
    )
    assert _traverse(migrated_db, _POINT_B, reach_old, include_history=False) == {}
    # ...and asking for history brings it back.
    assert _traverse(migrated_db, _POINT_B, reach_old, include_history=True) == {_OLD: False}

    # From the repealed document, the repealing one is reachable when history is requested.
    replaced = [EdgeType.REPLACED_BY]
    assert _REPEALING in _traverse(migrated_db, _OLD, replaced, include_history=True)


def test_a_stub_is_excluded_by_its_status_and_still_reaches_the_repealing_document_in_history(
    migrated_db: str, tmp_path: Path
):
    main(_args(_write_corpus(tmp_path), migrated_db, tmp_path))
    _approve(migrated_db, "edge_type IN ('replaced_by', 'references')")
    replaced = [EdgeType.REPLACED_BY]

    # Mechanism 2, the STATUS: a stub is 'unknown', never in_force, so a walk that lands on
    # it never reports it as current law -- with or without an approved REPLACED_BY.
    assert _traverse(migrated_db, _POINT_A, [EdgeType.REFERENCES], include_history=False) == {}
    assert _STUB_83 in _traverse(migrated_db, _POINT_A, [EdgeType.REFERENCES], include_history=True)
    # From the stub, history reaches the repealing document: 83/2017 -> 200/2025.
    assert _traverse(migrated_db, _STUB_83, replaced, include_history=True) == {_REPEALING: False}


# --- what a run prints -------------------------------------------------------------------------


def test_a_run_prints_the_repeal_edges_built_and_the_partial_repeals_it_counted_and_skipped(
    tmp_path: Path, capsys: Any
):
    corpus = _write_corpus(tmp_path)

    assert (
        main(["--corpus", str(corpus), "--references", str(tmp_path / "r.csv"), "--dry-run"]) == 0
    )
    out = capsys.readouterr().out

    assert (
        f"  INFO: {_REPEALING} 2 repeal edge(s) built (replaced_by, old -> new); "
        "1 partial repeal(s) counted and skipped, never an edge"
    ) in out
    assert f"INFO: {_OLD}" not in out  # a document with no repeal statement prints nothing
    assert "repeal statement(s) name a document" not in out


def test_a_statement_the_pass_cannot_read_is_a_warning_and_builds_nothing(
    tmp_path: Path, capsys: Any, migrated_db: str
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / MANIFEST_FILENAME).write_text(
        json.dumps(
            {"documents": [_entry("unreadable-decree-201-2025", "u.html", "201/2025/NĐ-CP")]}
        ),
        encoding="utf-8",
    )
    (corpus / "u.html").write_text(_UNREADABLE_HTML, encoding="utf-8")

    assert main(_args(corpus, migrated_db, tmp_path)) == 0
    out = capsys.readouterr().out

    assert "unreadable-decree-201-2025 1 repeal statement(s) name a document in a shape" in out
    assert _repeal_rows(migrated_db) == []
