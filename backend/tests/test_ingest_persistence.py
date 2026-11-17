"""Tests for ``ingest.run``'s opt-in persistence (build plan A6 Phase 1.5b, 2.1, 2.2).

Needs a live Postgres with pgvector (see ``tests/test_repositories.py``'s
module docstring). The central claim of this phase -- that running the CLI
twice over the same corpus leaves identical row counts, not doubled -- is
proven here empirically: ingest, count, ingest again, count again, compare.
A test that only checked the first run would prove nothing.

The fixture corpus mirrors ``test_ingest_cli.py``'s ``good_only_corpus``:
one document, one deliberately-empty article (Điều 3), so the ingest produces
a known, non-trivial unit/chunk count to assert against.
"""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.embedders.mock import MockEmbedder
from app.repositories.edges import EdgeRepository
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from ingest.manifest import MANIFEST_FILENAME
from ingest.run import main

_GOOD_ID = "test-gas-decree-2018"
"""Deliberately not the real ``gas-decree-87-2018`` id: A6 issue #51's
requirement cluster cites real units of that exact document, and this
fixture's synthetic content never reaches them -- reusing the real id would
make every persistence run here fail the new, correct cross-document check
in ``ingest/graph.py``'s ``build_requirement_graph``."""
_GOOD_HTML = """<html><body>
<p>Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015;</p>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Phạm vi điều chỉnh</p>
<p>Nghị định này quy định về điều kiện kinh doanh khí dầu mỏ hóa lỏng (LPG)
tại thị trường Việt Nam, bao gồm các cửa hàng bán lẻ và trạm chiết nạp khí.</p>
<p>Điều 2. Điều kiện đối với cửa hàng bán lẻ</p>
<p>1. Cửa hàng bán lẻ LPG chai phải đáp ứng các điều kiện sau đây theo quy
định của pháp luật hiện hành về phòng cháy chữa cháy và an toàn lao động.</p>
<p>a) Diện tích tối thiểu của khu vực bán hàng là 12 m2, bảo đảm khoảng cách
an toàn với các công trình lân cận và nguồn lửa, nguồn nhiệt.</p>
<p>Điều 3.</p>
</body></html>"""
"""Same fixture as test_ingest_cli.py's good_only_corpus: 1 chapter + 3
articles + 1 clause + 1 point = 6 units, 2 chunks (dieu-3 is empty)."""

_EXPECTED_UNITS = 6
_EXPECTED_CHUNKS = 2
_EXPECTED_EDGES = 6
"""One structural BELONGS_TO edge per unit (A6 Phase 2.1) -- always equal to
_EXPECTED_UNITS, kept as its own name so a test reads as "edges" not "units"."""

_NOANCHOR_ID = "noanchor-doc-2021"
_NOANCHOR_HTML = """<html><body>
<p>Đây là một trang không có cấu trúc điều khoản pháp luật nào cả, chỉ là văn
bản mô tả chung chung không mang tính quy phạm, có thể là nội dung bị lưu
nhầm từ một trang web khác không phải văn bản pháp luật thực sự cần trích
xuất, chẳng hạn như một trang chặn bot tự động của vbpl.vn.</p>
<p>Nội dung này tiếp tục không chứa bất kỳ từ khóa cấu trúc nào như chương,
điều, khoản, hoặc điểm được đánh số theo quy ước văn bản pháp luật Việt Nam.</p>
</body></html>"""
"""Extracts cleanly (well over the near-empty floor) but contains no
Chương/Điều/Khoản anchor at all -- parses to zero units. Mirrors
test_ingest_cli.py's noanchor_corpus."""

_CHAPTER_ONLY_ID = "chapter-only-doc"
_CHAPTER_ONLY_HTML = """<html><body>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Đây là nội dung giới thiệu chung của chương, không có điều khoản cụ thể
nào được liệt kê, chỉ mang tính chất mở đầu văn bản pháp luật để giải thích
bối cảnh và mục đích ban hành trước khi đi vào các quy định chi tiết.</p>
</body></html>"""
"""Parses to one real unit (the chapter) but build_chunks only chunks
articles and their descendants -- zero chunks. Mirrors test_ingest_cli.py's
chapter_only_corpus."""

_COUNT_QUERIES = {
    "documents": text("SELECT count(*) FROM documents"),
    "units": text("SELECT count(*) FROM units"),
    "chunks": text("SELECT count(*) FROM chunks"),
    "edges": text("SELECT count(*) FROM edges"),
}
"""One query per table, fixed at import time -- no string-built SQL (C3), not
even a table name interpolated from a Python-level constant."""


def _document_entry(
    doc_id: str, filename: str, *, number: str = "87/2018/NĐ-CP", doc_type: str = "decree"
) -> dict[str, Any]:
    """Return a minimal valid manifest document entry."""
    return {
        "id": doc_id,
        "number": number,
        "title_vi": "Nghị định thử nghiệm",
        "doc_type": doc_type,
        "status": "in_force",
        "effective": "2018-08-01",
        "path": f"corpus/{filename}",
    }


def _write_corpus(root: Path, documents: list[dict[str, Any]], files: dict[str, str]) -> Path:
    """Lay out ``<root>/corpus/manifest.json`` plus each named file; return the corpus dir."""
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / MANIFEST_FILENAME).write_text(json.dumps({"documents": documents}), encoding="utf-8")
    for name, content in files.items():
        (corpus / name).write_text(content, encoding="utf-8")
    return corpus


def _build_corpus(root: Path) -> Path:
    """Lay out a corpus with just the one good fixture document."""
    return _write_corpus(root, [_document_entry(_GOOD_ID, "good.html")], {"good.html": _GOOD_HTML})


def _build_noanchor_corpus(root: Path) -> Path:
    """Lay out a corpus with just the zero-unit fixture document."""
    return _write_corpus(
        root, [_document_entry(_NOANCHOR_ID, "noanchor.html")], {"noanchor.html": _NOANCHOR_HTML}
    )


def _build_chapter_only_corpus(root: Path) -> Path:
    """Lay out a corpus with just the units-but-zero-chunks fixture document."""
    return _write_corpus(
        root,
        [_document_entry(_CHAPTER_ONLY_ID, "chapter-only.html")],
        {"chapter-only.html": _CHAPTER_ONLY_HTML},
    )


def _build_mixed_corpus(root: Path) -> Path:
    """Lay out one good document plus both silent-loss shapes."""
    return _write_corpus(
        root,
        [
            _document_entry(_GOOD_ID, "good.html"),
            _document_entry(_NOANCHOR_ID, "noanchor.html"),
            _document_entry(_CHAPTER_ONLY_ID, "chapter-only.html"),
        ],
        {
            "good.html": _GOOD_HTML,
            "noanchor.html": _NOANCHOR_HTML,
            "chapter-only.html": _CHAPTER_ONLY_HTML,
        },
    )


_LAW_ID = "law-fire-prevention-2001"
_LAW_NUMBER = "27/2001/QH10"
_LAW_HTML = (
    "<html><body><p>Điều 1. Nội dung luật.</p>"
    "<p>Luật này quy định về phòng cháy và chữa cháy trên toàn quốc, "
    "bao gồm trách nhiệm của tổ chức, cá nhân trong hoạt động sản "
    "xuất, kinh doanh, sinh hoạt hằng ngày, cũng như trách nhiệm "
    "của các cơ quan quản lý nhà nước có liên quan trong việc "
    "thanh tra, kiểm tra và xử lý vi phạm.</p></body></html>"
)

_DEDUP_ID = "gas-decree-dedup-2018"
_DEDUP_HTML = """<html><body>
<p>Căn cứ Luật Phòng cháy và chữa cháy số 27/2001/QH10;</p>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Phạm vi điều chỉnh</p>
<p>Nghị định này quy định về điều kiện kinh doanh khí dầu mỏ hóa lỏng tại thị
trường Việt Nam, bao gồm các cửa hàng bán lẻ và trạm chiết nạp khí.</p>
<p>Căn cứ Luật Phòng cháy và chữa cháy số 27/2001/QH10.</p>
</body></html>"""
"""Cites the same external law twice from two different locations (the
preamble, and Điều 1's own body) -- both resolve to the same
(src=this document, dst=the law, edge_type=GUIDED_BY) edge. Without
``ingest/graph.py``'s own dedup, upserting both in the same transaction
would hit Postgres' "ON CONFLICT DO UPDATE command cannot affect row a
second time" (A6 Phase 2.2)."""


def _build_dedup_corpus(root: Path) -> Path:
    """Lay out the citing decree plus the law it cites twice."""
    return _write_corpus(
        root,
        [
            _document_entry(_DEDUP_ID, "dedup.html"),
            _document_entry(_LAW_ID, "law.html", number=_LAW_NUMBER, doc_type="law"),
        ],
        {"dedup.html": _DEDUP_HTML, "law.html": _LAW_HTML},
    )


_STALE_ID = "stale-basis-doc"
_STALE_HTML_WITH_CITATION = """<html><body>
<p>Căn cứ Luật Phòng cháy và chữa cháy số 27/2001/QH10;</p>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Phạm vi điều chỉnh</p>
<p>Nghị định này quy định về điều kiện kinh doanh khí dầu mỏ hóa lỏng tại thị
trường Việt Nam, bao gồm các cửa hàng bán lẻ và trạm chiết nạp khí.</p>
</body></html>"""
_STALE_HTML_WITHOUT_CITATION = """<html><body>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Phạm vi điều chỉnh</p>
<p>Nghị định này quy định về điều kiện kinh doanh khí dầu mỏ hóa lỏng tại thị
trường Việt Nam, bao gồm các cửa hàng bán lẻ và trạm chiết nạp khí, áp dụng
thống nhất trên toàn quốc cho mọi tổ chức, cá nhân tham gia hoạt động kinh
doanh thuộc phạm vi điều chỉnh của văn bản này.</p>
</body></html>"""
"""Same document, minus its one "Căn cứ ..." citation -- a second parse of
this version no longer produces the GUIDED_BY edge the first version did
(issue #75's stale-edge fixture)."""


def _build_stale_edge_corpus(root: Path) -> Path:
    """Lay out a decree citing a law (GUIDED_BY, decree -> law), plus the law itself."""
    return _write_corpus(
        root,
        [
            _document_entry(_STALE_ID, "stale.html"),
            _document_entry(_LAW_ID, "law.html", number=_LAW_NUMBER, doc_type="law"),
        ],
        {"stale.html": _STALE_HTML_WITH_CITATION, "law.html": _LAW_HTML},
    )


async def _document_ids(database_url: str) -> set[str]:
    """Return every id currently in ``documents``."""
    engine = create_engine(database_url)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT id FROM documents"))
            return {row[0] for row in result}
    finally:
        await engine.dispose()


async def _edge_exists(database_url: str, src_id: str, dst_id: str, edge_type: str) -> bool:
    """Return whether an edge with this exact triple currently has a row."""
    engine = create_engine(database_url)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text(
                    "SELECT 1 FROM edges WHERE src_id = :src AND dst_id = :dst "
                    "AND edge_type = :edge_type"
                ),
                {"src": src_id, "dst": dst_id, "edge_type": edge_type},
            )
            return result.first() is not None
    finally:
        await engine.dispose()


async def _edge_id(database_url: str, src_id: str, dst_id: str, edge_type: str) -> int:
    """Return the database row id of an edge with this exact triple."""
    engine = create_engine(database_url)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text(
                    "SELECT id FROM edges WHERE src_id = :src AND dst_id = :dst "
                    "AND edge_type = :edge_type"
                ),
                {"src": src_id, "dst": dst_id, "edge_type": edge_type},
            )
            return result.scalar_one()
    finally:
        await engine.dispose()


async def _edge_review_status(database_url: str, edge_id: int) -> str:
    """Return the current ``review_status`` of an edge, by row id."""
    engine = create_engine(database_url)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT review_status FROM edges WHERE id = :id"), {"id": edge_id}
            )
            return result.scalar_one()
    finally:
        await engine.dispose()


async def _set_review_status(database_url: str, edge_id: int, review_status: ReviewStatus) -> None:
    """Set an edge's ``review_status`` directly, as a human would through ``ingest.review``."""
    engine = create_engine(database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await EdgeRepository(session).set_review_status([edge_id], review_status)
    finally:
        await engine.dispose()


async def _seed_edge(database_url: str, edge: Edge) -> None:
    """Upsert one edge directly, bypassing the ingest pipeline (to seed a curated row)."""
    engine = create_engine(database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await EdgeRepository(session).upsert_many([edge])
    finally:
        await engine.dispose()


def _table_row(out: str, document_id: str) -> dict[str, str]:
    """Parse one printed stats-table row (including ``TOTAL``) into a ``{column: value}`` dict."""
    lines = out.splitlines()
    header = next(line for line in lines if line.startswith("document"))
    row = next(line for line in lines if line.startswith(document_id))
    return dict(zip(header.split(), row.split(), strict=True))


def _counts(database_url: str) -> dict[str, int]:
    """Return row counts for documents/units/chunks, in one engine/transaction."""

    async def _all() -> dict[str, int]:
        engine: AsyncEngine = create_engine(database_url)
        try:
            async with engine.begin() as conn:
                return {
                    table: (await conn.execute(query)).scalar_one()
                    for table, query in _COUNT_QUERIES.items()
                }
        finally:
            await engine.dispose()

    return asyncio.run(_all())


def test_running_the_cli_twice_leaves_identical_row_counts(migrated_db: str, tmp_path: Path):
    corpus = _build_corpus(tmp_path)
    references = tmp_path / "references.csv"
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(references),
    ]

    first_exit = main(args)
    first_counts = _counts(migrated_db)

    second_exit = main(args)
    second_counts = _counts(migrated_db)

    assert first_exit == 0
    assert second_exit == 0
    assert first_counts == {
        "documents": 1,
        "units": _EXPECTED_UNITS,
        "chunks": _EXPECTED_CHUNKS,
        "edges": _EXPECTED_EDGES,
    }
    assert second_counts == first_counts, "a second run must upsert in place, not duplicate rows"


def test_persisted_counts_match_the_printed_stats_table(
    migrated_db: str, tmp_path: Path, capsys: Any
):
    corpus = _build_corpus(tmp_path)
    main(
        [
            "--corpus",
            str(corpus),
            "--database-url",
            migrated_db,
            "--references",
            str(tmp_path / "references.csv"),
        ]
    )
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith(_GOOD_ID))
    header = next(line for line in out.splitlines() if line.startswith("document"))
    printed = dict(zip(header.split(), row.split(), strict=True))

    counts = _counts(migrated_db)
    assert int(printed["units"]) == counts["units"]
    assert int(printed["chunks"]) == counts["chunks"]
    assert int(printed["edges"]) == counts["edges"]


def test_dry_run_overrides_a_configured_database_url(migrated_db: str, tmp_path: Path):
    corpus = _build_corpus(tmp_path)
    main(
        [
            "--corpus",
            str(corpus),
            "--database-url",
            migrated_db,
            "--dry-run",
            "--references",
            str(tmp_path / "references.csv"),
        ]
    )

    assert _counts(migrated_db) == {"documents": 0, "units": 0, "chunks": 0, "edges": 0}


def test_re_ingesting_with_a_changed_document_updates_rather_than_duplicates(
    migrated_db: str, tmp_path: Path
):
    corpus = _build_corpus(tmp_path)
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]
    main(args)

    manifest_path = corpus / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][0]["status"] = "expired"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    main(args)

    async def _status() -> str:
        engine = create_engine(migrated_db)
        try:
            async with engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT status FROM documents WHERE id = :id"), {"id": _GOOD_ID}
                )
                return result.scalar_one()
        finally:
            await engine.dispose()

    assert asyncio.run(_status()) == "expired"
    assert _counts(migrated_db)["documents"] == 1


# --- a document the CLI reports as failed must not land in the database ---------


def test_a_zero_unit_document_is_not_persisted(migrated_db: str, tmp_path: Path):
    corpus = _build_noanchor_corpus(tmp_path)
    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--database-url",
            migrated_db,
            "--references",
            str(tmp_path / "references.csv"),
        ]
    )

    assert exit_code == 1
    assert _counts(migrated_db) == {"documents": 0, "units": 0, "chunks": 0, "edges": 0}


def test_a_units_but_zero_chunks_document_is_not_persisted(migrated_db: str, tmp_path: Path):
    corpus = _build_chapter_only_corpus(tmp_path)
    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--database-url",
            migrated_db,
            "--references",
            str(tmp_path / "references.csv"),
        ]
    )

    assert exit_code == 1
    assert _counts(migrated_db) == {"documents": 0, "units": 0, "chunks": 0, "edges": 0}


def test_a_mixed_run_persists_only_the_documents_that_did_not_fail(
    migrated_db: str, tmp_path: Path
):
    corpus = _build_mixed_corpus(tmp_path)
    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--database-url",
            migrated_db,
            "--references",
            str(tmp_path / "references.csv"),
        ]
    )

    assert exit_code == 1
    assert _counts(migrated_db) == {
        "documents": 1,
        "units": _EXPECTED_UNITS,
        "chunks": _EXPECTED_CHUNKS,
        "edges": _EXPECTED_EDGES,
    }
    assert asyncio.run(_document_ids(migrated_db)) == {_GOOD_ID}


def test_idempotency_holds_when_the_corpus_contains_a_failing_document(
    migrated_db: str, tmp_path: Path
):
    corpus = _build_mixed_corpus(tmp_path)
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]

    first_exit = main(args)
    first_counts = _counts(migrated_db)

    second_exit = main(args)
    second_counts = _counts(migrated_db)

    assert first_exit == 1
    assert second_exit == 1
    assert first_counts == {
        "documents": 1,
        "units": _EXPECTED_UNITS,
        "chunks": _EXPECTED_CHUNKS,
        "edges": _EXPECTED_EDGES,
    }
    assert second_counts == first_counts


def test_reingest_prunes_units_chunks_and_structural_edges_removed_by_new_parse(
    migrated_db: str, tmp_path: Path
):
    corpus = _build_corpus(tmp_path)
    document_path = corpus / "good.html"
    document_path.write_text(
        _GOOD_HTML.replace(
            "</body></html>", "<p>PHỤ LỤC I</p><p>1. Phụ lục thử nghiệm.</p></body></html>"
        ),
        encoding="utf-8",
    )
    references = tmp_path / "references.csv"
    args = ["--corpus", str(corpus), "--database-url", migrated_db, "--references", str(references)]
    assert main(args) == 0
    appendix_id = f"{_GOOD_ID}#phu-luc-1#muc-1"

    async def _has_appendix() -> bool:
        engine = create_engine(migrated_db)
        try:
            async with engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT 1 FROM units WHERE id = :id"), {"id": appendix_id}
                )
                return result.first() is not None
        finally:
            await engine.dispose()

    assert asyncio.run(_has_appendix())

    document_path.write_text(_GOOD_HTML, encoding="utf-8")
    assert main(args) == 0

    async def _remaining() -> tuple[int, int, int]:
        engine = create_engine(migrated_db)
        try:
            async with engine.begin() as conn:
                units = (
                    await conn.execute(
                        text("SELECT count(*) FROM units WHERE id LIKE :prefix"),
                        {"prefix": f"{_GOOD_ID}#phu-luc-%"},
                    )
                ).scalar_one()
                chunks = (
                    await conn.execute(
                        text("SELECT count(*) FROM chunks WHERE unit_id LIKE :prefix"),
                        {"prefix": f"{_GOOD_ID}#phu-luc-%"},
                    )
                ).scalar_one()
                edges = (
                    await conn.execute(
                        text("SELECT count(*) FROM edges WHERE origin_id LIKE :prefix"),
                        {"prefix": f"{_GOOD_ID}#phu-luc-%"},
                    )
                ).scalar_one()
                return units, chunks, edges
        finally:
            await engine.dispose()

    assert asyncio.run(_remaining()) == (0, 0, 0)


# --- chunks must prune by produced chunk ids, not surviving unit ids (issue #117) --
#
# A unit can survive a re-parse unchanged while the *chunk* built from it does
# not: ``build_chunks`` rebundles a shortened article's clauses into one
# article-level chunk instead of one per clause once the article's composed
# text drops at or under ``ARTICLE_SPLIT_THRESHOLD``. The bug this fixture
# exercises pruned chunks against the surviving *unit* ids, so the old
# clause-level chunks stayed embedded forever even though their units (and
# therefore the old "keep" list) never changed.

_RESHAPE_ID = "test-reshape-chunk-2020"
_RESHAPE_SENTENCE = (
    "Cửa hàng bán lẻ khí dầu mỏ hóa lỏng phải tuân thủ các quy định về phòng "
    "cháy chữa cháy theo quy định của pháp luật hiện hành ở mọi thời điểm hoạt động. "
)


def _reshape_html(repeat: int) -> str:
    """One article, two clauses, each repeating the filler sentence ``repeat`` times.

    ``repeat=8`` pushes the article's composed text (header + both clauses)
    well past ``ARTICLE_SPLIT_THRESHOLD`` (2000 chars), so ``build_chunks``
    splits it into one chunk per clause. ``repeat=1`` leaves it well under
    the threshold, so the whole article becomes a single chunk -- same
    three units (the article plus its two clauses) either way, only the
    chunking decision changes.
    """
    clause_text = _RESHAPE_SENTENCE * repeat
    return (
        "<html><body>"
        "<p>Điều 1. Điều kiện an toàn</p>"
        f"<p>1. {clause_text}</p>"
        f"<p>2. {clause_text}</p>"
        "</body></html>"
    )


def _build_reshape_corpus(root: Path) -> Path:
    """Lay out a corpus with just the reshape fixture document, split per clause."""
    return _write_corpus(
        root, [_document_entry(_RESHAPE_ID, "reshape.html")], {"reshape.html": _reshape_html(8)}
    )


async def _chunk_unit_ids(database_url: str, document_id: str) -> set[str]:
    """Return every ``chunks.unit_id`` currently stored for a document."""
    engine = create_engine(database_url)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT unit_id FROM chunks WHERE document_id = :document_id"),
                {"document_id": document_id},
            )
            return {row.unit_id for row in result}
    finally:
        await engine.dispose()


async def _unit_ids_of(database_url: str, document_id: str) -> set[str]:
    """Return every ``units.id`` currently stored for a document."""
    engine = create_engine(database_url)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT id FROM units WHERE document_id = :document_id"),
                {"document_id": document_id},
            )
            return {row.id for row in result}
    finally:
        await engine.dispose()


def test_reingest_prunes_a_chunk_whose_surviving_unit_stops_being_chunked_alone(
    migrated_db: str, tmp_path: Path
):
    corpus = _build_reshape_corpus(tmp_path)
    document_path = corpus / "reshape.html"
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]
    article_id = f"{_RESHAPE_ID}#dieu-1"
    clause_1_id = f"{article_id}#khoan-1"
    clause_2_id = f"{article_id}#khoan-2"

    assert main(args) == 0
    # First parse: the article is long enough to split -- one chunk per clause.
    assert asyncio.run(_chunk_unit_ids(migrated_db, _RESHAPE_ID)) == {clause_1_id, clause_2_id}

    document_path.write_text(_reshape_html(1), encoding="utf-8")
    assert main(args) == 0

    # Both clause units still exist: the second parse still produces them.
    assert {clause_1_id, clause_2_id} <= asyncio.run(_unit_ids_of(migrated_db, _RESHAPE_ID))
    # But the article is now short enough to stay one chunk, keyed on the
    # article's own id -- the two old clause-level chunks must be gone, not
    # left behind because their units happened to survive.
    assert asyncio.run(_chunk_unit_ids(migrated_db, _RESHAPE_ID)) == {article_id}


def test_reingest_reports_the_pruned_chunk_in_the_stats_table(
    migrated_db: str, tmp_path: Path, capsys: Any
):
    corpus = _build_reshape_corpus(tmp_path)
    document_path = corpus / "reshape.html"
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]
    main(args)
    capsys.readouterr()  # discard the first run's output

    document_path.write_text(_reshape_html(1), encoding="utf-8")
    main(args)
    out = capsys.readouterr().out
    row = _table_row(out, _RESHAPE_ID)

    assert row["units_pruned"] == "0"
    assert row["chunks_pruned"] == "2"


# --- a parse that would prune too much is refused (issue #117, review point 2) --

_LARGE_PRUNE_ID = "test-large-prune-2020"
_LARGE_PRUNE_HTML_FULL = (
    "<html><body><p>Điều 1. Điều kiện an toàn</p>"
    + "".join(
        f"<p>{i}. Quy định về điều kiện an toàn số {i} cho cửa hàng bán lẻ khí dầu mỏ hóa lỏng.</p>"
        for i in range(1, 7)
    )
    + "</body></html>"
)
"""One article, six clauses -- seven units in all."""

_LARGE_PRUNE_HTML_SHRUNK = (
    "<html><body><p>Điều 1. Điều kiện an toàn</p>"
    "<p>1. Quy định về điều kiện an toàn số 1 cho cửa hàng bán lẻ khí dầu mỏ hóa lỏng, "
    "áp dụng thống nhất trên toàn quốc cho mọi tổ chức, cá nhân tham gia hoạt động "
    "kinh doanh thuộc phạm vi điều chỉnh của văn bản này.</p></body></html>"
)
"""Same article, down to just its first clause -- two units, five fewer than
``_LARGE_PRUNE_HTML_FULL``'s seven, well over half."""


def _build_large_prune_corpus(root: Path) -> Path:
    """Lay out a corpus with just the large-prune fixture document, at its full size."""
    return _write_corpus(
        root,
        [_document_entry(_LARGE_PRUNE_ID, "large_prune.html")],
        {"large_prune.html": _LARGE_PRUNE_HTML_FULL},
    )


def test_a_parse_that_would_prune_more_than_half_of_existing_units_is_refused(
    migrated_db: str, tmp_path: Path
):
    corpus = _build_large_prune_corpus(tmp_path)
    document_path = corpus / "large_prune.html"
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]
    assert main(args) == 0
    first_counts = _counts(migrated_db)
    assert first_counts["units"] == 7

    document_path.write_text(_LARGE_PRUNE_HTML_SHRUNK, encoding="utf-8")
    exit_code = main(args)

    assert exit_code == 1
    # Refused before anything was written -- the database is untouched.
    assert _counts(migrated_db) == first_counts


def test_refusing_a_large_prune_reports_why(migrated_db: str, tmp_path: Path, capsys: Any):
    corpus = _build_large_prune_corpus(tmp_path)
    document_path = corpus / "large_prune.html"
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]
    main(args)
    capsys.readouterr()  # discard the first run's output

    document_path.write_text(_LARGE_PRUNE_HTML_SHRUNK, encoding="utf-8")
    main(args)
    out = capsys.readouterr().out

    assert f"ERROR: {_LARGE_PRUNE_ID}" in out
    assert "prune 5 of 7 existing units" in out
    assert "--allow-large-prune" in out


def test_allow_large_prune_lets_the_refused_parse_through(migrated_db: str, tmp_path: Path):
    corpus = _build_large_prune_corpus(tmp_path)
    document_path = corpus / "large_prune.html"
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]
    main(args)

    document_path.write_text(_LARGE_PRUNE_HTML_SHRUNK, encoding="utf-8")
    exit_code = main([*args, "--allow-large-prune"])

    assert exit_code == 0
    assert _counts(migrated_db)["units"] == 2


# --- structural edges (build plan A6 Phase 2.1) ---------------------------------


def test_persisted_edges_chain_from_a_point_up_to_the_document(migrated_db: str, tmp_path: Path):
    corpus = _build_corpus(tmp_path)
    main(
        [
            "--corpus",
            str(corpus),
            "--database-url",
            migrated_db,
            "--references",
            str(tmp_path / "references.csv"),
        ]
    )

    point_id = f"{_GOOD_ID}#dieu-2#khoan-1#diem-a"

    async def _walk_to_document() -> list[str]:
        engine = create_engine(migrated_db)
        try:
            chain = [point_id]
            node = point_id
            async with engine.begin() as conn:
                while node != _GOOD_ID:
                    result = await conn.execute(
                        text(
                            "SELECT dst_id FROM edges "
                            "WHERE src_id = :src AND edge_type = 'belongs_to'"
                        ),
                        {"src": node},
                    )
                    node = result.scalar_one()
                    chain.append(node)
            return chain
        finally:
            await engine.dispose()

    chain = asyncio.run(_walk_to_document())
    assert chain == [
        point_id,
        f"{_GOOD_ID}#dieu-2#khoan-1",
        f"{_GOOD_ID}#dieu-2",
        f"{_GOOD_ID}#chuong-1",
        _GOOD_ID,
    ]


# --- reference edges (build plan A6 Phase 2.2) -----------------------------------


def test_a_basis_cited_twice_persists_as_one_edge_not_a_conflict_error(
    migrated_db: str, tmp_path: Path
):
    corpus = _build_dedup_corpus(tmp_path)
    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--database-url",
            migrated_db,
            "--references",
            str(tmp_path / "references.csv"),
        ]
    )

    # Postgres would raise "ON CONFLICT DO UPDATE command cannot affect row a
    # second time" here if ingest/graph.py's dedup ever regressed -- a
    # non-zero exit would mean exactly that (see _ingest_document's broad
    # except), not just a wrong count.
    assert exit_code == 0

    async def _edge_type() -> str:
        engine = create_engine(migrated_db)
        try:
            async with engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT edge_type FROM edges WHERE src_id = :src AND dst_id = :dst"),
                    {"src": _DEDUP_ID, "dst": _LAW_ID},
                )
                return result.scalar_one()
        finally:
            await engine.dispose()

    assert asyncio.run(_edge_type()) == "guided_by"


# --- metadata-only documents (issue #59) -----------------------------------------
#
# A document declared with "acquired": false is part of the corpus but has no file
# on disk -- permanently for a copyrighted standard, or until someone downloads it
# (see SOURCES.md § Licensing, and lpg-qcvn-08-2012, kept only for its REPLACED_BY
# edge). It still becomes a documents row, so an edge naming it still resolves, but
# contributes zero units/chunks -- the text pipeline never runs for it at all.

_NOT_ACQUIRED_ID = "old-standard-2012"
_NOT_ACQUIRED_NUMBER = "10/2012/NĐ-CP"
"""A plain decree-style number, not the real QCVN-style ``10:2012/BKHCN`` format
SOURCES.md records for the document this fixture models (``lpg-qcvn-08-2012``):
``ingest/references.py``'s ``AMENDMENT_PATTERN`` only captures a ``_NUMBERED_DOC``
tail today, not a ``_STANDARD_DOC`` one -- a separate, pre-existing regex gap, out
of scope for #59. This fixture stays inside what the pattern already detects, so
the test exercises real, unmodified reference resolution rather than a hand-built
``Reference``."""

_CITING_ID = "gas-decree-repeal-2021"
_CITING_HTML = """<html><body>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Hiệu lực thi hành</p>
<p>Nghị định này bãi bỏ Nghị định số 10/2012/NĐ-CP kể từ ngày Nghị định này có
hiệu lực thi hành, đồng thời thay thế toàn bộ các quy định trước đây về cùng nội
dung để bảo đảm tính thống nhất của hệ thống pháp luật hiện hành.</p>
</body></html>"""
"""Real "bãi bỏ ... số N" text -- the same repeal cue and resolution machinery a
document replacing an expired standard would use; only the target's acquired
status is what #59 changes anything about."""


def _build_not_acquired_corpus(root: Path) -> Path:
    """Lay out the citing decree plus the not-acquired document it repeals."""
    return _write_corpus(
        root,
        [
            _document_entry(_CITING_ID, "citing.html"),
            {
                **_document_entry(
                    _NOT_ACQUIRED_ID, "old-standard.pdf", number=_NOT_ACQUIRED_NUMBER
                ),
                "status": "expired",
                "acquired": False,
            },
        ],
        {"citing.html": _CITING_HTML},
    )


def test_a_not_acquired_document_gets_a_row_but_zero_units_and_chunks(
    migrated_db: str, tmp_path: Path
):
    corpus = _build_not_acquired_corpus(tmp_path)
    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--database-url",
            migrated_db,
            "--references",
            str(tmp_path / "references.csv"),
        ]
    )
    assert exit_code == 0

    async def _row() -> tuple[int, int, int]:
        engine = create_engine(migrated_db)
        try:
            async with engine.begin() as conn:
                documents = (
                    await conn.execute(
                        text("SELECT count(*) FROM documents WHERE id = :id"),
                        {"id": _NOT_ACQUIRED_ID},
                    )
                ).scalar_one()
                units = (
                    await conn.execute(
                        text("SELECT count(*) FROM units WHERE document_id = :id"),
                        {"id": _NOT_ACQUIRED_ID},
                    )
                ).scalar_one()
                chunks = (
                    await conn.execute(
                        text("SELECT count(*) FROM chunks WHERE document_id = :id"),
                        {"id": _NOT_ACQUIRED_ID},
                    )
                ).scalar_one()
                return documents, units, chunks
        finally:
            await engine.dispose()

    documents, units, chunks = asyncio.run(_row())
    assert documents == 1
    assert units == 0
    assert chunks == 0


def test_an_edge_naming_a_not_acquired_document_still_resolves(migrated_db: str, tmp_path: Path):
    corpus = _build_not_acquired_corpus(tmp_path)
    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--database-url",
            migrated_db,
            "--references",
            str(tmp_path / "references.csv"),
        ]
    )
    assert exit_code == 0

    async def _replaced_by_edge() -> tuple[str, str] | None:
        engine = create_engine(migrated_db)
        try:
            async with engine.begin() as conn:
                result = await conn.execute(
                    text(
                        "SELECT src_id, dst_id FROM edges "
                        "WHERE edge_type = 'replaced_by' "
                        "AND (src_id = :id OR dst_id = :id)"
                    ),
                    {"id": _NOT_ACQUIRED_ID},
                )
                row = result.first()
                return (row.src_id, row.dst_id) if row is not None else None
        finally:
            await engine.dispose()

    edge = asyncio.run(_replaced_by_edge())
    assert edge is not None, "the repeal cue naming the not-acquired document must still resolve"
    # "bãi bỏ" is the reversed cue (ingest/graph.py's _resolve_amendment): the
    # document it names becomes the OLD side, per A5.2's OLD -> NEW direction.
    assert edge == (_NOT_ACQUIRED_ID, _CITING_ID)


# --- stale edges: a re-ingest must prune what the corpus stopped producing (issue #75) --
#
# #74 fixed a misclassification that had produced 14 wrong edges; nothing had removed
# them once the fix landed, because every endpoint still resolved -- they were not
# dangling, just wrong, and sat in the review queue looking exactly as plausible as a
# real edge. These tests reproduce the general shape directly: a document whose second
# parse stops producing an edge its first parse produced.


def test_a_stale_pending_edge_is_removed_when_the_second_parse_stops_producing_it(
    migrated_db: str, tmp_path: Path, capsys: Any
):
    corpus = _build_stale_edge_corpus(tmp_path)
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]
    edge = (_STALE_ID, _LAW_ID, "guided_by")

    first_exit = main(args)
    first_out = capsys.readouterr().out
    first_db_edges = _counts(migrated_db)["edges"]

    assert first_exit == 0
    assert asyncio.run(_edge_exists(migrated_db, *edge)) is True
    # The TOTAL row must equal the database's own count on this run too, not
    # only after a first, from-empty ingest (issue #75's second defect).
    assert int(_table_row(first_out, "TOTAL")["edges"]) == first_db_edges

    # Second parse: the citation is gone, so this document no longer produces
    # the edge at all -- not "unresolved", genuinely absent.
    (corpus / "stale.html").write_text(_STALE_HTML_WITHOUT_CITATION, encoding="utf-8")

    second_exit = main(args)
    second_out = capsys.readouterr().out
    second_db_edges = _counts(migrated_db)["edges"]

    assert second_exit == 0
    assert asyncio.run(_edge_exists(migrated_db, *edge)) is False
    assert int(_table_row(second_out, "TOTAL")["edges"]) == second_db_edges
    assert second_db_edges == first_db_edges - 1


def test_a_curated_edge_is_never_pruned_even_when_this_document_does_not_produce_it(
    migrated_db: str, tmp_path: Path
):
    corpus = _build_corpus(tmp_path)
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]
    assert main(args) == 0

    # A curated edge attributed to _GOOD_ID that its own text will never
    # produce -- the same shape as a requirements.json/document_edges.json
    # seed edge. If reconciliation ever treated "not produced by this parse"
    # as "delete" without excluding curated rows, this would vanish on the
    # very next re-ingest of the exact same corpus.
    curated = Edge(
        src_id=_GOOD_ID,
        dst_id="some-required-standard",
        edge_type=EdgeType.APPLIES_TO,
        evidence=None,
        confidence=None,
        review_status=ReviewStatus.PENDING,
        source=EdgeSource.CURATED,
    )
    asyncio.run(_seed_edge(migrated_db, curated))

    assert main(args) == 0

    assert (
        asyncio.run(_edge_exists(migrated_db, _GOOD_ID, "some-required-standard", "applies_to"))
        is True
    )


def test_limit_leaves_a_document_outside_its_scope_untouched(migrated_db: str, tmp_path: Path):
    corpus = _write_corpus(
        tmp_path,
        [
            _document_entry(_GOOD_ID, "good.html"),
            _document_entry(_STALE_ID, "stale.html"),
            _document_entry(_LAW_ID, "law.html", number=_LAW_NUMBER, doc_type="law"),
        ],
        {"good.html": _GOOD_HTML, "stale.html": _STALE_HTML_WITH_CITATION, "law.html": _LAW_HTML},
    )
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]
    edge = (_STALE_ID, _LAW_ID, "guided_by")

    assert main(args) == 0
    assert asyncio.run(_edge_exists(migrated_db, *edge)) is True

    # stale.html no longer cites the law at all, but --limit 1 narrows this
    # run to good.html alone (manifest order: good, stale, law) -- stale's
    # own edges must be left exactly as they were, not reconciled against a
    # document this run never touched (issue #75's scope trap).
    (corpus / "stale.html").write_text(_STALE_HTML_WITHOUT_CITATION, encoding="utf-8")

    assert main([*args, "--limit", "1"]) == 0
    assert asyncio.run(_edge_exists(migrated_db, *edge)) is True


def test_an_approved_stale_edge_is_reported_as_a_conflict_not_deleted(
    migrated_db: str, tmp_path: Path, capsys: Any
):
    corpus = _build_stale_edge_corpus(tmp_path)
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]
    assert main(args) == 0
    capsys.readouterr()  # discard the first run's output

    edge_id = asyncio.run(_edge_id(migrated_db, _STALE_ID, _LAW_ID, "guided_by"))
    asyncio.run(_set_review_status(migrated_db, edge_id, ReviewStatus.APPROVED))

    (corpus / "stale.html").write_text(_STALE_HTML_WITHOUT_CITATION, encoding="utf-8")
    assert main(args) == 0
    out = capsys.readouterr().out

    # A human already approved this edge; the evidence has since moved, but
    # that is a disagreement for a person to resolve, not something this
    # pass may resolve on its own by deleting it.
    assert asyncio.run(_edge_review_status(migrated_db, edge_id)) == "approved"
    assert f"edge id={edge_id}" in out
    assert "is approved but this parse no longer produces it" in out


# --- a failed run must not read as a productive one (issue #78) -----------------------
#
# Since #77, TOTAL ``edges`` is the database's row count. In a run where every document
# failed, that is the graph's existing size printed beside columns that are all zero --
# correct by its definition, but it reads as though the run produced it.


def _connection_refused(self: MockEmbedder, texts: Sequence[str]) -> list[list[float]]:
    """Fail the way ``BgeM3Embedder`` does when no Ollama daemon is listening (issue #78)."""
    raise ConnectionError("[Errno 61] Connection refused")


def test_total_edges_after_a_run_where_every_document_failed_says_what_it_counts(
    migrated_db: str, tmp_path: Path, capsys: Any, monkeypatch: Any
):
    corpus = _build_stale_edge_corpus(tmp_path)
    args = [
        "--corpus",
        str(corpus),
        "--database-url",
        migrated_db,
        "--references",
        str(tmp_path / "references.csv"),
    ]

    assert main(args) == 0
    first_out = capsys.readouterr().out
    edges_before = _counts(migrated_db)["edges"]
    assert edges_before > 0, "the zeros asserted below would be vacuous on an empty graph"
    # A run with no failure explains nothing and prints its summary as it always has.
    assert "INFO: TOTAL edges" not in first_out
    assert first_out.rstrip().splitlines()[-1] == "2 ingested, 0 metadata-only"

    monkeypatch.setattr(MockEmbedder, "embed", _connection_refused)
    assert main(args) == 1
    out = capsys.readouterr().out

    total = _table_row(out, "TOTAL")
    # The cell is the database's own count -- a failed run neither adds to nor prunes
    # it -- beside columns that say this run produced nothing.
    assert int(total["edges"]) == edges_before == _counts(migrated_db)["edges"]
    assert int(total["units"]) == 0
    assert int(total["chunks"]) == 0
    assert "INFO: TOTAL edges is the database's row count, not this run's output" in out
    assert out.rstrip().splitlines()[-1] == "0 ingested, 0 metadata-only, 2 failed"
