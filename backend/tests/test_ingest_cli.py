"""Tests for the ingestion CLI (``python -m ingest.run``).

The fixture corpus is built by this file, not committed: ``corpus/`` is
git-ignored for licensing reasons (TCVN standards are copyrighted), and the
issue requires the test to synthesize its own manifest and documents rather
than relying on a real statute. Every document here is synthetic text, not a
citable source.

Every ``main()`` call below omits ``--database-url``, on purpose: these tests
are about the dry-run pipeline and printed table, not persistence (build plan
A6 Phase 1.5b covers that in ``test_ingest_persistence.py``). Without
``_no_ambient_database``, that omission would not actually mean "no database"
whenever ``DATABASE_URL`` happens to be set in the environment -- exactly the
case for the "with a database" half of this issue's own acceptance run --
since ``--database-url`` then defaults to it (see ``ingest/run.py``).
"""

import csv
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.core.config import get_settings
from app.embedders.mock import MockEmbedder
from ingest.manifest import MANIFEST_FILENAME
from ingest.run import DocumentStats, _edges_total_note, _format_summary, main

_GOOD_ID = "test-gas-decree-2018"
"""Deliberately not the real ``gas-decree-87-2018`` id: A6 issue #51's
requirement cluster cites real units of that exact document, and this
fixture's synthetic content never reaches them -- reusing the real id would
make every test here fail the new, correct cross-document check in
``ingest/graph.py``'s ``build_requirement_graph`` instead of testing what
this file is actually about."""
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
"""Điều 3 is genuinely bare -- no heading text, no body, no clauses -- the
one shape ``build_chunks`` still (correctly) drops silently. A heading-only
article is a different, no-longer-dropped case; see ``heading_only_corpus``."""

_FAILING_ID = "too-short-doc"
_FAILING_HTML = "<html><body><p>Quá ngắn để trích xuất hợp lệ.</p></body></html>"
"""Under ingest/extract.py's _MIN_TEXT_LENGTH floor, so extract_text raises."""

_NOANCHOR_ID = "noanchor-doc-2021"
_NOANCHOR_HTML = """<html><body>
<p>Đây là một trang không có cấu trúc điều khoản pháp luật nào cả, chỉ là văn
bản mô tả chung chung không mang tính quy phạm, có thể là nội dung bị lưu
nhầm từ một trang web khác không phải văn bản pháp luật thực sự cần trích
xuất, chẳng hạn như một trang chặn bot tự động của vbpl.vn.</p>
<p>Nội dung này tiếp tục không chứa bất kỳ từ khóa cấu trúc nào như chương,
điều, khoản, hoặc điểm được đánh số theo quy ước văn bản pháp luật Việt Nam.</p>
</body></html>"""
"""Extracts well over the 200-character floor, so extract_text does not raise
-- but contains no Chương/Điều/Khoản/Điểm anchor at all, so it parses to zero
units. This is the review's own reproduction: a page saved by mistake (e.g.
vbpl.vn's anti-bot page) extracts cleanly and silently ingests to nothing."""

_CHAPTER_ONLY_ID = "chapter-only-doc"
_CHAPTER_ONLY_HTML = """<html><body>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Đây là nội dung giới thiệu chung của chương, không có điều khoản cụ thể
nào được liệt kê, chỉ mang tính chất mở đầu văn bản pháp luật để giải thích
bối cảnh và mục đích ban hành trước khi đi vào các quy định chi tiết.</p>
</body></html>"""
"""Parses to one real unit (the chapter) but zero chunks: build_chunks only
chunks articles and their descendants, and this document has no Điều at all."""

_HEADING_ONLY_ID = "test-lpg-cylinder-2013"
"""Deliberately not the real ``lpg-cylinder-qcvn-04-2013`` id -- same reason
as ``_GOOD_ID`` above: that is the other document A6 issue #51's requirement
cluster cites real units of."""
_HEADING_ONLY_HTML = """<html><body>
<p>Điều 16. Tổ chức, cá nhân sản xuất, nhập khẩu, sửa chữa, sở hữu, kiểm định,
giao nhận, vận chuyển, sử dụng chai chứa LPG bằng thép và các tổ chức, cá
nhân khác có liên quan phải tuân thủ các yêu cầu kỹ thuật an toàn quy định
tại Nghị định số 107/2009/NĐ-CP ngày 26 tháng 11 năm 2009 và Thông tư số
41/2011/TT-BCT ngày 16 tháng 12 năm 2011.</p>
</body></html>"""
"""Shaped after the real QCVN 04:2013/BCT (issue #22): a single-sentence
article with no separate body line and no clauses at all, its entire
substance -- citations included -- packed into ARTICLE_ANCHOR's heading
group. Every fixture before this issue put a body under every article,
which is exactly why this shape went unnoticed for four phases."""


@pytest.fixture(autouse=True)
def _no_ambient_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test in this file a true dry run, whatever the environment.

    ``get_settings()`` is process-wide cached (``lru_cache``), so clearing the
    env var alone is not enough once it has been read once already -- the
    cache has to be dropped too, in both directions, so a database-backed
    test file running earlier or later in the same session is unaffected.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _document_entry(doc_id: str, filename: str, **overrides: Any) -> dict[str, Any]:
    """Return a minimal valid manifest document entry, with any field overridden."""
    entry: dict[str, Any] = {
        "id": doc_id,
        "number": "87/2018/NĐ-CP",
        "title_vi": "Nghị định thử nghiệm",
        "doc_type": "decree",
        "status": "in_force",
        "effective": "2018-08-01",
        "path": f"corpus/{filename}",
    }
    entry.update(overrides)
    return entry


def _build_corpus(root: Path, documents: list[dict[str, Any]], files: dict[str, str]) -> Path:
    """Lay out ``<root>/corpus/manifest.json`` plus each named file; return the corpus dir."""
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / MANIFEST_FILENAME).write_text(json.dumps({"documents": documents}), encoding="utf-8")
    for name, content in files.items():
        (corpus / name).write_text(content, encoding="utf-8")
    return corpus


@pytest.fixture
def mixed_corpus(tmp_path: Path) -> Path:
    """One document that ingests cleanly (with one empty article) and one that fails."""
    documents = [
        _document_entry(_GOOD_ID, "good.html"),
        _document_entry(_FAILING_ID, "failing.html"),
    ]
    files = {"good.html": _GOOD_HTML, "failing.html": _FAILING_HTML}
    return _build_corpus(tmp_path, documents, files)


@pytest.fixture
def good_only_corpus(tmp_path: Path) -> Path:
    """A single document that ingests cleanly, no failures."""
    documents = [_document_entry(_GOOD_ID, "good.html")]
    return _build_corpus(tmp_path, documents, {"good.html": _GOOD_HTML})


@pytest.fixture
def noanchor_corpus(tmp_path: Path) -> Path:
    """One good document plus one that extracts cleanly but parses to zero units."""
    documents = [
        _document_entry(_GOOD_ID, "good.html"),
        _document_entry(_NOANCHOR_ID, "noanchor.html"),
    ]
    files = {"good.html": _GOOD_HTML, "noanchor.html": _NOANCHOR_HTML}
    return _build_corpus(tmp_path, documents, files)


@pytest.fixture
def chapter_only_corpus(tmp_path: Path) -> Path:
    """A single document with structure but no articles: units > 0, chunks == 0."""
    documents = [_document_entry(_CHAPTER_ONLY_ID, "chapter-only.html")]
    return _build_corpus(tmp_path, documents, {"chapter-only.html": _CHAPTER_ONLY_HTML})


@pytest.fixture
def heading_only_corpus(tmp_path: Path) -> Path:
    """A single document whose only article's entire content is its heading."""
    documents = [_document_entry(_HEADING_ONLY_ID, "heading-only.html")]
    return _build_corpus(tmp_path, documents, {"heading-only.html": _HEADING_ONLY_HTML})


_SELF_CITE_ID = "test-self-cite-doc"
_SELF_CITE_HTML = """<html><body>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Phạm vi điều chỉnh</p>
<p>Nghị định này quy định về điều kiện kinh doanh khí dầu mỏ hóa lỏng (LPG)
tại thị trường Việt Nam, không chia thành các khoản riêng biệt.</p>
<p>Điều 2. Điều kiện đối với cửa hàng bán lẻ</p>
<p>1. Cửa hàng bán lẻ LPG chai phải đáp ứng các điều kiện quy định tại khoản
5 Điều 1 và các quy định khác của pháp luật hiện hành về phòng cháy chữa
cháy.</p>
</body></html>"""
"""Điều 2 Khoản 1 cites "khoản 5 Điều 1" -- a real article (Điều 1 exists)
that genuinely has no clauses at all (its body is a single paragraph, no "N."
lines), so the citation cannot resolve. The same shape as the real corpus's
own self-citation misses (issue #69): article exists, clause does not."""


@pytest.fixture
def self_citation_corpus(tmp_path: Path) -> Path:
    """A document whose own internal citation names a clause that never existed."""
    documents = [_document_entry(_SELF_CITE_ID, "self-cite.html")]
    return _build_corpus(tmp_path, documents, {"self-cite.html": _SELF_CITE_HTML})


_SECOND_GOOD_ID = "test-gas-circular-2019"
_NOT_ACQUIRED_ID = "test-old-standard-2012"


def _summary_documents() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Two documents that ingest cleanly and one that is metadata-only (no file on disk)."""
    documents = [
        _document_entry(_GOOD_ID, "good.html"),
        _document_entry(_SECOND_GOOD_ID, "second.html", number="12/2019/TT-BCT"),
        _document_entry(
            _NOT_ACQUIRED_ID,
            "old-standard.pdf",
            number="10/2012/NĐ-CP",
            status="expired",
            acquired=False,
        ),
    ]
    return documents, {"good.html": _GOOD_HTML, "second.html": _GOOD_HTML}


@pytest.fixture
def summary_corpus(tmp_path: Path) -> Path:
    """Two documents that ingest, one metadata-only -- the shape of the real corpus, smaller."""
    documents, files = _summary_documents()
    return _build_corpus(tmp_path, documents, files)


@pytest.fixture
def summary_corpus_with_a_failure(tmp_path: Path) -> Path:
    """``summary_corpus`` plus a fourth document whose extraction raises."""
    documents, files = _summary_documents()
    documents.append(_document_entry(_FAILING_ID, "failing.html", number="13/2019/TT-BCT"))
    files["failing.html"] = _FAILING_HTML
    return _build_corpus(tmp_path, documents, files)


def _connection_refused(self: MockEmbedder, texts: Sequence[str]) -> list[list[float]]:
    """Fail the way ``BgeM3Embedder`` does when no Ollama daemon is listening (issue #78)."""
    raise ConnectionError("[Errno 61] Connection refused")


_SUMMARY_LINE = re.compile(r"^\d+ ingested, .*$", re.MULTILINE)


def _summary_line(output: str) -> str:
    """Return the ``N ingested, M metadata-only[, K failed]`` line the table ends with."""
    match = _SUMMARY_LINE.search(output)
    assert match is not None, "no summary line was printed"
    return match.group(0)


# --- table-parsing helper -----------------------------------------------------------


def _parse_table(output: str) -> dict[str, dict[str, int]]:
    """Parse the printed table's rows into ``{document_id: {column: value}}``.

    The header is located by content ("document ...") rather than assumed to
    be the first line, since progress/warnings go to the logger (per the
    module's own contract) and land on stdout too, ahead of the table. An
    "  ERROR: ...''/"  WARNING: ..."/"  INFO: ..." line is indented and
    skipped here. Parsing stops at the "TOTAL" row -- the summary line
    ``_print_table`` prints after it ("N ingested, M metadata-only", issue
    #59) is not a data row.
    """
    lines = output.splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("document"))
    header = lines[header_index].split()
    rows: dict[str, dict[str, int]] = {}
    for line in lines[header_index + 1 :]:
        if not line or line.startswith(" "):
            continue
        values = line.split()
        rows[values[0]] = dict(zip(header[1:], (int(v) for v in values[1:]), strict=True))
        if values[0] == "TOTAL":
            break
    return rows


# --- the table ---------------------------------------------------------------------


def test_prints_a_header_a_row_per_document_and_a_total(
    good_only_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["--corpus", str(good_only_corpus)])
    table = _parse_table(capsys.readouterr().out)

    assert _GOOD_ID in table
    assert "TOTAL" in table
    assert exit_code == 0


def test_units_and_chunks_columns_are_both_present_and_reported(
    good_only_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    main(["--corpus", str(good_only_corpus)])
    stats = _parse_table(capsys.readouterr().out)[_GOOD_ID]
    # 1 chapter + 3 articles + 1 clause + 1 point = 6 units.
    assert stats["units"] == 6
    # dieu-1 and dieu-2 chunk; dieu-3 is deliberately empty and produces none.
    assert stats["chunks"] == 2


def test_self_citation_misses_column_is_zero_when_every_internal_citation_resolves(
    good_only_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    main(["--corpus", str(good_only_corpus)])
    stats = _parse_table(capsys.readouterr().out)[_GOOD_ID]
    assert stats["self_cite_miss"] == 0


def test_self_citation_misses_counts_a_reference_to_a_clause_that_does_not_exist(
    self_citation_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    main(["--corpus", str(self_citation_corpus)])
    table = _parse_table(capsys.readouterr().out)
    assert table[_SELF_CITE_ID]["self_cite_miss"] == 1
    assert table["TOTAL"]["self_cite_miss"] == 1


def test_an_article_that_parsed_but_produced_no_chunk_is_named_explicitly(
    good_only_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    main(["--corpus", str(good_only_corpus)])
    out = capsys.readouterr().out
    assert f"{_GOOD_ID}#dieu-3 parsed but produced no chunk" in out


def test_an_article_that_does_produce_a_chunk_is_not_flagged(
    good_only_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    main(["--corpus", str(good_only_corpus)])
    out = capsys.readouterr().out
    assert f"{_GOOD_ID}#dieu-1 parsed but produced no chunk" not in out
    assert f"{_GOOD_ID}#dieu-2 parsed but produced no chunk" not in out


def test_total_row_sums_every_document(mixed_corpus: Path, capsys: pytest.CaptureFixture[str]):
    main(["--corpus", str(mixed_corpus)])
    table = _parse_table(capsys.readouterr().out)
    # The failing document contributes zero to every numeric column.
    assert table["TOTAL"]["units"] == table[_GOOD_ID]["units"]
    assert table["TOTAL"]["chunks"] == table[_GOOD_ID]["chunks"]


# --- failure handling: one bad document must not hide the others -----------------


def test_a_failing_document_is_reported_with_its_error(
    mixed_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    main(["--corpus", str(mixed_corpus)])
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert _FAILING_ID in out


def test_a_failing_document_does_not_stop_the_run(
    mixed_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    main(["--corpus", str(mixed_corpus)])
    good = _parse_table(capsys.readouterr().out)[_GOOD_ID]
    assert good["units"] == 6
    assert good["chunks"] == 2


def test_exit_code_is_nonzero_when_a_document_fails(mixed_corpus: Path):
    assert main(["--corpus", str(mixed_corpus)]) == 1


def test_exit_code_is_zero_when_every_document_succeeds(good_only_corpus: Path):
    assert main(["--corpus", str(good_only_corpus)]) == 0


def test_the_missing_real_corpus_fails_loudly_and_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # Mirrors corpus/ today: a manifest listing a document whose file was
    # never downloaded (Phase 0.1 outstanding) -- load_manifest's own
    # fail-loud check must stop the run before any document is touched.
    documents = [_document_entry(_GOOD_ID, "never-downloaded.html")]
    corpus = _build_corpus(tmp_path, documents, files={})

    exit_code = main(["--corpus", str(corpus)])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "never-downloaded.html" in out
    assert "missing source file" in out


# --- silent loss: a document contributing nothing is a failure too ---------------


def test_a_document_with_no_anchors_is_reported_and_fails_the_run(
    noanchor_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["--corpus", str(noanchor_corpus)])
    out = capsys.readouterr().out

    assert f"ERROR: {_NOANCHOR_ID} ingested to zero units" in out
    assert exit_code == 1


def test_a_document_with_units_but_no_chunks_is_reported_and_fails_the_run(
    chapter_only_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["--corpus", str(chapter_only_corpus)])
    out = capsys.readouterr().out

    assert f"ERROR: {_CHAPTER_ONLY_ID} parsed 1 units but produced zero chunks" in out
    assert exit_code == 1


def test_a_zero_unit_document_does_not_stop_the_good_one_from_ingesting(
    noanchor_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    main(["--corpus", str(noanchor_corpus)])
    table = _parse_table(capsys.readouterr().out)

    good = table[_GOOD_ID]
    assert good["units"] == 6
    assert good["chunks"] == 2
    assert table[_NOANCHOR_ID]["units"] == 0
    # The zero-unit document contributes nothing, so the total matches the
    # good document alone -- it is not silently folded into "success".
    assert table["TOTAL"]["units"] == good["units"]
    assert table["TOTAL"]["chunks"] == good["chunks"]


def test_a_raised_exception_is_still_distinguished_from_zero_units(
    mixed_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    main(["--corpus", str(mixed_corpus)])
    out = capsys.readouterr().out
    assert f"ERROR: {_FAILING_ID} raised:" in out


# --- issue #78: the summary line counts failures separately -------------------------
#
# The summary is the line a person actually reads. It used to compute "ingested" as
# every row that was not metadata-only, so seven documents that all failed at the
# embedding step still printed "7 ingested, 3 metadata-only" two lines below seven
# ERROR lines.


def test_a_run_where_every_document_fails_does_not_report_any_as_ingested(
    summary_corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(MockEmbedder, "embed", _connection_refused)

    exit_code = main(["--corpus", str(summary_corpus)])
    out = capsys.readouterr().out

    assert f"ERROR: {_GOOD_ID} raised: [Errno 61] Connection refused" in out
    assert _summary_line(out) == "0 ingested, 1 metadata-only, 2 failed"
    assert exit_code == 1


def test_one_failure_among_several_documents_is_counted_as_failed_not_ingested(
    summary_corpus_with_a_failure: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["--corpus", str(summary_corpus_with_a_failure)])
    out = capsys.readouterr().out

    assert f"ERROR: {_FAILING_ID} raised:" in out
    assert _summary_line(out) == "2 ingested, 1 metadata-only, 1 failed"
    assert exit_code == 1


def test_a_document_that_ingested_to_zero_units_is_counted_as_failed_too(
    noanchor_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    # It raises nothing -- ``error`` is set by the silent-loss check instead.
    main(["--corpus", str(noanchor_corpus)])

    assert _summary_line(capsys.readouterr().out) == "1 ingested, 0 metadata-only, 1 failed"


def test_a_fully_successful_run_keeps_its_summary_line_exactly(
    summary_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["--corpus", str(summary_corpus)])

    # Exactly: no ", 0 failed" is appended when nothing failed.
    assert _summary_line(capsys.readouterr().out) == "2 ingested, 1 metadata-only"
    assert exit_code == 0


def test_the_summary_leaves_out_the_seed_passes_pseudo_rows():
    rows = [
        DocumentStats(document_id="a"),
        DocumentStats(document_id="b", error="raised: boom"),
        DocumentStats(document_id="c", metadata_only=True),
        DocumentStats(document_id="requirements"),
        DocumentStats(document_id="document_edges"),
    ]

    assert _format_summary(rows) == "1 ingested, 1 metadata-only, 1 failed"


def test_the_total_edges_note_appears_only_for_a_persisted_run_with_a_failed_row():
    succeeded = [DocumentStats(document_id="a")]
    failed = [DocumentStats(document_id="a", error="raised: boom")]

    assert _edges_total_note(succeeded, edges_in_db=100) is None
    # A dry run has no database count for TOTAL edges to be confused with.
    assert _edges_total_note(failed, edges_in_db=None) is None
    assert _edges_total_note(failed, edges_in_db=100) is not None


# --- issue #22: a heading is content, not a label ---------------------------------


def test_a_heading_only_article_produces_a_chunk_and_no_gap_warning(
    heading_only_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    """Mirrors real QCVN 04:2013/BCT Điều 16 (issue #22).

    Before this fix, this exact shape parsed to one article and zero chunks,
    printing the "produced no chunk" warning below -- that was the "before";
    this test asserts the "after".
    """
    exit_code = main(["--corpus", str(heading_only_corpus)])
    out = capsys.readouterr().out
    table = _parse_table(out)

    assert table[_HEADING_ONLY_ID]["articles"] == 1
    assert table[_HEADING_ONLY_ID]["chunks"] == 1
    assert f"{_HEADING_ONLY_ID}#dieu-16 parsed but produced no chunk" not in out
    assert exit_code == 0


# --- flags ---------------------------------------------------------------------


def test_limit_ingests_only_the_first_n_documents(
    mixed_corpus: Path, capsys: pytest.CaptureFixture[str]
):
    main(["--corpus", str(mixed_corpus), "--limit", "1"])
    out = capsys.readouterr().out
    assert _GOOD_ID in out
    assert _FAILING_ID not in out


def test_references_are_written_to_the_default_csv_path(
    good_only_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    working_dir = tmp_path / "cwd"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)

    main(["--corpus", str(good_only_corpus)])
    with (working_dir / "references.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["src_id", "kind", "raw_text", "target_hint"]
    assert any(row[1] == "basis" for row in rows[1:])


def test_references_path_is_configurable(good_only_corpus: Path, tmp_path: Path):
    references_path = tmp_path / "custom-references.csv"
    main(["--corpus", str(good_only_corpus), "--references", str(references_path)])

    assert references_path.is_file()
    with references_path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["src_id", "kind", "raw_text", "target_hint"]
