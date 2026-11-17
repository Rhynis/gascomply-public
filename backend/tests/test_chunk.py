"""Tests for chunk building.

``long_article.txt`` is a synthetic fixture written specifically to push one
article's composed text past ``ARTICLE_SPLIT_THRESHOLD``, so the clause-split
path has real fixture data to run against; ``sample_article.txt`` (shared with
``test_parser.py``) stays comfortably under the threshold and exercises the
single-chunk-per-article path. Neither is a citable source.
"""

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app.schemas.chunk import Chunk
from app.schemas.document import DocStatus, DocType, Document
from ingest.chunk import ARTICLE_SPLIT_THRESHOLD, build_chunks
from ingest.parser import (
    APPENDIX_COLUMN_HEADER_MARKER,
    APPENDIX_SECTION_MARKER,
    ParseResult,
    parse_document,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_article.txt"
LONG_FIXTURE = Path(__file__).parent / "fixtures" / "long_article.txt"
DOC_ID = "gas-decree-87-2018"


def _make_document(**overrides: Any) -> Document:
    """Return a minimal valid `Document`, with any field overridden."""
    fields: dict[str, Any] = {
        "id": DOC_ID,
        "number": "87/2018/NĐ-CP",
        "title_vi": "Nghị định về kinh doanh khí",
        "doc_type": DocType.DECREE,
        "issued": date(2018, 6, 15),
        "effective": date(2018, 8, 1),
        "status": DocStatus.IN_FORCE,
        "path": "corpus/gas-decree-87-2018.pdf",
    }
    fields.update(overrides)
    return Document(**fields)


@pytest.fixture
def parsed() -> ParseResult:
    return parse_document(FIXTURE.read_text(encoding="utf-8"), DOC_ID)


@pytest.fixture
def long_parsed() -> ParseResult:
    return parse_document(LONG_FIXTURE.read_text(encoding="utf-8"), DOC_ID)


def _by_unit_id(chunks: tuple[Chunk, ...]) -> dict[str, Chunk]:
    return {chunk.unit_id: chunk for chunk in chunks}


# --- one chunk per article (under the threshold) ---------------------------------


def test_short_article_yields_one_chunk_not_per_clause(parsed: ParseResult):
    by_id = _by_unit_id(build_chunks(parsed, _make_document()))
    assert f"{DOC_ID}#dieu-38" in by_id
    assert f"{DOC_ID}#dieu-38#khoan-1" not in by_id


def test_article_chunk_recomposes_its_clauses_and_points(parsed: ParseResult):
    chunk = _by_unit_id(build_chunks(parsed, _make_document()))[f"{DOC_ID}#dieu-38"]
    assert "Cửa hàng bán lẻ LPG chai phải đáp ứng" in chunk.body
    assert "Diện tích tối thiểu của khu vực bán hàng là 12 m2" in chunk.body
    assert "Giá trưng bày chỉ được đặt chai LPG rỗng" in chunk.body
    assert "Thương nhân phải có hợp đồng mua LPG chai" in chunk.body


def test_article_chunk_carries_the_article_coordinates(parsed: ParseResult):
    chunk = _by_unit_id(build_chunks(parsed, _make_document()))[f"{DOC_ID}#dieu-38"]
    assert chunk.dieu_no == 38
    assert chunk.khoan_no is None


def test_composed_length_is_under_the_threshold(parsed: ParseResult):
    chunk = _by_unit_id(build_chunks(parsed, _make_document()))[f"{DOC_ID}#dieu-38"]
    assert len(chunk.body) <= ARTICLE_SPLIT_THRESHOLD


# --- context header (the query-shaped title must reach the embedded text) --------


def test_article_chunk_body_starts_with_the_article_heading(parsed: ParseResult):
    chunk = _by_unit_id(build_chunks(parsed, _make_document()))[f"{DOC_ID}#dieu-38"]
    assert chunk.body.startswith("Điều 38. Điều kiện đối với cửa hàng bán lẻ LPG chai\n")


def test_an_article_with_no_heading_has_no_literal_none_in_its_header():
    result = parse_document("Điều 9.\nNội dung không có tiêu đề.\n", DOC_ID)
    chunk = build_chunks(result, _make_document())[0]
    assert chunk.body.startswith("Điều 9\n")
    assert "None" not in chunk.body


# --- threshold is measured on the composed text WITH the header ------------------


def test_the_header_itself_can_push_a_borderline_article_over_the_threshold():
    # Body alone sits exactly at the threshold; the header pushes the total
    # over it, so this article must still split into per-clause chunks.
    heading = "Tiêu đề khá dài dùng để kiểm thử ngưỡng cắt " * 2
    body = "Nội dung. " * (ARTICLE_SPLIT_THRESHOLD // len("Nội dung. "))
    text = f"Điều 9. {heading}\n1. {body}\n2. Khoản thứ hai ngắn gọn.\n"
    result = parse_document(text, DOC_ID)
    by_id = _by_unit_id(build_chunks(result, _make_document()))
    assert f"{DOC_ID}#dieu-9" not in by_id
    assert f"{DOC_ID}#dieu-9#khoan-1" in by_id
    assert f"{DOC_ID}#dieu-9#khoan-2" in by_id


# --- one chunk per clause (over the threshold) ------------------------------------


def test_long_article_is_split_into_one_chunk_per_clause(long_parsed: ParseResult):
    by_id = _by_unit_id(build_chunks(long_parsed, _make_document()))
    assert f"{DOC_ID}#dieu-50" not in by_id
    for khoan_no in range(1, 6):
        assert f"{DOC_ID}#dieu-50#khoan-{khoan_no}" in by_id


def test_clause_chunk_recomposes_its_points(long_parsed: ParseResult):
    chunk = _by_unit_id(build_chunks(long_parsed, _make_document()))[f"{DOC_ID}#dieu-50#khoan-3"]
    assert "Cửa hàng phải lắp đặt hệ thống báo cháy tự động" in chunk.body
    assert "Phương án chữa cháy phải nêu rõ vị trí đặt bình chữa cháy" in chunk.body
    assert "Phương án chữa cháy phải được cập nhật" in chunk.body


def test_clause_chunk_carries_the_clause_coordinates(long_parsed: ParseResult):
    chunk = _by_unit_id(build_chunks(long_parsed, _make_document()))[f"{DOC_ID}#dieu-50#khoan-2"]
    assert chunk.dieu_no == 50
    assert chunk.khoan_no == 2


def test_every_clause_chunk_of_a_split_article_carries_the_article_heading(
    long_parsed: ParseResult,
):
    # A clause chunk carries none of Unit.body's article-level text (Unit.body
    # is own-text-only), so without the header it would say nothing about
    # which article -- or which store requirement -- it belongs to.
    by_id = _by_unit_id(build_chunks(long_parsed, _make_document()))
    heading = "Yêu cầu an toàn phòng cháy chữa cháy đối với cửa hàng bán lẻ LPG chai"
    for khoan_no in range(1, 6):
        chunk = by_id[f"{DOC_ID}#dieu-50#khoan-{khoan_no}"]
        assert chunk.body.startswith(f"Điều 50. {heading} — Khoản {khoan_no}\n")


def test_a_short_article_in_the_same_document_stays_unsplit(long_parsed: ParseResult):
    by_id = _by_unit_id(build_chunks(long_parsed, _make_document()))
    assert f"{DOC_ID}#dieu-51" in by_id


def test_total_chunk_count_for_the_long_fixture(long_parsed: ParseResult):
    # 5 clause chunks (dieu-50, split) + 1 article chunk (dieu-51, unsplit).
    assert len(build_chunks(long_parsed, _make_document())) == 6


# --- effective/status denormalization (the rule that matters most) ---------------


def test_every_chunk_carries_the_documents_effective_and_status(parsed: ParseResult):
    document = _make_document(status=DocStatus.EXPIRED, effective=date(2018, 8, 1))
    chunks = build_chunks(parsed, document)
    assert chunks  # sanity: the fixture actually produces chunks
    assert all(chunk.status is DocStatus.EXPIRED for chunk in chunks)
    assert all(chunk.effective == date(2018, 8, 1) for chunk in chunks)


def test_denormalized_fields_track_the_document_not_the_unit(parsed: ParseResult):
    in_force = build_chunks(parsed, _make_document(status=DocStatus.IN_FORCE))
    expired = build_chunks(parsed, _make_document(status=DocStatus.EXPIRED))
    # Same units, same composed bodies -- only the document's status differs,
    # and units carry no status of their own to leak through instead.
    assert [chunk.body for chunk in in_force] == [chunk.body for chunk in expired]
    assert all(chunk.status is DocStatus.IN_FORCE for chunk in in_force)
    assert all(chunk.status is DocStatus.EXPIRED for chunk in expired)


def test_a_document_with_no_effective_date_yields_chunks_with_none_effective(parsed: ParseResult):
    chunks = build_chunks(parsed, _make_document(effective=None))
    assert chunks
    assert all(chunk.effective is None for chunk in chunks)


def test_split_clause_chunks_also_carry_the_documents_status(long_parsed: ParseResult):
    document = _make_document(status=DocStatus.AMENDING)
    chunks = build_chunks(long_parsed, document)
    assert chunks
    assert all(chunk.status is DocStatus.AMENDING for chunk in chunks)


# --- unit_id / document_id integrity ----------------------------------------------


def test_every_chunk_unit_id_is_a_real_unit(parsed: ParseResult):
    real_ids = {unit.id for unit in parsed.units}
    chunks = build_chunks(parsed, _make_document())
    assert chunks
    assert all(chunk.unit_id in real_ids for chunk in chunks)


def test_every_split_clause_chunk_unit_id_is_a_real_unit(long_parsed: ParseResult):
    real_ids = {unit.id for unit in long_parsed.units}
    chunks = build_chunks(long_parsed, _make_document())
    assert chunks
    assert all(chunk.unit_id in real_ids for chunk in chunks)


def test_document_id_matches_the_document_argument(parsed: ParseResult):
    document = _make_document()
    chunks = build_chunks(parsed, document)
    assert chunks
    assert all(chunk.document_id == document.id for chunk in chunks)


# --- empty-chunk skipping -----------------------------------------------------------


def test_empty_parse_result_yields_no_chunks():
    result = parse_document("", DOC_ID)
    assert build_chunks(result, _make_document()) == ()


def test_a_genuinely_bare_article_yields_no_chunk():
    # No heading text and no body: a bare coordinate is not content.
    result = parse_document("Điều 9.\n", DOC_ID)
    assert build_chunks(result, _make_document()) == ()


def test_a_heading_only_article_yields_exactly_one_chunk_containing_it():
    # The heading IS the article's entire content (build plan issue #22) --
    # single-sentence Vietnamese drafting with no separate body line, no
    # clauses. Every fixture before this issue put a body under every
    # article, which is exactly why this shape went untested for so long.
    heading = "Tổ chức, cá nhân phải tuân thủ quy chuẩn kỹ thuật này."
    result = parse_document(f"Điều 9. {heading}\n", DOC_ID)
    chunks = build_chunks(result, _make_document())
    assert len(chunks) == 1
    assert chunks[0].unit_id == f"{DOC_ID}#dieu-9"
    assert heading in chunks[0].body


def test_an_over_threshold_article_with_no_clauses_is_not_split():
    # Nothing to split into, so it falls back to a single (long) article chunk.
    long_body = "Nội dung điều luật cần kiểm thử độ dài. " * 100
    result = parse_document(f"Điều 9. Tiêu đề\n{long_body}\n", DOC_ID)
    chunks = build_chunks(result, _make_document())
    assert len(chunks) == 1
    assert chunks[0].unit_id == f"{DOC_ID}#dieu-9"
    assert len(chunks[0].body) > ARTICLE_SPLIT_THRESHOLD


def test_an_empty_clause_is_skipped_once_the_article_is_split():
    long_body = "Nội dung điều luật cần kiểm thử độ dài. " * 100
    result = parse_document(f"Điều 9. Tiêu đề\n1. {long_body}\n2.\n", DOC_ID)
    by_id = _by_unit_id(build_chunks(result, _make_document()))
    assert f"{DOC_ID}#dieu-9#khoan-1" in by_id
    assert f"{DOC_ID}#dieu-9#khoan-2" not in by_id


# --- an appendix sub-row gets its own chunk, a sibling of its item's (#147) ------


def test_appendix_subitem_gets_its_own_chunk_not_folded_into_its_parent():
    # Appendix VI specifically: the subitem anchor is scoped to that one
    # appendix (``_PREMIUM_TABLE_APPENDIX_NO``, issue #147's own guard).
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "DANH MỤC MẪU\n"
        "35. Cơ sở công nghiệp thuộc hạng nguy hiểm cháy nổ.\n"
        "35.1 a) Cơ sở sản xuất công nghiệp hạng A, B N 0,2\n"
        "35.2 Cơ sở sản xuất công nghiệp hạng D, E M 0,15\n"
    )
    result = parse_document(text, DOC_ID)
    by_id = _by_unit_id(build_chunks(result, _make_document()))

    item_chunk = by_id[f"{DOC_ID}#phu-luc-6#muc-35"]
    sub1_chunk = by_id[f"{DOC_ID}#phu-luc-6#muc-35#muc-1"]
    sub2_chunk = by_id[f"{DOC_ID}#phu-luc-6#muc-35#muc-2"]

    # Each sub-row is its own, separate chunk -- not merged into item 35's.
    assert sub1_chunk is not sub2_chunk
    assert "a) Cơ sở sản xuất công nghiệp hạng A, B N 0,2" in sub1_chunk.body
    assert "Cơ sở sản xuất công nghiệp hạng D, E M 0,15" in sub2_chunk.body

    # The parent item's own chunk carries its own text and header only --
    # a sub-row's text is never duplicated into it.
    assert "Cơ sở công nghiệp thuộc hạng nguy hiểm cháy nổ." in item_chunk.body
    assert "N 0,2" not in item_chunk.body
    assert "M 0,15" not in item_chunk.body

    # The header names the sub-row with the same compound number the source
    # itself prints ("35.1"), not a separate, unfamiliar coordinate.
    assert sub1_chunk.body.startswith("Phụ lục 6 — Mục 35.1")
    assert sub2_chunk.body.startswith("Phụ lục 6 — Mục 35.2")


def test_an_appendix_points_text_still_folds_into_its_parent_item_unlike_a_subitem():
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC I\n"
        "DANH MỤC MẪU\n"
        "40. Cơ sở ví dụ.\n"
        "a) Điều kiện một.\n"
        "b) Điều kiện hai.\n"
    )
    result = parse_document(text, DOC_ID)
    by_id = _by_unit_id(build_chunks(result, _make_document()))

    # Unchanged, pre-existing behavior: a lettered point is not its own
    # chunk, it is folded into the single chunk its parent item already has.
    assert f"{DOC_ID}#phu-luc-1#muc-40#diem-a" not in by_id
    item_chunk = by_id[f"{DOC_ID}#phu-luc-1#muc-40"]
    assert "Điều kiện một." in item_chunk.body
    assert "Điều kiện hai." in item_chunk.body


# --- a premium table row/sub-row chunk carries its table's column header (#160) --

_H = APPENDIX_COLUMN_HEADER_MARKER
_COLUMN_HEADER = (
    "Loại hình cơ sở có nguy hiểm về cháy, nổ | Mức khấu trừ (loại) | "
    "Tỷ lệ phí bảo hiểm tối thiểu/năm (%)"
)


def test_premium_table_row_and_subrow_chunks_carry_the_column_header():
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "DANH MỤC MẪU\n"
        f"{_H}{_COLUMN_HEADER}\n"
        "17. Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg N 0,3\n"
        "35. Cơ sở công nghiệp thuộc hạng nguy hiểm cháy nổ.\n"
        "35.1 a) Cơ sở sản xuất công nghiệp hạng A, B N 0,2\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    by_id = _by_unit_id(
        build_chunks(result, _make_document(id="pccc-decree-105-2025", path="x.pdf"))
    )

    row_chunk = by_id["pccc-decree-105-2025#phu-luc-6#muc-17"]
    subrow_chunk = by_id["pccc-decree-105-2025#phu-luc-6#muc-35#muc-1"]
    assert f"(cột: {_COLUMN_HEADER})" in row_chunk.body
    assert f"(cột: {_COLUMN_HEADER})" in subrow_chunk.body
    # The header suffix is read from the source, never duplicating the body's
    # own cell text -- the row's own text is still present, just once.
    assert row_chunk.body.count("0,3") == 1


def test_a_phu_luc_vii_item_chunk_is_unaffected_by_the_column_header_feature():
    text = (
        "Điều 1. Nội dung\nPHỤ LỤC VII\nDANH MỤC MẪU\n15. Cơ sở kinh doanh khí đốt trên 500 kg.\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    by_id = _by_unit_id(
        build_chunks(result, _make_document(id="pccc-decree-105-2025", path="x.pdf"))
    )
    chunk = by_id["pccc-decree-105-2025#phu-luc-7#muc-15"]
    assert "cột:" not in chunk.body
    assert chunk.body == ("Phụ lục 7 — Mục 15\nCơ sở kinh doanh khí đốt trên 500 kg.")


def test_an_appendix_section_clause_chunk_is_unaffected_by_the_column_header_feature():
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "MỨC PHÍ BẢO HIỂM VÀ MỨC KHẤU TRỪ\n"
        f"{APPENDIX_SECTION_MARKER}II. MỨC KHẤU TRỪ BẢO HIỂM\n"
        f"{APPENDIX_SECTION_MARKER}II.1. Đối với cơ sở có nguy hiểm về cháy, nổ:\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    by_id = _by_unit_id(
        build_chunks(result, _make_document(id="pccc-decree-105-2025", path="x.pdf"))
    )
    chunk = by_id["pccc-decree-105-2025#phu-luc-6#muc-ii"]
    assert "cột:" not in chunk.body


def test_a_premium_row_with_no_captured_header_falls_back_and_logs(
    caplog: pytest.LogCaptureFixture,
):
    # No column-header marker line precedes this row -- the extraction-time
    # fallback (issue #160): the row's chunk keeps its unmodified header, and
    # the fallback is logged rather than silently dropped.
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "DANH MỤC MẪU\n"
        "17. Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg N 0,3\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    with caplog.at_level("WARNING"):
        by_id = _by_unit_id(
            build_chunks(result, _make_document(id="pccc-decree-105-2025", path="x.pdf"))
        )
    chunk = by_id["pccc-decree-105-2025#phu-luc-6#muc-17"]
    assert "cột:" not in chunk.body
    assert chunk.body == (
        "Phụ lục 6 — Mục 17\n"
        "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg N 0,3"
    )
    assert any(
        "no premium table column header was captured" in record.message for record in caplog.records
    )


# --- an appendix's own named sections are chunked like an article (#147) --------

_M = APPENDIX_SECTION_MARKER
_APPENDIX_6_DOC_ID = "pccc-decree-105-2025"


def test_a_short_appendix_section_is_one_chunk_with_its_clauses_and_points_folded_in():
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "MỨC PHÍ BẢO HIỂM VÀ MỨC KHẤU TRỪ\n"
        f"{_M}II. MỨC KHẤU TRỪ BẢO HIỂM\n"
        f"{_M}II.1. Đối với cơ sở có nguy hiểm về cháy, nổ (trừ cơ sở hạt nhân):\n"
        f"{_M}II.1.a) Đối với cơ sở ... thuộc loại M...\n"
        f"{_M}II.1.b) Đối với cơ sở ... thuộc loại N...\n"
        f"{_M}II.2. Đối với cơ sở có tổng số tiền bảo hiểm từ 1.000 tỷ đồng.\n"
    )
    result = parse_document(text, _APPENDIX_6_DOC_ID)
    chunks = build_chunks(result, _make_document(id=_APPENDIX_6_DOC_ID, path="corpus/x.pdf"))
    by_id = _by_unit_id(chunks)

    # One chunk for the whole section -- short enough to stay under
    # ARTICLE_SPLIT_THRESHOLD -- never one per clause.
    assert f"{_APPENDIX_6_DOC_ID}#phu-luc-6#muc-ii" in by_id
    assert f"{_APPENDIX_6_DOC_ID}#phu-luc-6#muc-ii#khoan-1" not in by_id
    assert f"{_APPENDIX_6_DOC_ID}#phu-luc-6#muc-ii#khoan-1#diem-a" not in by_id

    section_chunk = by_id[f"{_APPENDIX_6_DOC_ID}#phu-luc-6#muc-ii"]
    assert section_chunk.body.startswith("Phụ lục 6")
    assert "Mục II" in section_chunk.body
    assert "MỨC KHẤU TRỪ BẢO HIỂM" in section_chunk.body
    assert "Đối với cơ sở có nguy hiểm về cháy, nổ (trừ cơ sở hạt nhân):" in section_chunk.body
    assert "thuộc loại M" in section_chunk.body
    assert "thuộc loại N" in section_chunk.body
    assert "1.000 tỷ đồng" in section_chunk.body


def test_a_long_appendix_section_splits_into_one_chunk_per_clause():
    long_point = "Đối với cơ sở ... " + ("rất dài " * 400)
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "MỨC PHÍ BẢO HIỂM VÀ MỨC KHẤU TRỪ\n"
        f"{_M}II. MỨC KHẤU TRỪ BẢO HIỂM\n"
        f"{_M}II.1. Đối với cơ sở có nguy hiểm về cháy, nổ (trừ cơ sở hạt nhân):\n"
        f"{_M}II.1.a) {long_point}\n"
        f"{_M}II.2. Đối với cơ sở có tổng số tiền bảo hiểm từ 1.000 tỷ đồng.\n"
    )
    result = parse_document(text, _APPENDIX_6_DOC_ID)
    chunks = build_chunks(result, _make_document(id=_APPENDIX_6_DOC_ID, path="corpus/x.pdf"))
    by_id = _by_unit_id(chunks)

    assert f"{_APPENDIX_6_DOC_ID}#phu-luc-6#muc-ii" not in by_id
    khoan1_chunk = by_id[f"{_APPENDIX_6_DOC_ID}#phu-luc-6#muc-ii#khoan-1"]
    khoan2_chunk = by_id[f"{_APPENDIX_6_DOC_ID}#phu-luc-6#muc-ii#khoan-2"]
    # Points are still folded into their own clause's chunk, not split out --
    # the same way a lettered point folds into an article's clause chunk.
    assert f"{_APPENDIX_6_DOC_ID}#phu-luc-6#muc-ii#khoan-1#diem-a" not in by_id
    assert "rất dài" in khoan1_chunk.body
    assert "Khoản 1" in khoan1_chunk.body
    assert "Mục II" in khoan1_chunk.body
    assert "1.000 tỷ đồng" in khoan2_chunk.body
    assert "rất dài" not in khoan2_chunk.body
