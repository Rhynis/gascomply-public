"""Tests for the structural parser.

The fixture is a synthetic excerpt written in the shape of Nghị định 87/2018 --
real numbering and real requirement values, but deliberately seeded with three
anomalies the parser must skip and count. It is test data, not a citable source.
"""

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.unit import Unit, UnitType
from ingest.parser import (
    APPENDIX_COLUMN_HEADER_MARKER,
    APPENDIX_SECTION_ANCHOR,
    APPENDIX_SECTION_CLAUSE_ANCHOR,
    APPENDIX_SECTION_MARKER,
    APPENDIX_SECTION_POINT_ANCHOR,
    APPENDIX_SUBITEM_ANCHOR,
    CLAUSE_ANCHOR,
    ParseResult,
    _Parser,
    parse_document,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_article.txt"
DOC_ID = "gas-decree-87-2018"


@pytest.fixture
def sample_text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def parsed(sample_text: str) -> ParseResult:
    return parse_document(sample_text, DOC_ID)


def _by_id(result: ParseResult) -> dict[str, Unit]:
    return {unit.id: unit for unit in result.units}


# --- structure ------------------------------------------------------------------


def test_counts_every_level(parsed: ParseResult):
    assert parsed.document_id == DOC_ID
    assert parsed.stats.chapters == 2
    assert parsed.stats.articles == 4
    assert parsed.stats.clauses == 4
    assert parsed.stats.points == 5


def test_units_carry_the_locked_identifiers(parsed: ParseResult):
    units = _by_id(parsed)
    assert f"{DOC_ID}#chuong-1" in units
    assert f"{DOC_ID}#chuong-2" in units
    assert f"{DOC_ID}#dieu-38" in units
    assert f"{DOC_ID}#dieu-38#khoan-1" in units
    assert f"{DOC_ID}#dieu-38#khoan-1#diem-a" in units
    assert f"{DOC_ID}#dieu-38#khoan-1#diem-đ" in units


def test_identifiers_are_unique(parsed: ParseResult):
    identifiers = [unit.id for unit in parsed.units]
    assert len(identifiers) == len(set(identifiers))


def test_parent_chain_reaches_the_document(parsed: ParseResult):
    units = _by_id(parsed)
    point = units[f"{DOC_ID}#dieu-38#khoan-1#diem-a"]
    clause = units[point.parent_id or ""]
    article = units[clause.parent_id or ""]
    chapter = units[article.parent_id or ""]

    assert (clause.id, article.id, chapter.id) == (
        f"{DOC_ID}#dieu-38#khoan-1",
        f"{DOC_ID}#dieu-38",
        f"{DOC_ID}#chuong-2",
    )
    assert chapter.parent_id is None
    assert all(unit.document_id == DOC_ID for unit in (point, clause, article, chapter))


def test_ordinals_follow_reading_order(parsed: ParseResult):
    assert [unit.ordinal for unit in parsed.units] == list(range(len(parsed.units)))
    positions = {unit.id: unit.ordinal for unit in parsed.units}
    assert positions[f"{DOC_ID}#chuong-1"] < positions[f"{DOC_ID}#dieu-1"]
    assert positions[f"{DOC_ID}#dieu-38"] < positions[f"{DOC_ID}#dieu-38#khoan-1"]
    assert positions[f"{DOC_ID}#dieu-38#khoan-1"] < positions[f"{DOC_ID}#dieu-39"]


# --- text fidelity --------------------------------------------------------------


def test_headings_are_taken_from_the_source(parsed: ParseResult):
    units = _by_id(parsed)
    assert units[f"{DOC_ID}#chuong-2"].heading == "ĐIỀU KIỆN ĐỐI VỚI CỬA HÀNG BÁN LẺ LPG CHAI"
    assert units[f"{DOC_ID}#dieu-38"].heading == "Điều kiện đối với cửa hàng bán lẻ LPG chai"
    assert units[f"{DOC_ID}#dieu-38#khoan-1"].heading is None


def test_a_wrapped_sentence_is_never_split(parsed: ParseResult):
    point = _by_id(parsed)[f"{DOC_ID}#dieu-38#khoan-1#diem-b"]
    assert point.body == (
        "Khoảng cách từ nơi để chai LPG đến nguồn lửa, nguồn nhiệt tối thiểu là\n"
        "3 m, trừ trường hợp có tường ngăn cháy đạt yêu cầu theo quy chuẩn kỹ thuật\n"
        "quốc gia;"
    )


def test_body_excludes_child_units(parsed: ParseResult):
    units = _by_id(parsed)
    assert units[f"{DOC_ID}#dieu-38"].body == ""
    assert units[f"{DOC_ID}#dieu-38#khoan-1"].body == (
        "Cửa hàng bán lẻ LPG chai phải đáp ứng các điều kiện sau:"
    )
    assert units[f"{DOC_ID}#dieu-1"].body.startswith("Nghị định này quy định")
    assert units[f"{DOC_ID}#dieu-1"].body.endswith("Việt Nam.")


# --- anomalies ------------------------------------------------------------------


def test_anomalies_are_skipped_and_counted(parsed: ParseResult):
    assert parsed.stats.skipped == 3
    bodies = [unit.body for unit in parsed.units]
    assert not any("mồ côi" in body for body in bodies)
    assert not any("trùng số" in (unit.heading or "") for unit in parsed.units)


def test_skipped_anchors_are_logged(sample_text: str, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="ingest.parser"):
        parse_document(sample_text, DOC_ID)
    reasons = [record.getMessage() for record in caplog.records]
    assert sum("clause outside any article" in message for message in reasons) == 1
    assert sum("point outside any clause" in message for message in reasons) == 1
    assert sum("duplicate article" in message for message in reasons) == 1


def test_a_duplicate_article_keeps_the_first_occurrence(parsed: ParseResult):
    article = _by_id(parsed)[f"{DOC_ID}#dieu-38"]
    assert article.heading == "Điều kiện đối với cửa hàng bán lẻ LPG chai"


def test_malformed_input_never_raises():
    result = parse_document("Điều 5\n1.\na)\nĐiều 5\n", DOC_ID)
    assert result.stats.articles == 1
    assert result.stats.skipped == 1


# --- determinism ----------------------------------------------------------------


def test_parsing_is_deterministic(sample_text: str):
    assert parse_document(sample_text, DOC_ID) == parse_document(sample_text, DOC_ID)


def test_identifiers_do_not_depend_on_the_document(sample_text: str):
    other = parse_document(sample_text, "pccc-law-2024")
    assert all(unit.id.startswith("pccc-law-2024") for unit in other.units)
    assert [unit.id.split("#", 1)[-1] for unit in other.units] == [
        unit.id.split("#", 1)[-1] for unit in parse_document(sample_text, DOC_ID).units
    ]


# --- preamble --------------------------------------------------------------------


def test_preamble_holds_text_before_the_first_anchor(parsed: ParseResult):
    assert parsed.preamble == (
        "CHÍNH PHỦ\n\n"
        "NGHỊ ĐỊNH\n"
        "Về kinh doanh khí\n\n"
        "Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015;\n"
        "Theo đề nghị của Bộ trưởng Bộ Công Thương;"
    )


def test_preamble_excludes_an_orphan_anchor_and_everything_after_it(parsed: ParseResult):
    assert "mồ côi" not in parsed.preamble
    assert "Chương I" not in parsed.preamble


def test_a_document_with_no_preamble_yields_an_empty_string():
    result = parse_document("Điều 1. A\nNội dung.\n", DOC_ID)
    assert result.preamble == ""


def test_empty_text_yields_an_empty_preamble():
    assert parse_document("", DOC_ID).preamble == ""


# --- edge cases -----------------------------------------------------------------


def test_empty_text_yields_no_units():
    result = parse_document("", DOC_ID)
    assert result.units == ()
    assert result.stats == result.stats.model_copy(update={})
    assert result.stats.skipped == 0


def test_an_article_outside_any_chapter_has_no_parent():
    result = parse_document("Điều 7. Giải thích từ ngữ\nNội dung.\n", DOC_ID)
    assert result.units[0].parent_id is None
    assert result.units[0].unit_type is UnitType.ARTICLE


def test_arabic_chapter_numbers_are_accepted():
    result = parse_document("Chương 3\nQUY ĐỊNH CHUYỂN TIẾP\n", DOC_ID)
    assert result.units[0].id == f"{DOC_ID}#chuong-3"


def test_roman_chapter_numbers_use_subtractive_notation():
    result = parse_document("Chương IX\nĐIỀU KHOẢN THI HÀNH\n", DOC_ID)
    assert result.units[0].id == f"{DOC_ID}#chuong-9"


def test_a_period_after_an_arabic_chapter_number_is_not_read_as_the_heading():
    result = parse_document("Chương 1.\nQUY ĐỊNH CHUNG\n", DOC_ID)
    assert result.units[0].id == f"{DOC_ID}#chuong-1"
    assert result.units[0].heading == "QUY ĐỊNH CHUNG"


def test_a_period_after_a_roman_chapter_number_is_not_read_as_the_heading():
    result = parse_document("Chương I.\nQUY ĐỊNH CHUNG\n", DOC_ID)
    assert result.units[0].id == f"{DOC_ID}#chuong-1"
    assert result.units[0].heading == "QUY ĐỊNH CHUNG"


def test_an_invalid_document_id_is_rejected():
    with pytest.raises(ValueError, match="Invalid document slug"):
        parse_document("Điều 1. A\n", "Gas Decree 87")


# --- unit model invariants ------------------------------------------------------


def test_unit_rejects_an_identifier_that_contradicts_its_coordinates():
    with pytest.raises(ValidationError, match="id must be"):
        Unit(
            id=f"{DOC_ID}#dieu-9",
            document_id=DOC_ID,
            unit_type=UnitType.ARTICLE,
            dieu_no=8,
            ordinal=0,
        )


def test_unit_rejects_a_wrong_parent():
    with pytest.raises(ValidationError, match="parent_id must be"):
        Unit(
            id=f"{DOC_ID}#dieu-8#khoan-1",
            document_id=DOC_ID,
            unit_type=UnitType.CLAUSE,
            dieu_no=8,
            khoan_no=1,
            parent_id=f"{DOC_ID}#dieu-7",
            ordinal=0,
        )


def test_unit_rejects_a_coordinate_from_another_level():
    with pytest.raises(ValidationError, match="must not set"):
        Unit(
            id=f"{DOC_ID}#dieu-8",
            document_id=DOC_ID,
            unit_type=UnitType.ARTICLE,
            dieu_no=8,
            khoan_no=1,
            ordinal=0,
        )


def test_unit_rejects_a_missing_coordinate():
    with pytest.raises(ValidationError, match="required for this unit type"):
        Unit(id=f"{DOC_ID}#dieu-8", document_id=DOC_ID, unit_type=UnitType.ARTICLE, ordinal=0)


def test_appendix_form_unit_builds_with_its_own_coordinates():
    unit = Unit(
        id=f"{DOC_ID}#phu-luc-1#mau-01",
        document_id=DOC_ID,
        unit_type=UnitType.APPENDIX_FORM,
        appendix_no=1,
        form_code="01",
        parent_id=f"{DOC_ID}#phu-luc-1",
        ordinal=0,
    )
    assert unit.id == f"{DOC_ID}#phu-luc-1#mau-01"


def test_appendix_form_unit_rejects_an_item_no_coordinate():
    with pytest.raises(ValidationError, match="must not set"):
        Unit(
            id=f"{DOC_ID}#phu-luc-1#mau-01",
            document_id=DOC_ID,
            unit_type=UnitType.APPENDIX_FORM,
            appendix_no=1,
            item_no=1,
            form_code="01",
            parent_id=f"{DOC_ID}#phu-luc-1",
            ordinal=0,
        )


def test_appendix_subitem_unit_builds_with_its_own_coordinates():
    unit = Unit(
        id=f"{DOC_ID}#phu-luc-1#muc-35#muc-1",
        document_id=DOC_ID,
        unit_type=UnitType.APPENDIX_SUBITEM,
        appendix_no=1,
        item_no=35,
        subitem_no=1,
        parent_id=f"{DOC_ID}#phu-luc-1#muc-35",
        ordinal=0,
    )
    assert unit.id == f"{DOC_ID}#phu-luc-1#muc-35#muc-1"
    assert unit.parent_id == f"{DOC_ID}#phu-luc-1#muc-35"


def test_appendix_item_unit_accepts_a_column_header():
    # issue #160: informational only, never a coordinate -- does not affect
    # the id derivation or trip `_reject_foreign_coordinates`.
    unit = Unit(
        id=f"{DOC_ID}#phu-luc-6#muc-17",
        document_id=DOC_ID,
        unit_type=UnitType.APPENDIX_ITEM,
        appendix_no=6,
        item_no=17,
        column_header="Loại hình cơ sở | Mức khấu trừ (loại) | Tỷ lệ phí...",
        parent_id=f"{DOC_ID}#phu-luc-6",
        ordinal=0,
    )
    assert unit.column_header == "Loại hình cơ sở | Mức khấu trừ (loại) | Tỷ lệ phí..."


def test_a_clause_unit_rejects_a_column_header():
    # issue #160: only an appendix_item/appendix_subitem row of Phụ lục VI's
    # own premium table may ever carry a captured column header.
    with pytest.raises(ValidationError, match="must not set column_header"):
        Unit(
            id=f"{DOC_ID}#dieu-1#khoan-1",
            document_id=DOC_ID,
            unit_type=UnitType.CLAUSE,
            dieu_no=1,
            khoan_no=1,
            column_header="x",
            parent_id=f"{DOC_ID}#dieu-1",
            ordinal=0,
        )


def test_appendix_subitem_unit_rejects_a_diem_coordinate():
    with pytest.raises(ValidationError, match="must not set"):
        Unit(
            id=f"{DOC_ID}#phu-luc-1#muc-35#muc-1",
            document_id=DOC_ID,
            unit_type=UnitType.APPENDIX_SUBITEM,
            appendix_no=1,
            item_no=35,
            subitem_no=1,
            diem="a",
            parent_id=f"{DOC_ID}#phu-luc-1#muc-35",
            ordinal=0,
        )


def test_an_article_parent_must_be_a_chapter():
    with pytest.raises(ValidationError, match="parent must be a chapter"):
        Unit(
            id=f"{DOC_ID}#dieu-8",
            document_id=DOC_ID,
            unit_type=UnitType.ARTICLE,
            dieu_no=8,
            parent_id=f"{DOC_ID}#dieu-7",
            ordinal=0,
        )


# --- duplicate anchors at every level -------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_units"),
    [
        ("Chương I\nA\nChương I\nB\n", 1),
        ("Điều 1. A\n1. x\n1. y\n", 2),
        ("Điều 1. A\n1. x\na) p\na) q\n", 3),
    ],
    ids=["chapter", "clause", "point"],
)
def test_a_duplicate_anchor_is_skipped_at_every_level(text: str, expected_units: int):
    result = parse_document(text, DOC_ID)
    assert len(result.units) == expected_units
    assert result.stats.skipped == 1


# --- figure legends misread as clauses (issue #16) ------------------------------
#
# QCVN-style documents carry numbered figure legends under diagrams, e.g.
# "1. Các vị trí ... 2. Mẫu thử ..." followed by "Hình 5. ...". Read naively each
# numbered caption matches CLAUSE_ANCHOR exactly like a real clause, and a legend
# that claims a clause id first makes the real clause with that number look like
# the duplicate. The fixtures below reproduce that shape (not the original real
# document, which is not available here -- see the PR body).


def test_a_figure_legend_does_not_evict_the_real_clause_it_impersonates():
    text = (
        "Điều 8. Yêu cầu thử nghiệm\n"
        "1. Các mẫu thử kéo được lấy theo quy định.\n"
        "2. Các mẫu thử uốn được lấy từ mối hàn.\n"
        "1. Các vị trí thay đổi cho mẫu thử kéo 2. Mẫu thử cho thử kéo\n"
        "3. Mẫu thử cho thử uốn (mặt đỉnh mối hàn) 4. Mẫu thử cho thử uốn (mặt chân mối hàn)\n"
        "Hình 5. Các mẫu thử được lấy từ các chai gồm hai mảnh\n"
        "3. Thử nổ thủy lực\n"
        "a) Áp suất thử không nhỏ hơn 1,5 lần áp suất làm việc;\n"
        "b) Thời gian duy trì áp suất thử không nhỏ hơn 30 giây.\n"
    )
    result = parse_document(text, DOC_ID)
    units = _by_id(result)

    real_clause = units[f"{DOC_ID}#dieu-8#khoan-3"]
    assert real_clause.body == "Thử nổ thủy lực"
    assert f"{DOC_ID}#dieu-8#khoan-3#diem-a" in units
    assert f"{DOC_ID}#dieu-8#khoan-3#diem-b" in units
    assert result.stats.clauses == 3
    assert result.stats.points == 2
    assert result.stats.skipped == 2


def test_figure_legend_lines_are_logged_as_rejected_not_as_duplicates(
    caplog: pytest.LogCaptureFixture,
):
    text = (
        "Điều 8. Yêu cầu thử nghiệm\n"
        "1. Các mẫu thử kéo được lấy theo quy định.\n"
        "2. Các mẫu thử uốn được lấy từ mối hàn.\n"
        "1. Các vị trí thay đổi cho mẫu thử kéo 2. Mẫu thử cho thử kéo\n"
        "3. Mẫu thử cho thử uốn (mặt đỉnh mối hàn) 4. Mẫu thử cho thử uốn (mặt chân mối hàn)\n"
        "Hình 5. Các mẫu thử được lấy từ các chai gồm hai mảnh\n"
        "3. Thử nổ thủy lực\n"
        "a) Áp suất thử không nhỏ hơn 1,5 lần áp suất làm việc;\n"
        "b) Thời gian duy trì áp suất thử không nhỏ hơn 30 giây.\n"
    )
    with caplog.at_level(logging.WARNING, logger="ingest.parser"):
        parse_document(text, DOC_ID)
    reasons = [record.getMessage() for record in caplog.records]
    assert sum("rejected as a figure legend" in message for message in reasons) == 2
    assert not any("duplicate clause" in message for message in reasons)


def test_a_single_number_legend_followed_by_a_hinh_caption_is_rejected():
    """Signal 2 alone (a Hình caption on the next line) catches a legend.

    Unlike the fixture above, this legend has exactly one number per line, so
    only the "followed by a Hình/Bảng caption" signal can catch it.
    """
    text = (
        "Điều 9. Ký hiệu\n"
        "1. Nội dung khoản một.\n"
        "1. Xem minh họa dưới đây\n"
        "Hình 1. Sơ đồ minh họa\n"
        "2. Nội dung khoản hai thật sự.\n"
    )
    result = parse_document(text, DOC_ID)
    units = _by_id(result)
    assert units[f"{DOC_ID}#dieu-9#khoan-2"].body == "Nội dung khoản hai thật sự."
    assert result.stats.skipped == 1


def test_a_backward_clause_number_with_no_other_signal_is_still_rejected():
    """Signal 3 (monotonicity) alone catches a regression signals 1 and 2 miss."""
    text = (
        "Điều 10. Thử nghiệm\n"
        "1. Khoản một.\n"
        "2. Khoản hai.\n"
        "3. Khoản ba.\n"
        "1. Một dòng đánh số ngược lại, không có Hình đi kèm.\n"
        "4. Khoản bốn thật sự.\n"
    )
    result = parse_document(text, DOC_ID)
    units = _by_id(result)
    assert f"{DOC_ID}#dieu-10#khoan-4" in units
    assert result.stats.clauses == 4
    assert result.stats.skipped == 1


def test_claim_evicts_an_earlier_clause_when_a_later_one_is_preferred(
    caplog: pytest.LogCaptureFixture,
):
    """Direct test of the duplicate-clause safety net described in issue #16.

    ``_open_clause``'s figure-legend/monotonic pre-filter is expected to
    prevent a real id collision from ever reaching ``_claim`` in practice --
    that is the whole point of fixing the legend misclassification first. This
    test exercises ``_claim``'s own eviction logic directly, in isolation, as
    the safety net it is meant to be for a future document shape the
    pre-filter does not catch.
    """
    parser = _Parser(DOC_ID)
    stale_id = f"{DOC_ID}#dieu-1#khoan-1"
    stale_unit = Unit(
        id=stale_id,
        document_id=DOC_ID,
        unit_type=UnitType.CLAUSE,
        dieu_no=1,
        khoan_no=1,
        parent_id=f"{DOC_ID}#dieu-1",
        body="stale occurrence that should be evicted",
        ordinal=0,
    )
    parser._units.append(stale_unit)
    parser._seen_ids.add(stale_id)

    with caplog.at_level(logging.WARNING, logger="ingest.parser"):
        blocked = parser._claim(stale_id, 42, "duplicate clause", prefer_new=True)

    assert blocked is False
    assert stale_unit not in parser._units
    assert parser._skipped == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("evicted" in message and stale_id in message for message in messages)


# --- a promulgating circular's own articles ahead of Chương I (issue #54) -------
#
# A "Công báo" extract of a QCVN standard often carries the tail of the Thông tư
# that promulgates it -- its own closing articles -- ahead of the standard's own
# Chương I in the extracted text, e.g. "Điều 1. Ban hành kèm theo Thông tư này...".
# Read naively, ARTICLE_ANCHOR matches those lines as this document's own Điều 1-4,
# and first-occurrence-wins then rejects the standard's real, later Điều 1-4 as
# duplicates -- silently dropping its actual content (real shape: PR body for #54).


def test_an_article_before_the_first_chapter_is_folded_into_preamble():
    text = (
        "Điều 1.  Ban hành kèm theo Thông tư này Quy chuẩn kỹ thuật quốc gia.\n"
        "Điều 2.  Hiệu lực thi hành\n"
        "Điều 3. Điều khoản chuyển tiếp\n"
        "Điều 4.  Ủy ban nhân dân các tỉnh chịu trách nhiệm thi hành.\n"
        "Chương I\n"
        "QUY ĐỊNH CHUNG\n"
        "Điều 1. Phạm vi điều chỉnh\n"
        "Quy chuẩn này quy định các yêu cầu an toàn.\n"
        "Điều 3. Giải thích từ ngữ\n"
        "1. Khí dầu mỏ hóa lỏng là sản phẩm hydrocacbon.\n"
    )
    result = parse_document(text, DOC_ID)
    units = _by_id(result)

    real_article_1 = units[f"{DOC_ID}#dieu-1"]
    assert real_article_1.heading == "Phạm vi điều chỉnh"
    assert real_article_1.body == "Quy chuẩn này quy định các yêu cầu an toàn."

    real_article_3 = units[f"{DOC_ID}#dieu-3"]
    assert real_article_3.heading == "Giải thích từ ngữ"
    assert f"{DOC_ID}#dieu-3#khoan-1" in units
    assert units[f"{DOC_ID}#dieu-3#khoan-1"].body == "Khí dầu mỏ hóa lỏng là sản phẩm hydrocacbon."

    assert result.stats.articles == 2
    assert "Ban hành kèm theo Thông tư này" in result.preamble
    assert "Ủy ban nhân dân các tỉnh chịu trách nhiệm thi hành" in result.preamble


def test_the_circulars_articles_do_not_log_as_duplicates(caplog: pytest.LogCaptureFixture):
    text = (
        "Điều 3. Ủy ban nhân dân các tỉnh chịu trách nhiệm thi hành.\n"
        "Chương I\n"
        "QUY ĐỊNH CHUNG\n"
        "Điều 3. Giải thích từ ngữ\n"
        "Nội dung thật.\n"
    )
    with caplog.at_level(logging.WARNING, logger="ingest.parser"):
        result = parse_document(text, DOC_ID)
    assert result.stats.skipped == 0
    messages = [record.getMessage() for record in caplog.records]
    assert not any("duplicate article" in message for message in messages)


def test_a_chapterless_documents_first_article_still_opens_normally():
    # No Chương anywhere in this text -- "before the first chapter" is not a
    # meaningful position, so the article must open exactly as it always has.
    text = "Điều 1.  Ban hành kèm theo Thông tư này Quy chuẩn kỹ thuật quốc gia.\n"
    result = parse_document(text, DOC_ID)
    assert result.stats.articles == 1
    assert f"{DOC_ID}#dieu-1" in _by_id(result)


# --- a citation to another document's article read as a heading (issue #56) -----
#
# "...quy định tại Điều 17 Luật Phòng cháy, chữa cháy..." has no period after the
# number, and can fall at the start of a physical line (an ordinary PDF line-wrap).
# Read naively it opens this document's own Điều 17, and the real "Điều 17. Thủ tục
# huy động..." heading that follows is then rejected as a duplicate (real shape:
# PR body for #56).


def test_a_citation_to_another_documents_article_does_not_open_a_new_article(
    caplog: pytest.LogCaptureFixture,
):
    text = (
        "Điều 16. Trách nhiệm huấn luyện\n"
        "Nội dung của điều 16.\n"
        "Điều 17 Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ đối với dự án đầu tư.\n"
        "Điều 17. Thủ tục huy động lực lượng\n"
        "Nội dung thật của điều 17.\n"
    )
    with caplog.at_level(logging.WARNING, logger="ingest.parser"):
        result = parse_document(text, DOC_ID)
    units = _by_id(result)

    real_article = units[f"{DOC_ID}#dieu-17"]
    assert real_article.heading == "Thủ tục huy động lực lượng"
    assert real_article.body == "Nội dung thật của điều 17."
    assert result.stats.articles == 2

    messages = [record.getMessage() for record in caplog.records]
    assert not any("duplicate article" in message for message in messages)


def test_the_citation_line_is_folded_into_the_surrounding_body():
    text = (
        "Điều 16. Trách nhiệm huấn luyện\n"
        "Nội dung của điều 16.\n"
        "Điều 17 Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ đối với dự án đầu tư.\n"
        "Điều 17. Thủ tục huy động lực lượng\n"
        "Nội dung thật của điều 17.\n"
    )
    result = parse_document(text, DOC_ID)
    article_16 = _by_id(result)[f"{DOC_ID}#dieu-16"]
    assert "Điều 17 Luật Phòng cháy" in article_16.body


def test_an_articles_own_heading_without_a_period_is_still_recognized():
    # No legal-instrument noun here -- this is a genuine heading that simply
    # was not punctuated, a real and common shape in the corpus. Must not be
    # rejected just because it lacks a period.
    text = "Điều 1 Phạm vi điều chỉnh\nNội dung.\n"
    result = parse_document(text, DOC_ID)
    assert result.stats.articles == 1
    assert _by_id(result)[f"{DOC_ID}#dieu-1"].heading == "Phạm vi điều chỉnh"


def test_a_period_protects_a_heading_that_happens_to_start_with_a_legal_noun():
    # The period is what marks this as the document's own heading, even
    # though the heading text itself starts with "Luật".
    text = "Điều 5. Luật Đầu tư có liên quan như sau:\nNội dung.\n"
    result = parse_document(text, DOC_ID)
    assert result.stats.articles == 1
    assert _by_id(result)[f"{DOC_ID}#dieu-5"].heading == "Luật Đầu tư có liên quan như sau:"


def test_appendix_is_separate_from_last_article_and_signature_is_not_content():
    text = """Điều 47. Trách nhiệm thi hành
1. Nội dung của điều.
Nơi nhận: các cơ quan liên quan
TM. CHÍNH PHỦ
THỦ TƯỚNG
Nguyễn Văn A
PHỤ LỤC I
DANH MỤC CƠ SỞ (Kèm theo Nghị định số 105/2025/NĐ-CP)
13. Cơ sở kinh doanh khí đốt; cửa hàng xăng dầu.
a) Điểm chi tiết.
"""
    result = parse_document(text, "pccc-decree-105-2025")
    by_id = {unit.id: unit for unit in result.units}
    assert by_id["pccc-decree-105-2025#dieu-47#khoan-1"].body == "Nội dung của điều."
    item = by_id["pccc-decree-105-2025#phu-luc-1#muc-13"]
    assert item.body == "Cơ sở kinh doanh khí đốt; cửa hàng xăng dầu."
    all_body = "\n".join(unit.body for unit in result.units)
    assert "Nơi nhận" not in all_body
    assert "Nguyễn Văn A" not in all_body
    assert by_id["pccc-decree-105-2025#phu-luc-1#muc-13#diem-a"].parent_id == item.id


def test_unnumbered_appendix_without_items_is_one_unit():
    result = parse_document("Điều 1. Nội dung\nPHỤ LỤC\nGhi chú chung.\n", DOC_ID)
    appendices = [unit for unit in result.units if unit.unit_type is UnitType.APPENDIX]
    assert len(appendices) == 1
    assert appendices[0].id == f"{DOC_ID}#phu-luc-1"
    assert appendices[0].body == "Ghi chú chung."


# --- form appendices (issue #131) ------------------------------------------------


def test_a_form_becomes_its_own_unit_not_an_appendix_item():
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC\n"
        "Mẫu số 01\n"
        "GIẤY ĐỀ NGHỊ\n"
        "1. Tên thương nhân: ...............\n"
        "2. Địa chỉ: ...............\n"
    )
    result = parse_document(text, DOC_ID)
    by_id = _by_id(result)
    form = by_id[f"{DOC_ID}#phu-luc-1#mau-01"]
    assert form.unit_type is UnitType.APPENDIX_FORM
    assert form.form_code == "01"
    assert form.parent_id == f"{DOC_ID}#phu-luc-1"
    assert (
        form.body
        == "GIẤY ĐỀ NGHỊ\n1. Tên thương nhân: ...............\n2. Địa chỉ: ..............."
    )
    assert not any(unit.unit_type is UnitType.APPENDIX_ITEM for unit in result.units)
    assert f"{DOC_ID}#phu-luc-1#muc-1" not in by_id
    assert f"{DOC_ID}#phu-luc-1#muc-2" not in by_id


def test_a_form_code_with_letters_is_lowercased():
    text = "Điều 1. Nội dung\nPHỤ LỤC VIII\nMẫu số PC01\nPHIẾU THÔNG TIN CỦA CƠ SỞ\n"
    result = parse_document(text, "pccc-decree-105-2025")
    by_id = _by_id(result)
    form = by_id["pccc-decree-105-2025#phu-luc-8#mau-pc01"]
    assert form.form_code == "pc01"
    assert form.body == "PHIẾU THÔNG TIN CỦA CƠ SỞ"


def test_two_forms_in_one_appendix_each_become_their_own_unit():
    # Shaped after gas-decree-87-2018's own real Phụ lục: a table of contents
    # naming each form by code and title, immediately followed by the forms'
    # own real content, each opening with a bare "Mẫu số N" line.
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC\n"
        "(Kèm theo Nghị định này)\n"
        "Mẫu số 01\n"
        "Giấy đề nghị cấp Giấy chứng nhận đủ điều kiện\n"
        "Mẫu số 02\n"
        "Giấy chứng nhận đủ điều kiện\n"
        "Mẫu số 01\n"
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
        "GIẤY ĐỀ NGHỊ\n"
        "Mẫu số 02\n"
        "BỘ CÔNG THƯƠNG\n"
        "GIẤY CHỨNG NHẬN\n"
    )
    result = parse_document(text, DOC_ID)
    by_id = _by_id(result)
    forms = [unit for unit in result.units if unit.unit_type is UnitType.APPENDIX_FORM]
    assert {form.id for form in forms} == {
        f"{DOC_ID}#phu-luc-1#mau-01",
        f"{DOC_ID}#phu-luc-1#mau-02",
    }
    # The table of contents entry is evicted by the real content that follows it.
    assert (
        by_id[f"{DOC_ID}#phu-luc-1#mau-01"].body
        == "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nGIẤY ĐỀ NGHỊ"
    )
    assert by_id[f"{DOC_ID}#phu-luc-1#mau-02"].body == "BỘ CÔNG THƯƠNG\nGIẤY CHỨNG NHẬN"


def test_a_numbered_table_of_contents_entry_naming_a_form_is_not_an_item():
    # Shaped after circular 36/2025's own real Phụ lục VI: its table of
    # contents numbers each entry, but every entry names a form, not a
    # genuine appendix item.
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "CÁC BIỂU MẪU\n"
        "1. Mẫu số 01: Sổ theo dõi phương tiện.\n"
        "2. Mẫu số 02: Phiếu chiến thuật.\n"
        "Mẫu số 01\n"
        "SỔ THEO DÕI PHƯƠNG TIỆN\n"
    )
    result = parse_document(text, "pccc-circular-36-2025")
    by_id = _by_id(result)
    assert not any(unit.unit_type is UnitType.APPENDIX_ITEM for unit in result.units)
    assert "pccc-circular-36-2025#phu-luc-6#muc-1" not in by_id
    assert "pccc-circular-36-2025#phu-luc-6#muc-2" not in by_id
    form = by_id["pccc-circular-36-2025#phu-luc-6#mau-01"]
    assert form.body == "SỔ THEO DÕI PHƯƠNG TIỆN"
    # The second table-of-contents entry still opens its own form, kept as-is
    # since nothing in this excerpt supplies its real content.
    assert by_id["pccc-circular-36-2025#phu-luc-6#mau-02"].body == "Phiếu chiến thuật."


def test_a_forms_own_placeholder_signature_does_not_drop_the_next_form():
    # Issue #131: a sample form's own letterhead/signer placeholder
    # ("Nơi nhận:", "ĐẠI DIỆN CƠ SỞ") must not be read as the document's real
    # final signature -- that reading previously dropped every form after
    # the first one entirely (measured on gas-decree-87-2018's real forms).
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC\n"
        "Mẫu số 01\n"
        "GIẤY ĐỀ NGHỊ\n"
        "Nơi nhận:\n"
        "- Như trên;\n"
        "ĐẠI DIỆN THƯƠNG NHÂN\n"
        "Mẫu số 02\n"
        "GIẤY CHỨNG NHẬN\n"
    )
    result = parse_document(text, DOC_ID)
    by_id = _by_id(result)
    form_01 = by_id[f"{DOC_ID}#phu-luc-1#mau-01"]
    assert form_01.body == "GIẤY ĐỀ NGHỊ\nNơi nhận:\n- Như trên;\nĐẠI DIỆN THƯƠNG NHÂN"
    form_02 = by_id[f"{DOC_ID}#phu-luc-1#mau-02"]
    assert form_02.body == "GIẤY CHỨNG NHẬN"


def test_a_genuine_numbered_appendix_is_unaffected_by_form_detection():
    # No "Mẫu số" line anywhere: this appendix must split into items exactly
    # as it always has (trap #1 -- a non-form appendix must stay byte-identical).
    text = (
        "Điều 47. Trách nhiệm thi hành\n"
        "1. Nội dung của điều.\n"
        "PHỤ LỤC I\n"
        "DANH MỤC CƠ SỞ (Kèm theo Nghị định số 105/2025/NĐ-CP)\n"
        "13. Cơ sở kinh doanh khí đốt; cửa hàng xăng dầu.\n"
        "a) Điểm chi tiết.\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    by_id = _by_id(result)
    assert not any(unit.unit_type is UnitType.APPENDIX_FORM for unit in result.units)
    item = by_id["pccc-decree-105-2025#phu-luc-1#muc-13"]
    assert item.body == "Cơ sở kinh doanh khí đốt; cửa hàng xăng dầu."
    assert by_id["pccc-decree-105-2025#phu-luc-1#muc-13#diem-a"].parent_id == item.id


# --- a thousands group is never an item/clause anchor (issue #141) --------------
#
# Decree 105/2025's own Phụ lục II, "Danh mục cơ sở có nguy hiểm về cháy, nổ", is a
# table of area/volume thresholds ("1.000 m²", "30.000 m³"). Before this fix,
# CLAUSE_ANCHOR read a thousands group's own dot as if it opened a numbered item:
# "1.000 m² đến dưới" matched as item "1", with "000 m² đến dưới" as its body,
# merging the table's real, STT-numbered rows out of order.
#
# Narrowed per PR #142's review: the first version of this fix rejected *any*
# digit right after the dot, which also rejected hierarchical sub-numbering like
# "1.1." and a citation like "4.2.7.2" -- neither is a thousands group (their
# groups are not all exactly three digits), and both must keep matching exactly
# as they always have on `main`. Only ``_THOUSANDS_GROUP`` (one to three leading
# digits, then one or more ".DDD" groups of exactly three digits) is excluded.


@pytest.mark.parametrize(
    "line",
    [
        "1.000 m² đến dưới",
        "30.000 m3 trở lên",
        "1.000 tỷ đồng trở lên",
        "1.000.000 đồng",
    ],
)
def test_clause_anchor_never_matches_a_thousands_group(line: str):
    assert CLAUSE_ANCHOR.match(line) is None


@pytest.mark.parametrize(
    "line",
    [
        "1. Nhà chung cư, nhà ở tập thể.",
        "17.",
        "47. Hạ tầng kỹ thuật của khu đô thị.",
        "1.1. Phương tiện chữa cháy cơ giới:",  # hierarchical sub-numbering, not a
        # thousands group (only one digit in the second group) -- behaves exactly
        # as on main: read as a clause/item "1" whose rest text happens to start
        # with "1.", not rejected.
        "4.2.7.2 TCVN 7441: 2004",  # a citation to a sub-section of a standard, not
        # a thousands group either (no group is exactly three digits) -- also
        # behaves exactly as on main.
    ],
)
def test_clause_anchor_still_matches_a_genuine_numbered_line(line: str):
    assert CLAUSE_ANCHOR.match(line) is not None


def test_appendix_item_body_keeps_a_thousands_separator_instead_of_opening_a_fake_item():
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC II\n"
        "DANH MỤC CƠ SỞ CÓ NGUY HIỂM VỀ CHÁY, NỔ\n"
        "1. Nhà chung cư, nhà ở tập thể Có nhà cao từ 7 tầng trở lên hoặc tổng diện "
        "tích sàn từ 3.000 m² trở lên Có nhà cao từ 5 tầng đến dưới 7 tầng hoặc tổng "
        "diện tích sàn từ 1.000 m² đến dưới 3.000 m².\n"
        "2. Nhà trẻ, trường mẫu giáo, trường mầm non.\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    by_id = _by_id(result)
    # No fake "mục 1" from "1.000", no fake "mục 3" from "3.000": exactly the two
    # real, STT-numbered items, in order.
    item_units = [u for u in result.units if u.unit_type is UnitType.APPENDIX_ITEM]
    assert [u.item_no for u in item_units] == [1, 2]
    item1 = by_id["pccc-decree-105-2025#phu-luc-2#muc-1"]
    assert "3.000 m² trở lên" in item1.body
    assert "1.000 m² đến dưới" in item1.body
    assert result.stats.skipped == 0


# --- Phụ lục VI's premium table has its own hierarchical row numbering (#147) ----


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("35.1 a) Cơ sở sản xuất công nghiệp...", (35, 1)),
        ("14.1 Chợ  N 0,5", (14, 1)),
        ("36.2 Kho chứa hàng hóa...", (36, 2)),
        # Never a thousands group: the lookahead already rejects a third digit.
        ("5.000 m3 trở lên", None),
        ("1.000.000 đồng", None),
        # Never Phụ lục IV's own real hierarchical numbering, which always
        # carries a *trailing* dot -- this anchor's lookahead excludes it too
        # (a dot is neither whitespace nor end-of-line).
        ("1.1. Phương tiện chữa cháy cơ giới:", None),
        ("4.2.7.2 TCVN 7441: 2004", None),
    ],
)
def test_appendix_subitem_anchor_matches_only_a_compound_row_with_no_trailing_dot(
    line: str, expected: tuple[int, int] | None
):
    match = APPENDIX_SUBITEM_ANCHOR.match(line)
    if expected is None:
        assert match is None
    else:
        assert match is not None
        assert (int(match.group("item")), int(match.group("subitem"))) == expected


_PREMIUM_TABLE_TEXT = (
    "Điều 1. Nội dung\n"
    "PHỤ LỤC VI\n"
    "MỨC PHÍ BẢO HIỂM VÀ MỨC KHẤU TRỪ BẢO HIỂM CHÁY, NỔ BẮT BUỘC\n"
    "17. Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg N 0,3\n"
    "18. Cửa hàng xăng dầu N 0,3\n"
    "35. Cơ sở công nghiệp có nhà phục vụ sản xuất thuộc hạng nguy hiểm cháy nổ A, B.\n"
    "35.1 a) Cơ sở sản xuất công nghiệp có nhà phục vụ sản xuất thuộc hạng nguy "
    "hiểm cháy nổ A, B N 0,2\n"
    "35.2 Cơ sở sản xuất công nghiệp có nhà phục vụ sản xuất thuộc hạng nguy hiểm "
    "cháy D, E có khối tích từ 5.000m3 trở lên M 0,15\n"
    "36. Nhà ở kết hợp sản xuất, kinh doanh.\n"
)


def test_a_hierarchical_row_becomes_a_child_of_its_own_item_not_a_duplicate():
    result = parse_document(_PREMIUM_TABLE_TEXT, "pccc-decree-105-2025")
    by_id = _by_id(result)

    # The simple, single-level rows -- the gas business and the gas station --
    # are untouched: each is its own item, by its own real STT.
    assert by_id["pccc-decree-105-2025#phu-luc-6#muc-17"].body == (
        "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg N 0,3"
    )
    assert by_id["pccc-decree-105-2025#phu-luc-6#muc-18"].body == "Cửa hàng xăng dầu N 0,3"

    # "35.1"/"35.2" are children of item 35, never a duplicate-and-skipped
    # second occurrence of it (the bug this issue exists to fix).
    assert "pccc-decree-105-2025#phu-luc-6#muc-35" in by_id
    sub1 = by_id["pccc-decree-105-2025#phu-luc-6#muc-35#muc-1"]
    sub2 = by_id["pccc-decree-105-2025#phu-luc-6#muc-35#muc-2"]
    assert sub1.unit_type is UnitType.APPENDIX_SUBITEM
    assert sub1.parent_id == "pccc-decree-105-2025#phu-luc-6#muc-35"
    assert sub1.appendix_no == 6
    assert sub1.item_no == 35
    assert sub1.subitem_no == 1
    assert sub1.body == (
        "a) Cơ sở sản xuất công nghiệp có nhà phục vụ sản xuất thuộc hạng nguy "
        "hiểm cháy nổ A, B N 0,2"
    )
    assert sub2.subitem_no == 2
    # The rate and the threshold value are both still in the cell's own text.
    assert "5.000m3 trở lên" in sub2.body
    assert "M 0,15" in sub2.body

    # Row 36, right after the sub-rows, opens as its own ordinary item -- the
    # sub-row sequence does not leak into unrelated, later top-level rows.
    assert by_id["pccc-decree-105-2025#phu-luc-6#muc-36"].body == (
        "Nhà ở kết hợp sản xuất, kinh doanh."
    )
    assert result.stats.skipped == 0


def test_a_compound_row_number_outside_phu_luc_vi_is_not_read_as_a_subitem():
    # The appendix-6 guard (`_PREMIUM_TABLE_APPENDIX_NO`) means this anchor
    # never fires for any other appendix, even one with the exact same
    # coincidental "35" then "35.1" shape -- Phụ lục V here, numbered 5, not 6.
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC V\n"
        "DANH MỤC KHÁC\n"
        "35. Mục giả định trong Phụ lục V.\n"
        "35.1 Dòng tiếp theo, vẫn đọc như mục 35 bị trùng.\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    by_id = _by_id(result)
    assert "pccc-decree-105-2025#phu-luc-5#muc-35#muc-1" not in by_id
    assert all(unit.unit_type is not UnitType.APPENDIX_SUBITEM for unit in result.units)
    # Unchanged, pre-existing behavior: a duplicate item number is skipped.
    assert result.stats.skipped == 1


def test_a_compound_row_whose_leading_number_does_not_match_the_open_item_falls_through():
    # Even inside Phụ lục VI, "5.1" cannot attach under item 35 -- the anchor
    # only fires when its own "item" group equals the appendix item currently
    # open (``self._item_no``), never from appendix_no alone.
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "MỨC PHÍ BẢO HIỂM\n"
        "35. Mục giả định.\n"
        "5.1 Dòng không thuộc mục 35.\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    assert all(unit.unit_type is not UnitType.APPENDIX_SUBITEM for unit in result.units)


# --- Phụ lục VI's own Mục I/Mục II prose, outside the table (issue #147's fix round) ---

_M = APPENDIX_SECTION_MARKER


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            f"{_M}I. MỨC PHÍ BẢO HIỂM (CHƯA BAO GỒM THUẾ GTGT)",
            ("I", "MỨC PHÍ BẢO HIỂM (CHƯA BAO GỒM THUẾ GTGT)"),
        ),
        (f"{_M}II. MỨC KHẤU TRỪ BẢO HIỂM", ("II", "MỨC KHẤU TRỪ BẢO HIỂM")),
        # Never matched without the marker -- the table's own "I." row text
        # (if it existed) would be read as an ordinary clause, not a section.
        ("I. MỨC PHÍ BẢO HIỂM", None),
    ],
)
def test_appendix_section_anchor_requires_the_sentinel_marker(
    line: str, expected: tuple[str, str] | None
):
    match = APPENDIX_SECTION_ANCHOR.match(line)
    if expected is None:
        assert match is None
    else:
        assert match is not None
        assert (match.group("section"), match.group("heading")) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (f"{_M}I.1. Đối với cơ sở ...", ("I", 1, "Đối với cơ sở ...")),
        (f"{_M}II.3. Đối với cơ sở hạt nhân...", ("II", 3, "Đối với cơ sở hạt nhân...")),
        # Never matched without the marker -- the table's own "1." row text
        # keeps matching the plain CLAUSE_ANCHOR unchanged.
        ("1. Trụ sở làm việc...", None),
    ],
)
def test_appendix_section_clause_anchor_requires_the_sentinel_marker(
    line: str, expected: tuple[str, int, str] | None
):
    match = APPENDIX_SECTION_CLAUSE_ANCHOR.match(line)
    if expected is None:
        assert match is None
    else:
        assert match is not None
        section, khoan, rest = expected
        assert (match.group("section"), int(match.group("khoan")), match.group("rest")) == (
            section,
            khoan,
            rest,
        )


def test_appendix_section_point_anchor_is_checked_before_the_clause_anchor():
    line = f"{_M}II.1.a) Đối với cơ sở có nguy hiểm về cháy, nổ thuộc loại M..."
    point_match = APPENDIX_SECTION_POINT_ANCHOR.match(line)
    assert point_match is not None
    assert (
        point_match.group("section"),
        point_match.group("khoan"),
        point_match.group("letter"),
    ) == (
        "II",
        "1",
        "a",
    )
    # The clause anchor's own permissive "rest" group would otherwise just
    # as happily swallow the "a) ..." tail -- ``feed`` checks the point
    # anchor first for exactly this reason (see its own docstring).
    clause_match = APPENDIX_SECTION_CLAUSE_ANCHOR.match(line)
    assert clause_match is not None
    assert clause_match.group("rest") == "a) Đối với cơ sở có nguy hiểm về cháy, nổ thuộc loại M..."


def test_appendix_section_opens_with_its_own_heading_and_parent():
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "MỨC PHÍ BẢO HIỂM VÀ MỨC KHẤU TRỪ\n"
        f"{_M}I. MỨC PHÍ BẢO HIỂM (CHƯA BAO GỒM THUẾ GTGT)\n"
        f"{_M}I.1. Đối với cơ sở có nguy hiểm về cháy, nổ...\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    by_id = _by_id(result)

    section = by_id["pccc-decree-105-2025#phu-luc-6#muc-i"]
    assert section.unit_type is UnitType.APPENDIX_SECTION
    assert section.heading == "MỨC PHÍ BẢO HIỂM (CHƯA BAO GỒM THUẾ GTGT)"
    assert section.parent_id == "pccc-decree-105-2025#phu-luc-6"

    clause = by_id["pccc-decree-105-2025#phu-luc-6#muc-i#khoan-1"]
    assert clause.unit_type is UnitType.APPENDIX_SECTION_CLAUSE
    assert clause.parent_id == section.id
    assert clause.body == "Đối với cơ sở có nguy hiểm về cháy, nổ..."
    assert result.stats.skipped == 0


def test_appendix_section_clause_split_by_the_table_merges_back_into_one_unit():
    # Khoản 1 Mục I's own body is split by the premium table itself: its
    # intro sits before the table, its post-table text sits after the
    # table's last row -- both reach the same sentinel (issue #147's review
    # of PR #151), and must merge into one unit, not two, with the table's
    # own row 1 claiming "#muc-1" for itself, never colliding with either half.
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "MỨC PHÍ BẢO HIỂM VÀ MỨC KHẤU TRỪ\n"
        f"{_M}I. MỨC PHÍ BẢO HIỂM (CHƯA BAO GỒM THUẾ GTGT)\n"
        f"{_M}I.1. Đối với cơ sở ... tỷ lệ phí bảo hiểm tối thiểu sau:\n"
        "1. Trụ sở làm việc của cơ quan nhà nước... M 0,05\n"
        f"{_M}I.1. Trường hợp thời hạn bảo hiểm khác 01 năm...\n"
        f"{_M}I.2. Đối với cơ sở có tổng số tiền bảo hiểm...\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    by_id = _by_id(result)

    row1 = by_id["pccc-decree-105-2025#phu-luc-6#muc-1"]
    assert row1.unit_type is UnitType.APPENDIX_ITEM
    assert row1.body == "Trụ sở làm việc của cơ quan nhà nước... M 0,05"

    khoan1 = by_id["pccc-decree-105-2025#phu-luc-6#muc-i#khoan-1"]
    assert khoan1.unit_type is UnitType.APPENDIX_SECTION_CLAUSE
    assert "tỷ lệ phí bảo hiểm tối thiểu sau" in khoan1.body
    assert "Trường hợp thời hạn bảo hiểm khác 01 năm" in khoan1.body
    # Only one unit for this id -- the merge replaced the first occurrence,
    # it did not leave a second, duplicate one behind.
    assert sum(1 for unit in result.units if unit.id == khoan1.id) == 1

    khoan2 = by_id["pccc-decree-105-2025#phu-luc-6#muc-i#khoan-2"]
    assert khoan2.body == "Đối với cơ sở có tổng số tiền bảo hiểm..."
    assert result.stats.skipped == 0


def test_appendix_section_point_attaches_to_the_current_clause_not_a_table_item():
    # Issue #147's review: Mục II's own lettered points must never attach to
    # whichever table row happened to be open (the "#muc-37#diem-a" bug).
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "MỨC PHÍ BẢO HIỂM VÀ MỨC KHẤU TRỪ\n"
        f"{_M}II. MỨC KHẤU TRỪ BẢO HIỂM\n"
        f"{_M}II.1. Đối với cơ sở có nguy hiểm về cháy, nổ (trừ cơ sở hạt nhân):\n"
        f"{_M}II.1.a) Đối với cơ sở ... thuộc loại M...\n"
        f"{_M}II.1.b) Đối với cơ sở ... thuộc loại N...\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    by_id = _by_id(result)

    clause = by_id["pccc-decree-105-2025#phu-luc-6#muc-ii#khoan-1"]
    point_a = by_id["pccc-decree-105-2025#phu-luc-6#muc-ii#khoan-1#diem-a"]
    point_b = by_id["pccc-decree-105-2025#phu-luc-6#muc-ii#khoan-1#diem-b"]
    assert point_a.unit_type is UnitType.APPENDIX_SECTION_POINT
    assert point_a.parent_id == clause.id
    assert point_b.parent_id == clause.id
    assert "#muc-37" not in point_a.id
    assert result.stats.skipped == 0


def test_appendix_section_anchors_are_gated_to_appendix_6() -> None:
    # The sentinel marker alone is not trusted -- ``feed`` also requires
    # ``self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO`` (mirroring
    # ``APPENDIX_SUBITEM_ANCHOR``'s own guard), so even a line that already
    # carries the marker is never opened as a section outside Phụ lục VI.
    text = f"Điều 1. Nội dung\nPHỤ LỤC I\nDANH MỤC KHÁC\n{_M}I. MỤC GIẢ ĐỊNH\n"
    result = parse_document(text, "pccc-decree-105-2025")
    assert all(unit.unit_type is not UnitType.APPENDIX_SECTION for unit in result.units)


# --- the premium table's own column header is carried onto its rows (#160) -------

_H = APPENDIX_COLUMN_HEADER_MARKER
_HEADER_TEXT = "Loại hình cơ sở có nguy hiểm về cháy, nổ | Mức khấu trừ (loại) | Tỷ lệ phí..."


def test_column_header_sentinel_attaches_to_the_next_row_and_subrow():
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "DANH MỤC MẪU\n"
        f"{_H}{_HEADER_TEXT}\n"
        "17. Cơ sở kinh doanh khí đốt trên 500 kg N 0,3\n"
        "35. Cơ sở công nghiệp.\n"
        "35.1 a) Cơ sở sản xuất A, B N 0,2\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    units = {unit.id: unit for unit in result.units}
    assert units["pccc-decree-105-2025#phu-luc-6#muc-17"].column_header == _HEADER_TEXT
    assert units["pccc-decree-105-2025#phu-luc-6#muc-35"].column_header == _HEADER_TEXT
    assert units["pccc-decree-105-2025#phu-luc-6#muc-35#muc-1"].column_header == _HEADER_TEXT


def test_column_header_sentinel_does_not_disturb_the_pending_unit():
    # The sentinel is pure state, never body text -- it must not appear in,
    # or truncate, whatever unit is currently being collected.
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "DANH MỤC MẪU\n"
        "17. Cơ sở kinh doanh khí đốt trên 500 kg N 0,3\n"
        f"{_H}{_HEADER_TEXT}\n"
        "18. Cửa hàng xăng dầu N 0,3\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    units = {unit.id: unit for unit in result.units}
    item_17 = units["pccc-decree-105-2025#phu-luc-6#muc-17"]
    assert item_17.body == "Cơ sở kinh doanh khí đốt trên 500 kg N 0,3"
    assert "‡" not in item_17.body
    assert _HEADER_TEXT not in item_17.body
    # The row opened *after* the sentinel reads the now-current header.
    assert units["pccc-decree-105-2025#phu-luc-6#muc-18"].column_header == _HEADER_TEXT


def test_column_header_sentinel_is_gated_to_appendix_6():
    # Outside Phụ lục VI, the marker is never consulted as an anchor -- it is
    # just an ordinary (if bizarre) line of body text, and no column header
    # state is ever set from it.
    text = f"Điều 1. Nội dung\nPHỤ LỤC I\nDANH MỤC KHÁC\n{_H}{_HEADER_TEXT}\n40. Mục thường.\n"
    result = parse_document(text, "pccc-decree-105-2025")
    units = {unit.id: unit for unit in result.units}
    item_40 = units["pccc-decree-105-2025#phu-luc-1#muc-40"]
    assert item_40.column_header is None


def test_column_header_does_not_leak_into_the_next_appendix():
    text = (
        "Điều 1. Nội dung\n"
        "PHỤ LỤC VI\n"
        "DANH MỤC MẪU\n"
        f"{_H}{_HEADER_TEXT}\n"
        "17. Cơ sở kinh doanh khí đốt trên 500 kg N 0,3\n"
        "PHỤ LỤC VII\n"
        "DANH MỤC KHÁC\n"
        "5. Một mục khác.\n"
    )
    result = parse_document(text, "pccc-decree-105-2025")
    units = {unit.id: unit for unit in result.units}
    assert units["pccc-decree-105-2025#phu-luc-7#muc-5"].column_header is None
