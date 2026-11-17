"""Tests for the repeal-statement reader (issue #68).

Pure: no database. Every fixture is the real text of a real unit of the real
corpus, PDF artifacts included ("s au đây", "hiệ u lực", "N Đ-CP"), keyed by the
unit's real id -- a sentence written from the issue's description alone would not
carry the artifacts that decide whether the reader works. The corpus itself is
git-ignored, so the text is copied here.

The five traps the issue names each have a test named for them, and Trap D (a
partial repeal is not a replacement) is the one that must never regress: emitting
REPLACED_BY for "Bãi bỏ khoản 1 và 2 Điều 30" would mark a live decree superseded
and remove it from every answer.
"""

from pathlib import Path

import pytest

from app.core.ids import article_id, clause_id, point_id
from app.schemas.unit import Unit, UnitType
from ingest.extract import _rejoin_split_document_numbers
from ingest.parser import ParseResult, parse_document
from ingest.repeals import RepealScan, scan_repeals

# --- fixtures: real unit text -------------------------------------------------------

_DECREE_105 = "pccc-decree-105-2025"
_CIRCULAR_36 = "pccc-circular-36-2025"
_LAW_2024 = "pccc-law-2024"
_DECREE_87 = "gas-decree-87-2018"
_DECREE_17 = "gas-decree-17-2020"


def _extracted(text: str) -> str:
    """Return ``text`` as ``extract_text`` hands it on: its document numbers already whole.

    A fixture below is the real PDF text, artifacts included. In the pipeline it reaches
    ``scan_repeals`` only after extraction has repaired the numbers in it (issue #80), so
    the unit builders apply the same step instead of the reader repairing them itself.
    """
    return _rejoin_split_document_numbers(text, Path("fixture.pdf"))


def _result(document: str, *units: Unit) -> ParseResult:
    empty = parse_document("", document)
    return ParseResult(document_id=document, units=tuple(units), stats=empty.stats, preamble="")


def _article(document: str, dieu: int, *, heading: str | None = None, body: str = "") -> Unit:
    return Unit(
        id=article_id(document, dieu),
        document_id=document,
        unit_type=UnitType.ARTICLE,
        dieu_no=dieu,
        heading=None if heading is None else _extracted(heading),
        body=_extracted(body),
        ordinal=dieu * 100,
    )


def _clause(document: str, dieu: int, khoan: int, body: str) -> Unit:
    return Unit(
        id=clause_id(document, dieu, khoan),
        document_id=document,
        unit_type=UnitType.CLAUSE,
        dieu_no=dieu,
        khoan_no=khoan,
        parent_id=article_id(document, dieu),
        body=_extracted(body),
        ordinal=dieu * 100 + khoan,
    )


def _point(document: str, dieu: int, khoan: int, diem: str, body: str) -> Unit:
    return Unit(
        id=point_id(document, dieu, khoan, diem),
        document_id=document,
        unit_type=UnitType.POINT,
        dieu_no=dieu,
        khoan_no=khoan,
        diem=diem,
        parent_id=clause_id(document, dieu, khoan),
        body=_extracted(body),
        ordinal=dieu * 100 + khoan * 10 + ord(diem) % 10,
    )


def _scan_clause(document: str, dieu: int, khoan: int, body: str) -> RepealScan:
    return scan_repeals(
        _result(document, _article(document, dieu), _clause(document, dieu, khoan, body))
    )


_105_LEAD_IN = (
    "Các Nghị định sau đây hết hiệu lực kể từ ngày Nghị định này có hiệu lực thi hành, "
    "trừ trường hợp quy định tại Điều 46 của Nghị định này:"
)
_105_A = (
    "Nghị định số 83/2017/NĐ-CP ngày 18 tháng 7 năm 2017 quy định về công tác cứu nạn, "
    "cứu hộ của lực lượng phòng cháy và chữa cháy;"
)
_105_B = (
    "Nghị định số 136/2020/NĐ-CP ngày 24 tháng 11 năm 2020 quy định chi tiết một số điều "
    "và biện pháp thi hành Luật Phòng cháy và chữa cháy và Luật sửa đổi, bổ sung một số "
    "điều của Luật Phòng cháy và chữa cháy;"
)
_105_C = (
    "Nghị định số 50/2024/NĐ-CP ngày 10 tháng 5 năm  2024 sửa đổi, bổ sung một số điều của "
    "Nghị định số 136/2020/NĐ-CP ngày 24 tháng 11 năm 2020 của Chính 72   CÔNG BÁO/Số 717 + "
    "718/Ngày 31-5-2025 phủ quy định chi tiết một số điều và biện pháp thi hành Luật Phòng "
    "cháy và chữa cháy và Luật sửa đổi, bổ sung một số điều của Luật Phòng cháy và chữ a "
    "cháy và Nghị định số 83/2017/NĐ-CP ngày 18 tháng 7 năm 2017 của Chính phủ quy định về "
    "công tác cứu nạn, cứu hộ của lực lượng phòng cháy và chữa cháy;"
)
_105_D = (
    "Nghị định số 78/2011/NĐ-CP ngày 01 tháng 9 năm 2011 quy định việc phối hợp giữa Bộ "
    "Công an với Bộ Quốc phòng trong tổ chức thực hiện công tác phòng cháy và chữa cháy "
    "đối với cơ sở quốc phòng."
)


def _decree_105_article_45() -> ParseResult:
    d = _DECREE_105
    return _result(
        d,
        _article(d, 45),
        _clause(d, 45, 2, _105_LEAD_IN),
        _point(d, 45, 2, "a", _105_A),
        _point(d, 45, 2, "b", _105_B),
        _point(d, 45, 2, "c", _105_C),
        _point(d, 45, 2, "d", _105_D),
    )


_36_LEAD_IN = (
    "Các Thông tư s au đây hết hiệu lực kể từ ngày Thông tư này có hiệ u lực thi hành, "
    "trừ trường hợp quy định tại khoản 3 Điều này:"
)
_36_A = (
    "Thông tư số  17/2021/TT-BCA ngày 05 tháng 02 năm 2021 của Bộ trưởng Bộ Công an quy "
    "đị nh về quản lý, bảo quản, bảo dưỡng phương tiện phòng cháy, chữa cháy và cứu nạn, "
    "cứu hộ;"
)
_36_B = (
    "Thông tư số 150/2020/TT-BCA ngày 31 tháng 12 năm 2020 của Bộ trưởng Bộ Công an quy "
    "định về trang bị phương tiện phòng cháy, chữ a cháy và cứu nạn, cứu hộ cho lực lượng "
    "dân phòng, lực lượng phòng cháy và chữa cháy cơ sở, lực lượng phòng cháy và  chữa "
    "cháy chuyên ngành, đư ợc sửa đổi, bổ sung theo quy định tại Thông tư số 55/2024/TT-BCA "
    "ngày 31 tháng 10 năm 2024 của Bộ trưởng Bộ Công an sửa đổi, bổ sung một số điều của "
    "Thông tư số 141/2020/TT-BCA ngày 23 tháng 12 năm 2020 của Bộ trưởng Bộ Công an quy "
    "định công tác kiểm tra về phòng cháy, chữ a cháy và cứu nạn, cứu hộ của lực lư ợng "
    "Công an nhân dân; Thông tư số 150/2020/TT-BCA ngày 31 tháng 12 năm 2020 của Bộ trưởng "
    "Bộ Công an quy định về trang bị phương tiện phòng cháy, chữa cháy và cứu nạn, cứu hộ "
    "cho lực lượng dân phòng, lực lượng phòng cháy và chữa cháy cơ sở, lực lượng phòng "
    "cháy và chữa cháy chuyên ngành; Thông tư số 82/2021/TT-BCA ngày 06 tháng 8 năm 2021 "
    "của Bộ trưởng Bộ Công an quy định về tiêu chuẩn, nhiệm vụ, tập huấn, kiểm tra nghi ệp "
    "vụ thẩm duyệt thiết kế, nghi ệm thu về phòng cháy và chữa cháy của lực lượ ng Cảnh sát "
    "phòng cháy, chữ a cháy và cứu nạn, cứu hộ; Thông tư  số 06/2022/TT-BCA ngày 17 tháng "
    "01 năm 2022 của Bộ trưởng Bộ Công an quy đị nh quy trình thực hiện nhiệm vụ công tác "
    "phòng cháy, chữa cháy, cứu nạn, cứu hộ trong Công an nhân dân;"
)
_36_C = (
    "Thông tư số 149/2020/TT-BCA ngày 31 tháng 12 năm 2020 của Bộ trưởng Bộ Công an quy đị "
    "nh chi tiết một số điều và biện pháp thi hành Luậ t Phòng cháy và chữa cháy và Luật "
    "sửa đổi, bổ sung một số điều của Luật Phòng cháy và chữa cháy và Nghị định số "
    "136/2020/NĐ-CP ngày 24 tháng 11 năm 2020 của Chính phủ quy định chi tiết một số điều "
    "và biện pháp thi hành Luật Phòng cháy và chữa cháy và Luật sửa đổi, bổ sung một số "
    "điều của Luật Phòng cháy và chữa cháy;"
)
_36_D = (
    "Thông tư số 32/2024/TT-BCA ngày 10 tháng 7 năm 2024 của Bộ trưởng Bộ Công an sửa đổi, "
    "bổ sung một số điều của Thông tư số 149/2020/TT-BCA ngày 31 tháng 12 năm 2020 của Bộ "
    "trưởng Bộ Công an quy định chi tiế t một số điều và CÔNG BÁO/Số 743 + 744/Ngày "
    "12-6-2025 33 biện pháp thi hành Luật Phòng cháy và chữa cháy và Luật sửa đổi, bổ sung "
    "một số điều của Luật Phòng cháy và chữa cháy và Nghị định số 136/2020/NĐ-CP ngày 24 "
    "tháng 11 năm 2020 của Chính phủ quy định chi tiết một số điều và biệ n pháp thi hành "
    "Luật Phòng cháy và chữa cháy và Luật sửa đổi, bổ sung một số điều của Luật Phòng cháy "
    "và chữa cháy và Thông tư số 08/2018/TT-BCA ngày 05 tháng 3 năm 2018 của Bộ trưởng Bộ "
    "Công an quy đị nh chi ti ết một số điều của Nghị định số 83/2017/NĐ-CP ngày 18 tháng 7 "
    "năm 2017 của Chính phủ quy định về công tác cứu nạn, cứu hộ của lực lượng phòng cháy "
    "và chữa cháy."
)


def _circular_36_article_14() -> ParseResult:
    d = _CIRCULAR_36
    return _result(
        d,
        _article(d, 14),
        _clause(d, 14, 2, _36_LEAD_IN),
        _point(d, 14, 2, "a", _36_A),
        _point(d, 14, 2, "b", _36_B),
        _point(d, 14, 2, "c", _36_C),
        _point(d, 14, 2, "d", _36_D),
    )


_LAW_54_K2 = (
    "Luật Phòng cháy và chữa cháy số 27/2001/QH10 đã được sửa đổi, bổ sung một số điều theo "
    "Luật số 40/2013/QH13 và Luật số 30/2023/QH15 hết hiệu lực từ ngày Luật này có hiệu lực "
    "thi hành, trừ trường hợp quy định tại Điều 55 của Luật này."
)
_LAW_54_K3 = (
    "Nghị quyết số 99/2019/QH14 ngày 27/11/2019 của Quốc hội về tiếp tục hoàn thiện, nâng "
    "cao hiệu lực, hiệu quả thực hiện chính sách, pháp luật về phòng cháy và chữa cháy hết "
    "hiệu lực từ ngày Luật này có hiệu lực thi hành."
)
_87_K2 = (
    "Nghị định này thay thế Nghị định số 19/2016/NĐ-CP ngày 22 tháng 3 năm 2016 của Chính "
    "phủ về kinh doanh khí. Bãi bỏ Chương V của Nghị định số 77/2016/NĐ-CP ngày 01 tháng 7 "
    "năm 2016 của Chính phủ ban hành Nghị định sửa đổi, bổ sung một số quy định về điều "
    "kiện đầu tư kinh doanh trong lĩnh vực mua bán hàng hóa quốc tế, hóa chất; vật liệu nổ "
    "công nghiệp, phân bón, kinh doanh khí, kinh doanh thực phẩm thuộc phạm vi quản lý nhà "
    "nước của Bộ Công Thương."
)


def _numbers(scan: RepealScan) -> list[str]:
    return [repeal.number for repeal in scan.repeals]


# --- the genuine whole-document repeals ---------------------------------------------


def test_a_list_clause_repeals_the_document_each_of_its_points_opens_with():
    scan = scan_repeals(_decree_105_article_45())

    assert _numbers(scan) == ["83/2017/NĐ-CP", "136/2020/NĐ-CP", "50/2024/NĐ-CP", "78/2011/NĐ-CP"]
    assert [repeal.origin_id for repeal in scan.repeals] == [
        f"{_DECREE_105}#dieu-45#khoan-2#diem-{letter}" for letter in "abcd"
    ]
    assert (scan.partial, scan.unclear) == (0, 0)


def test_a_repeal_carries_the_lead_in_and_its_point_as_evidence():
    scan = scan_repeals(_decree_105_article_45())

    evidence = scan.repeals[1].evidence
    assert evidence.startswith(_105_LEAD_IN)
    assert "Nghị định số 136/2020/NĐ-CP ngày 24 tháng 11 năm 2020" in evidence


def test_the_lead_in_and_every_point_are_statement_units():
    scan = scan_repeals(_decree_105_article_45())

    assert scan.statement_units == frozenset(
        [f"{_DECREE_105}#dieu-45#khoan-2"]
        + [f"{_DECREE_105}#dieu-45#khoan-2#diem-{letter}" for letter in "abcd"]
    )


def test_the_circular_list_survives_the_pdf_artifacts_in_its_own_lead_in():
    scan = scan_repeals(_circular_36_article_14())

    # "s au đây" and "hiệ u lực" are split in the real lead-in; "Thông tư số  17/2021"
    # has two spaces. None of it may stop the list being read.
    assert _numbers(scan) == [
        "17/2021/TT-BCA",
        "150/2020/TT-BCA",
        "149/2020/TT-BCA",
        "32/2024/TT-BCA",
    ]


def test_trap_a_only_the_subject_of_an_expiry_sentence_is_repealed_not_the_laws_that_amended_it():
    scan = _scan_clause(_LAW_2024, 54, 2, _LAW_54_K2)

    assert _numbers(scan) == ["27/2001/QH10"]
    assert "40/2013/QH13" not in _numbers(scan)
    assert "30/2023/QH15" not in _numbers(scan)


def test_a_resolution_is_repealed_when_its_expiry_sentence_names_only_it():
    scan = _scan_clause(_LAW_2024, 54, 3, _LAW_54_K3)

    assert _numbers(scan) == ["99/2019/QH14"]


def test_trap_b_one_unit_holds_one_replacement_and_one_partial_repeal():
    scan = _scan_clause(_DECREE_87, 61, 2, _87_K2)

    assert _numbers(scan) == ["19/2016/NĐ-CP"]
    assert "77/2016/NĐ-CP" not in _numbers(scan)
    assert scan.partial == 1  # "Bãi bỏ Chương V của ..." -- counted, never an edge
    assert scan.repeals[0].evidence == (
        "Nghị định này thay thế Nghị định số 19/2016/NĐ-CP ngày 22 tháng 3 năm 2016 của "
        "Chính phủ về kinh doanh khí."
    )


def test_trap_c_a_point_repeals_its_own_document_not_the_ones_it_describes():
    scan = scan_repeals(_circular_36_article_14())
    by_point = {repeal.origin_id.rsplit("-", 1)[-1]: repeal.number for repeal in scan.repeals}

    # diem-c is TT 149/2020, which "quy định chi tiết ... Nghị định số 136/2020/NĐ-CP".
    assert by_point["c"] == "149/2020/TT-BCA"
    assert "136/2020/NĐ-CP" not in _numbers(scan)
    # diem-b and diem-d name four more documents each in their descriptions.
    for described in ("55/2024/TT-BCA", "141/2020/TT-BCA", "82/2021/TT-BCA", "06/2022/TT-BCA"):
        assert described not in _numbers(scan)
    assert "08/2018/TT-BCA" not in _numbers(scan)
    assert _numbers(scan).count("149/2020/TT-BCA") == 1  # diem-d only *amends* it


# --- trap D: a partial repeal is never a replacement --------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "Bãi bỏ khoản 1 và 2 Điều 30.",
        "Bãi bỏ Điều 7.",
        "Bãi bỏ điểm b, e và g khoản 1 Điều 15;",
        "Bãi bỏ khoản 11 Điều 3.",
        "Bãi bỏ điểm c khoản 1, điểm c và đ khoản 2 Điều 9.",
        (
            "Bãi bỏ điểm đ khoản 5, điểm c kho ản 8, điểm b và d kho ản 9, kho ản 10 và "
            "khoản 11 Điều 34."
        ),
        "Bãi bỏ cụm từ “hoặc pháp luật về phòng cháy chữa cháy” tại điểm c khoản 6 Điều 19;",
        "Bãi bỏ một số điều, khoản Nghị định số 38/2014/NĐ-CP như sau:",
        "Sửa đổi, bổ sung, bãi bỏ một số điều của Nghị định số 67/2023/NĐ-CP:",
        (
            "Bãi bỏ điểm đ khoản 1 Điều 15 Nghị định số 38/2014/NĐ-CP đã được sửa đổi tại "
            "Điều 9 Nghị định số 77/2016/NĐ-CP."
        ),
        (
            "Bãi bỏ một số hóa chất thu ộc Danh mục hóa chất sản xu ất, kinh doanh có điều "
            "ki ện trong lĩnh vực công nghi ệp của Phụ lục I kèm theo Nghị định số "
            "113/2017/N Đ-CP ngày 09 tháng 10 năm 2017 của Chính phủ như sau:"
        ),
    ],
)
def test_trap_d_a_partial_repeal_is_counted_and_never_becomes_a_repeal(sentence: str):
    scan = _scan_clause(_DECREE_17, 11, 4, sentence)

    assert scan.repeals == ()
    assert scan.partial == 1
    assert scan.unclear == 0
    assert scan.statement_units == frozenset()


def test_trap_d_the_split_cue_the_real_corpus_prints_is_still_a_partial_repeal():
    # NĐ 105/2025 Điều 44's heading, verbatim: "bã i  bỏ" is split, and so is "Nghị đị nh".
    heading = (
        "Sử a đổi, bổ sung,  bã i  bỏ một số điều của Nghị đị nh số 67/2023/NĐ-CP ngày 06 "
        "tháng 9 năm 2023"
    )
    scan = scan_repeals(_result(_DECREE_105, _article(_DECREE_105, 44, heading=heading)))

    assert scan.repeals == ()
    assert scan.partial == 1


def test_trap_d_a_whole_amending_decree_of_partial_repeals_yields_no_repeal_at_all():
    d = _DECREE_17
    units = [
        _article(
            d,
            2,
            heading="Bãi bỏ một số điều, khoản của Nghị định số 116/2017/NĐ-CP ngày 17",
            body=(
                "tháng 10 năm 2017 của Chính phủ quy định điều ki ện sản xu ất, lắp ráp, nhập "
                "khẩu và kinh doanh dịch vụ bảo hành, bảo dưỡng ô tô"
            ),
        ),
        _clause(d, 2, 1, "Bãi bỏ khoản 11 Điều 3."),
        _clause(d, 2, 2, "Bãi bỏ khoản 2, 3, 4 và 5 Điều 7."),
        _clause(d, 2, 3, "Bãi bỏ điểm e và h khoản 2 Điều 8."),
        _clause(d, 2, 4, "Bãi bỏ khoản 8, 9 và 10 Điều 21."),
    ]

    scan = scan_repeals(_result(d, *units))

    assert scan.repeals == ()
    assert scan.partial == 5  # the article heading announces them; each act is counted too
    assert scan.unclear == 0


# --- guards: an unrecognised shape is counted and never becomes an edge -------------


def test_a_certificate_expiring_is_not_a_document_repeal_and_is_not_counted():
    # Real: gas-decree-17-2020#dieu-13#khoan-3. The AMENDMENT_PATTERN matches it.
    body = (
        "Trường hợp cấp lại do cơ sở thay đổi địa điểm sản xuất, kinh doanh; thay đổi, bổ "
        "sung quy trình sản xuất, mặt hàng kinh doanh và khi Giấy chứng nhận hết hiệu lực. "
        "Trình tự, thủ tục cấp Giấy chứng nhận thực hi ện theo quy định tại kho ản 1 Điều này."
    )

    scan = _scan_clause(_DECREE_17, 13, 3, body)

    assert scan == RepealScan()


def test_a_document_declaring_its_own_expiry_names_no_other_document_so_builds_and_counts_nothing():
    body = "Thông tư này hết hiệu lực từ ngày Nghị định này có hiệu lực thi hành."

    assert _scan_clause(_DECREE_87, 20, 1, body) == RepealScan()


def test_an_ordinary_replacement_of_equipment_is_not_a_repeal():
    body = "Thường xuyên kiểm tra, bảo dưỡng, thay thế các thiết bị của trạm nạp đảm bảo an toàn."

    assert _scan_clause(_DECREE_87, 23, 8, body) == RepealScan()


@pytest.mark.parametrize(
    "sentence",
    [
        (
            "Thay thế Phụ lục II kèm theo Nghị định số 67/2023/NĐ-CP bằng Phụ lục VI kèm "
            "theo Nghị định này."
        ),
        (
            "Thay thế mẫu số 05 của Phụ lục ban hành kèm theo Nghị định số 116/2017/ NĐ-CP "
            "bằng mẫu số 15 Mục III tại Phụ lục ban hành kèm theo Nghị định này."
        ),
        (
            "Thay thế từ “thẩm duyệt” bằng từ “thẩm định” tại điểm c khoản 3 Điều 56 của "
            "Luật Xây dựng số 50/2014/QH13."
        ),
    ],
)
def test_replacing_a_part_of_a_document_is_neither_a_repeal_nor_a_partial_repeal(sentence: str):
    assert _scan_clause(_DECREE_17, 1, 9, sentence) == RepealScan()


@pytest.mark.parametrize(
    "sentence",
    [
        "Nghị định này thay thế Nghị định số 19/2016/NĐ-CP, trừ Điều 5 và Điều 6.",
        "Bãi bỏ Nghị định số 19/2016/NĐ-CP, trừ khoản 2 Điều 3.",
        "Nghị định này thay thế Nghị định số 19/2016/NĐ-CP và Nghị định số 20/2017/NĐ-CP.",
        "Nghị định này thay thế Nghị định về kinh doanh khí.",
        (
            "Nghị định này có hiệu lực từ ngày 01 tháng 7 năm 2025, thay thế Nghị định số "
            "19/2016/NĐ-CP."
        ),
        "Nghị định số 12/2019/NĐ-CP hết hiệu lực từ ngày 01 tháng 1 năm 2026.",
        (
            "Nghị định số 12/2019/NĐ-CP và Nghị định số 13/2019/NĐ-CP hết hiệu lực từ ngày "
            "Nghị định này có hiệu lực."
        ),
        "Nghị định số 12/2019/NĐ-CP hết hiệu lực từ ngày Nghị định số 99/2025/NĐ-CP có hiệu lực.",
        "Nghị định số 12/2019/NĐ-CP hết hiệu lực từ ngày Nghị định này có hiệu lực, trừ Điều 5.",
    ],
)
def test_a_statement_that_fits_no_recognised_shape_is_counted_unclear_and_builds_nothing(
    sentence: str,
):
    scan = _scan_clause(_DECREE_87, 61, 2, sentence)

    assert scan.repeals == ()
    assert scan.unclear == 1
    assert scan.statement_units == frozenset()


def test_a_bare_imperative_repeal_of_one_whole_document_is_recognised():
    scan = _scan_clause(
        _DECREE_87, 61, 3, "Bãi bỏ Nghị định số 19/2016/NĐ-CP ngày 22 tháng 3 năm 2016."
    )

    assert _numbers(scan) == ["19/2016/NĐ-CP"]


def test_a_document_that_repeals_another_with_the_verb_bai_bo_is_recognised():
    body = (
        "Nghị định này bãi bỏ Nghị định số 10/2012/NĐ-CP kể từ ngày Nghị định này có hiệu "
        "lực thi hành."
    )

    assert _numbers(_scan_clause(_DECREE_87, 1, 1, body)) == ["10/2012/NĐ-CP"]


def test_a_list_with_a_point_that_opens_with_no_document_counts_that_point_unclear():
    d = _DECREE_105
    scan = scan_repeals(
        _result(
            d,
            _article(d, 45),
            _clause(d, 45, 2, _105_LEAD_IN),
            _point(d, 45, 2, "a", _105_A),
            _point(d, 45, 2, "b", "Điều 5 và Điều 6 của Nghị định số 136/2020/NĐ-CP;"),
        )
    )

    assert _numbers(scan) == ["83/2017/NĐ-CP"]
    assert scan.unclear == 1


def test_a_list_lead_in_with_no_points_beneath_it_names_nothing():
    scan = _scan_clause(_DECREE_105, 45, 2, _105_LEAD_IN)

    assert scan.repeals == ()
    assert scan.unclear == 1


def test_a_list_lead_in_with_an_exception_makes_every_point_unclear_rather_than_repealed():
    d = _DECREE_105
    lead_in = (
        "Các Nghị định sau đây hết hiệu lực kể từ ngày Nghị định này có hiệu lực thi hành, "
        "trừ Điều 5:"
    )
    scan = scan_repeals(
        _result(
            d,
            _article(d, 45),
            _clause(d, 45, 2, lead_in),
            _point(d, 45, 2, "a", _105_A),
            _point(d, 45, 2, "b", _105_B),
        )
    )

    assert scan.repeals == ()
    assert scan.unclear == 2


def test_a_list_lead_in_naming_a_successor_document_is_not_read():
    d = _DECREE_105
    lead_in = (
        "Các Nghị định sau đây hết hiệu lực kể từ ngày Nghị định này có hiệu lực thi hành "
        "theo Nghị quyết số 10/2025/QH15:"
    )
    scan = scan_repeals(
        _result(d, _article(d, 45), _clause(d, 45, 2, lead_in), _point(d, 45, 2, "a", _105_A))
    )

    assert scan.repeals == ()
    assert scan.unclear == 1


def test_the_points_of_a_recognised_list_are_not_read_a_second_time():
    d = _DECREE_105
    scan = scan_repeals(
        _result(
            d,
            _article(d, 45),
            _clause(d, 45, 2, _105_LEAD_IN),
            _point(d, 45, 2, "a", "Nghị định số 83/2017/NĐ-CP bãi bỏ khoản 3 Điều 5;"),
        )
    )

    assert _numbers(scan) == ["83/2017/NĐ-CP"]
    assert scan.partial == 0


def test_a_preamble_is_never_read_it_is_web_page_furniture_not_the_documents_text():
    # Real: the HTML copy of NĐ 87/2018 has a "Văn bản bị thay thế" summary before Điều 1.
    empty = parse_document("", _DECREE_87)
    preamble = (
        "Nghị định 87/2018/NĐ-CP có hiệu lực thi hành từ ngày 01 tháng 8 năm 2018, thay thế "
        "hoàn toàn Nghị định số 19/2016/NĐ-CP ngày 22 tháng 3 năm 2016 của Chính phủ về kinh "
        "doanh khí và bãi bỏ Chương V của Nghị định số 77/2016/NĐ-CP."
    )

    scan = scan_repeals(
        ParseResult(document_id=_DECREE_87, units=(), stats=empty.stats, preamble=preamble)
    )

    assert scan == RepealScan()


# --- trap E: a number the PDF split ---------------------------------------------------


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        ("113/2017/N Đ-CP", "113/2017/NĐ-CP"),
        ("113/2017/NĐ -CP", "113/2017/NĐ-CP"),
        ("113/2017/NĐ- CP", "113/2017/NĐ-CP"),
        ("116/2017/ NĐ-CP", "116/2017/NĐ-CP"),
        ("68/ 2006/QH11", "68/2006/QH11"),
    ],
)
def test_trap_e_a_number_the_pdf_split_is_repaired_before_it_is_used(printed: str, expected: str):
    # The repair is ingest.extract's (issue #80); this pins that the reader is handed whole numbers.
    sentence = (
        f"Nghị định này thay thế Nghị định số {printed} ngày 09 tháng 10 năm 2017 của Chính phủ."
    )

    scan = _scan_clause(_DECREE_17, 9, 3, sentence)

    assert _numbers(scan) == [expected]


def test_trap_e_no_scan_of_the_real_partial_repeal_that_carries_a_split_number_names_it_truncated():
    # Real: gas-decree-17-2020#dieu-9#khoan-3 -- "113/2017/N Đ-CP". A naive scan reads "113/2017/N".
    body = (
        "Bãi bỏ một số hóa chất thu ộc Danh mục hóa chất sản xu ất, kinh doanh có điều ki ện "
        "trong lĩnh vực công nghi ệp của Phụ lục I kèm theo Nghị định số 113/2017/N Đ-CP ngày "
        "09 tháng 10 năm 2017 của Chính phủ như sau: STT Tên hóa chất theo tiếng Việt"
    )

    scan = _scan_clause(_DECREE_17, 9, 3, body)

    assert scan.repeals == ()
    assert not any(repeal.number.endswith("/N") for repeal in scan.repeals)


def test_scanning_is_deterministic_and_in_reading_order():
    assert scan_repeals(_decree_105_article_45()) == scan_repeals(_decree_105_article_45())
