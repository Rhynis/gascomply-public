"""Tests for the gold evaluation set (build plan A6 Phase 4.1, issue #95).

Two different questions are being answered here, and they need two different
kinds of test:

1. Is ``data/gold.yaml`` internally well-formed -- right shape, right category
   mix, no row citing the corrected TT 48/2011 error, every emergency row
   using a real phrase from ``tests/test_safety.py`` and the real hard-coded
   response? These run with no database at all.
2. Does every ``source_unit_id`` the file cites actually resolve to a row in
   ``units``? This is the acceptance criterion that stops the gold set from
   rotting silently the next time the parser changes (it already has, four
   times, across Phases 1-2). CI has no access to the real, reviewed corpus --
   ``corpus/`` is git-ignored and several of its documents are copyrighted --
   so this cannot mean "ingest the real corpus" here the way the issue's own
   suggestion imagines. What it means instead: a small fixture corpus, built
   from the exact real clause text this file's own author read from the dev
   database while writing each row (see the PR for that verification, unit id
   by unit id), run through the real ``ingest.parser``/``ingest.run``
   pipeline -- so a future change to the ID scheme or the anchor regexes
   still fails this test, exactly as it would against the real corpus, just
   without needing the real corpus present. ``tests/test_ingest_persistence.py``
   already establishes this "small fixture, real pipeline, migrated_db"
   pattern; this file reuses it rather than inventing a second way to prove
   the same thing. Needs a live Postgres: set ``TEST_DATABASE_URL``.

No ``pytest-asyncio`` (C12): the one async body drives its own
``asyncio.run()`` call, matching ``tests/test_repositories.py``.
"""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from sqlalchemy import text

from app.core.db import create_engine
from app.schemas.ask import AskOutcome
from app.services.safety import EMERGENCY_RESPONSE_VI
from eval.gold import GoldCategory, GoldRow, load_gold_set
from eval.metrics import answer_check
from ingest.manifest import MANIFEST_FILENAME
from ingest.run import main


def _load_sibling_module(filename: str) -> ModuleType:
    """Load a sibling test module by file path, independent of package/sys.path state.

    ``tests/`` has no ``__init__.py`` (a namespace package at best, and not even that
    under pytest's default "prepend" import mode without one), so ``from tests.test_safety
    import ...`` resolves locally under some invocations (e.g. ``python -m pytest``, which
    puts the backend/ working directory on ``sys.path``) and fails under others (a bare
    ``pytest -q``, exactly how CI's own "Tests + coverage gate" step invokes it) with
    ``ModuleNotFoundError: No module named 'tests'``. Loading by explicit path sidesteps
    that entirely -- it works the same way regardless of how pytest was invoked.
    """
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(f"gold_sibling_{path.stem}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive only
        raise ImportError(f"cannot load sibling test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALL_EMERGENCY: tuple[str, ...] = _load_sibling_module("test_safety.py").ALL_EMERGENCY
"""The exact phrase tuple ``tests/test_safety.py`` (issue #91) was built and measured
against -- loaded, not copy-pasted, so this file can never silently drift onto a phrase
``app.services.safety`` was never actually proven to catch."""

# --- documents whose text was never ingested (SOURCES.md, issue #95) -------------------
#
# A row may only cite a unit belonging to one of the seven acquired documents. Citing any
# of these three -- or a bare "cited-*" stub, which by definition has no text either
# (issue #67) -- would mean an "answerable" row secretly depends on text nobody has.
_UNACQUIRED_DOCUMENT_IDS = frozenset(
    {"lpg-store-tcvn-6223-2017", "lpg-qcvn-08-2019", "lpg-qcvn-08-2012"}
)

_ROWS = load_gold_set()


# --- structural checks: no database needed ----------------------------------------------


def test_the_real_gold_file_has_at_least_30_rows():
    assert len(_ROWS) >= 30


def test_the_real_gold_file_has_at_most_50_rows():
    # Not itself an acceptance criterion, but a file that silently grew past
    # the issue's own 30-50 target is worth noticing in review.
    assert len(_ROWS) <= 50


def test_at_least_five_rows_are_marked_multi_hop():
    assert sum(row.multi_hop for row in _ROWS) >= 5


def test_at_least_three_rows_are_marked_expected_refusal():
    assert sum(row.expected_refusal for row in _ROWS) >= 3


def test_every_row_id_is_unique():
    ids = [row.id for row in _ROWS]
    assert len(ids) == len(set(ids))


def test_every_compliance_and_multi_hop_row_has_at_least_one_source_unit_id():
    answerable = (GoldCategory.COMPLIANCE, GoldCategory.MULTI_HOP)
    for row in _ROWS:
        if row.category in answerable:
            assert len(row.source_unit_ids) >= 1, row.id


def test_every_refusal_and_emergency_row_cites_no_source_units():
    unanswerable = (GoldCategory.REFUSAL, GoldCategory.EMERGENCY)
    for row in _ROWS:
        if row.category in unanswerable:
            assert row.source_unit_ids == (), row.id


def test_no_row_cites_tt_48_2011_for_cylinder_inspection():
    # BUILD_PLAN.md's own seed question 6 names TT 48/2011 as a source for
    # cylinder inspection; SOURCES.md's 2026-09-01 verification log records
    # that document as investigated and dropped -- it governs group-2 goods
    # quality management generally, not cylinder inspection. Checked against
    # both the source ids (the id a citation to it would take, "cited-48-2011-
    # tt-bct" -- it is not even a manifest document) and the answer text
    # (in case a wrong number were merely mentioned in prose).
    forbidden = ("48/2011", "48-2011-tt-bct")
    for row in _ROWS:
        haystacks = (*row.source_unit_ids, row.reference_answer, row.question)
        for needle in forbidden:
            for haystack in haystacks:
                assert needle not in haystack, (
                    f"{row.id} references TT 48/2011, the corrected error: {haystack!r}"
                )


def test_no_answerable_row_depends_on_an_unacquired_document():
    answerable = (GoldCategory.COMPLIANCE, GoldCategory.MULTI_HOP)
    for row in _ROWS:
        if row.category not in answerable:
            continue
        for unit_id in row.source_unit_ids:
            document_id = unit_id.split("#", maxsplit=1)[0]
            assert document_id not in _UNACQUIRED_DOCUMENT_IDS, (
                f"{row.id} cites {unit_id!r}, whose document has no ingested text"
            )
            assert not document_id.startswith("cited-"), (
                f"{row.id} cites {unit_id!r}, a text-free citation stub (issue #67)"
            )


def test_cylinder_inspection_row_cites_qcvn_04_2013_instead():
    # The positive half of the TT 48/2011 correction: the real source is on
    # the record, not just the wrong one's absence.
    row = next(row for row in _ROWS if row.id == "compliance-cylinder-inspection")
    assert row.source_unit_ids == ("lpg-cylinder-qcvn-04-2013#dieu-10#khoan-1#diem-a",)
    assert "QCVN 04:2013" in row.reference_answer


def test_every_emergency_row_uses_a_real_phrase_from_the_safety_test_suite():
    # The issue is explicit: reuse a representative sample of #91's own
    # phrasings, never invent new ones. Checked against the exact tuples
    # tests/test_safety.py was built and measured against, so this file
    # cannot silently drift onto a phrase app.services.safety was never
    # actually proven to catch.
    for row in _ROWS:
        if row.category is GoldCategory.EMERGENCY:
            assert row.question in ALL_EMERGENCY, row.id


def test_every_emergency_row_answer_is_byte_identical_to_the_hardcoded_response():
    for row in _ROWS:
        if row.category is GoldCategory.EMERGENCY:
            assert row.reference_answer == EMERGENCY_RESPONSE_VI, row.id


def test_every_emergency_phrase_actually_escalates():
    # Belt and suspenders: not just "is this phrase in the safety test suite"
    # but "does the real detector agree this is an emergency", exercised
    # directly against the shipped app.services.safety.check_emergency.
    from app.services.safety import check_emergency

    for row in _ROWS:
        if row.category is GoldCategory.EMERGENCY:
            assert check_emergency(row.question).escalated, row.id


# --- GoldRow / load_gold_set: the loader's own validation ------------------------------


def test_a_multi_hop_row_with_one_source_unit_is_rejected():
    with pytest.raises(ValueError, match="multi_hop"):
        GoldRow(
            id="x",
            category=GoldCategory.MULTI_HOP,
            question="q",
            reference_answer="a",
            source_unit_ids=("doc#dieu-1",),
            multi_hop=True,
        )


def test_a_refusal_row_with_a_source_unit_is_rejected():
    with pytest.raises(ValueError, match="refusal"):
        GoldRow(
            id="x",
            category=GoldCategory.REFUSAL,
            question="q",
            reference_answer="a",
            source_unit_ids=("doc#dieu-1",),
            expected_refusal=True,
        )


def test_a_compliance_row_with_no_source_unit_is_rejected():
    with pytest.raises(ValueError, match="compliance"):
        GoldRow(id="x", category=GoldCategory.COMPLIANCE, question="q", reference_answer="a")


def test_a_malformed_unit_id_is_rejected():
    with pytest.raises(ValueError, match="does not look like a unit id"):
        GoldRow(
            id="x",
            category=GoldCategory.COMPLIANCE,
            question="q",
            reference_answer="a",
            source_unit_ids=("Not A Valid Id!",),
        )


def test_a_duplicate_source_unit_id_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        GoldRow(
            id="x",
            category=GoldCategory.COMPLIANCE,
            question="q",
            reference_answer="a",
            source_unit_ids=("doc#dieu-1", "doc#dieu-1"),
        )


def test_an_invalid_answer_pattern_is_rejected():
    with pytest.raises(ValueError, match="invalid answer pattern"):
        GoldRow(
            id="x",
            category=GoldCategory.COMPLIANCE,
            question="q",
            reference_answer="a",
            source_unit_ids=("doc#dieu-1",),
            answer_must_match=["("],
        )


def test_a_refusal_row_with_an_answer_pattern_is_rejected():
    with pytest.raises(ValueError, match="refusal row cannot set"):
        GoldRow(
            id="x",
            category=GoldCategory.REFUSAL,
            question="q",
            reference_answer="a",
            expected_refusal=True,
            answer_must_match=["500\\s*kg"],
        )


def test_a_refusal_row_with_an_answer_must_not_match_pattern_is_rejected():
    with pytest.raises(ValueError, match="refusal row cannot set"):
        GoldRow(
            id="x",
            category=GoldCategory.REFUSAL,
            question="q",
            reference_answer="a",
            expected_refusal=True,
            answer_must_not_match=["xăng dầu"],
        )


def test_answer_patterns_default_to_empty():
    row = GoldRow(
        id="x",
        category=GoldCategory.COMPLIANCE,
        question="q",
        reference_answer="a",
        source_unit_ids=("doc#dieu-1",),
    )
    assert row.answer_must_match == ()
    assert row.answer_must_not_match == ()


def test_answer_patterns_are_compiled_case_insensitively():
    row = GoldRow(
        id="x",
        category=GoldCategory.COMPLIANCE,
        question="q",
        reference_answer="a",
        source_unit_ids=("doc#dieu-1",),
        answer_must_match=["TRÊN 500\\s*KG"],
    )
    (pattern,) = row.answer_must_match
    assert pattern.search("Có trên 500 kg khí đốt.") is not None


def test_load_gold_set_rejects_a_duplicate_row_id(tmp_path: Path):
    path = tmp_path / "dup.yaml"
    path.write_text(
        "- id: x\n"
        "  category: refusal\n"
        "  question: q1\n"
        "  reference_answer: a1\n"
        "  expected_refusal: true\n"
        "- id: x\n"
        "  category: refusal\n"
        "  question: q2\n"
        "  reference_answer: a2\n"
        "  expected_refusal: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate gold row id"):
        load_gold_set(path)


def test_load_gold_set_rejects_a_non_list_top_level(tmp_path: Path):
    path = tmp_path / "notalist.yaml"
    path.write_text("rows: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a top-level YAML list"):
        load_gold_set(path)


# --- the two insurance rows' answer_must_match accepts "không thuộc diện/danh mục ...
# phải mua" (issue #197) -------------------------------------------------------------------
#
# Runs the real eval.metrics.answer_check against the rows exactly as load_gold_set parses
# them from the committed data/gold.yaml, so this tests the shipped patterns, not a copy of
# them. The five answer strings are copied verbatim from the issue, which in turn copied
# them from real qwen3:8b samples recorded in gascomply-archive/probes/
# out-pr194-answer-samples.txt and out-pr196-500kg-samples.txt.


def _insurance_row(row_id: str) -> GoldRow:
    return next(row for row in _ROWS if row.id == row_id)


def test_300kg_insurance_row_accepts_khong_phai_mua():
    row = _insurance_row("multihop-gas-store-300kg-insurance")
    answer = (
        "Không, cửa hàng gas chứa 300 kg khí không phải mua bảo hiểm cháy, nổ bắt buộc vì "
        "300 kg nhỏ hơn ngưỡng 500 kg …"
    )
    assert (
        answer_check(AskOutcome.ANSWER, answer, row.answer_must_match, row.answer_must_not_match)
        is True
    )


def test_500kg_insurance_row_accepts_khong_thuoc_dien_phai_mua():
    row = _insurance_row("multihop-gas-store-500kg-insurance")
    answer = (
        "Không. Cửa hàng gas chứa 500 kg khí không thuộc diện phải mua bảo hiểm cháy, nổ bắt "
        "buộc vì ngưỡng quy định là 'trên 500 kg' …"
    )
    assert (
        answer_check(AskOutcome.ANSWER, answer, row.answer_must_match, row.answer_must_not_match)
        is True
    )


def test_500kg_insurance_row_accepts_khong_thuoc_danh_muc_co_so_phai_mua():
    row = _insurance_row("multihop-gas-store-500kg-insurance")
    answer = (
        "Không. Cửa hàng gas chứa 500 kg khí không thuộc danh mục cơ sở phải mua bảo hiểm "
        "cháy, nổ bắt buộc theo Phụ lục VII, mục 15 … 'trên 500 kg' …"
    )
    assert (
        answer_check(AskOutcome.ANSWER, answer, row.answer_must_match, row.answer_must_not_match)
        is True
    )


def test_300kg_insurance_row_still_rejects_the_wrong_co():
    row = _insurance_row("multihop-gas-store-300kg-insurance")
    answer = (
        "Có, cửa hàng gas chứa 300 kg khí phải mua bảo hiểm cháy, nổ bắt buộc vì lượng khí "
        "đốt tồn chứa vượt quá ngưỡng 500 kg …"
    )
    assert (
        answer_check(AskOutcome.ANSWER, answer, row.answer_must_match, row.answer_must_not_match)
        is False
    )


def test_500kg_insurance_row_still_rejects_the_wrong_co():
    row = _insurance_row("multihop-gas-store-500kg-insurance")
    answer = (
        "Có, cửa hàng gas chứa 500 kg khí phải mua bảo hiểm cháy, nổ bắt buộc vì cơ sở kinh "
        "doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg …"
    )
    assert (
        answer_check(AskOutcome.ANSWER, answer, row.answer_must_match, row.answer_must_not_match)
        is False
    )


# --- source_unit_ids resolve to real units (DB-backed) ----------------------------------
#
# A minimal fixture corpus, one document per real document cited anywhere in
# data/gold.yaml, each carrying only the specific articles/clauses/points the
# gold set actually cites -- copied verbatim from the real, reviewed dev
# database (see the PR for the psql output this was read from). Ingested
# through the real pipeline (ingest.run.main, mock embedder) so the resulting
# `units` table carries the same ids the parser would produce against the
# real corpus.

_GAS_DECREE_87_2018_HTML = """<html><body>
<p>Điều 8. Điều kiện đối với thương nhân kinh doanh mua bán khí</p>
<p>1. Điều kiện đối với thương nhân kinh doanh mua bán khí gồm:</p>
<p>a) Là thương nhân được thành lập theo quy định của pháp luật;</p>
<p>b) Có bồn chứa khí đáp ứng các quy định về an toàn hoặc có chai LPG đáp ứng đủ điều kiện
lưu thông trên thị trường hoặc có hợp đồng thuê bồn, thuê chai LPG;</p>
<p>c) Đáp ứng các điều kiện về phòng cháy và chữa cháy theo quy định của pháp luật.</p>
<p>Điều 9. Pha chế khí</p>
<p>2. Thương nhân pha chế khí phải có phòng thử nghiệm chất lượng khí hoặc có hợp đồng thuê
tối thiểu 01 năm với tổ chức đủ năng lực thử nghiệm các chỉ tiêu theo quy định tại quy chuẩn
kỹ thuật quốc gia.</p>
<p>3. Thương nhân thực hiện pha chế khí phải đăng ký cơ sở pha chế với Bộ Khoa học và Công
nghệ theo quy định của pháp luật hiện hành.</p>
<p>Điều 10. Điều kiện đối với cửa hàng bán lẻ LPG chai</p>
<p>2. Có hợp đồng tối thiểu 1 năm bán LPG chai với thương nhân có giấy chứng nhận đủ điều
kiện còn hiệu lực theo quy định tại Nghị định này, trừ trường hợp cửa hàng trực thuộc của
thương nhân.</p>
<p>Điều 43. Trình tự, thủ tục cấp Giấy chứng nhận đủ điều kiện</p>
<p>1. Thương nhân đáp ứng đầy đủ các điều kiện quy định tại Nghị định này gửi 01 bộ hồ sơ đề
nghị cấp Giấy chứng nhận đủ điều kiện trực tiếp hoặc qua đường bưu điện hoặc qua mạng điện tử
đến cơ quan nhà nước có thẩm quyền.</p>
<p>3. Trong thời hạn 15 ngày làm việc kể từ ngày nhận được hồ sơ đầy đủ và hợp lệ, cơ quan
nhà nước có thẩm quyền có trách nhiệm xem xét, thẩm định và cấp Giấy chứng nhận đủ điều kiện
cho thương nhân. Trường hợp từ chối cấp Giấy chứng nhận phải trả lời bằng văn bản và nêu rõ lý
do.</p>
<p>5. Giấy chứng nhận đủ điều kiện có thời hạn hiệu lực là 10 năm kể từ ngày cấp mới.</p>
<p>Điều 53. Điều kiện đối với cửa hàng bán lẻ LPG chai</p>
<p>1. Cửa hàng phải có diện tích tối thiểu là 12 m2.</p>
<p>2. Cửa hàng phải cách ly với các nguồn gây cháy ít nhất 03 m về phía không có tường chịu
lửa; nếu có tường chịu lửa thì không yêu cầu khoảng cách giữa nguồn gây cháy và cửa hàng. Kết
cấu xây dựng, bậc chịu lửa của cửa hàng tối thiểu bậc II. Thiết bị điện của cửa hàng phải là
loại phòng nổ, cách chai LPG tối thiểu 1,5 m.</p>
<p>3. Chỉ được phép trưng bày trên giá quảng cáo những chai không chứa LPG.</p>
</body></html>"""

_GAS_DECREE_17_2020_HTML = """<html><body>
<p>Điều 15. Sửa đổi, bổ sung một số điều của Nghị định số 87/2018/NĐ-CP</p>
<p>ngày 15 tháng 6 năm 2018 của Chính phủ về kinh doanh khí.</p>
<p>2. Điểm b khoản 1 Điều 8 được sửa đổi như sau: b) Có bồn chứa khí hoặc hợp đồng thuê bồn
chứa khí hoặc có chai LPG hoặc hợp đồng thuê chai LPG;</p>
</body></html>"""

_QCVN_04_2013_HTML = """<html><body>
<p>Điều 10. Yêu cầu kiểm định chai LPG</p>
<p>1. Quy định về an toàn đối với việc nạp LPG vào chai</p>
<p>a) Không được nạp LPG vào chai, đồng thời phải thực hiện kiểm định khi phát hiện một
trong các trường hợp sau: - Chai quá thời hạn kiểm định; - Chai chưa được kiểm định hoặc
không thể xác định được thời hạn kiểm định.</p>
</body></html>"""

_QCVN_10_2012_HTML = """<html><body>
<p>Điều 14. Quy định về an toàn điện và phòng cháy</p>
<p>2. Thiết bị điện</p>
<p>b) Thiết bị điện sử dụng trong các vùng nguy hiểm phải là loại phòng nổ và có cấp nhiệt độ
lớn nhất trên bề mặt phù hợp với yêu cầu của từng vị trí lắp đặt, phù hợp với điện áp và tần
số danh định của lưới điện.</p>
</body></html>"""

_PCCC_LAW_2024_HTML = """<html><body>
<p>Điều 41. Nhiệm vụ của lực lượng Cảnh sát phòng cháy, chữa cháy và cứu nạn, cứu hộ</p>
<p>5. Thẩm định thiết kế, kiểm tra công tác nghiệm thu về phòng cháy và chữa cháy theo thẩm
quyền.</p>
<p>Điều 54. Hiệu lực thi hành</p>
<p>1. Luật này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2025, trừ trường hợp quy định tại
khoản 4 Điều này.</p>
<p>2. Luật Phòng cháy và chữa cháy số 27/2001/QH10 đã được sửa đổi, bổ sung một số điều theo
Luật số 40/2013/QH13 và Luật số 30/2023/QH15 hết hiệu lực từ ngày Luật này có hiệu lực thi
hành, trừ trường hợp quy định tại Điều 55 của Luật này.</p>
</body></html>"""

_PCCC_DECREE_105_2025_HTML = """<html><body>
<!--
Phụ lục VI below (issue #160) embeds ``ingest/parser.py``'s own
APPENDIX_SECTION_MARKER ("§") directly in the fixture text, rather than
through ingest.extract's PDF-only _restructure_appendix_6: that function
never runs over HTML, but the parser's anchors read the marker's shape
alone, regardless of which extractor produced it -- so this still proves
the real parser pipeline resolves #muc-ii#khoan-1#diem-b the same way it
would against the real PDF's own restructured text.
-->
<p>Điều 35. Bảo hiểm cháy, nổ bắt buộc</p>
<p>1. Cơ sở thuộc danh mục quy định tại Phụ lục VII kèm theo Nghị định này phải mua bảo hiểm
cháy, nổ bắt buộc đối với tài sản của cơ sở, trừ cơ sở thuộc phạm vi quản lý của Bộ Quốc phòng,
Bộ Công an phục vụ mục đích quân sự, quốc phòng, an ninh, trật tự. Khuyến khích các cơ sở khác
mua bảo hiểm cháy, nổ bắt buộc.</p>
<p>Điều 45. Hiệu lực thi hành</p>
<p>1. Nghị định này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2025.</p>
<p>2. Các Nghị định sau đây hết hiệu lực kể từ ngày Nghị định này có hiệu lực thi hành, trừ
trường hợp quy định tại Điều 46 của Nghị định này:</p>
<p>a) Nghị định số 83/2017/NĐ-CP ngày 18 tháng 7 năm 2017 quy định về công tác cứu nạn, cứu
hộ của lực lượng phòng cháy và chữa cháy;</p>
<p>b) Nghị định số 136/2020/NĐ-CP ngày 24 tháng 11 năm 2020 quy định chi tiết một số điều và
biện pháp thi hành Luật Phòng cháy và chữa cháy và Luật sửa đổi, bổ sung một số điều của Luật
Phòng cháy và chữa cháy;</p>
<p>PHỤ LỤC I</p>
<p>DANH MỤC CƠ SỞ THUỘC DIỆN QUẢN LÝ VỀ PHÒNG CHÁY, CHỮA CHÁY</p>
<p>13. Cơ sở kinh doanh khí đốt; cửa hàng xăng dầu.</p>
<p>PHỤ LỤC VI</p>
<p>MỨC PHÍ BẢO HIỂM VÀ MỨC KHẤU TRỪ BẢO HIỂM CHÁY, NỔ BẮT BUỘC</p>
<p>17. Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg N 0,3</p>
<p>§II. MỨC KHẤU TRỪ BẢO HIỂM</p>
<p>§II.1. Đối với cơ sở có nguy hiểm về cháy, nổ (trừ cơ sở hạt nhân):</p>
<p>§II.1.b) Đối với cơ sở có nguy hiểm về cháy, nổ thuộc loại N quy định tại khoản 1 Mục I
Phụ lục này: mức khấu trừ bảo hiểm tối đa là 10% số tiền bảo hiểm và không thấp hơn mức khấu
trừ bảo hiểm quy định tại điểm c khoản này;</p>
<p>PHỤ LỤC VII</p>
<p>DANH MỤC CƠ SỞ PHẢI MUA BẢO HIỂM CHÁY, NỔ BẮT BUỘC</p>
<p>15. Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg.</p>
</body></html>"""

_FIXTURE_DOCUMENTS: tuple[tuple[str, str, str, str, str, str], ...] = (
    # (document_id, filename, number, doc_type, effective, html)
    ("gas-decree-87-2018", "gas-87.html", "87/2018/NĐ-CP", "decree", "2018-08-01"),
    ("gas-decree-17-2020", "gas-17.html", "17/2020/NĐ-CP", "decree", "2020-03-22"),
    ("lpg-cylinder-qcvn-04-2013", "qcvn-04.html", "QCVN 04:2013/BCT", "qcvn", "2013-01-01"),
    ("lpg-station-qcvn-10-2012", "qcvn-10.html", "QCVN 10:2012/BCT", "qcvn", "2012-01-01"),
    ("pccc-law-2024", "pccc-law.html", "55/2024/QH15", "law", "2025-07-01"),
    ("pccc-decree-105-2025", "pccc-decree.html", "105/2025/NĐ-CP", "decree", "2025-07-01"),
)
_FIXTURE_HTML = {
    "gas-decree-87-2018": _GAS_DECREE_87_2018_HTML,
    "gas-decree-17-2020": _GAS_DECREE_17_2020_HTML,
    "lpg-cylinder-qcvn-04-2013": _QCVN_04_2013_HTML,
    "lpg-station-qcvn-10-2012": _QCVN_10_2012_HTML,
    "pccc-law-2024": _PCCC_LAW_2024_HTML,
    "pccc-decree-105-2025": _PCCC_DECREE_105_2025_HTML,
}

_METADATA_ONLY_DOCUMENT_IDS: tuple[tuple[str, str, str], ...] = (
    # (document_id, number, doc_type) -- no gold row cites a unit of these, but
    # the real, committed data/document_edges.json (loaded by every
    # ingest.run.main call, not swappable per test) names them as one side of
    # a curated document-level edge, so ingest fails loudly unless they too
    # have a `documents` row. Metadata-only (`acquired: false`, no file on
    # disk), the same shape SOURCES.md itself gives lpg-qcvn-08-2012/2019 and
    # test_ingest_persistence.py's own not-acquired fixture uses -- nothing
    # here claims these three are unacquired in the *real* corpus (one of
    # them, pccc-circular-36-2025, has real text there), only that this
    # fixture does not need their text to prove what this test proves.
    ("pccc-circular-36-2025", "36/2025/TT-BCA", "circular"),
    ("lpg-qcvn-08-2012", "QCVN 08:2012/BKHCN", "qcvn"),
    ("lpg-qcvn-08-2019", "QCVN 08:2019/BKHCN", "qcvn"),
)


def _build_fixture_corpus(root: Path) -> Path:
    """Lay out a corpus with one fixture document per real document the gold set cites.

    Also seeds metadata-only rows for every document ``data/document_edges.json``
    names but this fixture has no text for (see ``_METADATA_ONLY_DOCUMENT_IDS``),
    so that file's curated edges resolve during this same ingest run.
    """
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, Any]] = []
    for document_id, filename, number, doc_type, effective in _FIXTURE_DOCUMENTS:
        documents.append(
            {
                "id": document_id,
                "number": number,
                "title_vi": f"Fixture for {document_id} (tests/test_gold.py)",
                "doc_type": doc_type,
                "status": "in_force",
                "effective": effective,
                "path": f"corpus/{filename}",
            }
        )
        (corpus / filename).write_text(_FIXTURE_HTML[document_id], encoding="utf-8")
    for document_id, number, doc_type in _METADATA_ONLY_DOCUMENT_IDS:
        documents.append(
            {
                "id": document_id,
                "number": number,
                "title_vi": f"Metadata-only fixture for {document_id} (tests/test_gold.py)",
                "doc_type": doc_type,
                "status": "in_force",
                "path": f"corpus/{document_id}.pdf",
                "acquired": False,
            }
        )
    (corpus / MANIFEST_FILENAME).write_text(json.dumps({"documents": documents}), encoding="utf-8")
    return corpus


async def _existing_unit_ids(database_url: str, unit_ids: frozenset[str]) -> frozenset[str]:
    """Return the subset of ``unit_ids`` that has a row in ``units``."""
    engine = create_engine(database_url)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT id FROM units WHERE id = ANY(:ids)"), {"ids": list(unit_ids)}
            )
            return frozenset(row[0] for row in result)
    finally:
        await engine.dispose()


def test_every_gold_source_unit_id_resolves_to_a_real_unit(migrated_db: str, tmp_path: Path):
    all_source_unit_ids = frozenset(unit_id for row in _ROWS for unit_id in row.source_unit_ids)
    assert all_source_unit_ids, "the gold set cites no source units at all -- nothing to prove"

    corpus = _build_fixture_corpus(tmp_path)
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
    assert exit_code == 0, "the fixture corpus failed to ingest cleanly"

    found = asyncio.run(_existing_unit_ids(migrated_db, all_source_unit_ids))
    missing = all_source_unit_ids - found
    assert not missing, (
        f"gold.yaml cites {len(missing)} unit id(s) that do not resolve after a real ingest "
        f"of their fixture text -- the parser's id scheme or the fixture has drifted: "
        f"{sorted(missing)}"
    )


def test_a_representative_source_unit_carries_the_expected_real_text(
    migrated_db: str, tmp_path: Path
):
    """Spot-check that the fixture is not vacuous: the cited body says what the row claims."""
    corpus = _build_fixture_corpus(tmp_path)
    assert (
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
        == 0
    )

    async def _body(unit_id: str) -> str:
        engine = create_engine(migrated_db)
        try:
            async with engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT body FROM units WHERE id = :id"), {"id": unit_id}
                )
                return result.scalar_one()
        finally:
            await engine.dispose()

    expectations = {
        "gas-decree-87-2018#dieu-53#khoan-1": "12 m2",
        "gas-decree-87-2018#dieu-53#khoan-3": "không chứa LPG",
        "lpg-cylinder-qcvn-04-2013#dieu-10#khoan-1#diem-a": "quá thời hạn kiểm định",
        "pccc-decree-105-2025#dieu-45#khoan-1": "01 tháng 7 năm 2025",
        "pccc-decree-105-2025#dieu-35#khoan-1": "bảo hiểm cháy, nổ bắt buộc",
        "pccc-decree-105-2025#phu-luc-1#muc-13": "Cơ sở kinh doanh khí đốt",
        "pccc-decree-105-2025#phu-luc-7#muc-15": "trên 500 kg",
        "pccc-decree-105-2025#phu-luc-6#muc-17": "0,3",
        "pccc-decree-105-2025#phu-luc-6#muc-ii#khoan-1#diem-b": "tối đa là 10%",
    }
    for unit_id, expected_substring in expectations.items():
        body = asyncio.run(_body(unit_id))
        assert expected_substring in body, (unit_id, body)
