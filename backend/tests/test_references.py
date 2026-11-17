"""Tests for the regex reference extractor.

The fixture is a synthetic excerpt seeding one citation of every kind, in the
phrasing real Vietnamese statutes use (smallest structural unit first, e.g.
"điểm a khoản 1 Điều 38"; a "Căn cứ" line closed by "." rather than ";"), plus
the near-misses a naive regex would wrongly match: ordinary prose using "điều"
as a common noun, a bare year, and a fraction. It is test data, not a citable
source.

``test_realistic_citation_forms`` below reproduces, one probe per case, the
citation forms PR #8's review found broken in the first cut of this extractor
(GitHub PR #8 review comment) -- a fixture written from the issue's spec alone
could not have caught them, since it shared the spec's same wrong assumptions.
"""

import csv
from collections.abc import Iterable
from pathlib import Path

import pytest

from app.schemas.reference import Reference, ReferenceKind
from app.schemas.unit import Unit, UnitType
from ingest.parser import ParseResult, parse_document
from ingest.references import extract_references, write_references_csv

FIXTURE = Path(__file__).parent / "fixtures" / "sample_references.txt"
DOC_ID = "gas-decree-87-2018"


@pytest.fixture
def sample_text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def parsed(sample_text: str) -> ParseResult:
    return parse_document(sample_text, DOC_ID)


@pytest.fixture
def references(parsed: ParseResult) -> tuple[Reference, ...]:
    return extract_references(parsed)


def _by_kind(references: tuple[Reference, ...], kind: ReferenceKind) -> list[Reference]:
    return [reference for reference in references if reference.kind is kind]


def _by_src(references: tuple[Reference, ...], src_id: str) -> list[Reference]:
    return [reference for reference in references if reference.src_id == src_id]


def _unit(unit_id: str, dieu_no: int, body: str = "", heading: str | None = None) -> Unit:
    return Unit(
        id=unit_id,
        document_id=DOC_ID,
        unit_type=UnitType.ARTICLE,
        dieu_no=dieu_no,
        heading=heading,
        body=body,
        ordinal=dieu_no,
    )


def _references_for(text: str, document_titles: Iterable[str] = ()) -> tuple[Reference, ...]:
    """Extract references from ``text`` as the sole body of a one-unit document."""
    result = ParseResult(
        document_id=DOC_ID,
        units=(_unit(f"{DOC_ID}#dieu-1", 1, text),),
        stats=parse_document("", DOC_ID).stats,
        preamble="",
    )
    return extract_references(result, document_titles)


def _references_for_unit(*, heading: str | None = None, body: str = "") -> tuple[Reference, ...]:
    """Extract references from a single article unit with the given heading/body."""
    result = ParseResult(
        document_id=DOC_ID,
        units=(_unit(f"{DOC_ID}#dieu-1", 1, body=body, heading=heading),),
        stats=parse_document("", DOC_ID).stats,
        preamble="",
    )
    return extract_references(result)


# --- every kind -------------------------------------------------------------------


def test_every_kind_is_found(references: tuple[Reference, ...]):
    for kind in ReferenceKind:
        assert _by_kind(references, kind), f"no {kind.value} reference was found"


def test_basis_rows_come_from_the_preamble(references: tuple[Reference, ...]):
    basis = _by_kind(references, ReferenceKind.BASIS)
    assert len(basis) == 2
    assert all(reference.src_id == DOC_ID for reference in basis)


def test_a_basis_citation_without_a_document_number_uses_the_clause_as_its_target(
    references: tuple[Reference, ...],
):
    basis = _by_kind(references, ReferenceKind.BASIS)
    plain = next(r for r in basis if "Tổ chức Chính phủ" in r.raw_text)
    assert plain.target_hint == "Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015"


def test_a_basis_citation_ending_in_a_period_is_one_row_not_two(
    references: tuple[Reference, ...],
):
    basis = _by_kind(references, ReferenceKind.BASIS)
    numbered = next(r for r in basis if "55/2024/QH15" in r.raw_text)
    assert numbered.target_hint == "55/2024/QH15"
    assert numbered.raw_text == "Căn cứ Luật Thương mại số 55/2024/QH15."
    assert not any(
        reference.kind is ReferenceKind.EXTERNAL and reference.target_hint == "55/2024/QH15"
        for reference in references
    )


# --- BASIS: a real citation by name vs. ordinary prose reusing "Căn cứ" ----------
#
# Real corpus samples (build plan A6 issue #9): "Căn cứ" also introduces
# ordinary Vietnamese prose ("based on the situation ...", "based on the
# province's decision ..."), never naming an actual document. Only a match
# whose object is a legal-instrument noun or a bare number is a citation.


@pytest.mark.parametrize(
    "text",
    [
        "Căn cứ tình hình, yêu cầu bảo đảm an toàn phòng cháy, chữa cháy;",
        "Căn cứ quyết định của Ủy ban nhân dân cấp tỉnh quy định tại điểm c khoản này;",
        "Căn cứ khả năng cân đối của ngân sách địa phương;",
        "Căn cứ vào văn bản đề nghị của thương nhân;",
        "Căn cứ vào yêu cầu quản lý, Cục Kỹ thuật an toàn;",
    ],
)
def test_can_cu_followed_by_prose_not_a_document_is_not_a_reference(text: str):
    assert _references_for(text) == ()


@pytest.mark.parametrize(
    "text",
    [
        "Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam;",
        "Căn cứ Nghị quyết số 99/2019/QH14;",
        "Căn cứ vào Nghị định số 87/2018/NĐ-CP;",
    ],
)
def test_can_cu_followed_by_a_legal_instrument_or_number_is_still_a_reference(text: str):
    references = _references_for(text)
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.BASIS


def test_external_document_numbers_are_found(references: tuple[Reference, ...]):
    targets = {reference.target_hint for reference in _by_kind(references, ReferenceKind.EXTERNAL)}
    assert targets == {
        "QCVN 04:2013/BCT",
        "TCVN 6223:2017",
        "87/2018/NĐ-CP",
        "48/2011/TT-BCT",
        "87/2018/NĐ-CP Điều 5",
    }


def test_an_amendment_with_a_following_document_captures_it_as_the_target(
    references: tuple[Reference, ...],
):
    amendment = next(
        r for r in _by_kind(references, ReferenceKind.AMENDMENT) if r.target_hint == "17/2020/NĐ-CP"
    )
    assert amendment.src_id == f"{DOC_ID}#dieu-2#khoan-1"
    assert "\n" in amendment.raw_text, "the reference wraps onto the next line in the fixture"


def test_an_amendment_cue_with_no_following_document_falls_back_to_the_cue_text(
    references: tuple[Reference, ...],
):
    amendment = _by_kind(references, ReferenceKind.AMENDMENT)
    bare = {r.target_hint for r in amendment if r.raw_text == r.target_hint}
    assert bare == {"bãi bỏ", "hết hiệu lực"}


def test_internal_reference_smallest_unit_first_and_line_wrap(references: tuple[Reference, ...]):
    internal = _by_kind(references, ReferenceKind.INTERNAL)
    assert len(internal) == 3
    full = next(r for r in internal if r.src_id == f"{DOC_ID}#dieu-3#khoan-1")
    assert full.target_hint == "Điều 38 Khoản 1 Điểm a"
    assert full.raw_text == "quy định tại điểm a khoản 1 Điều 38\ncủa Nghị định này"


def test_internal_reference_without_khoan_diem_or_suffix_keeps_just_the_article(
    references: tuple[Reference, ...],
):
    article_only = next(
        r
        for r in _by_kind(references, ReferenceKind.INTERNAL)
        if r.src_id == f"{DOC_ID}#dieu-3#khoan-2"
    )
    assert article_only.target_hint == "Điều 40"
    assert article_only.raw_text == "quy định tại Điều 40"


def test_internal_reference_with_khoan_after_dieu_normalizes_the_same_way(
    references: tuple[Reference, ...],
):
    largest_first = next(
        r
        for r in _by_kind(references, ReferenceKind.INTERNAL)
        if r.src_id == f"{DOC_ID}#dieu-3#khoan-4"
    )
    assert largest_first.target_hint == "Điều 42 Khoản 2"


def test_a_cross_document_article_reference_is_external_not_internal(
    references: tuple[Reference, ...],
):
    cross_doc = next(
        r for r in _by_kind(references, ReferenceKind.EXTERNAL) if "Điều 5" in r.target_hint
    )
    assert cross_doc.src_id == f"{DOC_ID}#dieu-3#khoan-3"
    assert cross_doc.target_hint == "87/2018/NĐ-CP Điều 5"
    assert not any(
        reference.kind is ReferenceKind.INTERNAL and reference.src_id == cross_doc.src_id
        for reference in references
    )


# --- external references cited by title, not by number (issue #71) --------------
#
# Real corpus samples (issue #71, surveyed via #70's self_cite_miss column):
# Vietnamese statutes routinely cite each other by exact title alone, with no
# document number and often no "của" either -- pccc-decree-105-2025 cites
# pccc-law-2024's title this way twelve times. ``document_titles`` is the
# manifest's own closed set; matching is exact, never a guess at what a
# legal title looks like.

_PCCC_LAW_TITLE = "Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ"


def test_a_title_cited_with_no_cua_at_all_is_external():
    # The corpus's actual, highest-volume shape (pccc-decree-105-2025 Điều
    # 29 khoản 1, verbatim): the unit chain is followed straight by the
    # target's exact manifest title, no "của" in between at all.
    text = (
        "... người làm việc trên phương tiện giao thông quy định tại khoản 10 Điều 2 "
        "Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ được huấn luyện, bồi dưỡng."
    )
    references = _references_for(text, document_titles=(_PCCC_LAW_TITLE,))
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.EXTERNAL
    # Issue #121: the unit chain is kept, the same way a numbered external
    # citation keeps it, so ``ingest/graph.py`` can resolve into the law's
    # own unit, not just the law document.
    assert references[0].target_hint == f"{_PCCC_LAW_TITLE} Điều 2 Khoản 10"
    assert references[0].raw_text == (
        "quy định tại khoản 10 Điều 2 Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ"
    )


def test_a_title_cited_after_cua_is_also_external():
    # The same citation, "của" form (pccc-decree-105-2025 Điều 28 khoản 5
    # điểm e, verbatim) -- both forms occur in the real corpus and both must
    # resolve to the same kind and the same target_hint.
    text = (
        "... người làm việc trên phương tiện giao thông quy định tại khoản 10 Điều 2 "
        "của Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ được huấn luyện."
    )
    references = _references_for(text, document_titles=(_PCCC_LAW_TITLE,))
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.EXTERNAL
    assert references[0].target_hint == f"{_PCCC_LAW_TITLE} Điều 2 Khoản 10"


def test_a_title_citation_resolves_down_to_its_own_unit_chain():
    # "điểm c khoản 1 Điều 17" of the law is what the sentence actually
    # names (pccc-decree-105-2025 Điều 7 khoản 1, verbatim). Issue #71
    # originally kept this document-level only (no corpus-wide unit index
    # at extraction time); issue #121 revisits that, since the real corpus
    # has exactly this shape resolving wrong otherwise (dev database id
    # 19036) -- the unit chain is now appended to target_hint, same as a
    # numbered external citation, for ``ingest/graph.py`` to resolve.
    text = (
        "Việc thẩm định thực hiện theo quy định tại điểm c khoản 1 Điều 17 "
        "Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ."
    )
    references = _references_for(text, document_titles=(_PCCC_LAW_TITLE,))
    assert len(references) == 1
    assert references[0].target_hint == f"{_PCCC_LAW_TITLE} Điều 17 Khoản 1 Điểm c"


def test_quy_dinh_tai_dieu_46_cua_nghi_dinh_nay_stays_internal_even_with_titles_loaded():
    # The self-reference boundary issue #71 must not disturb: "của Nghị
    # định này" always means *this* document, title-matching or not.
    text = "Việc bảo quản thực hiện theo quy định tại Điều 46 của Nghị định này."
    references = _references_for(text, document_titles=(_PCCC_LAW_TITLE,))
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.INTERNAL
    assert references[0].target_hint == "Điều 46"


def test_a_title_naming_no_manifest_document_produces_an_unresolved_external_reference():
    # A title that resolves to nothing in the manifest is not a citation
    # this module can see by title -- but it still opens with "Luật", a
    # legal-instrument noun not followed by "này", so issue #121 requires
    # it stay external (unresolved downstream) rather than fall back to
    # internal: "a reference to a different instrument must never become
    # an edge into this document" is the whole point of #121, regardless
    # of whether that instrument can be identified.
    text = "Việc thực hiện theo quy định tại Điều 5 Luật Không Tồn Tại Trong Bộ Hồ Sơ."
    references = _references_for(text, document_titles=(_PCCC_LAW_TITLE,))
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.EXTERNAL
    assert references[0].target_hint == "Luật Điều 5"


def test_with_no_document_titles_given_a_noun_only_citation_is_still_external():
    # ``document_titles`` defaults to empty (e.g. a caller with no manifest
    # at hand), so the title itself is invisible to the pattern -- but
    # issue #121's ``ext_noun`` fallback does not need the manifest: "Luật"
    # alone, not followed by "này", is enough to keep this external rather
    # than silently internal (the exact defect #121 exists to close).
    text = (
        "... quy định tại khoản 10 Điều 2 Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ "
        "được huấn luyện."
    )
    references = _references_for(text)
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.EXTERNAL
    assert references[0].target_hint == "Luật Điều 2 Khoản 10"


# --- real dev-database false positives (issue #121) ------------------------------
#
# Five real regex ``references`` edges GitHub issue #121 found approved on the
# dev database into a wrong, same-document edge: a citation of *another*
# document's article that the old suffix pattern -- "của" or an exact,
# space-for-space title, nothing else -- could not see, so it fell through to
# this module's internal default. Reproduced verbatim (ids and source units
# from the issue), one per case, plus the three forms that must keep staying
# internal.

_PCCC_LAW_TITLE_121 = "Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ"


@pytest.mark.parametrize(
    "text",
    [
        # id 1224: pccc-decree-105-2025#dieu-46#khoan-5#diem-a
        "quy định tại Điều 17 Nghị định số 136/2020/NĐ-CP, được sửa đổi",
        # id 1335: pccc-circular-36-2025#dieu-10
        "quy định tại điểm a khoản 2 Điều 13 Nghị định số 105/2025/NĐ-CP theo trình tự",
        # id 1336: pccc-circular-36-2025#dieu-10
        "quy định tại khoản 3 Điều 14 Nghị định số 105/2025/NĐ-CP.",
        # id 19036: pccc-decree-105-2025#phu-luc-3#muc-16#diem-b -- the title
        # here is exactly what PDF extraction produced for this unit (a stray
        # space splits "cứu" into "cứ u").
        "quy định tại khoản 1 Điều 16 Luật Phòng cháy, chữa cháy và cứ u nạn, cứu hộ",
        # id 19040: pccc-decree-105-2025#phu-luc-6#muc-3
        "quy định tại khoản 3 Điều 26 Nghị định số 67/2023/NĐ-CP.",
    ],
)
def test_the_five_dev_database_false_positives_are_external_not_internal(text: str):
    references = _references_for(text, document_titles=(_PCCC_LAW_TITLE_121,))
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.EXTERNAL


def test_the_title_with_a_pdf_split_word_still_resolves_to_the_canonical_title():
    # id 19036 again: the matched text itself carries the PDF artifact, but
    # target_hint carries the clean, manifest-exact title -- what
    # ``ingest/graph.py``'s exact dict lookup needs, not whatever whitespace
    # variant the source text actually had.
    text = "quy định tại khoản 1 Điều 16 Luật Phòng cháy, chữa cháy và cứ u nạn, cứu hộ"
    references = _references_for(text, document_titles=(_PCCC_LAW_TITLE_121,))
    assert len(references) == 1
    assert references[0].target_hint == f"{_PCCC_LAW_TITLE_121} Điều 16 Khoản 1"


@pytest.mark.parametrize(
    "text",
    [
        "quy định tại Điều 5 Nghị định này",
        "quy định tại Điều 5 Luật này",
        "quy định tại Điều 5",
    ],
)
def test_a_self_reference_or_a_plain_unit_chain_stays_internal(text: str):
    references = _references_for(text, document_titles=(_PCCC_LAW_TITLE_121,))
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.INTERNAL


# --- appendix citations (issue #130) ----------------------------------------------
#
# Every sentence below is verbatim from the real corpus (measured with
# ``pdftotext``/the real HTML extractor before writing the pattern, not
# invented from the issue's spec alone), one per acceptance-criteria shape:
# internal (self-reference), external in the corpus, external outside it
# (must stay unresolved downstream, never become an edge into the citing
# document -- issue #121's lesson generalized to appendices), item-level,
# and a form citation.

_DECREE_105_NUMBER = "105/2025/NĐ-CP"


def test_appendix_internal_self_reference():
    # pccc-decree-105-2025#dieu-35#khoan-1, verbatim -- the exact sentence
    # issue #130 was filed over: no edge from here to Phụ lục VII is the
    # whole reason the 500 kg rule never surfaces.
    text = (
        "Cơ sở thuộc danh mục quy định tại Phụ lục VII kèm theo Nghị định này phải "
        "mua bảo hiểm cháy, nổ bắt buộc đối với tài sản của cơ sở."
    )
    references = _references_for(text)
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.INTERNAL
    assert references[0].target_hint == "Phụ lục VII"


def test_appendix_item_level_internal():
    # pccc-decree-105-2025#dieu-33#khoan-1 (point c), verbatim: "mục 16 Phụ
    # lục III", lower-case "mục", still self-referencing this document.
    text = (
        "phương tiện thủy nội địa vận tải hành khách có sức chở người từ 50 người "
        "trở lên quy định tại mục 16 Phụ lục III kèm theo Nghị định này;"
    )
    references = _references_for(text)
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.INTERNAL
    assert references[0].target_hint == "Phụ lục III Mục 16"


def test_appendix_external_in_corpus_item_level():
    # pccc-circular-36-2025#dieu-11, verbatim: cites decree 105/2025's own
    # Phụ lục V Mục 1, a document this corpus actually has.
    text = (
        "cấp giấy phép lưu thông đối với phương tiện phòng cháy, chữa cháy, cứu "
        f"nạn, cứu hộ quy định tại Mục 1 Phụ lục V kèm theo Nghị định số "
        f"{_DECREE_105_NUMBER}."
    )
    references = _references_for(text)
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.EXTERNAL
    assert references[0].target_hint == f"{_DECREE_105_NUMBER} Phụ lục V Mục 1"


def test_appendix_external_outside_corpus_stays_unresolvable_into_this_document():
    # gas-decree-17-2020#dieu-9#khoan-3, verbatim: a repeal clause naming
    # 113/2017/NĐ-CP's own Phụ lục I -- 113/2017 is not in the corpus.
    # target_hint still names that document's appendix, never this one's,
    # which is what keeps ``ingest/graph.py`` from ever resolving it into
    # the citing document (#121's lesson, applied to appendices).
    text = (
        "Bãi bỏ một số hóa chất thuộc Danh mục hóa chất sản xuất, kinh doanh có "
        "điều kiện trong lĩnh vực công nghiệp của Phụ lục I kèm theo Nghị định số "
        "113/2017/NĐ-CP ngày 09 tháng 10 năm 2017 của Chính phủ như sau:"
    )
    references = _references_for(text)
    appendix_refs = [r for r in references if "Phụ lục I" in r.target_hint]
    assert len(appendix_refs) == 1
    assert appendix_refs[0].kind is ReferenceKind.EXTERNAL
    assert appendix_refs[0].target_hint == "113/2017/NĐ-CP Phụ lục I"


def test_appendix_form_citation_resolves_to_the_appendix_not_an_item():
    # gas-decree-87-2018 Điều 37 Khoản 1, verbatim (real extracted body,
    # via ``ingest.extract``): a bare "Phụ lục" with no number -- the
    # document has only one -- and a leading form number ("Mẫu số 01")
    # that must be discarded, never promoted into target_hint (issue
    # #131 is what splits actual "#mau-..." units; here the citation must
    # stay pointed at the appendix as a whole).
    text = (
        "Giấy đề nghị cấp Giấy chứng nhận đủ điều kiện thương nhân xuất khẩu, "
        "nhập khẩu LPG/LNG/CNG theo Mẫu số 01 tại Phụ lục kèm theo Nghị định này."
    )
    references = _references_for(text)
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.INTERNAL
    assert references[0].target_hint == "Phụ lục 1"


def test_appendix_group_filler_is_discarded_and_stays_external():
    # pccc-circular-36-2025#dieu-10, verbatim: "thuộc nhóm 1 Phụ lục II" --
    # the group number is metadata about the facility, not a sub-unit of
    # the appendix, and must not leak into target_hint.
    text = (
        "thực hiện kiểm tra tại các cơ sở thuộc nhóm 1 Phụ lục II kèm theo "
        f"Nghị định số {_DECREE_105_NUMBER};"
    )
    references = _references_for(text)
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.EXTERNAL
    assert references[0].target_hint == f"{_DECREE_105_NUMBER} Phụ lục II"


def test_appendix_definition_heading_is_not_itself_a_reference():
    # pccc-decree-105-2025#dieu-2#khoan-1, verbatim: this is Điều 2 defining
    # what each appendix number means, not a citation -- no cue word
    # ("quy định tại"/"thuộc"/"theo"/"của"/"tại") sits before "Phụ lục I"
    # here, so it must not be captured.
    text = "Phụ lục I: Danh mục cơ sở thuộc diện quản lý về phòng cháy, chữa cháy."
    assert _references_for(text) == ()


# --- "ban hành kèm theo" connector (PR #135 review) -------------------------------
#
# The review on PR #135 found three real edges wrongly pointing into the citing
# document's own appendix: every sentence above uses the bare connector "kèm theo",
# but these four real sentences use the longer "ban hành kèm theo" ("issued and
# promulgated together with") instead, which the first cut of this pattern did not
# recognize at all -- so the suffix past it (the actual cited document, outside the
# corpus in every case) never matched, and the match fell through to the internal
# default. That is #121's exact mistake, applied to appendices, which issue #130
# itself exists to rule out.


def test_appendix_ban_hanh_kem_theo_external_standard_stays_unresolved():
    # pccc-circular-36-2025#dieu-14#khoan-3, verbatim: a standard (149/2020/TT-BCA)
    # outside the corpus, named after "ban hành kèm theo" -- must stay external,
    # never fall back to this document's own #phu-luc-1.
    text = (
        "Tem kiểm định phương tiện phòng cháy và chữa cháy theo Mẫu số 03 quy "
        "định tại Phụ lục ban hành kèm theo Thông tư số 149/2020/TT-BCA và việc "
        "quản lý, sử dụng, in, phát hành tem kiểm định phương tiện phòng cháy và "
        "chữa cháy theo quy định tại khoản 6 Điều 1 Thông tư số 32/2024/TT-BCA "
        "được tiếp tục thực hiện đến hết ngày 30 tháng 6 năm 2026."
    )
    appendix_refs = [r for r in _references_for(text) if "149/2020" in r.target_hint]
    assert len(appendix_refs) == 1
    assert appendix_refs[0].kind is ReferenceKind.EXTERNAL
    assert appendix_refs[0].target_hint == "149/2020/TT-BCA Phụ lục 1"


def test_appendix_ban_hanh_kem_theo_external_decree_stays_unresolved():
    # gas-decree-17-2020#dieu-21#khoan-4, verbatim: 81/2018/NĐ-CP is outside the
    # corpus -- must stay external, never resolve into gas-decree-17-2020's own
    # #phu-luc-1.
    text = (
        'bãi bỏ cụm từ "việc sử dụng danh nghĩa Quốc gia Việt Nam" tại số thứ tự '
        "1 Mẫu số 10 của Phụ lục ban hành kèm theo Nghị định số 81/2018/NĐ-CP "
        "ngày 22 tháng 5 năm 2018 của Chính phủ quy định chi tiết Luật Thương "
        "mại về hoạt động xúc tiến thương mại."
    )
    appendix_refs = [r for r in _references_for(text) if "81/2018" in r.target_hint]
    assert len(appendix_refs) == 1
    assert appendix_refs[0].kind is ReferenceKind.EXTERNAL
    assert appendix_refs[0].target_hint == "81/2018/NĐ-CP Phụ lục 1"


def test_appendix_ban_hanh_kem_theo_two_mentions_one_sentence():
    # gas-decree-17-2020#dieu-1#khoan-9, verbatim: two appendix mentions in one
    # sentence -- the first names 116/2017/NĐ-CP (outside the corpus, must stay
    # external/unresolved), the second is a bare self-reference ("... Nghị định
    # này", internal). The review: "the evidence must be the second mention's
    # text, not 'của Phụ lục'" -- i.e. the two must not collapse into one match.
    text = (
        "Thay thế mẫu số 05 của Phụ lục ban hành kèm theo Nghị định số "
        "116/2017/NĐ-CP bằng mẫu số 15 Mục III tại Phụ lục ban hành kèm theo "
        "Nghị định này."
    )
    references = _references_for(text)
    external = [r for r in references if r.kind is ReferenceKind.EXTERNAL]
    internal = [r for r in references if r.kind is ReferenceKind.INTERNAL]
    assert len(external) == 1
    assert external[0].target_hint == "116/2017/NĐ-CP Phụ lục 1"
    assert len(internal) == 1
    assert internal[0].target_hint == "Phụ lục 1"
    assert internal[0].raw_text == "tại Phụ lục ban hành kèm theo Nghị định này"


def test_appendix_ban_hanh_kem_theo_self_reference_with_form_and_section_stays_internal():
    # gas-decree-17-2020 Điều 12, verbatim ("Đơn đề nghị theo Mẫu số 01a Mục I
    # tại Phụ lục ban hành kèm theo Nghị định này"): a form and section named
    # before the appendix, self-referencing via the longer connector -- must
    # stay internal, same as the plain "kèm theo" form-citation case.
    text = "Đơn đề nghị theo Mẫu số 01a Mục I tại Phụ lục ban hành kèm theo Nghị định này."
    references = _references_for(text)
    assert len(references) == 1
    assert references[0].kind is ReferenceKind.INTERNAL
    assert references[0].target_hint == "Phụ lục 1"


# --- realistic citation forms (PR #8 review probes) -----------------------------------

# Each case reproduces one line of the review's probe table verbatim: the
# first nine are the review's own findings (four were silent MISSes, two came
# out with the wrong kind), the last two are the amended issue's added
# examples (chapter-level, cross-document).
_PROBE_CASES = [
    pytest.param(
        "Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015;",
        ReferenceKind.BASIS,
        "Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015",
        id="basis-terminated-by-semicolon",
    ),
    pytest.param(
        "Căn cứ Luật Doanh nghiệp ngày 17 tháng 6 năm 2020",
        ReferenceKind.BASIS,
        "Luật Doanh nghiệp ngày 17 tháng 6 năm 2020",
        id="basis-with-no-terminator-at-all",
    ),
    pytest.param(
        "Căn cứ Luật Thương mại số 55/2024/QH15.",
        ReferenceKind.BASIS,
        "55/2024/QH15",
        id="basis-terminated-by-period",
    ),
    pytest.param(
        "Việc bảo quản thực hiện theo quy định tại Điều 38 của Nghị định này.",
        ReferenceKind.INTERNAL,
        "Điều 38",
        id="internal-largest-first-full-suffix",
    ),
    pytest.param(
        "Việc bảo quản thực hiện theo quy định tại khoản 1 Điều 38 của Nghị định này.",
        ReferenceKind.INTERNAL,
        "Điều 38 Khoản 1",
        id="internal-khoan-before-dieu",
    ),
    pytest.param(
        "Việc bảo quản thực hiện theo quy định tại điểm a khoản 1 Điều 38.",
        ReferenceKind.INTERNAL,
        "Điều 38 Khoản 1 Điểm a",
        id="internal-smallest-unit-first-no-suffix",
    ),
    pytest.param(
        "Việc bảo quản thực hiện theo quy định tại Điều 38.",
        ReferenceKind.INTERNAL,
        "Điều 38",
        id="internal-bare-article-no-suffix",
    ),
    pytest.param(
        "Điều 5 được sửa đổi, bổ sung bởi Nghị định số 12/2020/NĐ-CP.",
        ReferenceKind.AMENDMENT,
        "12/2020/NĐ-CP",
        id="amendment-sua-doi-bo-sung",
    ),
    pytest.param(
        "Điều 5 được sửa đổi bởi Nghị định số 12/2020/NĐ-CP.",
        ReferenceKind.AMENDMENT,
        "12/2020/NĐ-CP",
        id="amendment-sua-doi-without-bo-sung",
    ),
    pytest.param(
        "Việc cấp phép thực hiện theo quy định tại Chương II.",
        ReferenceKind.INTERNAL,
        "Chương II",
        id="internal-chapter-level",
    ),
    pytest.param(
        "Việc cấp phép thực hiện theo quy định tại Điều 5 của Nghị định số 87/2018/NĐ-CP.",
        ReferenceKind.EXTERNAL,
        "87/2018/NĐ-CP Điều 5",
        id="external-cross-document-article",
    ),
]


@pytest.mark.parametrize(("text", "expected_kind", "expected_target_hint"), _PROBE_CASES)
def test_realistic_citation_forms(
    text: str, expected_kind: ReferenceKind, expected_target_hint: str
):
    references = _references_for(text)
    assert len(references) == 1, f"expected exactly one reference, got {references}"
    assert references[0].kind is expected_kind
    assert references[0].target_hint == expected_target_hint


# --- near-misses --------------------------------------------------------------------


def test_ordinary_prose_using_dieu_as_a_common_noun_is_not_matched(
    references: tuple[Reference, ...],
):
    assert _by_src(references, f"{DOC_ID}#dieu-4#khoan-1") == []


def test_a_bare_year_and_a_fraction_are_not_read_as_document_numbers(
    references: tuple[Reference, ...],
):
    assert _by_src(references, f"{DOC_ID}#dieu-4#khoan-2") == []


# --- structural guarantees -----------------------------------------------------------


def test_every_raw_text_appears_verbatim_in_the_source(
    sample_text: str, references: tuple[Reference, ...]
):
    for reference in references:
        assert reference.raw_text in sample_text


def test_every_src_id_is_a_real_unit_or_the_document(
    parsed: ParseResult, references: tuple[Reference, ...]
):
    valid_ids = {unit.id for unit in parsed.units} | {parsed.document_id}
    for reference in references:
        assert reference.src_id in valid_ids


def test_references_are_returned_in_reading_order(
    parsed: ParseResult, references: tuple[Reference, ...]
):
    ordinal = {unit.id: unit.ordinal for unit in parsed.units}
    ordinal[parsed.document_id] = -1
    positions = [ordinal[reference.src_id] for reference in references]
    assert positions == sorted(positions)


def test_extraction_is_deterministic(parsed: ParseResult):
    assert extract_references(parsed) == extract_references(parsed)


# --- deduplication -------------------------------------------------------------------


def test_a_repeated_reference_in_one_unit_is_deduplicated():
    body = "Xem TCVN 6223:2017. Đối chiếu lại với TCVN 6223:2017 lần nữa."
    references = _references_for(body)
    assert len(references) == 1
    assert references[0].target_hint == "TCVN 6223:2017"


def test_the_same_reference_in_two_units_is_two_rows():
    body = "Xem TCVN 6223:2017."
    result = ParseResult(
        document_id=DOC_ID,
        units=(
            _unit(f"{DOC_ID}#dieu-1", 1, body),
            _unit(f"{DOC_ID}#dieu-2", 2, body),
        ),
        stats=parse_document("", DOC_ID).stats,
        preamble="",
    )
    references = extract_references(result)
    assert len(references) == 2
    assert {reference.src_id for reference in references} == {
        f"{DOC_ID}#dieu-1",
        f"{DOC_ID}#dieu-2",
    }


# --- headings are content too (issue #22) -----------------------------------------


def test_a_citation_inside_a_heading_is_detected():
    # Vietnamese drafting sometimes packs a single-sentence article's entire
    # substance, citations included, into ARTICLE_ANCHOR's heading group,
    # with no separate body line at all -- real QCVN 04:2013/BCT Điều 16.
    references = _references_for_unit(
        heading="Phải tuân thủ quy định tại Nghị định số 107/2009/NĐ-CP."
    )
    assert any(reference.target_hint == "107/2009/NĐ-CP" for reference in references)
    assert all(reference.src_id == f"{DOC_ID}#dieu-1" for reference in references)


def test_the_same_citation_in_both_heading_and_body_is_one_row_not_two():
    references = _references_for_unit(
        heading="Phải tuân thủ quy định tại Nghị định số 107/2009/NĐ-CP.",
        body="Nghị định số 107/2009/NĐ-CP nêu tại tiêu đề được nhắc lại ở đây.",
    )
    matches = [reference for reference in references if reference.target_hint == "107/2009/NĐ-CP"]
    assert len(matches) == 1


def test_a_body_only_citation_is_unaffected_by_scanning_the_heading():
    references = _references_for_unit(
        heading="Điều kiện đối với cửa hàng bán lẻ", body="Xem TCVN 6223:2017."
    )
    assert [reference.target_hint for reference in references] == ["TCVN 6223:2017"]


# --- CSV output ------------------------------------------------------------------------


def test_write_references_csv_round_trips(tmp_path: Path, references: tuple[Reference, ...]):
    csv_path = tmp_path / "references.csv"
    write_references_csv(references, csv_path)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["src_id", "kind", "raw_text", "target_hint"]
    assert len(rows) == len(references) + 1
    first = rows[1]
    assert first == [
        references[0].src_id,
        references[0].kind.value,
        references[0].raw_text,
        references[0].target_hint,
    ]


def test_write_references_csv_handles_no_references(tmp_path: Path):
    csv_path = tmp_path / "empty.csv"
    write_references_csv((), csv_path)
    assert csv_path.read_text(encoding="utf-8").splitlines() == ["src_id,kind,raw_text,target_hint"]
