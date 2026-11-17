"""Tests for reference-edge resolution (build plan A5.2, A6 Phase 2.2).

Every reference here comes from real regex extraction (``extract_references``)
over one synthetic document, not hand-built ``Reference`` objects, so a test
failure means the resolver genuinely disagrees with what the extractor really
produces -- not with a guess at its shape. Every citation-bearing sentence is
kept on one physical line: ``BASIS_PATTERN``/``AMENDMENT_PATTERN`` deliberately
never cross a newline (``ingest/references.py``), so a wrapped line would
silently truncate the match before reaching this module's own concern.

The self document's own number ("200/2018/NĐ-CP") is deliberately never the
same as any cited number -- an early draft of this fixture reused
"87/2018/NĐ-CP" for both and produced a same-document REFERENCES edge that
had nothing to do with what was being tested.
"""

from pathlib import Path
from typing import Any

import pytest

from app.schemas.document import DocStatus, DocType, Document
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from app.schemas.reference import Reference, ReferenceKind
from ingest.graph import ReferenceEdges, build_reference_edges
from ingest.manifest import Manifest
from ingest.parser import ParseResult, parse_document
from ingest.references import extract_references

DOC_ID = "gas-decree-87-2018"

TEXT = """CHÍNH PHỦ

NGHỊ ĐỊNH
Về kinh doanh khí

Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015;
Căn cứ Luật Phòng cháy và chữa cháy số 27/2001/QH10;
Theo đề nghị của Bộ trưởng Bộ Công Thương;

Chương I
QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Nghị định này quy định về điều kiện kinh doanh khí dầu mỏ hóa lỏng tại Việt Nam.
Căn cứ Luật Phòng cháy và chữa cháy số 27/2001/QH10.

Chương II
ĐIỀU KIỆN ĐỐI VỚI CỬA HÀNG BÁN LẺ LPG CHAI

Điều 38. Điều kiện đối với cửa hàng bán lẻ LPG chai
1. Cửa hàng bán lẻ LPG chai phải đáp ứng các điều kiện sau đây theo quy định.
a) Diện tích tối thiểu của khu vực bán hàng là 12 m2.

Điều 39. Xử lý văn bản liên quan
1. Điều 5 của Nghị định số 87/2018/NĐ-CP được sửa đổi bởi Nghị định số 17/2020/NĐ-CP.
2. Nghị định này được thay thế bởi Nghị định số 105/2025/NĐ-CP.
3. Nghị định này bãi bỏ Nghị định số 136/2020/NĐ-CP.

Điều 40. Dẫn chiếu
1. Việc bảo quản chai LPG thực hiện theo quy định tại điểm a khoản 1 Điều 38 của Nghị định này.
2. Tổ chức thực hiện theo quy định tại Chương II của Nghị định này.
3. Hồ sơ đề nghị cấp phép thực hiện theo quy định tại Điều 99.
4. Diện tích khu vực bán hàng thực hiện theo quy định tại Điều 5 của Nghị định số 87/2018/NĐ-CP.
5. Việc lắp đặt phải tuân thủ QCVN 04:2013/BCT.
"""
"""Deliberately covers, in one document (empirically checked against the real
extractor, not hand-derived):
  - an unresolvable basis (prose, no document number)          -> unresolved
  - the same resolvable basis cited from both the preamble and
    Điều 1's own body (self=decree, target=law)                -> GUIDED_BY, deduped to one edge
  - "được sửa đổi bởi"                                          -> AMENDED_BY, forward
  - "được thay thế bởi"                                         -> REPLACED_BY, forward
  - "bãi bỏ"                                                    -> REPLACED_BY, REVERSED (the trap)
  - an internal point reference (điểm a khoản 1 Điều 38)        -> resolves
  - an internal chapter reference in Roman numerals (Chương II) -> resolves
  - an internal reference to an article that does not exist     -> unresolved
  - an external reference with an internal unit chain appended
    ("... Điều 5 của Nghị định số 87/2018/NĐ-CP")               -> resolves to the document only
  - a bare external standard number (QCVN)                      -> resolves
"""

SELF_NUMBER = "200/2018/NĐ-CP"
LAW_NUMBER = "27/2001/QH10"
OTHER_DECREE_NUMBER = "87/2018/NĐ-CP"
AMENDING_NUMBER = "17/2020/NĐ-CP"
REPLACING_NUMBER = "105/2025/NĐ-CP"
REPEALED_NUMBER = "136/2020/NĐ-CP"
QCVN_NUMBER = "QCVN 04:2013/BCT"


def _document(doc_id: str, number: str, doc_type: DocType, **overrides: Any) -> Document:
    fields: dict[str, Any] = {
        "id": doc_id,
        "number": number,
        "title_vi": f"Văn bản {number}",
        "doc_type": doc_type,
        "status": DocStatus.IN_FORCE,
        "path": f"corpus/{doc_id}.html",
    }
    fields.update(overrides)
    return Document(**fields)


SELF_DOCUMENT = _document(DOC_ID, SELF_NUMBER, DocType.DECREE)
LAW_DOCUMENT = _document("law-fire-prevention-2001", LAW_NUMBER, DocType.LAW)
OTHER_DECREE = _document("other-decree-87-2018", OTHER_DECREE_NUMBER, DocType.DECREE)
AMENDING_DECREE = _document("amending-decree-17-2020", AMENDING_NUMBER, DocType.DECREE)
REPLACING_DECREE = _document("replacing-decree-105-2025", REPLACING_NUMBER, DocType.DECREE)
REPEALED_DECREE = _document("repealed-decree-136-2020", REPEALED_NUMBER, DocType.DECREE)
QCVN_DOCUMENT = _document("qcvn-04-2013", QCVN_NUMBER, DocType.QCVN)

MANIFEST = Manifest(
    root=Path("."),
    documents=(
        SELF_DOCUMENT,
        LAW_DOCUMENT,
        OTHER_DECREE,
        AMENDING_DECREE,
        REPLACING_DECREE,
        REPEALED_DECREE,
        QCVN_DOCUMENT,
    ),
)


@pytest.fixture(scope="module")
def parsed() -> ParseResult:
    return parse_document(TEXT, DOC_ID)


@pytest.fixture(scope="module")
def references(parsed: ParseResult) -> tuple[Reference, ...]:
    return extract_references(parsed)


@pytest.fixture(scope="module")
def resolution(references: tuple[Reference, ...], parsed: ParseResult) -> ReferenceEdges:
    return build_reference_edges(references, MANIFEST, parsed)


def _by_dst(edges: tuple[Edge, ...], dst_id: str) -> Edge:
    return next(edge for edge in edges if edge.dst_id == dst_id)


def _by_src_and_dst(edges: tuple[Edge, ...], src_id: str, dst_id: str) -> Edge:
    return next(edge for edge in edges if edge.src_id == src_id and edge.dst_id == dst_id)


# --- every edge is pending, regex-sourced, evidenced ------------------------------


def test_every_edge_is_pending_and_regex_sourced(resolution: ReferenceEdges):
    assert resolution.edges
    for edge in resolution.edges:
        assert edge.review_status is ReviewStatus.PENDING
        assert edge.source is EdgeSource.REGEX
        assert edge.evidence


# --- internal references ----------------------------------------------------------


def test_an_internal_reference_to_a_point_resolves(resolution: ReferenceEdges):
    point_id = f"{DOC_ID}#dieu-38#khoan-1#diem-a"
    edge = _by_src_and_dst(resolution.edges, f"{DOC_ID}#dieu-40#khoan-1", point_id)
    assert edge.edge_type is EdgeType.REFERENCES


def test_an_internal_chapter_reference_in_roman_numerals_resolves(resolution: ReferenceEdges):
    chapter_id = f"{DOC_ID}#chuong-2"
    edge = _by_src_and_dst(resolution.edges, f"{DOC_ID}#dieu-40#khoan-2", chapter_id)
    assert edge.edge_type is EdgeType.REFERENCES


def test_an_internal_reference_to_a_nonexistent_article_is_unresolved(
    resolution: ReferenceEdges,
):
    # Điều 99 does not exist in this document -- inventing an edge here would
    # hide whichever of the source or the parser is actually wrong.
    assert not any(edge.src_id == f"{DOC_ID}#dieu-40#khoan-3" for edge in resolution.edges)


# --- external references ----------------------------------------------------------


def test_a_bare_external_reference_resolves_to_the_manifest_document(
    resolution: ReferenceEdges,
):
    edge = _by_src_and_dst(resolution.edges, f"{DOC_ID}#dieu-39#khoan-1", OTHER_DECREE.id)
    assert edge.edge_type is EdgeType.REFERENCES


def test_an_external_reference_with_an_internal_unit_suffix_resolves_into_that_unit(
    resolution: ReferenceEdges,
):
    # target_hint is "87/2018/NĐ-CP Điều 5" -- issue #121: the unit chain is
    # now resolved into the target document's own deterministic unit id,
    # not left as a document-level edge. No existence check against
    # OTHER_DECREE's own units (this phase never parses another document's
    # tree), so the id is built the same way ``_resolve_internal`` builds
    # one for this document's own units.
    edge = _by_src_and_dst(
        resolution.edges, f"{DOC_ID}#dieu-40#khoan-4", f"{OTHER_DECREE.id}#dieu-5"
    )
    assert edge.edge_type is EdgeType.REFERENCES


def test_a_bare_standard_number_resolves(resolution: ReferenceEdges):
    edge = _by_src_and_dst(resolution.edges, f"{DOC_ID}#dieu-40#khoan-5", QCVN_DOCUMENT.id)
    assert edge.edge_type is EdgeType.REFERENCES


# --- a unit reference cited by title, not by number (issue #71) ------------------


def test_a_title_only_unit_reference_resolves_end_to_end_to_its_own_unit():
    """Extraction plus resolution together (issues #71, #121).

    The same two-step pipeline ``ingest.run`` drives -- proves the real
    defect issue #71 reports: a citation like "...Điều 2 Luật Phòng cháy,
    chữa cháy và cứu nạn, cứu hộ..." now becomes a REFERENCES edge to the
    law, not a silently-dropped, misclassified internal miss.

    Issue #121 revisits #71's original document-level-only landing: the
    edge now resolves into the law's own "Điều 2 Khoản 10" unit, the same
    deterministic id ``_resolve_internal`` would build for a same-document
    reference, with no existence check against the law's own unit tree
    (this pass never parses another document's units).
    """
    law = _document(
        "law-fire-prevention-2024",
        "55/2024/QH15",
        DocType.LAW,
        title_vi="Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ",
    )
    manifest = Manifest(root=Path("."), documents=(SELF_DOCUMENT, law))
    text = (
        "Điều 41. Trách nhiệm huấn luyện\n"
        "Đối tượng quy định tại khoản 10 Điều 2 Luật Phòng cháy, chữa cháy và cứu "
        "nạn, cứu hộ được huấn luyện, bồi dưỡng nghiệp vụ.\n"
    )
    result = parse_document(text, DOC_ID)
    references = extract_references(result, (law.title_vi,))
    resolution = build_reference_edges(references, manifest, result)

    assert resolution.unresolved == 0
    edge = resolution.edges[0]
    assert edge.edge_type is EdgeType.REFERENCES
    assert edge.src_id == result.units[0].id
    assert edge.dst_id == f"{law.id}#dieu-2#khoan-10"


def test_a_title_only_unit_reference_with_no_manifest_match_is_unresolved():
    """The manifest-side mirror of an extraction-level test in ``test_references.py``.

    See its ``test_a_title_naming_no_manifest_document_produces_no_external_reference``:
    a title naming no manifest document was never classified external in
    the first place, so resolution sees an ordinary unresolved internal
    reference here -- not a fabricated edge to nowhere.
    """
    manifest = Manifest(root=Path("."), documents=(SELF_DOCUMENT,))
    other_law_title = "Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ"
    text = "Điều 41. Trách nhiệm\nThực hiện theo quy định tại Điều 5 Luật Không Tồn Tại.\n"
    result = parse_document(text, DOC_ID)
    # A real manifest title is offered, just not the one this text names.
    references = extract_references(result, (other_law_title,))
    resolution = build_reference_edges(references, manifest, result)

    assert resolution.edges == ()
    assert resolution.unresolved == 1


# --- basis: BASED_ON vs GUIDED_BY, and an unresolvable citation -------------------


def test_a_basis_citation_with_no_document_number_is_unresolved(resolution: ReferenceEdges):
    assert not any(
        edge.dst_id == LAW_DOCUMENT.id and edge.evidence and "Tổ chức Chính phủ" in edge.evidence
        for edge in resolution.edges
    )


def test_a_basis_citation_by_title_and_date_resolves_by_exact_title(resolution: ReferenceEdges):
    # "Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015" carries no number and
    # LAW_DOCUMENT's title_vi is "Văn bản 27/2001/QH10" -- no manifest title
    # matches, so this specific citation stays unresolved even with title
    # matching in place (see the previous test). This test proves the title
    # path itself works, with its own manifest carrying a real matching
    # title (build plan A6 issue #9).
    law = _document(
        "law-organization-of-government-2025",
        "76/2025/QH15",
        DocType.LAW,
        title_vi="Luật Tổ chức Chính phủ",
    )
    manifest = Manifest(root=Path("."), documents=(SELF_DOCUMENT, law))
    result = ParseResult(
        document_id=DOC_ID, units=(), stats=parse_document("", DOC_ID).stats, preamble=""
    )
    reference = Reference(
        src_id=DOC_ID,
        kind=ReferenceKind.BASIS,
        raw_text="Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015;",
        target_hint="Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015",
    )
    resolution = build_reference_edges((reference,), manifest, result)
    assert resolution.unresolved == 0
    edge = _by_dst(resolution.edges, law.id)
    assert edge.edge_type is EdgeType.GUIDED_BY
    assert edge.src_id == DOC_ID


def test_a_title_match_requires_the_date_suffix_to_be_stripped_exactly(resolution: ReferenceEdges):
    # A title that only partially matches (a real different statute, or a
    # citation whose date-suffix shape is not the exact one stripped) must
    # never resolve -- an over-eager match would build a wrong edge.
    law = _document(
        "law-organization-of-government-2025",
        "76/2025/QH15",
        DocType.LAW,
        title_vi="Luật Tổ chức Chính phủ và Chính quyền địa phương",
    )
    manifest = Manifest(root=Path("."), documents=(SELF_DOCUMENT, law))
    result = ParseResult(
        document_id=DOC_ID, units=(), stats=parse_document("", DOC_ID).stats, preamble=""
    )
    reference = Reference(
        src_id=DOC_ID,
        kind=ReferenceKind.BASIS,
        raw_text="Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015;",
        target_hint="Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015",
    )
    resolution = build_reference_edges((reference,), manifest, result)
    assert resolution.unresolved == 1
    assert resolution.edges == ()


def test_a_decree_citing_a_law_as_basis_is_guided_by_not_based_on(resolution: ReferenceEdges):
    edge = _by_dst(resolution.edges, LAW_DOCUMENT.id)
    assert edge.edge_type is EdgeType.GUIDED_BY
    assert edge.src_id == DOC_ID


def test_the_same_basis_cited_twice_from_different_units_deduplicates_to_one_edge(
    resolution: ReferenceEdges,
):
    matches = [edge for edge in resolution.edges if edge.dst_id == LAW_DOCUMENT.id]
    assert len(matches) == 1


# --- the direction trap: REPLACED_BY in both voices, pinned with real ids --------


def test_thay_the_boi_means_this_document_is_old_replaced_by_the_named_one(
    resolution: ReferenceEdges,
):
    """Cue "được thay thế bởi X": this document is OLD, X is NEW -- no swap."""
    edge = _by_dst(resolution.edges, REPLACING_DECREE.id)
    assert edge.edge_type is EdgeType.REPLACED_BY
    assert edge.src_id == DOC_ID
    assert edge.dst_id == REPLACING_DECREE.id


def test_bai_bo_means_the_named_document_is_old_replaced_by_this_one(
    resolution: ReferenceEdges,
):
    """Cue "bãi bỏ X": X is OLD, this document is NEW -- src/dst swap from the default.

    This is the trap: the sentence names the OLD document as its object, but
    A5.2 fixes REPLACED_BY as OLD -> NEW, so the OLD document -- not this one
    -- must be src. Getting this backwards would make a repealed decree look
    current and the current one look superseded, with nothing raising
    anywhere to catch it.
    """
    edge = _by_dst(resolution.edges, DOC_ID)
    assert edge.edge_type is EdgeType.REPLACED_BY
    assert edge.src_id == REPEALED_DECREE.id
    assert edge.dst_id == DOC_ID


def test_sua_doi_boi_means_amended_by_forward_never_replaced_by(resolution: ReferenceEdges):
    edge = _by_dst(resolution.edges, AMENDING_DECREE.id)
    assert edge.edge_type is EdgeType.AMENDED_BY
    assert edge.src_id == DOC_ID
    assert edge.dst_id == AMENDING_DECREE.id


# --- unresolved count and the underlying reference count -------------------------


def test_unresolved_count_matches_the_two_unresolvable_citations(
    resolution: ReferenceEdges,
):
    assert resolution.unresolved == 2


def test_edge_count_after_dedup(resolution: ReferenceEdges):
    # 12 references, 2 unresolved, 10 resolved, 2 of which (the duplicated
    # basis) collapse to 1 -- see the module docstring for the full tally.
    assert len(resolution.edges) == 9


# --- resolving against the manifest, not ingestion order --------------------------


def test_a_target_listed_after_self_in_the_manifest_still_resolves():
    # OTHER_DECREE sits after SELF_DOCUMENT in MANIFEST.documents -- resolution
    # must not depend on manifest order or on what has been ingested so far.
    reordered = Manifest(root=Path("."), documents=(OTHER_DECREE, SELF_DOCUMENT))
    result = ParseResult(
        document_id=DOC_ID,
        units=(),
        stats=parse_document("", DOC_ID).stats,
        preamble="",
    )
    reference = Reference(
        src_id=DOC_ID,
        kind=ReferenceKind.EXTERNAL,
        raw_text=OTHER_DECREE_NUMBER,
        target_hint=OTHER_DECREE_NUMBER,
    )
    resolution = build_reference_edges((reference,), reordered, result)
    assert resolution.edges[0].dst_id == OTHER_DECREE.id


# --- an unrecognized amendment cue is a bug, not a corpus quirk ------------------


def test_an_unrecognized_amendment_cue_raises_rather_than_guessing_a_direction():
    result = ParseResult(
        document_id=DOC_ID, units=(), stats=parse_document("", DOC_ID).stats, preamble=""
    )
    reference = Reference(
        src_id=DOC_ID,
        kind=ReferenceKind.AMENDMENT,
        raw_text=f"some new cue bởi {AMENDING_NUMBER}",
        target_hint=AMENDING_NUMBER,
    )
    with pytest.raises(ValueError, match="unrecognized amendment cue"):
        build_reference_edges((reference,), MANIFEST, result)


# --- appendix references, end to end (issue #130) --------------------------------
#
# Each test parses its own small document with ``parse_document`` -- a real
# appendix section included, not a hand-built ``Reference`` -- and resolves
# through the full ``extract_references`` -> ``build_reference_edges``
# pipeline, the same two-step pipeline ``ingest.run`` drives, following the
# precedent set by ``test_a_title_only_unit_reference_resolves_end_to_end_to_its_own_unit``
# above. This is what proves acceptance criterion #2: an edge only ever
# lands on an appendix unit the parser actually placed.

APPENDIX_TEXT = """Điều 1. Phạm vi điều chỉnh
Nghị định này quy định về kinh doanh khí.

Điều 35. Bảo hiểm cháy, nổ bắt buộc
1. Cơ sở thuộc danh mục quy định tại Phụ lục VII kèm theo Nghị định này
phải mua bảo hiểm cháy, nổ bắt buộc.
2. Cơ sở thuộc nhóm nguy hiểm quy định tại Mục 1 Phụ lục VII kèm theo
Nghị định này phải báo cáo định kỳ.
3. Cơ sở không thuộc danh mục quy định tại Phụ lục IX kèm theo Nghị định
này không phải mua bảo hiểm.
4. Hồ sơ đăng ký lập theo Mẫu số 01 tại Phụ lục kèm theo Nghị định này.
5. Việc thẩm định thực hiện theo quy định tại Mục 2 Phụ lục III kèm theo
Nghị định số 87/2018/NĐ-CP.
6. Hóa chất thuộc Danh mục của Phụ lục I kèm theo Nghị định số 999/2099/NĐ-CP
ngày 01 tháng 01 năm 2099 của Chính phủ không được kinh doanh.

PHỤ LỤC I
MẪU ĐĂNG KÝ KINH DOANH

1. Nội dung mẫu đăng ký.

PHỤ LỤC VII
DANH MỤC CƠ SỞ PHẢI MUA BẢO HIỂM CHÁY, NỔ BẮT BUỘC

1. Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg.
2. Cơ sở kinh doanh hóa chất nguy hiểm.
"""


@pytest.fixture(scope="module")
def appendix_parsed() -> ParseResult:
    return parse_document(APPENDIX_TEXT, DOC_ID)


@pytest.fixture(scope="module")
def appendix_references(appendix_parsed: ParseResult) -> tuple[Reference, ...]:
    return extract_references(appendix_parsed)


@pytest.fixture(scope="module")
def appendix_resolution(
    appendix_references: tuple[Reference, ...], appendix_parsed: ParseResult
) -> ReferenceEdges:
    return build_reference_edges(appendix_references, MANIFEST, appendix_parsed)


def test_the_appendix_and_its_item_are_real_parsed_units(appendix_parsed: ParseResult):
    # Sanity check on the fixture itself: if the parser did not place these
    # units, every assertion below would pass for the wrong reason (nothing
    # to dangle a reference against).
    unit_ids = {unit.id for unit in appendix_parsed.units}
    assert f"{DOC_ID}#phu-luc-7" in unit_ids
    assert f"{DOC_ID}#phu-luc-7#muc-1" in unit_ids
    assert f"{DOC_ID}#phu-luc-1" in unit_ids


def test_appendix_internal_self_reference_resolves(appendix_resolution: ReferenceEdges):
    edge = _by_src_and_dst(
        appendix_resolution.edges, f"{DOC_ID}#dieu-35#khoan-1", f"{DOC_ID}#phu-luc-7"
    )
    assert edge.edge_type is EdgeType.REFERENCES


def test_appendix_item_level_internal_reference_resolves(appendix_resolution: ReferenceEdges):
    edge = _by_src_and_dst(
        appendix_resolution.edges, f"{DOC_ID}#dieu-35#khoan-2", f"{DOC_ID}#phu-luc-7#muc-1"
    )
    assert edge.edge_type is EdgeType.REFERENCES


def test_appendix_citation_of_a_nonexistent_appendix_is_unresolved(
    appendix_resolution: ReferenceEdges,
):
    # Phụ lục IX does not exist in this document -- acceptance criterion #2:
    # never invent an edge to a unit the parser never placed.
    assert not any(edge.src_id == f"{DOC_ID}#dieu-35#khoan-3" for edge in appendix_resolution.edges)


def test_appendix_form_citation_resolves_to_the_bare_appendix(
    appendix_resolution: ReferenceEdges,
):
    # "theo Mẫu số 01 tại Phụ lục kèm theo Nghị định này" -- bare, no
    # number, defaults to appendix 1 (the same default
    # ``ingest/parser.py``'s own ``_open_appendix`` uses) and must land on
    # Phụ lục I as a whole, never on "#muc-1" or any form-specific unit.
    edge = _by_src_and_dst(
        appendix_resolution.edges, f"{DOC_ID}#dieu-35#khoan-4", f"{DOC_ID}#phu-luc-1"
    )
    assert edge.edge_type is EdgeType.REFERENCES


def test_appendix_external_in_corpus_item_resolves_into_that_document(
    appendix_resolution: ReferenceEdges,
):
    # "Mục 2 Phụ lục III kèm theo Nghị định số 87/2018/NĐ-CP" -- OTHER_DECREE
    # is a real manifest document; no existence check against its own unit
    # tree (this pass never parses another document), same convention as
    # every other external unit-chain resolution in this file.
    edge = _by_src_and_dst(
        appendix_resolution.edges,
        f"{DOC_ID}#dieu-35#khoan-5",
        f"{OTHER_DECREE.id}#phu-luc-3#muc-2",
    )
    assert edge.edge_type is EdgeType.REFERENCES


def test_appendix_external_outside_the_corpus_never_points_into_this_document(
    appendix_resolution: ReferenceEdges,
):
    # 999/2099/NĐ-CP is not in MANIFEST -- issue #121's lesson, applied to
    # appendices: this must become a stub-document edge (or stay
    # unresolved), never an edge into this document's own Phụ lục I.
    assert not any(
        edge.dst_id in (f"{DOC_ID}#phu-luc-1", f"{DOC_ID}#phu-luc-1#muc-1")
        for edge in appendix_resolution.edges
        if edge.src_id == f"{DOC_ID}#dieu-35#khoan-6"
    )
    stub_edges = [
        edge for edge in appendix_resolution.edges if edge.src_id == f"{DOC_ID}#dieu-35#khoan-6"
    ]
    assert len(stub_edges) == 1
    assert stub_edges[0].dst_id == "cited-999-2099-nd-cp"
    assert any(stub.id == "cited-999-2099-nd-cp" for stub in appendix_resolution.stubs)
