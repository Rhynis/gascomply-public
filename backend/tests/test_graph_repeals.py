"""Tests for the REPLACED_BY edges a document's repeal statements build (issue #68).

Pure: real ``extract_references`` output and the real ``build_reference_edges``
over hand-built units, no database. What is under test here is the graph half --
direction, stubs, the complete-number rule, what the stats count -- while
``tests/test_repeals.py`` covers reading the statements themselves.
"""

from pathlib import Path
from typing import Any

import pytest

from app.core.ids import article_id, clause_id, point_id, stub_document_id
from app.schemas.document import DocStatus, DocType, Document
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from app.schemas.unit import Unit, UnitType
from ingest.extract import _rejoin_split_document_numbers
from ingest.graph import ReferenceEdges, build_reference_edges
from ingest.manifest import Manifest
from ingest.parser import ParseResult, parse_document
from ingest.references import extract_references

DOC = "pccc-decree-105-2025"
DOC_NUMBER = "105/2025/NĐ-CP"

_LEAD_IN = (
    "Các Nghị định sau đây hết hiệu lực kể từ ngày Nghị định này có hiệu lực thi hành, "
    "trừ trường hợp quy định tại Điều 46 của Nghị định này:"
)
_POINT_83 = (
    "Nghị định số 83/2017/NĐ-CP ngày 18 tháng 7 năm 2017 quy định về công tác cứu nạn, "
    "cứu hộ của lực lượng phòng cháy và chữa cháy;"
)
_POINT_136 = (
    "Nghị định số 136/2020/NĐ-CP ngày 24 tháng 11 năm 2020 quy định chi tiết một số điều "
    "và biện pháp thi hành Luật Phòng cháy và chữa cháy;"
)


def _document(doc_id: str, number: str, **overrides: Any) -> Document:
    fields: dict[str, Any] = {
        "id": doc_id,
        "number": number,
        "title_vi": f"Văn bản {number}",
        "doc_type": DocType.DECREE,
        "status": DocStatus.IN_FORCE,
        "path": f"corpus/{doc_id}.pdf",
    }
    fields.update(overrides)
    return Document(**fields)


def _unit(unit_id: str, unit_type: UnitType, body: str, **coordinates: Any) -> Unit:
    # As extract_text hands text on: document numbers already repaired (issue #80).
    return Unit(
        id=unit_id,
        document_id=DOC,
        unit_type=unit_type,
        body=_rejoin_split_document_numbers(body, Path("fixture.pdf")),
        ordinal=len(unit_id),
        **coordinates,
    )


def _units(lead_in: str, *point_bodies: str) -> tuple[Unit, ...]:
    units = [
        _unit(article_id(DOC, 45), UnitType.ARTICLE, "", dieu_no=45),
        _unit(
            clause_id(DOC, 45, 2),
            UnitType.CLAUSE,
            lead_in,
            dieu_no=45,
            khoan_no=2,
            parent_id=article_id(DOC, 45),
        ),
    ]
    for letter, body in zip("abcdefgh", point_bodies, strict=False):
        units.append(
            _unit(
                point_id(DOC, 45, 2, letter),
                UnitType.POINT,
                body,
                dieu_no=45,
                khoan_no=2,
                diem=letter,
                parent_id=clause_id(DOC, 45, 2),
            )
        )
    return tuple(units)


def _resolve(
    units: tuple[Unit, ...], *documents: Document, self_number: str = DOC_NUMBER
) -> ReferenceEdges:
    result = ParseResult(
        document_id=DOC,
        units=units,
        stats=parse_document("", DOC).stats,
        preamble="",
    )
    manifest = Manifest(root=Path("."), documents=(_document(DOC, self_number), *documents))
    return build_reference_edges(extract_references(result), manifest, result)


def _replaced_by(resolution: ReferenceEdges) -> list[Edge]:
    return [edge for edge in resolution.edges if edge.edge_type is EdgeType.REPLACED_BY]


# --- direction, provenance, evidence ---------------------------------------------------


def test_a_repeal_edge_runs_old_to_new_from_the_repealed_stub_to_the_repealing_document():
    resolution = _resolve(_units(_LEAD_IN, _POINT_83, _POINT_136))

    edges = {(edge.src_id, edge.dst_id) for edge in _replaced_by(resolution)}

    # 136/2020 REPLACED_BY 105/2025 -- never the reverse, which would invert the in-force
    # filter: the live decree would look superseded and the repealed one current.
    assert edges == {
        (stub_document_id("83/2017/NĐ-CP"), DOC),
        (stub_document_id("136/2020/NĐ-CP"), DOC),
    }
    assert (DOC, stub_document_id("136/2020/NĐ-CP")) not in edges


def test_every_repeal_edge_is_pending_regex_with_its_statement_as_evidence_and_its_source_unit():
    resolution = _resolve(_units(_LEAD_IN, _POINT_83, _POINT_136))

    by_dst = {edge.src_id: edge for edge in _replaced_by(resolution)}
    edge = by_dst[stub_document_id("136/2020/NĐ-CP")]
    assert edge.review_status is ReviewStatus.PENDING
    assert edge.source is EdgeSource.REGEX
    assert edge.origin_id == point_id(DOC, 45, 2, "b")
    assert edge.evidence is not None
    assert edge.evidence.startswith("Các Nghị định sau đây hết hiệu lực")
    assert "Nghị định số 136/2020/NĐ-CP ngày 24 tháng 11 năm 2020" in edge.evidence
    assert all(edge.evidence for edge in _replaced_by(resolution))


def test_the_stub_a_repeal_needs_is_returned_beside_its_edge_so_it_is_persisted_first():
    resolution = _resolve(_units(_LEAD_IN, _POINT_83, _POINT_136))

    stub_ids = {stub.id for stub in resolution.stubs}
    assert {edge.src_id for edge in _replaced_by(resolution)} <= stub_ids
    assert resolution.repeal_edges == 2


def test_a_repealed_document_the_manifest_lists_resolves_to_it_and_makes_no_stub():
    old = _document("gas-decree-136-2020", "136/2020/NĐ-CP")

    resolution = _resolve(_units(_LEAD_IN, _POINT_136), old)

    (edge,) = _replaced_by(resolution)
    assert (edge.src_id, edge.dst_id) == (old.id, DOC)
    assert stub_document_id("136/2020/NĐ-CP") not in {stub.id for stub in resolution.stubs}


def test_a_document_that_names_itself_as_repealed_builds_no_edge_and_is_counted_unclear():
    resolution = _resolve(_units(_LEAD_IN, _POINT_136), self_number="136/2020/NĐ-CP")

    assert _replaced_by(resolution) == []
    assert resolution.unclear_repeals == 1


# --- the complete-number rule is shared with citations, not loosened for repeals --------


@pytest.mark.parametrize(
    ("printed", "rejected"),
    [
        ("78/2021/NĐ-C P", "78/2021/NĐ-C"),  # a split between two plain capitals: a guess
        ("41/2011/TT-B CA", "41/2011/TT-B"),  # an issuer whose own letters are split
        ("12/2015/QĐ-TTg", "12/2015/QĐ-TT"),  # extraction-style truncation at a lower-case g
    ],
)
def test_a_repealed_number_the_pdf_damaged_beyond_repair_gets_no_stub_no_edge_and_is_named(
    printed: str, rejected: str
):
    point = f"Nghị định số {printed} ngày 31 tháng 12 năm 2021 quy định về kinh doanh khí;"

    resolution = _resolve(_units(_LEAD_IN, point))

    assert _replaced_by(resolution) == []
    assert resolution.stubs == ()
    assert resolution.unclear_repeals == 1
    assert resolution.rejected_numbers == (rejected,)
    assert resolution.repeal_edges == 0


def test_a_repealed_standard_builds_no_edge_and_is_not_named_as_a_defect():
    body = "Quy chuẩn này thay thế Quy chuẩn QCVN 10:2012/BCT về an toàn."
    units = (_unit(article_id(DOC, 1), UnitType.ARTICLE, body, dieu_no=1),)

    resolution = _resolve(units)

    assert _replaced_by(resolution) == []
    assert resolution.rejected_numbers == ()


def test_trap_e_a_number_split_by_the_pdf_becomes_the_complete_number_never_a_truncated_one():
    body = (
        "Nghị định này thay thế Nghị định số 113/2017/N Đ-CP ngày 09 tháng 10 năm 2017 của "
        "Chính phủ quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hóa chất."
    )
    units = (_unit(article_id(DOC, 61), UnitType.ARTICLE, body, dieu_no=61),)

    resolution = _resolve(units)

    (edge,) = _replaced_by(resolution)
    assert edge.src_id == stub_document_id("113/2017/NĐ-CP") == "cited-113-2017-nd-cp"
    assert [stub.number for stub in resolution.stubs] == ["113/2017/NĐ-CP"]
    every_id = {stub.id for stub in resolution.stubs} | {
        end for edge in resolution.edges for end in (edge.src_id, edge.dst_id)
    }
    assert not any(identifier.removeprefix("cited-") == "113-2017-n" for identifier in every_id)
    # The citation path reads the same repaired text, so it names no truncated number
    # either (before issue #80 it reported "113/2017/N" for this very sentence).
    assert resolution.rejected_numbers == ()


# --- what a run counts ------------------------------------------------------------------


def test_trap_d_a_document_of_partial_repeals_builds_no_replaced_by_and_counts_each():
    body = "Bãi bỏ khoản 1 và 2 Điều 30."
    units = (
        _unit(article_id(DOC, 11), UnitType.ARTICLE, "", dieu_no=11),
        _unit(
            clause_id(DOC, 11, 4),
            UnitType.CLAUSE,
            body,
            dieu_no=11,
            khoan_no=4,
            parent_id=article_id(DOC, 11),
        ),
        _unit(
            clause_id(DOC, 11, 5),
            UnitType.CLAUSE,
            "Bãi bỏ Điều 7.",
            dieu_no=11,
            khoan_no=5,
            parent_id=article_id(DOC, 11),
        ),
    )

    resolution = _resolve(units)

    assert _replaced_by(resolution) == []
    assert resolution.repeal_edges == 0
    assert resolution.partial_repeals == 2
    assert resolution.stubs == ()


def test_an_amendment_cue_in_a_unit_that_produced_a_repeal_is_resolved_not_unresolved():
    with_points = _resolve(_units(_LEAD_IN, _POINT_83, _POINT_136))
    without_points = _resolve(_units(_LEAD_IN))

    # The cue "hết hiệu lực" captures a predicate, not a target: with nothing beneath
    # the lead-in to read, it is exactly the unresolved amendment it always was...
    assert without_points.no_stub == 1
    assert without_points.repeal_edges == 0
    # ...and once the points are read, the same cue is resolved by the repeal edges.
    assert with_points.no_stub == 0
    assert with_points.repeal_edges == 2
    # Both fixtures also miss the lead-in's own "Điều 46": that internal miss is unrelated
    # and identical, so the only difference in `unresolved` is the one cue.
    assert without_points.unresolved - with_points.unresolved == 1


def test_an_amendment_cue_in_a_unit_that_repeals_nothing_stays_unresolved():
    # Real: pccc-decree-105-2025#dieu-44#khoan-1 -- a partial repeal.
    body = "Sửa đổi, bổ sung, bãi bỏ một số điều của Nghị định số 67/2023/NĐ-CP:"
    units = (
        _unit(article_id(DOC, 44), UnitType.ARTICLE, "", dieu_no=44),
        _unit(
            clause_id(DOC, 44, 1),
            UnitType.CLAUSE,
            body,
            dieu_no=44,
            khoan_no=1,
            parent_id=article_id(DOC, 44),
        ),
    )

    resolution = _resolve(units)

    assert resolution.no_stub == 1  # the cue built no edge and is still counted
    assert resolution.unresolved == 1
    assert resolution.partial_repeals == 1
    assert _replaced_by(resolution) == []


def test_a_repeal_edge_wins_the_dedup_against_the_short_evidence_of_the_older_cue_path():
    # "bãi bỏ Nghị định số N" is also matched by AMENDMENT_PATTERN when N is a manifest
    # document, which builds the same triple with just the cue as evidence. The repeal
    # pass runs first, so the sentence survives the dedup.
    old = _document("gas-decree-10-2012", "10/2012/NĐ-CP")
    body = (
        "Nghị định này bãi bỏ Nghị định số 10/2012/NĐ-CP kể từ ngày Nghị định này có hiệu "
        "lực thi hành."
    )
    units = (_unit(article_id(DOC, 1), UnitType.ARTICLE, body, dieu_no=1),)

    resolution = _resolve(units, old)

    (edge,) = _replaced_by(resolution)
    assert edge.evidence == body
    assert (edge.src_id, edge.dst_id) == (old.id, DOC)


def test_every_edge_this_document_builds_names_the_unit_or_document_that_produced_it():
    units = _units(_LEAD_IN, _POINT_83)
    resolution = _resolve(units)

    unit_ids = {unit.id for unit in units}
    assert resolution.edges
    for edge in resolution.edges:
        assert edge.origin_id is not None
        assert edge.origin_id in unit_ids | {DOC}
