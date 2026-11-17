"""Tests for structural BELONGS_TO edge building (build plan A5.2, A6 Phase 2.1).

Most cases here build a ``ParseResult`` by hand from ``Unit`` objects, using
``app.core.ids`` to compute every id and parent id -- never hand-formatted,
per the locked A5.1 scheme -- so each test can isolate exactly one tree shape
(a top-level article, a chapter, a full four-level chain) without depending
on the parser. ``sample_article.txt`` (shared with ``test_chunk.py``) is used
once at the end for a realistically parsed document.
"""

from pathlib import Path
from typing import Any

from app.core.ids import article_id, chapter_id, clause_id, point_id
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from app.schemas.unit import Unit, UnitType
from ingest.graph import build_structural_edges
from ingest.parser import ParseResult, ParseStats, parse_document

FIXTURE = Path(__file__).parent / "fixtures" / "sample_article.txt"
DOC_ID = "gas-decree-87-2018"


def _chapter(chapter_no: int = 1, **overrides: Any) -> Unit:
    fields: dict[str, Any] = {
        "id": chapter_id(DOC_ID, chapter_no),
        "document_id": DOC_ID,
        "unit_type": UnitType.CHAPTER,
        "chapter_no": chapter_no,
        "ordinal": 0,
    }
    fields.update(overrides)
    return Unit(**fields)


def _article(dieu_no: int = 1, parent_id: str | None = None, **overrides: Any) -> Unit:
    fields: dict[str, Any] = {
        "id": article_id(DOC_ID, dieu_no),
        "document_id": DOC_ID,
        "unit_type": UnitType.ARTICLE,
        "dieu_no": dieu_no,
        "parent_id": parent_id,
        "ordinal": 1,
    }
    fields.update(overrides)
    return Unit(**fields)


def _clause(dieu_no: int = 1, khoan_no: int = 1, **overrides: Any) -> Unit:
    fields: dict[str, Any] = {
        "id": clause_id(DOC_ID, dieu_no, khoan_no),
        "document_id": DOC_ID,
        "unit_type": UnitType.CLAUSE,
        "dieu_no": dieu_no,
        "khoan_no": khoan_no,
        "parent_id": article_id(DOC_ID, dieu_no),
        "ordinal": 2,
    }
    fields.update(overrides)
    return Unit(**fields)


def _point(dieu_no: int = 1, khoan_no: int = 1, diem: str = "a", **overrides: Any) -> Unit:
    fields: dict[str, Any] = {
        "id": point_id(DOC_ID, dieu_no, khoan_no, diem),
        "document_id": DOC_ID,
        "unit_type": UnitType.POINT,
        "dieu_no": dieu_no,
        "khoan_no": khoan_no,
        "diem": diem,
        "parent_id": clause_id(DOC_ID, dieu_no, khoan_no),
        "ordinal": 3,
    }
    fields.update(overrides)
    return Unit(**fields)


def _result(*units: Unit) -> ParseResult:
    return ParseResult(document_id=DOC_ID, units=units, stats=ParseStats(), preamble="")


def _by_src(edges: tuple[Edge, ...]) -> dict[str, Edge]:
    return {edge.src_id: edge for edge in edges}


# --- shape: one edge per unit, every edge is BELONGS_TO/approved/structural -----


def test_one_edge_per_unit():
    result = _result(_chapter(), _article(dieu_no=1, parent_id=chapter_id(DOC_ID, 1)))
    edges = build_structural_edges(result)
    assert len(edges) == len(result.units)


def test_every_edge_is_belongs_to_approved_structural_with_no_evidence_or_confidence():
    result = _result(_chapter(), _clause())
    for edge in build_structural_edges(result):
        assert edge.edge_type is EdgeType.BELONGS_TO
        assert edge.review_status is ReviewStatus.APPROVED
        assert edge.source is EdgeSource.STRUCTURAL
        assert edge.evidence is None
        assert edge.confidence is None


# --- the direction convention is the point of this phase ------------------------


def test_belongs_to_direction_is_pinned_child_to_parent_not_the_reverse():
    """Explicit, hardcoded assertion of the A5.2 direction convention.

    src is always the CHILD, dst is always the PARENT. Getting this backwards
    would not raise anywhere -- it would silently invert the whole graph,
    since every later edge type inherits this same reading.
    """
    chapter = _chapter(chapter_no=2)
    article = _article(dieu_no=38, parent_id=chapter.id)
    clause = _clause(dieu_no=38, khoan_no=1)
    point = _point(dieu_no=38, khoan_no=1, diem="a")
    edges = _by_src(build_structural_edges(_result(chapter, article, clause, point)))

    assert edges[clause.id].src_id == clause.id
    assert edges[clause.id].dst_id == article.id
    assert edges[article.id].src_id == article.id
    assert edges[article.id].dst_id == chapter.id
    assert edges[point.id].src_id == point.id
    assert edges[point.id].dst_id == clause.id


# --- a unit whose parent is the document (parent_id is None) --------------------


def test_a_chapter_belongs_to_the_document():
    chapter = _chapter()
    edges = _by_src(build_structural_edges(_result(chapter)))
    assert edges[chapter.id].dst_id == DOC_ID


def test_a_top_level_article_with_no_chapter_belongs_to_the_document():
    # parent_id=None here means "no chapter above it", not "no parent at all" --
    # units.parent_id only ever references another unit, so this article's
    # real parent -- the document -- has nowhere else to be recorded.
    article = _article(dieu_no=1, parent_id=None)
    edges = _by_src(build_structural_edges(_result(article)))
    assert edges[article.id].dst_id == DOC_ID


# --- every unit must chain to its document ---------------------------------------


def test_the_full_chain_from_a_point_reaches_the_document():
    chapter = _chapter(chapter_no=2)
    article = _article(dieu_no=38, parent_id=chapter.id)
    clause = _clause(dieu_no=38, khoan_no=1)
    point = _point(dieu_no=38, khoan_no=1, diem="a")
    edges = _by_src(build_structural_edges(_result(chapter, article, clause, point)))

    assert edges[point.id].dst_id == clause.id
    assert edges[clause.id].dst_id == article.id
    assert edges[article.id].dst_id == chapter.id
    assert edges[chapter.id].dst_id == DOC_ID


# --- a realistically parsed document ---------------------------------------------


def test_a_real_parsed_document_produces_one_edge_per_unit():
    result = parse_document(FIXTURE.read_text(encoding="utf-8"), DOC_ID)
    edges = build_structural_edges(result)
    assert len(edges) == len(result.units)
    assert len(result.units) > 0


def test_every_unit_in_a_real_parsed_document_chains_to_the_document():
    result = parse_document(FIXTURE.read_text(encoding="utf-8"), DOC_ID)
    edges = _by_src(build_structural_edges(result))

    for unit in result.units:
        node = unit.id
        hops = {node}
        while node != DOC_ID:
            node = edges[node].dst_id
            assert node not in hops, f"cycle detected walking {unit.id} to the document"
            hops.add(node)
        assert node == DOC_ID
