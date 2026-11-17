"""Build graph edges from a parsed document (build plan A5.2, A6 Phase 2.1/2.2/2.4).

Four families live here.

``build_structural_edges`` (Phase 2.1) builds BELONGS_TO: the one edge type
that needs no human judgment, since a clause belongs to its article because
the parser read it there, not because a regex or an LLM guessed it. That is
what makes it safe to auto-approve, and it gives Phase 2.5's recursive
traversal something real to walk long before any other edge type is trusted.
Direction is CHILD -> PARENT, always:

    clause  #dieu-38#khoan-1  --BELONGS_TO-->  article  #dieu-38
    article #dieu-38          --BELONGS_TO-->  chapter  #chuong-2
    chapter #chuong-2         --BELONGS_TO-->  document gas-decree-87-2018

``build_reference_edges`` (Phase 2.2) resolves ``ingest/references.py``'s
draft ``Reference`` rows into REFERENCES/BASED_ON/GUIDED_BY/AMENDED_BY/
REPLACED_BY edges. Every one of them is *inferred*, never observed like
BELONGS_TO, so every edge here comes out ``review_status=PENDING``,
``source=REGEX`` -- none of it is traversable until Phase 2.3's review CLI
approves it. An external or basis citation of a document the manifest does not
list resolves to a *stub* -- a ``documents`` row with no text and an unknown
status (issue #67) -- but only when it carries a complete document number; see
``_stub_outcome`` for what earns one and what never does.

The same function also turns a document's *repeal statements* into REPLACED_BY
edges (issue #68): the whole documents its text repeals, read by
``ingest/repeals.py`` from a clause together with its child points -- which the
reference regexes cannot do, since they capture the cue ("hết hiệu lực") and not
its target. Those edges run OLD -> NEW, so they start at the repealed document
(a manifest document, or a stub) and end at the document doing the repealing.
Each carries ``origin_id``, the unit whose text produced it, which is what
``EdgeRepository.reconcile_document_edges`` scopes by.

``build_requirement_graph`` (Phase 2.4) turns ``data/requirements.json`` --
a human-curated file, not something derived -- into one Concept entity, one
Requirement entity per row, an APPLIES_TO edge from each requirement to the
concept, and a REFERENCES edge from each requirement to the real unit that
states it. Spans multiple documents (the requirement cluster for one
business concept cites more than one statute), so it runs once, against the
union of every unit ingested this run, rather than once per document like
the two families above.

``build_document_edges`` (issue #63) turns ``data/document_edges.json`` --
also human-curated, same review model as the requirement cluster -- into
document-level GUIDED_BY/AMENDED_BY/REPLACED_BY edges the corpus cannot
produce any other way: `lpg-qcvn-08-2012`'s REPLACED_BY edge above all,
since both QCVN 08 endpoints are ``acquired: false`` (#62) and no text on
disk states the repeal. Validates every endpoint against the document ids
*this run actually persisted* -- not against the manifest, which can
declare more than a ``--limit``-truncated or partially-failed run ever
ingests (issue #63's review). A metadata-only document (#62) still counts,
since it still gets a ``documents`` row regardless of ``acquired``; only a
document this specific run never reached does not.

Getting any family's direction backwards would not fail anything -- it
would silently invert part of the graph, since every later edge type
inherits the same ``src --edge_type--> dst`` reading (A5.2).
"""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.ids import (
    EntityType,
    appendix_id,
    appendix_item_id,
    article_id,
    chapter_id,
    clause_id,
    entity_id,
    point_id,
)
from app.schemas.document import DocType, Document, DocumentStub
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from app.schemas.entity import Entity
from app.schemas.reference import Reference, ReferenceKind
from app.schemas.unit import Unit
from ingest.manifest import Manifest
from ingest.parser import ParseResult, chapter_numeral_to_int
from ingest.references import find_document_number
from ingest.repeals import Repeal, RepealScan, scan_repeals

logger = logging.getLogger(__name__)


def build_structural_edges(result: ParseResult) -> tuple[Edge, ...]:
    """Build one BELONGS_TO edge per unit, from child to parent.

    A unit whose own ``parent_id`` is ``None`` -- a top-level chapter, or an
    article/clause/point with no chapter above it -- belongs directly to the
    document: ``units.parent_id`` only ever references another unit
    (``REFERENCES units(id)``), so a unit whose real parent IS the document
    has nowhere else to record that fact. Building edges only from units with
    a non-``None`` ``parent_id`` would silently detach every such unit from
    its document and fragment the graph into orphan subtrees -- this
    function always emits exactly one edge per unit, falling back to the
    document id when ``parent_id`` is unset.

    Args:
        result: Parsed units to build edges from.

    Returns:
        One BELONGS_TO edge per unit in ``result.units``, in the same order.
    """
    return tuple(_belongs_to_edge(unit, result.document_id) for unit in result.units)


def _belongs_to_edge(unit: Unit, document_id: str) -> Edge:
    """Build one unit's BELONGS_TO edge, pointing at its parent or its document."""
    dst_id = unit.parent_id if unit.parent_id is not None else document_id
    return Edge(
        src_id=unit.id,
        dst_id=dst_id,
        edge_type=EdgeType.BELONGS_TO,
        review_status=ReviewStatus.APPROVED,
        source=EdgeSource.STRUCTURAL,
        origin_id=unit.id,
    )


# --- reference edges (build plan A6 Phase 2.2) -----------------------------------
#
# RESOLVE FROM THE IN-MEMORY Reference OBJECTS ONLY -- never re-read
# references.csv, which stays a human-readable dump, not an input.
#
# RESOLVE AGAINST THE MANIFEST, not against what has been ingested so far:
# the manifest lists every document upfront, so a reference to a document
# later in the run still resolves. An edge whose target later fails to
# ingest is still emitted -- the citation genuinely exists in the text,
# there are no foreign keys to violate, and Phase 2.5's traversal will
# simply find nothing behind it. Honest data beats a silently dropped edge.


@dataclass(frozen=True)
class ReferenceEdges:
    """Result of resolving one document's draft references into graph edges.

    Attributes:
        edges: Edges built from references that resolved to a real unit, a
            manifest document, or -- for an external or basis citation -- a
            stub (issue #67).
        unresolved: Count of references that built no edge. Reported in the
            CLI's stats table so the gap between "citations found" and
            "edges built" stays visible in the same place every other
            silent-loss gap in this pipeline does. Every unresolved
            reference is either an internal miss (``self_cite_miss`` in the
            table) or counted in ``no_stub``.
        stubs: The stub documents ``edges`` point at, distinct, ordered by id
            -- a document cited by a number this manifest does not list. The
            caller persists them beside the edges, so no edge is ever written
            pointing at a node that does not exist.
        no_stub: Unresolved external/basis/amendment references -- every
            unresolved reference that is not an internal miss -- that got no
            stub, for any reason: cited by title only, a TCVN/QCVN standard,
            an amendment reference (never stubbed), or a number that failed
            the strict pattern (``rejected_numbers``). Only a *complete*
            ``N/YYYY/CODE`` number earns a stub -- a junk stub is a junk node
            forever.
        rejected_numbers: The distinct numbers that failed the strict
            pattern, sorted -- named rather than only counted, since each one
            is an extraction defect (a number cut short by a PDF artifact)
            somebody can go and fix. Covers a number named by a repeal
            statement as well as by a citation.
        repeal_edges: How many distinct whole-document repeals this
            document's text states and built into REPLACED_BY edges (issue
            #68), included in ``edges``. An ``amendment`` reference sitting in a
            unit that produced one is resolved, so it is not in ``unresolved``.
        partial_repeals: Repeal acts whose object is not a whole document -- a
            clause, a point, a chapter, a phrase -- counted and never built
            into an edge: repealing part of a decree is not replacing it, and a
            REPLACED_BY there would remove a live decree from every answer.
        unclear_repeals: Statements about a document that fit no recognised
            shape (or whose number failed the strict pattern), counted and
            never built into an edge.
    """

    edges: tuple[Edge, ...]
    unresolved: int
    stubs: tuple[DocumentStub, ...] = ()
    no_stub: int = 0
    rejected_numbers: tuple[str, ...] = ()
    repeal_edges: int = 0
    partial_repeals: int = 0
    unclear_repeals: int = 0


_ARTICLE_HINT = re.compile(
    r"^Điều (?P<dieu>\d+)(?: Khoản (?P<khoan>\d+))?(?: Điểm (?P<diem>[a-zđ]))?$"
)
_CHAPTER_HINT = re.compile(r"^Chương (?P<chuong>[IVXLCDM]+|\d+)$")
_APPENDIX_HINT = re.compile(r"^Phụ lục (?P<phu_luc>[IVXLCDM]+|\d+)(?: Mục (?P<muc>\d+))?$")
"""Match ``ingest/references.py``'s own normalized ``target_hint`` output for
an INTERNAL reference (``_unit_chain_target``/``_appendix_chain_target``) --
never raw source text, so a string that matches none of these patterns
indicates a bug upstream, not a corpus quirk. Treated as unresolved either
way: this module counts what it cannot place rather than raising over it,
the same stance the parser itself takes on anchors it cannot place
(``ParseStats.skipped``). ``_APPENDIX_HINT`` is issue #130's own addition,
checked ahead of ``_ARTICLE_HINT`` in ``_internal_target_id`` since "Phụ
lục" never starts with "Điều" or "Chương" -- the three patterns are mutually
exclusive by construction, so trying them in any order gives the same
result; this order simply matches how the three hint shapes were added."""

_AMENDED_BY_CUES = ("sửa đổi", "bổ sung")
_REPLACED_BY_REVERSED_CUES = ("bãi bỏ",)
_REPLACED_BY_FORWARD_CUES = ("thay thế bởi", "hết hiệu lực")
"""Cue phrases from ``ingest/references.py``'s ``AMENDMENT_PATTERN``, grouped
by what edge they imply. The pattern's own alternation carries no named
groups to tell them apart, so which cue actually matched is read back from
``Reference.raw_text`` here -- see ``_amendment_edge_type``."""


@dataclass(frozen=True)
class _DocumentLookup:
    """Resolves a manifest document by number or, as a fallback, by exact title.

    Bundled into one object, rather than two separate parameters, to keep
    every resolver function that needs it within C10's 5-parameter limit.
    """

    by_number: dict[str, Document]
    by_title: dict[str, Document]


_TRAILING_CITATION_DATE = re.compile(r"\s+ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}$")
"""Vietnamese law routinely cites a statute by title and date instead of a
number ("Luật Tổ chức Chính phủ ngày 18 tháng 02 năm 2025") -- the corpus's
own manifest never records that date, only the bare ``title_vi``. Stripping
this exact trailing shape is the one normalization applied before a title
lookup (build plan A6 issue #9): narrow enough to never touch anything but
this specific citation convention, so a title match either lands on the
statute's real, unmodified title or does not match at all."""


def _strip_citation_date(target_hint: str) -> str:
    """Return ``target_hint`` with a trailing "ngày D tháng M năm YYYY" removed, if any."""
    return _TRAILING_CITATION_DATE.sub("", target_hint)


def build_reference_edges(
    references: Sequence[Reference], manifest: Manifest, result: ParseResult
) -> ReferenceEdges:
    """Resolve one document's draft references into graph edges.

    Every edge built here is inferred, never observed like BELONGS_TO, so
    every one comes out ``review_status=PENDING``, ``source=REGEX`` --
    nothing here is traversable until a human approves it in Phase 2.3.

    Args:
        references: Draft references from ``extract_references(result)``.
        manifest: The full corpus manifest -- external references and the
            BASIS/GUIDED_BY document-type check resolve against every
            document it lists, not just the ones ingested so far.
        result: The parse result ``references`` was extracted from, for its
            own document id and unit ids (internal references resolve
            against these, never against another document's tree).

    Returns:
        The resolved edges, how many references resolved to nothing, and --
        for an external or basis citation of a document the manifest does
        not list, or a whole document the text repeals -- the stubs those
        edges point at (issue #67, see ``_stub_outcome``; issue #68, see
        ``_build_repeal_edges``).
    """
    unit_ids = {unit.id for unit in result.units}
    documents = _DocumentLookup(
        by_number={document.number: document for document in manifest.documents},
        by_title={document.title_vi: document for document in manifest.documents},
    )
    self_document = _self_document(manifest, result.document_id)
    scan = scan_repeals(result)
    repeals = _build_repeal_edges(scan, documents, result.document_id)

    edges: list[Edge] = list(repeals.edges)
    stubs: dict[str, DocumentStub] = {stub.id: stub for stub in repeals.stubs}
    rejected: set[str] = set(repeals.rejected)
    unresolved = 0
    no_stub = 0
    for reference in references:
        if reference.kind is ReferenceKind.AMENDMENT and reference.src_id in scan.statement_units:
            continue
        edge = _resolve(reference, unit_ids, documents, self_document, result.document_id)
        if edge is None and reference.kind is not ReferenceKind.INTERNAL:
            outcome = _stub_outcome(reference, self_document, result.document_id)
            edge = outcome.edge
            no_stub += _record_stub_outcome(outcome, stubs, rejected)
        if edge is None:
            unresolved += 1
        else:
            edges.append(edge)
    return ReferenceEdges(
        edges=_dedup_edges(edges),
        unresolved=unresolved,
        stubs=tuple(sorted(stubs.values(), key=lambda stub: stub.id)),
        no_stub=no_stub,
        rejected_numbers=tuple(sorted(rejected)),
        repeal_edges=len({(edge.src_id, edge.dst_id) for edge in repeals.edges}),
        partial_repeals=scan.partial,
        unclear_repeals=repeals.unclear,
    )


def _record_stub_outcome(
    outcome: "_StubOutcome", stubs: dict[str, DocumentStub], rejected: set[str]
) -> int:
    """Record a stub outcome's stub or rejected number; return 1 when it earned no stub."""
    if outcome.stub is not None:
        stubs[outcome.stub.id] = outcome.stub
        return 0
    if outcome.rejected_number is not None:
        rejected.add(outcome.rejected_number)
    return 1


def _self_document(manifest: Manifest, document_id: str) -> Document:
    """Return the manifest's own entry for the document currently being ingested."""
    for document in manifest.documents:
        if document.id == document_id:
            return document
    msg = f"{document_id} is not in the manifest"
    raise ValueError(msg)


def _resolve(
    reference: Reference,
    unit_ids: set[str],
    documents: _DocumentLookup,
    self_document: Document,
    document_id: str,
) -> Edge | None:
    """Dispatch one reference to its kind-specific resolver."""
    if reference.kind is ReferenceKind.INTERNAL:
        return _resolve_internal(reference, unit_ids, document_id)
    if reference.kind is ReferenceKind.EXTERNAL:
        return _resolve_external(reference, documents)
    if reference.kind is ReferenceKind.BASIS:
        return _resolve_basis(reference, documents, self_document, document_id)
    return _resolve_amendment(reference, documents, document_id)


def _resolve_internal(reference: Reference, unit_ids: set[str], document_id: str) -> Edge | None:
    """Resolve a same-document unit reference; unresolved if the unit does not exist.

    A reference to a unit the parser never placed (a typo in the source, or
    an anomaly the parser skipped) must not become an edge: inventing one
    would hide whichever of the two is actually wrong.
    """
    target_id = _internal_target_id(reference.target_hint, document_id)
    if target_id is None or target_id not in unit_ids:
        return None
    return _reference_edge(reference.src_id, target_id, reference.raw_text)


def _internal_target_id(target_hint: str, document_id: str) -> str | None:
    """Return the unit id ``target_hint`` names, or ``None`` for an unrecognized shape."""
    if chapter_match := _CHAPTER_HINT.match(target_hint):
        chuong_no = chapter_numeral_to_int(chapter_match.group("chuong"))
        return chapter_id(document_id, chuong_no)

    if appendix_match := _APPENDIX_HINT.match(target_hint):
        phu_luc_no = chapter_numeral_to_int(appendix_match.group("phu_luc"))
        muc = appendix_match.group("muc")
        if muc is None:
            return appendix_id(document_id, phu_luc_no)
        return appendix_item_id(document_id, phu_luc_no, int(muc))

    unit_match = _ARTICLE_HINT.match(target_hint)
    if unit_match is None:
        return None
    dieu_no = int(unit_match.group("dieu"))
    khoan = unit_match.group("khoan")
    if khoan is None:
        return article_id(document_id, dieu_no)
    khoan_no = int(khoan)
    diem = unit_match.group("diem")
    if diem is None:
        return clause_id(document_id, dieu_no, khoan_no)
    return point_id(document_id, dieu_no, khoan_no, diem)


def _resolve_external(reference: Reference, documents: _DocumentLookup) -> Edge | None:
    """Resolve a reference to another manifest document, by number or by title.

    When the citation also named a unit inside that document -- an
    unqualified unit chain appended after the document's number or title,
    issue #121 -- resolve straight into that unit with
    ``_internal_target_id``, the same deterministic id builder
    ``_resolve_internal`` uses for this document's own units. No existence
    check against the target document's own unit tree: this pass never
    parses another document (see the module docstring's "RESOLVE AGAINST
    THE MANIFEST" note above), so a unit the citation names wrong still
    becomes an honest edge that traversal will simply find nothing behind
    -- the same stance already taken for a stub document's id. Falls back
    to a document-level edge when there is no unit suffix, or it does not
    parse as one -- a correct document-level edge beats a wrong unit-level
    one.
    """
    target_document, unit_suffix = _target_document_and_suffix(reference.target_hint, documents)
    if target_document is None:
        return None
    if unit_suffix:
        unit_id = _internal_target_id(unit_suffix, target_document.id)
        if unit_id is not None:
            return _reference_edge(reference.src_id, unit_id, reference.raw_text)
    return _reference_edge(reference.src_id, target_document.id, reference.raw_text)


def _resolve_basis(
    reference: Reference,
    documents: _DocumentLookup,
    self_document: Document,
    document_id: str,
) -> Edge | None:
    """Resolve a "Căn cứ ..." citation to BASED_ON, or GUIDED_BY for decree/circular -> law."""
    target_document = _target_document(reference.target_hint, documents)
    if target_document is None:
        return None
    return _basis_edge(reference, target_document, self_document, document_id)


def _basis_edge(
    reference: Reference,
    target: Document | DocumentStub,
    self_document: Document,
    document_id: str,
) -> Edge:
    """Build one basis edge to ``target``, a manifest document or a stub."""
    return Edge(
        src_id=document_id,
        dst_id=target.id,
        edge_type=_basis_edge_type(self_document, target),
        evidence=reference.raw_text,
        review_status=ReviewStatus.PENDING,
        source=EdgeSource.REGEX,
        origin_id=reference.src_id,
    )


def _basis_edge_type(self_document: Document, target_document: Document | DocumentStub) -> EdgeType:
    """Return GUIDED_BY for a decree/circular citing a law as basis, else BASED_ON.

    A stub's type is ``DocType.UNKNOWN`` -- its number's ``QH`` suffix names a
    law *or* a National Assembly resolution -- so a basis edge to a stub is
    always BASED_ON, never a GUIDED_BY that would claim the target is a law.
    """
    subordinate_types = (DocType.DECREE, DocType.CIRCULAR)
    if self_document.doc_type in subordinate_types and target_document.doc_type is DocType.LAW:
        return EdgeType.GUIDED_BY
    return EdgeType.BASED_ON


def _resolve_amendment(
    reference: Reference, documents: _DocumentLookup, document_id: str
) -> Edge | None:
    """Resolve an amendment/repeal cue to AMENDED_BY or REPLACED_BY, direction per the cue.

    A5.2 fixes REPLACED_BY as OLD -> NEW, but Vietnamese states it in two
    voices that point opposite ways:

        "... được thay thế bởi Nghị định 105/2025 ..."  (this doc IS replaced)
            this document is OLD  ->  edge: this_document --REPLACED_BY--> 105/2025

        "... bãi bỏ Nghị định 136/2020 ..."              (this doc DOES the repealing)
            136/2020 is OLD       ->  edge: 136/2020 --REPLACED_BY--> this_document

    Getting this backwards would silently invert the in-force rule: a
    repealed decree would look current and the current one superseded, with
    nothing raising anywhere to catch it.
    """
    target_document = _target_document(reference.target_hint, documents)
    if target_document is None:
        return None
    edge_type, reversed_direction = _amendment_edge_type(reference.raw_text)
    src_id, dst_id = (
        (target_document.id, document_id)
        if reversed_direction
        else (document_id, target_document.id)
    )
    return Edge(
        src_id=src_id,
        dst_id=dst_id,
        edge_type=edge_type,
        evidence=reference.raw_text,
        review_status=ReviewStatus.PENDING,
        source=EdgeSource.REGEX,
        origin_id=reference.src_id,
    )


def _amendment_edge_type(raw_text: str) -> tuple[EdgeType, bool]:
    """Return the edge type for an amendment cue, and whether its direction must flip.

    ``True`` means src/dst must swap from the (this document, target
    document) default -- see ``_resolve_amendment`` for why "bãi bỏ" is the
    one cue that flips.

    Raises:
        ValueError: If ``raw_text`` names none of ``AMENDMENT_PATTERN``'s
            cue phrases -- a closed set by construction, so this is an
            upstream bug, not a corpus quirk.
    """
    if any(cue in raw_text for cue in _AMENDED_BY_CUES):
        return EdgeType.AMENDED_BY, False
    if any(cue in raw_text for cue in _REPLACED_BY_REVERSED_CUES):
        return EdgeType.REPLACED_BY, True
    if any(cue in raw_text for cue in _REPLACED_BY_FORWARD_CUES):
        return EdgeType.REPLACED_BY, False
    msg = f"unrecognized amendment cue: {raw_text!r}"
    raise ValueError(msg)


def _target_document(target_hint: str, documents: _DocumentLookup) -> Document | None:
    """Return the manifest document ``target_hint`` names, by number or, failing that, title.

    A thin wrapper over ``_target_document_and_suffix`` for the two callers
    (``_resolve_basis``, ``_resolve_amendment``) that only ever need the
    document, never a trailing unit suffix -- a basis or amendment
    ``target_hint`` never carries one.
    """
    document, _unit_suffix = _target_document_and_suffix(target_hint, documents)
    return document


def _target_document_and_suffix(
    target_hint: str, documents: _DocumentLookup
) -> tuple[Document | None, str]:
    """Return the manifest document ``target_hint`` names, plus any unit chain after it.

    ``target_hint`` is sometimes a bare document number or title (a plain
    external citation, a basis, an amendment cue) and sometimes a number or
    title followed by an internal unit chain, one space apart (an external
    "quy định tại ..." citation -- issue #9 for the number shape, #121 for
    "của" being optional and for the title shape carrying a suffix too).
    ``find_document_number`` finds the number anywhere in the hint and
    normalizes its whitespace first, so "68/ 2006/QH11" looks up the same
    document as "68/2006/QH11" (issue #67); whatever in ``target_hint``
    comes after it is the unit suffix, or "" when there is none.

    When no number is present at all, Vietnamese law often cites a statute
    by title and date instead (build plan A6 issue #9): the title lookup is
    tried next, against ``target_hint`` with only its trailing citation date
    stripped (``_strip_citation_date``) -- an exact match on the manifest's
    own ``title_vi``, nothing fuzzy, tried longest title first so one that
    is a textual prefix of another can never shadow the more specific
    match. No normalization beyond that in either path: an over-eager match
    would build a wrong edge, which is worse than an unresolved one a human
    can see.

    Returns:
        The document, or ``None`` if nothing in ``target_hint`` resolves;
        paired with the unit-chain text (if any) that followed the number
        or title that resolved it, trimmed, or "" otherwise.
    """
    if number := find_document_number(target_hint):
        document = documents.by_number.get(number)
        if document is None:
            return None, ""
        suffix = target_hint[len(number) :] if target_hint.startswith(number) else ""
        return document, suffix.strip()
    stripped = _strip_citation_date(target_hint)
    for title in sorted(documents.by_title, key=len, reverse=True):
        if stripped == title:
            return documents.by_title[title], ""
        if stripped.startswith(f"{title} "):
            return documents.by_title[title], stripped[len(title) :].strip()
    return None, ""


def _reference_edge(src_id: str, dst_id: str, raw_text: str) -> Edge:
    """Build one REFERENCES edge, pending review, with its matched text as evidence."""
    return Edge(
        src_id=src_id,
        dst_id=dst_id,
        edge_type=EdgeType.REFERENCES,
        evidence=raw_text,
        review_status=ReviewStatus.PENDING,
        source=EdgeSource.REGEX,
        origin_id=src_id,
    )


def _dedup_edges(edges: list[Edge]) -> tuple[Edge, ...]:
    """Collapse edges sharing a conflict key, keeping the first occurrence.

    Postgres' ``ON CONFLICT DO UPDATE`` cannot affect the same row twice
    within one statement, and two different references legitimately can
    resolve to the same ``(src_id, dst_id, edge_type)`` -- e.g. the same
    basis document cited once from the preamble and again from an article's
    own text.
    """
    seen: set[tuple[str, str, EdgeType]] = set()
    deduped: list[Edge] = []
    for edge in edges:
        key = (edge.src_id, edge.dst_id, edge.edge_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return tuple(deduped)


# --- stubs: documents known only from a citation (issue #67) ---------------------
#
# WHAT EARNS A STUB. A citation of a document the manifest does not list gets a
# node only if its number is a *complete* N/YYYY/CODE number. That is stricter
# than "a number was found", on purpose: a number can still arrive cut short at a
# PDF artifact -- "12/2020/NĐ-C P" is captured as "12/2020/NĐ-C" -- and a stub
# built from that is a junk node forever, often a second node for a document that
# has a right one. ``ingest/extract.py`` repairs the artifacts the real corpus
# shows (issue #80: "N Đ-CP", "NĐạCP", "68/ 2006/QH11", ...) before anything is
# captured, which is what keeps "87/2018/N" from shadowing NĐ 87/2018; this rule
# is the backstop for whatever it does not cover. Everything refused is counted
# (``ReferenceEdges.no_stub``) and every refused *number* is named
# (``rejected_numbers``), so the gap shows in the stats output instead of
# silently shrinking the graph.
#
# WHAT NEVER DOES. A citation with no number (by title and date only: "Luật
# thương mại ngày 14 tháng 6 năm 2005") has nothing to derive a stable id from,
# and inventing one from free text is how two nodes are born from one capital
# letter. A TCVN/QCVN standard has a number, but its identity is number *and
# edition* ("TCVN 7762" and "TCVN 7762:2007" are both cited; the manifest holds
# "QCVN 10:2012/BCT" while the text says "QCVN 10") -- the same unsolved
# identity problem as a title. Neither gets a stub here.
#
# WHICH KINDS. External and basis only. An amendment *reference* keeps
# manifest-only resolution: it captures a cue, not a target, so it has no number
# to derive a stub from. The edge that *does* start at a stub -- REPLACED_BY, OLD
# -> NEW, with the repealed document as ``src_id`` -- is built by the repeal
# pass (``_build_repeal_edges``), which reads the target out of the repeal clause
# and its points and goes through the same rule below. That edge is why
# ``reconcile_document_edges`` scopes by ``origin_id`` and not by ``src_id``
# (issue #68): no ingested document owns a stub, so by ``src_id`` a stale one
# would never have been pruned.

_STUB_KINDS = (ReferenceKind.EXTERNAL, ReferenceKind.BASIS)

_COMPLETE_DOCUMENT_NUMBER = re.compile(
    r"^(?:0[1-9]|[1-9]\d{0,3})/(?:19|20)\d{2}/(?:NĐ-CP|QH\d{1,2}|TT-[A-ZĐ]{2,})$"
)
"""A number that is certainly complete: a serial from 1 (zero-padded to two
digits is how 1-9 are printed), a plausible year, and one of the three codes
this corpus cites -- a Government decree (``NĐ-CP``), a National Assembly law or
resolution (``QH15``), or a ministerial circular (``TT-BCA``). A closed list, so
a shape nobody has looked at is rejected and reported, never guessed at. Not
listed on purpose: ``QĐ-TTg`` and ``CT-TTg`` -- extraction captures only capital
letters, so it would hand this pattern ``QĐ-TT``, which looks complete and is
not. Add a code only together with the extraction that captures it whole."""

_STANDARD_PREFIXES = ("TCVN", "QCVN")


@dataclass(frozen=True)
class _StubOutcome:
    """What one unresolved external/basis reference becomes.

    Exactly one of three: an ``edge`` to a ``stub``, a ``rejected_number`` (the
    citation carried a number that failed the strict pattern), or neither (it
    carried no usable number at all, or it is an amendment reference).
    """

    edge: Edge | None = None
    stub: DocumentStub | None = None
    rejected_number: str | None = None


def _stub_outcome(reference: Reference, self_document: Document, document_id: str) -> _StubOutcome:
    """Decide whether an unresolved external/basis reference earns a stub.

    Only called once the manifest lookup has already failed, so a number found
    here names a document the manifest does not list. An amendment reference
    never earns a stub (see WHICH KINDS above) but is still a citation of a
    document that got none, so it lands in ``no_stub`` like any other.
    """
    if reference.kind not in _STUB_KINDS:
        return _StubOutcome()
    number = find_document_number(reference.target_hint)
    if number is None:
        return _StubOutcome()
    outcome = _stub_for_number(number)
    if outcome.stub is None:
        return outcome
    if reference.kind is ReferenceKind.EXTERNAL:
        edge = _reference_edge(reference.src_id, outcome.stub.id, reference.raw_text)
    else:
        edge = _basis_edge(reference, outcome.stub, self_document, document_id)
    return _StubOutcome(edge=edge, stub=outcome.stub)


def _stub_for_number(number: str) -> _StubOutcome:
    """Apply the complete-number rule: a stub, a rejected number, or neither (a standard).

    The one place that decides what earns a stub, shared by a citation
    (``_stub_outcome``) and a repeal statement (``_build_repeal_edges``): a
    repealed document must not be allowed a looser rule than a cited one.
    """
    if number.startswith(_STANDARD_PREFIXES):
        return _StubOutcome()
    if _COMPLETE_DOCUMENT_NUMBER.match(number) is None:
        return _StubOutcome(rejected_number=number)
    return _StubOutcome(stub=DocumentStub(number=number))


@dataclass(frozen=True)
class _RepealEdges:
    """The REPLACED_BY edges one document's repeal statements build."""

    edges: tuple[Edge, ...] = ()
    stubs: tuple[DocumentStub, ...] = ()
    rejected: tuple[str, ...] = ()
    unclear: int = 0


def _build_repeal_edges(
    scan: RepealScan, documents: _DocumentLookup, document_id: str
) -> _RepealEdges:
    """Turn the whole documents a text repeals into REPLACED_BY edges, OLD -> NEW.

    The repealed document is the source and ``document_id`` -- the text that
    states the repeal -- the destination: "136/2020 REPLACED_BY 105/2025", never
    the reverse, which would invert the in-force filter (A5.2). A repealed
    document the manifest lists resolves to it; one it does not gets a stub, but
    only under ``_COMPLETE_DOCUMENT_NUMBER`` -- a number a PDF cut short, or a
    document naming itself, builds no edge and is counted as unclear (and named,
    for the first), never guessed at.
    """
    edges: list[Edge] = []
    stubs: dict[str, DocumentStub] = {}
    rejected: set[str] = set()
    unclear = scan.unclear
    for repeal in scan.repeals:
        repealed = documents.by_number.get(repeal.number)
        if repealed is not None and repealed.id != document_id:
            edges.append(_repeal_edge(repealed.id, document_id, repeal))
            continue
        outcome = _stub_for_number(repeal.number) if repealed is None else _StubOutcome()
        if outcome.stub is None:
            unclear += 1
            if outcome.rejected_number is not None:
                rejected.add(outcome.rejected_number)
            continue
        stubs[outcome.stub.id] = outcome.stub
        edges.append(_repeal_edge(outcome.stub.id, document_id, repeal))
    return _RepealEdges(
        edges=tuple(edges),
        stubs=tuple(stubs.values()),
        rejected=tuple(sorted(rejected)),
        unclear=unclear,
    )


def _repeal_edge(repealed_id: str, repealing_id: str, repeal: Repeal) -> Edge:
    """Build one REPLACED_BY edge, pending review, its statement as evidence."""
    return Edge(
        src_id=repealed_id,
        dst_id=repealing_id,
        edge_type=EdgeType.REPLACED_BY,
        evidence=repeal.evidence,
        review_status=ReviewStatus.PENDING,
        source=EdgeSource.REGEX,
        origin_id=repeal.origin_id,
    )


# --- requirement cluster (build plan A5.6, A6 Phase 2.4) --------------------------
#
# data/requirements.json IS THE REVIEW. A domain expert located every
# source_unit_id in the real corpus and checked it against the unit's own
# body before writing it down -- adding, correcting, or removing an entry in
# that file is the actual act of review, not something this module infers.
# EdgeSource.CURATED records that provenance; it still lands review_status
# PENDING like REGEX/LLM (not auto-approved like STRUCTURAL), because typing
# a fact into a seed file is not the same act as a reviewer confirming it
# through `ingest.review` with the source unit's own text in front of them --
# one uniform gate for every non-structural edge is easier to reason about,
# and safer, than a second, silent exception to "inferred edges wait for
# review". This also protects against the file being wrong: A5.6 itself
# names one requirement (cylinder inspection) whose exact clause needed
# pinning by hand rather than being handed down settled, so nothing here
# should skip the same review every regex-inferred edge gets.


@dataclass(frozen=True)
class RequirementSeed:
    """One row of ``data/requirements.json``, before it becomes graph nodes."""

    slug: str
    label_vi: str
    value: str
    source_unit_id: str


@dataclass(frozen=True)
class ConceptSeed:
    """The one business concept ``data/requirements.json``'s requirements apply to."""

    slug: str
    label_vi: str


@dataclass(frozen=True)
class RequirementCluster:
    """A concept plus every requirement curated against it."""

    concept: ConceptSeed
    requirements: tuple[RequirementSeed, ...]

    def target_document_ids(self) -> set[str]:
        """Return the id of every document a requirement's ``source_unit_id`` names.

        Used to decide whether a given ingest run is even in scope for this
        cluster (``ingest/run.py``): a run over a manifest that includes none
        of these documents at all has nothing to validate the cluster
        against, which is a different situation from a run that includes one
        of them but is somehow missing the specific unit.
        """
        return {requirement.source_unit_id.split("#", 1)[0] for requirement in self.requirements}


def load_requirement_cluster(path: Path) -> RequirementCluster:
    """Read and parse the human-curated requirement seed file.

    Args:
        path: Path to ``data/requirements.json``.

    Returns:
        The parsed concept and requirements, in file order.
    """
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    concept = ConceptSeed(**raw["concept"])
    requirements = tuple(RequirementSeed(**item) for item in raw["requirements"])
    return RequirementCluster(concept=concept, requirements=requirements)


def build_requirement_graph(
    cluster: RequirementCluster, unit_ids: set[str]
) -> tuple[tuple[Entity, ...], tuple[Edge, ...]]:
    """Build the concept and requirement entities, and their edges.

    Args:
        cluster: The curated seed data (``load_requirement_cluster``).
        unit_ids: Every unit id ingested this run, across all documents --
            the requirement cluster spans more than one statute, so this is
            checked against the *union* of units, not one document's own.

    Returns:
        One Concept entity plus one Requirement entity per row, and two
        edges per requirement: REQUIREMENT --APPLIES_TO--> CONCEPT, and
        REQUIREMENT --REFERENCES--> its source unit.

    Raises:
        ValueError: If any requirement's ``source_unit_id`` names no real
            ingested unit. A citation to nothing must fail the ingest, the
            same rule every earlier phase of this pipeline has needed
            (build plan A6 issue #51) -- writing the edge anyway would be a
            requirement that cites a unit that does not exist, silently.
    """
    concept_id = entity_id(EntityType.CONCEPT, cluster.concept.slug)
    entities: list[Entity] = [
        Entity(id=concept_id, entity_type=EntityType.CONCEPT, label_vi=cluster.concept.label_vi)
    ]
    edges: list[Edge] = []
    for requirement in cluster.requirements:
        if requirement.source_unit_id not in unit_ids:
            msg = (
                f"requirement {requirement.slug!r} cites {requirement.source_unit_id!r}, "
                "which is not a real unit in the ingested corpus"
            )
            raise ValueError(msg)
        entities.append(_requirement_entity(requirement))
        edges.append(_applies_to_edge(requirement, concept_id))
        edges.append(_requirement_reference_edge(requirement))
    return tuple(entities), tuple(edges)


def _requirement_entity(requirement: RequirementSeed) -> Entity:
    """Build one Requirement entity, its value carried in ``meta`` (A5.3's own example)."""
    return Entity(
        id=entity_id(EntityType.REQUIREMENT, requirement.slug),
        entity_type=EntityType.REQUIREMENT,
        label_vi=requirement.label_vi,
        meta={"value": requirement.value},
    )


def _applies_to_edge(requirement: RequirementSeed, concept_id: str) -> Edge:
    """Build one REQUIREMENT --APPLIES_TO--> CONCEPT edge."""
    return Edge(
        src_id=entity_id(EntityType.REQUIREMENT, requirement.slug),
        dst_id=concept_id,
        edge_type=EdgeType.APPLIES_TO,
        review_status=ReviewStatus.PENDING,
        source=EdgeSource.CURATED,
    )


def _requirement_reference_edge(requirement: RequirementSeed) -> Edge:
    """Build one REQUIREMENT --REFERENCES--> source unit edge."""
    return Edge(
        src_id=entity_id(EntityType.REQUIREMENT, requirement.slug),
        dst_id=requirement.source_unit_id,
        edge_type=EdgeType.REFERENCES,
        evidence=f"{requirement.label_vi}: {requirement.value}",
        review_status=ReviewStatus.PENDING,
        source=EdgeSource.CURATED,
    )


# --- curated document-level edges (issue #63) --------------------------------------
#
# data/document_edges.json IS THE REVIEW, the same way requirements.json is (see
# above): a human located each src_id/dst_id pair in the manifest and checked its
# direction against EdgeType's own docstring in app/schemas/edge.py, not against
# SOURCES.md's arrows on trust. EdgeSource.CURATED, review_status=PENDING -- the
# same uniform gate as every other non-structural edge; a fact typed into a seed
# file is not the same act as a human approving it through ingest.review.
#
# One of these five, `pccc-decree-105-2025 GUIDED_BY pccc-law-2024`, can also be
# produced by the regex pass when both documents' text is on disk (it already is,
# for this pair). `edges` is UNIQUE on (src_id, dst_id, edge_type), so a curated
# and a regex-inferred edge for the same triple upsert onto the same row, not two
# rows. This module deliberately leaves the *decision* of which `source` wins to
# call order -- see ingest/run.py's `_ingest_document_edges`, which runs this pass
# after the per-document loop specifically so a colliding regex edge's `source` is
# overwritten to CURATED, never the reverse. Curated provenance is a human having
# looked at the corpus and the manifest together and decided this edge belongs;
# a regex match is a pattern firing on one document's own text. The stronger claim
# should be what a reader sees, deterministically, not by accident of which pass
# happened to run last. This makes the guarantee entirely a property of *call
# order*, not of anything in this module: moving `_ingest_document_edges`'s call
# earlier in `ingest/run.py`'s `_amain` moves the outcome with it.


@dataclass(frozen=True)
class DocumentEdgeSeed:
    """One row of ``data/document_edges.json``, before it becomes a graph edge."""

    src_id: str
    dst_id: str
    edge_type: EdgeType
    note: str | None = None


def load_document_edges(path: Path) -> tuple[DocumentEdgeSeed, ...]:
    """Read and parse the human-curated document-edge seed file.

    Args:
        path: Path to ``data/document_edges.json``.

    Returns:
        The parsed edges, in file order.
    """
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        DocumentEdgeSeed(
            src_id=item["src_id"],
            dst_id=item["dst_id"],
            edge_type=EdgeType(item["edge_type"]),
            note=item.get("note"),
        )
        for item in raw["edges"]
    )


def document_edge_target_ids(seeds: Sequence[DocumentEdgeSeed]) -> set[str]:
    """Return every document id a curated edge seed's ``src_id`` or ``dst_id`` names.

    Used by ``ingest/run.py`` to decide whether the current manifest is even
    in scope for these seeds, the same way
    ``RequirementCluster.target_document_ids`` does for the requirement
    cluster (issue #51): a run over an unrelated manifest -- this project's
    own test suite, or a single other statute -- has nothing to validate
    these seeds against.
    """
    ids: set[str] = set()
    for seed in seeds:
        ids.add(seed.src_id)
        ids.add(seed.dst_id)
    return ids


def build_document_edges(
    seeds: Sequence[DocumentEdgeSeed], document_ids: set[str]
) -> tuple[Edge, ...]:
    """Build curated document-level edges, validating every endpoint against ``document_ids``.

    Args:
        seeds: The curated seed data (``load_document_edges``).
        document_ids: Every document id that actually has a ``documents``
            row after *this run* -- the same check ``build_requirement_graph``
            makes against its own ``unit_ids``, for the same reason. The
            full manifest is the wrong set to check here (issue #63's
            review): a manifest can declare a document -- #62's
            ``acquired: false`` included, that part is fine -- that this
            particular run never actually persisted, most often because of
            ``--limit``. Validating against the manifest instead of what
            was actually ingested is exactly what let a truncated run write
            edges whose endpoints had no node anywhere in the database.

    Returns:
        One edge per seed row, ``review_status=PENDING``, ``source=CURATED``.

    Raises:
        ValueError: If any seed's ``src_id`` or ``dst_id`` is not in
            ``document_ids``, listing every offending id. ``edges`` carries
            no foreign key on either endpoint (build plan
            ``migrations/002_graph.sql``), so nothing else would catch a
            dangling one -- issue #61's review is what a dangling endpoint
            does downstream once traversal reaches it.
    """
    offenders = sorted(
        {seed.src_id for seed in seeds if seed.src_id not in document_ids}
        | {seed.dst_id for seed in seeds if seed.dst_id not in document_ids}
    )
    if offenders:
        msg = (
            f"document edge seed cites {len(offenders)} document id(s) not persisted "
            f"this run: {', '.join(offenders)}"
        )
        raise ValueError(msg)
    return tuple(
        Edge(
            src_id=seed.src_id,
            dst_id=seed.dst_id,
            edge_type=seed.edge_type,
            evidence=seed.note,
            review_status=ReviewStatus.PENDING,
            source=EdgeSource.CURATED,
        )
        for seed in seeds
    )
