"""Typed graph edges, mirroring the ``edges`` table (build plan A5.2, A5.3).

An edge is one directed link of the citation graph, always read
``src --edge_type--> dst``. Retrieval (Phase 2.5+) will only ever traverse an
edge whose ``review_status`` is ``APPROVED`` -- the graph must stay a set of
human-approved claims, never a set of things a parser or a model merely
guessed.

``EdgeType`` lists every relationship this project's graph will ever carry,
not just the one this phase (2.1) produces. Locking the full set -- and each
type's direction -- now means every later phase inherits one convention
instead of inventing its own.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EdgeType(StrEnum):
    """Relationship type between two graph nodes (build plan A5.2).

    Attributes:
        BELONGS_TO: A unit belongs to its parent unit, or to its document
            when it has no unit parent. Direction: CHILD -> PARENT, e.g.
            a clause -> its article, an article -> its chapter or document.
            The only type Phase 2.1 emits; every other type below is
            reserved for the phase that emits it.
        BASED_ON: A document cites another as its legal basis ("Căn cứ
            ..."). Direction: CITING document -> CITED document (Phase 2.2).
        GUIDED_BY: A subordinate document cites a law as its legal basis --
            the same relationship as BASED_ON, just labelled more precisely
            when the citing document is a decree/circular and the cited one
            is a law. Direction: CITING document -> CITED (law) document,
            same as BASED_ON (Phase 2.2). Corrected from an earlier,
            backwards "higher-level -> guiding" phrasing that predated a
            concrete resolution rule: "src GUIDED_BY dst" must read "src is
            guided by dst", which puts the subordinate document first.
        AMENDED_BY: A document is amended by a later one. Direction:
            ORIGINAL document -> AMENDING document (Phase 2.2).
        REPLACED_BY: A document is replaced by a later one. Direction: OLD
            document -> NEW document (Phase 2.2).
        REFERENCES: An in-text cross-reference to another unit or document,
            internal or external. Direction: REFERRING unit -> REFERENCED
            unit/document (Phase 2.2).
        APPLIES_TO: A requirement entity applies to a concept entity, e.g.
            ``req:min-store-area`` applies to ``concept:lpg-retail-store``.
            Direction: REQUIREMENT -> CONCEPT (Phase 2.4). Corrected from an
            earlier "ENTITY -> unit" phrasing that predated a concrete seed
            data set: a requirement's target is another entity (the business
            concept it constrains), never a legal-text unit directly -- a
            requirement reaches its supporting statute through its own
            REFERENCES edge instead, kept separate so "what must a store
            satisfy" and "where does that rule come from" stay two edges,
            not one conflated relationship.
    """

    BELONGS_TO = "belongs_to"
    BASED_ON = "based_on"
    GUIDED_BY = "guided_by"
    AMENDED_BY = "amended_by"
    REPLACED_BY = "replaced_by"
    REFERENCES = "references"
    APPLIES_TO = "applies_to"


class ReviewStatus(StrEnum):
    """Human review state of an edge.

    Retrieval traverses ``APPROVED`` edges only -- an edge sitting at
    ``PENDING`` or ``REJECTED`` is invisible to the graph, on purpose.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class EdgeSource(StrEnum):
    """What produced an edge -- provenance, not a confidence signal."""

    STRUCTURAL = "structural"
    REGEX = "regex"
    LLM = "llm"
    CURATED = "curated"
    """A human wrote this edge directly into a seed data file (build plan A6
    issue #51, ``data/requirements.json``) -- neither the parser (STRUCTURAL)
    nor a pattern (REGEX) nor a model (LLM) produced it. Still landed
    ``review_status=PENDING``, the same as REGEX/LLM: a fact typed into a
    file is not the same act as a reviewer confirming it through
    ``ingest.review`` with the source unit's own text in front of them, and
    one uniform gate for every non-structural edge is easier to reason about
    than a second, silent exception to "inferred edges wait for review" --
    see ``ingest/graph.py``'s requirement pass for the full argument."""


class Edge(BaseModel):
    """One directed edge of the citation graph.

    Attributes:
        src_id: Source node identifier (A5.1).
        dst_id: Destination node identifier (A5.1). See ``EdgeType`` for what
            "source" and "destination" mean for each relationship.
        edge_type: Relationship type.
        evidence: Verbatim text supporting the edge, when the source has any
            (e.g. a regex match's span). ``None`` for a structural edge,
            which needs no evidence beyond the parser having read the unit
            there.
        confidence: Confidence score in ``[0, 1]``, when the source produces
            one (e.g. an LLM-proposed edge). ``None`` for a structural edge,
            which is certain by construction.
        review_status: Human review state; only ``APPROVED`` edges are safe
            to traverse.
        source: What produced this edge.
        origin_id: The unit -- or the document, for a citation in a preamble
            -- whose text produced this edge (issue #68). Kept apart from
            ``src_id`` because they only coincide for an edge that starts
            where it is stated: a REPLACED_BY edge starts at the repealed
            document, while the sentence that repeals it lives in the
            repealing one. Reconciliation scopes by this, never by
            ``src_id``. ``None`` for a curated edge, which no document's text
            produces.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    src_id: str = Field(min_length=1)
    dst_id: str = Field(min_length=1)
    edge_type: EdgeType
    evidence: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    review_status: ReviewStatus
    source: EdgeSource
    origin_id: str | None = Field(default=None, min_length=1)
