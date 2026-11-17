"""Typed results and settings of hybrid retrieval (build plan A6 Phase 3.1-3.4, issue #89).

Retrieval hands the generator three kinds of thing, kept apart because each is used
differently:

* ``Passage`` -- text to quote. A chunk of a document, with the unit ids inside it that the
  graph pointed at.
* ``RetrievedEntity`` -- a fact to cite. A requirement (``req:min-store-area``: "≥ 12 m²") or the
  concept it applies to. An entity has no chunk and no embedding, so it cannot be a passage,
  but it is the obligation set the graph exists to find; it carries the units that state it.
* ``Supersession`` -- a warning. A returned unit cites a document that has been replaced, and
  here is what replaced it.

Every item says *why* it is in the result (``Why``), since "this clause, because a requirement
cites it" is better context to generate from than an unexplained list.
"""

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.ids import EntityType


class Why(StrEnum):
    """Why an item is in the result. Says what found it, never how sure anyone is of it."""

    SIMILAR = "similar"
    """Among the ``k`` chunks nearest the question by embedding."""

    CITED_BY_HIT = "cited_by_hit"
    """A hit (or a unit inside one) references it: the legal basis or the article a hit sends
    the reader to (``REFERENCES``, forwards)."""

    CITES_HIT = "cites_hit"
    """It references a hit (or a unit inside one): the requirement that a clause states, or a
    unit that points at the clause (``REFERENCES``, backwards)."""

    SAME_CONCEPT = "same_concept"
    """Reached through a concept that a reached requirement applies to: the concept itself, or
    another requirement of it (``APPLIES_TO``, both ways)."""

    SOURCE_OF_REQUIREMENT = "source_of_requirement"
    """The clause a reached requirement cites as its legal basis, so the fact can be quoted."""

    SUCCESSOR = "successor"
    """The in-force replacement of a retired document that a returned unit cites."""


class RetrievalConfig(BaseModel):
    """Every tunable of a retrieval, in one validated place.

    Attributes:
        k: How many chunks the vector search returns. These are the seeds of the graph walk.
        cite_hops: How far forward ``REFERENCES`` chains are followed from a hit (a decree that
            cites a law that cites a decree). Backward ``REFERENCES`` is always one hop: it fans
            out (a popular article is cited by many), and a second step would reach units that
            merely cite the same thing.
        decay: The graph-proximity factor of ``rerank.score``: an item ``hop`` edges from a hit
            keeps ``decay ** hop`` of its similarity. ``1.0`` ignores the graph, ``0.0`` keeps
            only the hits.
        passage_limit: Most passages returned, after graph-proximity ranking. This is also the
            pool a cross-encoder reranks (issue #114): the candidates never number much more than
            ``passage_limit`` (measured: 13.5 per question on average with the limit raised to
            30), so widening the pool bought no recall.
        rerank_top_n: How many passages survive the cross-encoder, when ``Retriever`` was given
            a reranker (issue #114). ``None`` keeps every passage in the pool and only reorders
            them. Ignored when there is no reranker. Fewer passages means fewer wrong options
            for the generator to pick from, at the price of possibly dropping the one holding
            the gold clause -- the number is chosen by measurement (see the ADR).
        entity_limit: Most entities returned, after reranking.
        successor_k: How many chunks to fetch from an in-force replacement of a retired document.
        descend_char_budget: Characters of chunked descendant text to offer when the graph
            reaches a container unit (an appendix, a long article split into clause chunks)
            that has no chunk of its own (issue #136, re-applied with a budget in #140). Ranked
            by the descendants' own chunk embeddings against the question, which is already
            computed -- no extra model call -- descendants are taken in that order until the
            running total of their ``body`` lengths would exceed the budget; the first one is
            always kept even alone over budget, so a container is never silently emptied.
            A fixed top-k (#138) cut Phụ lục VII's 44 short items down to 3 and lost the one
            that answered the question; a character budget instead lets a container of many
            short items (an appendix) come in close to whole while still bounding a container
            of few long items (a split article). 6,000 characters holds Phụ lục VII (5,117
            characters total) whole with room to spare, and #129's 16,384-token context window
            holds that easily (the largest gold prompt today is 12,373 tokens).
    """

    model_config = ConfigDict(frozen=True)

    k: int = Field(default=5, ge=1)
    cite_hops: int = Field(default=2, ge=1)
    decay: float = Field(default=0.9, gt=0.0, le=1.0)
    passage_limit: int = Field(default=10, ge=1)
    rerank_top_n: int | None = Field(default=None, ge=1)
    entity_limit: int = Field(default=10, ge=1)
    successor_k: int = Field(default=3, ge=1)
    descend_char_budget: int = Field(default=6000, ge=1)


class Passage(BaseModel):
    """One chunk of a document, ranked.

    Attributes:
        unit_id: The chunk's unit id: an article, or a clause of a long article. What to cite
            when the fine-grained ``focus_units`` are not enough.
        document_id: Owning document.
        dieu_no: Article number, for a citation.
        khoan_no: Clause number, set only on a clause-level chunk.
        effective: When the document took effect.
        text: The chunk's text.
        similarity: Cosine similarity of the chunk's embedding to the question.
        hop: Fewest edges between this chunk and a vector hit, not counting containment: ``0``
            for a hit, and for a clause that is part of one (a chunk *is* its clauses' text).
        score: ``similarity`` discounted by ``hop`` -- what the list is sorted by, unless a
            cross-encoder reranked it (see ``rerank_score``).
        rerank_score: The cross-encoder's raw score for this (question, passage) pair, when
            ``Retriever`` reranked (issue #114); the list is then sorted by this, best first.
            A logit, not a probability: comparable only among the passages of one result.
            ``None`` when nothing reranked, or when the reranker failed and the retriever
            fell back to ``score`` order.
        why: What found it. More than one when several routes lead here.
        focus_units: Real units *inside* this chunk, from the graph (a requirement cites the
            clause; a hit cites the point) or simply from the corpus's own structure (issue
            #106: every unit ``units`` has nested under this chunk's id, whether or not any edge
            points at it) -- so a passage that holds a whole article can still say which clause
            an obligation comes from, or which clause a model may cite at its own precision.
            Empty when the chunk itself is the finest unit here (a clause-level chunk with no
            children, or an article the corpus never subdivided further).
        superseded: ``True`` only under ``include_history=True``, for a chunk whose document is
            superseded (A5.2: not in force, or replaced).
        superseded_by: The documents that replace this chunk's document, when it is superseded.
        possibly_superseded: The document has a ``REPLACED_BY`` edge pending review.
        replaces: For a passage from an in-force replacement, the retired documents it replaces
            that a returned unit cites (``Why.SUCCESSOR``).
    """

    model_config = ConfigDict(frozen=True)

    unit_id: str
    document_id: str
    dieu_no: int | None = None
    khoan_no: int | None = None
    effective: date | None = None
    text: str
    similarity: float
    hop: int
    score: float
    rerank_score: float | None = None
    why: tuple[Why, ...]
    focus_units: tuple[str, ...] = ()
    superseded: bool = False
    superseded_by: tuple[str, ...] = ()
    possibly_superseded: bool = False
    replaces: tuple[str, ...] = ()


class RetrievedEntity(BaseModel):
    """A requirement or concept the graph reached, with the units that state it.

    Attributes:
        entity_id: ``req:...`` or ``concept:...`` (A5.1).
        entity_type: Its type.
        label_vi: Vietnamese label.
        meta: Structured data, e.g. ``{"value": "≥ 12 m²"}`` for a requirement.
        similarity: Of the clause that states it -- the best of its sources -- because an
            entity has no embedding of its own: an obligation is as relevant as its legal basis
            is. A concept has no source, so it takes the best of the requirements found with it.
        hop: Fewest edges from a vector hit; containment is not counted.
        score: ``similarity`` discounted by ``hop``.
        why: What found it.
        sources: The unit ids it cites as its legal basis (its ``REFERENCES`` edges, approved
            only): the citation. Empty for a concept.
    """

    model_config = ConfigDict(frozen=True)

    entity_id: str
    entity_type: EntityType
    label_vi: str
    meta: dict[str, Any]
    similarity: float
    hop: int
    score: float
    why: tuple[Why, ...]
    sources: tuple[str, ...] = ()


class Supersession(BaseModel):
    """A warning that a returned unit cites a document that has been replaced.

    The real case is a *stub*: ``cited-136-2020-nd-cp`` is known only from citations, holds no
    text and so can never be a passage, yet eight approved ``REFERENCES`` edges point at it and
    an approved ``REPLACED_BY`` says ``pccc-decree-105-2025`` replaced it.

    Attributes:
        retired_document: The document that is no longer law (the source of an approved
            ``REPLACED_BY`` edge, or an ``expired`` one).
        replaced_by: The in-force documents at the end of its ``REPLACED_BY`` chain -- empty when
            nothing in force is known to have replaced it.
        declared_in: The unit whose text declares the repeal (the edge's ``origin_id``).
        evidence: The sentence that says so.
        cited_by: The returned units that cite the retired document.
    """

    model_config = ConfigDict(frozen=True)

    retired_document: str
    replaced_by: tuple[str, ...]
    declared_in: str | None = None
    evidence: str | None = None
    cited_by: tuple[str, ...]


class RetrievalResult(BaseModel):
    """What one question retrieves.

    Attributes:
        question: The question, as asked.
        passages: Chunks, best first.
        entities: Requirements and concepts, best first.
        supersessions: Retired documents that a returned unit cites, each with its replacement.
    """

    model_config = ConfigDict(frozen=True)

    question: str
    passages: tuple[Passage, ...]
    entities: tuple[RetrievedEntity, ...] = ()
    supersessions: tuple[Supersession, ...] = ()
