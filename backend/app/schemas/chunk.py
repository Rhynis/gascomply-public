"""Typed embedding-ready chunks, mirroring the ``chunks`` table (build plan A5.3).

A ``Chunk`` is the text unit that gets embedded and, in Phase 1.5b, written to
pgvector. This model carries every column of the ``chunks`` table except its
database identity (``id``) and the embedding vector itself, which an
``Embedder`` (``app/embedders/base.py``) produces separately and is not part
of this model.

``effective`` and ``status`` are denormalized from the owning ``Document``,
never derived from the ``Unit`` a chunk was built from -- a ``Unit`` carries no
in-force state at all. This is deliberate (A5.3): Phase 3.3's effective-date
filter reads these two chunk columns alone to drop superseded law, with no
join back to ``documents`` at query time. ``ingest/chunk.py`` is the only
place these fields are set, and it copies them from the ``Document`` verbatim.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.document import DocStatus


class Chunk(BaseModel):
    """One unit of text prepared for embedding.

    Attributes:
        unit_id: Identifier (A5.1) of the unit this chunk was composed from --
            an article for an article-level chunk, a clause for a clause-level
            chunk.
        document_id: Owning document identifier.
        dieu_no: Article number, denormalized for citation display.
        khoan_no: Clause number, denormalized for citation display. Set only
            on a clause-level chunk; unset (``None``) on an article-level
            chunk.
        effective: Effective date, denormalized from the owning ``Document`` --
            never re-derived from anything else.
        status: In-force status, denormalized from the owning ``Document`` --
            never re-derived from anything else. Phase 3.3's date filter reads
            this field alone to decide whether a chunk represents superseded
            law that should be excluded from an answer.
        body: The composed text that gets embedded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    dieu_no: int | None = None
    khoan_no: int | None = None
    effective: date | None = None
    status: DocStatus
    body: str = Field(min_length=1)


class ChunkMatch(Chunk):
    """A stored chunk together with how well it matches one question (issue #89).

    What ``ChunkRepository.search`` and ``ChunkRepository.score`` return: the chunk's own
    columns, plus the two things retrieval needs to know about it that no ``Chunk``
    carries -- how near it is to the question, and whether its document is on its way out.

    Attributes:
        similarity: Cosine similarity of the chunk's embedding to the question's, i.e.
            ``1 - cosine_distance`` (pgvector's ``<=>``): ``1.0`` for the same direction,
            ``0.0`` for orthogonal, ``-1.0`` for opposite. ``0.0`` for a chunk that has no
            embedding, which cannot be ranked by it.
        replaced_by: The documents an *approved* ``REPLACED_BY`` edge says replace this
            chunk's document (A5.2, OLD -> NEW); empty for a document nothing replaces. Only
            ever non-empty on a chunk returned with ``include_history=True``, since the date
            filter drops it otherwise.
        possibly_superseded: ``True`` when this chunk's document has a ``REPLACED_BY`` edge
            that is still ``pending`` review. Not a reason to drop the chunk -- an unreviewed
            edge is a machine's guess -- but a reason to warn a reader (same rule as
            ``TraversalHit.possibly_superseded``).
    """

    similarity: float
    replaced_by: tuple[str, ...] = ()
    possibly_superseded: bool = False
