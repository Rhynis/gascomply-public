"""Async repository for the ``chunks`` table (build plan A5.3, A6 Phase 1.5b, 3.1, 3.3).

Persists the existing ``Chunk`` model plus the embedding vector an
``Embedder`` produced separately (``Chunk`` itself carries no vector -- see
``app/schemas/chunk.py``), and reads it back by similarity to a question. No second,
ORM-shaped domain object; every query is parameterized (C3).
"""

import logging
from collections.abc import Sequence
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Row, bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.chunk import Chunk, ChunkMatch

logger = logging.getLogger(__name__)

CHUNKS_EMBEDDING_DIMENSION = 1024
"""Must match ``migrations/001_init.sql``'s ``chunks.embedding vector(1024)``
column. The vector dimension is embedder-dependent (build plan A2), so this
is the one place that dimension is pinned for persistence; every embedding
passed to ``upsert_many`` is checked against it before any SQL runs, so a
mismatched embedder fails loudly here rather than letting Postgres reject
rows partway through a corpus."""

_UPSERT_SQL = text(
    """
    INSERT INTO chunks (unit_id, document_id, dieu_no, khoan_no, effective,
                         status, body, embedding)
    VALUES (:unit_id, :document_id, :dieu_no, :khoan_no, :effective,
            :status, :body, :embedding)
    ON CONFLICT (unit_id) DO UPDATE SET
        document_id = EXCLUDED.document_id,
        dieu_no = EXCLUDED.dieu_no,
        khoan_no = EXCLUDED.khoan_no,
        effective = EXCLUDED.effective,
        status = EXCLUDED.status,
        body = EXCLUDED.body,
        embedding = EXCLUDED.embedding
    """
).bindparams(bindparam("embedding", type_=Vector(CHUNKS_EMBEDDING_DIMENSION)))
"""pgvector's SQLAlchemy ``Vector`` type must be bound explicitly for a raw
``text()`` query -- it is what lets a Python ``list[float]`` reach the
``vector`` column at all; without it asyncpg has no codec for the type and
the insert fails."""

_REPLACED_BY_EDGES = (
    "FROM edges r WHERE r.src_id = c.document_id AND r.edge_type = 'replaced_by' "
    "AND r.review_status = '{review_status}'"
)

_MATCH_COLUMNS = f"""
    c.unit_id, c.document_id, c.dieu_no, c.khoan_no, c.effective, c.status, c.body,
    COALESCE(1 - (c.embedding <=> :query_vector), 0.0) AS similarity,
    ARRAY(SELECT r.dst_id {_REPLACED_BY_EDGES.format(review_status="approved")}
          ORDER BY r.dst_id) AS replaced_by,
    EXISTS (SELECT 1 {_REPLACED_BY_EDGES.format(review_status="pending")}) AS possibly_superseded
"""

_IS_CURRENT = f"""(
    CAST(:include_history AS boolean)
    OR (c.status = 'in_force'
        AND NOT EXISTS (SELECT 1 {_REPLACED_BY_EDGES.format(review_status="approved")}))
)"""
"""The effective-date filter (A5.2, issue #89), in one place so vector search and the by-id
lookup cannot drift apart: a chunk is current unless its own ``status`` is not ``in_force`` or
its document is the source of an approved ``REPLACED_BY``. ``status`` is read from the chunk,
where it was denormalised from the document at ingest for exactly this (A5.3), and the
replacement from ``edges`` (indexed by ``src_id``). ``'approved'`` is a literal, as in
``graph_traversal``: an unreviewed edge is a guess, and never drops a chunk (a *pending* one
only sets ``possibly_superseded``). ``include_history`` turns the filter off entirely.

The two fragments above are static text, never built from input: every value in the queries
below is a bind parameter (C3)."""

_SEARCH_SQL = text(
    f"""
    SELECT {_MATCH_COLUMNS}
    FROM chunks c
    WHERE c.embedding IS NOT NULL
      AND {_IS_CURRENT}
      AND (cardinality(CAST(:document_ids AS text[])) = 0
           OR c.document_id = ANY(CAST(:document_ids AS text[])))
    ORDER BY c.embedding <=> :query_vector, c.unit_id
    LIMIT :limit
    """
).bindparams(bindparam("query_vector", type_=Vector(CHUNKS_EMBEDDING_DIMENSION)))
"""Exact cosine nearest neighbours, no index: ``chunks`` has none on purpose (migration 003 and
ADR 0001 -- an approximate index returned the wrong top hit at this size). ``unit_id`` breaks
ties, so equal similarities always come back in the same order. A chunk with no embedding cannot
be ranked and is skipped."""

_SCORE_SQL = text(
    f"""
    SELECT {_MATCH_COLUMNS}
    FROM chunks c
    WHERE c.unit_id = ANY(CAST(:unit_ids AS text[])) AND {_IS_CURRENT}
    """
).bindparams(bindparam("query_vector", type_=Vector(CHUNKS_EMBEDDING_DIMENSION)))


class ChunkRepository:
    """Persists ``Chunk`` rows plus their embeddings, upserted by stable unit id."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository.

        Args:
            session: Session to run queries on. The caller owns its
                transaction (see ``app.core.db.session_scope``).
        """
        self._session = session

    async def prune_document(self, document_id: str, keep_ids: Sequence[str]) -> tuple[str, ...]:
        """Delete chunks for a document that the current parse no longer produces."""
        result = await self._session.execute(
            text("SELECT unit_id FROM chunks WHERE document_id = :document_id"),
            {"document_id": document_id},
        )
        stale = [row.unit_id for row in result if row.unit_id not in set(keep_ids)]
        if stale:
            await self._session.execute(
                text("DELETE FROM chunks WHERE unit_id = ANY(:unit_ids)"),
                {"unit_ids": stale},
            )
        return tuple(stale)

    async def upsert_many(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None:
        """Insert or update chunks and their embeddings, keyed on ``unit_id``.

        ``chunks.unit_id`` carries a ``UNIQUE`` constraint added in the
        migration (the DDL in the build plan does not declare one) precisely
        so this upsert has a conflict target -- ``build_chunks`` never
        produces two chunks for the same unit, so the constraint reflects a
        rule already true in practice.

        Args:
            chunks: Chunks to persist, in the same order as ``embeddings``.
            embeddings: One embedding vector per chunk, same order and
                length as ``chunks``.

        Raises:
            ValueError: If ``chunks`` and ``embeddings`` differ in length, or
                any embedding's length does not match
                ``CHUNKS_EMBEDDING_DIMENSION``.
        """
        if len(chunks) != len(embeddings):
            msg = f"got {len(chunks)} chunk(s) but {len(embeddings)} embedding(s)"
            raise ValueError(msg)

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            _require_dimension(chunk.unit_id, embedding)
            await self._session.execute(
                _UPSERT_SQL,
                {
                    "unit_id": chunk.unit_id,
                    "document_id": chunk.document_id,
                    "dieu_no": chunk.dieu_no,
                    "khoan_no": chunk.khoan_no,
                    "effective": chunk.effective,
                    "status": chunk.status.value,
                    "body": chunk.body,
                    "embedding": embedding,
                },
            )

    async def search(
        self,
        query_embedding: list[float],
        *,
        limit: int,
        include_history: bool = False,
        document_ids: Sequence[str] = (),
    ) -> list[ChunkMatch]:
        """Return the ``limit`` chunks nearest ``query_embedding`` by cosine similarity.

        Exact, not approximate (see ``_SEARCH_SQL``). The effective-date filter runs inside the
        query, before ``LIMIT``, so a superseded chunk never uses up one of the ``limit`` slots.

        Args:
            query_embedding: The question's vector, ``CHUNKS_EMBEDDING_DIMENSION`` wide.
            limit: Most chunks to return; ``0`` or less returns none.
            include_history: When ``True``, skips the effective-date filter and returns
                superseded chunks too, each carrying ``replaced_by`` -- for a question about
                law that has been replaced.
            document_ids: When given, only chunks of these documents are considered -- how the
                in-force replacement of a retired document is searched for what the question asks.

        Returns:
            Matches, most similar first.

        Raises:
            ValueError: If ``query_embedding`` is not ``CHUNKS_EMBEDDING_DIMENSION`` wide.
        """
        _require_dimension("query", query_embedding)
        if limit < 1:
            return []
        result = await self._session.execute(
            _SEARCH_SQL,
            {
                "query_vector": query_embedding,
                "include_history": include_history,
                "document_ids": list(document_ids),
                "limit": limit,
            },
        )
        return [_to_match(row) for row in result]

    async def score(
        self,
        query_embedding: list[float],
        unit_ids: Sequence[str],
        *,
        include_history: bool = False,
    ) -> list[ChunkMatch]:
        """Return the chunks named by ``unit_ids`` with their similarity to ``query_embedding``.

        Unlike ``search`` this does not choose which chunks; it measures chunks somebody else
        chose (a graph walk reached them). The same effective-date filter applies, so a chunk of
        a superseded document is absent from the answer rather than present and flagged, and an id
        that names no chunk is simply absent too.

        Args:
            query_embedding: The question's vector.
            unit_ids: Unit ids to look up as chunks.
            include_history: As for ``search``.

        Returns:
            One match per id that names a current chunk, in no particular order.

        Raises:
            ValueError: If ``query_embedding`` is not ``CHUNKS_EMBEDDING_DIMENSION`` wide.
        """
        _require_dimension("query", query_embedding)
        if not unit_ids:
            return []
        result = await self._session.execute(
            _SCORE_SQL,
            {
                "query_vector": query_embedding,
                "include_history": include_history,
                "unit_ids": list(unit_ids),
            },
        )
        return [_to_match(row) for row in result]


def _to_match(row: Row[Any]) -> ChunkMatch:
    """Build a ``ChunkMatch`` from one row of ``_MATCH_COLUMNS``."""
    return ChunkMatch(
        unit_id=row.unit_id,
        document_id=row.document_id,
        dieu_no=row.dieu_no,
        khoan_no=row.khoan_no,
        effective=row.effective,
        status=row.status,
        body=row.body,
        similarity=float(row.similarity),
        replaced_by=tuple(row.replaced_by),
        possibly_superseded=row.possibly_superseded,
    )


def _require_dimension(unit_id: str, embedding: list[float]) -> None:
    """Raise if ``embedding``'s length does not match the ``chunks`` schema."""
    if len(embedding) != CHUNKS_EMBEDDING_DIMENSION:
        msg = (
            f"{unit_id}: embedding has dimension {len(embedding)}, "
            f"but the chunks table expects {CHUNKS_EMBEDDING_DIMENSION}"
        )
        raise ValueError(msg)
