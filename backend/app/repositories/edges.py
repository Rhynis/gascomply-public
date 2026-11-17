"""Async repository for the ``edges`` table (build plan A5.3, A6 Phase 2.1/2.3a).

Persists the existing ``Edge`` model verbatim -- no second, ORM-shaped
domain object. Every query is parameterized (C3).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import CursorResult, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.edge import Edge, EdgeType, ReviewStatus

logger = logging.getLogger(__name__)

_UPSERT_SQL = text(
    """
    INSERT INTO edges (src_id, dst_id, edge_type, evidence, confidence,
                        review_status, source, origin_id)
    VALUES (:src_id, :dst_id, :edge_type, :evidence, :confidence,
            :review_status, :source, :origin_id)
    ON CONFLICT (src_id, dst_id, edge_type) DO UPDATE SET
        evidence = EXCLUDED.evidence,
        confidence = EXCLUDED.confidence,
        source = EXCLUDED.source,
        origin_id = EXCLUDED.origin_id
    """
)
"""``review_status`` is deliberately absent from the ``DO UPDATE SET`` list
(build plan A6 issue #45). Every inferred edge ``build_reference_edges``
produces is stamped ``PENDING`` unconditionally -- that is what "inferred,
not yet human-approved" means -- so if a re-ingest overwrote
``review_status`` on conflict, a human's approval or rejection would be
silently reset to ``PENDING`` and re-presented for review on the very next
``python -m ingest.run``, forever. Verified, not just reasoned about: ingest,
reject an edge, re-ingest the same corpus, confirm it is still rejected (see
``tests/test_graph_review.py`` and the PR body). ``evidence``/``confidence``/
``source`` still refresh to the latest extraction -- only the human decision
is sticky. ``origin_id`` refreshes with them (issue #68): it records what the
*latest* parse says produced the row, and is ``NULL`` when a curated seed
overwrites a regex-produced one, the same way ``source`` flips to ``curated``."""

_SET_REVIEW_STATUS_SQL = text(
    "UPDATE edges SET review_status = :review_status WHERE id = ANY(:edge_ids)"
)

_STATS_SQL = text(
    """
    SELECT edge_type, review_status, source, count(*) AS count
    FROM edges
    GROUP BY edge_type, review_status, source
    ORDER BY edge_type, review_status, source
    """
)


_APPROVED_TARGETS_SQL = text(
    """
    SELECT src_id, dst_id FROM edges
    WHERE src_id = ANY(:src_ids) AND edge_type = :edge_type AND review_status = 'approved'
    ORDER BY src_id, dst_id
    """
)
"""``'approved'`` is a literal, as in the traversal queries: no argument reaches it."""

_RETIRED_CITATIONS_SQL = text(
    """
    SELECT ref.src_id AS citing_unit,
           d.id AS retired_document,
           rep.dst_id AS replaced_by,
           rep.origin_id AS declared_in,
           rep.evidence AS evidence
    FROM edges ref
    JOIN documents d ON d.id = split_part(ref.dst_id, '#', 1)
    LEFT JOIN edges rep ON rep.src_id = d.id AND rep.edge_type = 'replaced_by'
                       AND rep.review_status = 'approved'
    WHERE ref.src_id = ANY(:unit_ids) AND ref.edge_type = 'references'
      AND ref.review_status = 'approved'
      AND (rep.id IS NOT NULL OR d.status = 'expired')
    ORDER BY ref.src_id, d.id, rep.dst_id
    """
)
"""A citation of a retired document (A5.2): an approved ``REFERENCES`` edge whose target's
document is the source of an approved ``REPLACED_BY`` (OLD -> NEW), or is marked ``expired``.
Both edges are approved-only literals. ``JOIN documents`` drops a target with no document row (a
dangling edge, whose status is unknown and so is nothing to warn about). A stub is a document
row with status ``unknown``: it counts only through a ``REPLACED_BY``, never for being unknown,
so the 63 stubs merely cited are not all reported as retired.

Both lookups are by ``src_id`` (``edges_src_type``), per returned unit -- never a scan."""


@dataclass(frozen=True)
class RetiredCitation:
    """One returned unit citing one retired document, with what replaced it.

    Attributes:
        citing_unit: The unit whose approved ``REFERENCES`` edge points at the retired document.
        retired_document: The document that is no longer law.
        replaced_by: The document the approved ``REPLACED_BY`` edge names as its replacement
            (its first step -- it may itself have been replaced since); ``None`` for a document
            that is merely marked ``expired``.
        declared_in: The unit whose text declares the repeal (the edge's ``origin_id``).
        evidence: The sentence that says so.
    """

    citing_unit: str
    retired_document: str
    replaced_by: str | None
    declared_in: str | None
    evidence: str | None


@dataclass(frozen=True)
class ReviewRow:
    """One edge, with enough context for a human to decide whether to approve it.

    Attributes:
        id: The edge's database row id -- what ``--approve``/``--reject``
            take, since ``(src_id, dst_id, edge_type)`` is verbose to type
            and ``Edge`` itself carries no id (build plan A6 issue #45).
        src_id: Source node id: where the edge *starts*. A unit, or a bare
            document id for a preamble-level BASIS/AMENDMENT citation -- or,
            for a REPLACED_BY edge, the repealed document (issue #83), which
            is usually a stub with no text.
        dst_id: Destination node id.
        edge_type: Relationship type, as its raw stored string.
        evidence: The exact citation text the edge was inferred from.
        confidence: Confidence score, when the source produces one.
        review_status: Current review state, as its raw stored string.
        source: What produced the edge, as its raw stored string.
        source_body: The full body of the unit whose text produced the edge
            -- ``origin_id`` when the edge records one, else ``src_id`` (issue
            #83) -- or ``None`` when that is a bare document id with no
            matching unit row (a citation in a document's preamble, before
            any unit begins).
        origin_id: The unit -- or, for a preamble citation, the document --
            whose text produced the edge, as stored (issue #68); ``None`` for
            a curated edge, which no document's text produces. Carried so a
            reviewer can be told *which* unit ``source_body`` is when it is
            not the one the edge starts at.
    """

    id: int
    src_id: str
    dst_id: str
    edge_type: str
    evidence: str | None
    confidence: float | None
    review_status: str
    source: str
    source_body: str | None
    origin_id: str | None = None


@dataclass(frozen=True)
class EdgeStatsRow:
    """One row of the ``(edge_type, review_status, source)`` count breakdown."""

    edge_type: str
    review_status: str
    source: str
    count: int


@dataclass(frozen=True)
class StaleEdgeConflict:
    """A stale edge this run no longer produces, but a human already reviewed.

    Attributes:
        id: Database row id, for a human to look up directly.
        src_id: Source node id.
        dst_id: Destination node id.
        edge_type: Relationship type, as its raw stored string.
        review_status: ``approved`` or ``rejected`` -- never ``pending``,
            which ``reconcile_document_edges`` removes outright instead of
            reporting.
    """

    id: int
    src_id: str
    dst_id: str
    edge_type: str
    review_status: str


@dataclass(frozen=True)
class ReconcileResult:
    """Outcome of reconciling one document's edges against what a run just produced.

    Attributes:
        removed: How many stale, ``pending`` edges were deleted -- safe,
            since nothing had judged them yet.
        conflicts: Stale edges at ``approved``/``rejected`` that were left
            untouched. A human decided these once and the evidence has since
            moved; that is a disagreement for a person to resolve, not
            something this pass may resolve on its own by deleting or
            re-trusting a row it can no longer verify.
    """

    removed: int
    conflicts: tuple[StaleEdgeConflict, ...]


class EdgeRepository:
    """Persists ``Edge`` rows, upserted by ``(src_id, dst_id, edge_type)``."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository.

        Args:
            session: Session to run queries on. The caller owns its
                transaction (see ``app.core.db.session_scope``).
        """
        self._session = session

    async def delete_structural_for_units(self, unit_ids: Sequence[str]) -> None:
        """Remove observed structural edges for units pruned from the parser tree."""
        if not unit_ids:
            return
        await self._session.execute(
            text("DELETE FROM edges WHERE source = 'structural' AND origin_id = ANY(:unit_ids)"),
            {"unit_ids": list(unit_ids)},
        )

    async def upsert_many(self, edges: Sequence[Edge]) -> None:
        """Insert or update edges, keyed on ``(src_id, dst_id, edge_type)``.

        Unlike ``units`` and ``chunks``, an edge carries no foreign key to
        order around (see ``migrations/002_graph.sql``), so every edge is
        upserted in a single batched statement rather than one at a time.
        A row that already exists keeps its own ``review_status`` -- see
        ``_UPSERT_SQL``.

        Args:
            edges: Edges to persist.
        """
        if not edges:
            return
        await self._session.execute(
            _UPSERT_SQL,
            [
                {
                    "src_id": edge.src_id,
                    "dst_id": edge.dst_id,
                    "edge_type": edge.edge_type.value,
                    "evidence": edge.evidence,
                    "confidence": edge.confidence,
                    "review_status": edge.review_status.value,
                    "source": edge.source.value,
                    "origin_id": edge.origin_id,
                }
                for edge in edges
            ],
        )

    async def approved_targets(
        self, src_ids: Sequence[str], edge_type: EdgeType
    ) -> dict[str, tuple[str, ...]]:
        """Return the ids each source points at by an approved edge of ``edge_type``.

        A direct lookup of one step's out-edges, not a walk: what retrieval uses to ask which
        clauses a requirement cites, when it must know *which* requirement cites *which* clause
        and ``GraphTraversalService`` (which reports only what was reached) cannot say.
        ``approved`` only, as everywhere in retrieval.

        Args:
            src_ids: Source node ids.
            edge_type: The edge type to follow, forwards.

        Returns:
            ``{src_id: (dst_id, ...)}``, for each source that has at least one such edge.
        """
        if not src_ids:
            return {}
        result = await self._session.execute(
            _APPROVED_TARGETS_SQL, {"src_ids": list(src_ids), "edge_type": edge_type.value}
        )
        targets: dict[str, list[str]] = {}
        for row in result:
            targets.setdefault(row.src_id, []).append(row.dst_id)
        return {src_id: tuple(dst_ids) for src_id, dst_ids in targets.items()}

    async def retired_citations(self, unit_ids: Sequence[str]) -> list[RetiredCitation]:
        """Return every approved citation, from these units, of a document that was retired.

        Args:
            unit_ids: Units to check, exactly (their own edges; not their descendants').

        Returns:
            One row per (citing unit, retired document, replacement) -- several when a document
            was replaced by more than one, or a unit cites it more than once.
        """
        if not unit_ids:
            return []
        result = await self._session.execute(_RETIRED_CITATIONS_SQL, {"unit_ids": list(unit_ids)})
        return [RetiredCitation(**row._mapping) for row in result]

    async def existing_triples(self, edges: Sequence[Edge]) -> set[tuple[str, str, str]]:
        """Return which of ``edges``' ``(src_id, dst_id, edge_type)`` already have a row.

        Read-only, and deliberately separate from ``upsert_many``'s own
        batched execution rather than folded into it (e.g. via ``RETURNING``
        on the upsert itself): a hand-written multi-row ``text()`` statement
        has no reliable way to collect one ``RETURNING`` row per input row
        across drivers, and every other caller of ``upsert_many`` depends on
        its single-batch performance, which this must not change. Used by
        ``ingest/run.py`` to report how many of a curated seed's edges were
        genuinely new rows versus an update to one the regex pass already
        produced (issue #63's review) -- the stats table must report what
        was actually persisted, not what was attempted.

        Args:
            edges: Candidate edges to check.

        Returns:
            The ``(src_id, dst_id, edge_type)`` triples among ``edges`` that
            already exist. Over-fetches by ``src_id`` alone and filters the
            full triple in Python rather than a composite-tuple SQL
            comparison, since the caller's edge count is always small
            (curated seed data, never corpus-scale).
        """
        if not edges:
            return set()
        result = await self._session.execute(
            text("SELECT src_id, dst_id, edge_type FROM edges WHERE src_id = ANY(:src_ids)"),
            {"src_ids": list({edge.src_id for edge in edges})},
        )
        return {(row.src_id, row.dst_id, row.edge_type) for row in result}

    async def reconcile_document_edges(
        self, document_id: str, produced: Sequence[Edge]
    ) -> ReconcileResult:
        """Remove edges this document no longer produces -- issue #75.

        A re-ingest upserts every edge the current parse produces, but never
        removed a row from an *earlier* parse that the corpus, or the
        parser's understanding of it, has since stopped producing (e.g. a
        citation once misclassified as internal, corrected by issue #74).
        Nothing else notices: the row's endpoints still resolve, so it is not
        dangling, and it sits in the review queue looking exactly as
        plausible as a real edge -- approving it would launder a parser
        mistake into a human decision. This closes that gap by comparing
        ``document_id``'s current DB edges against ``produced``, this run's
        freshly built set for the same document.

        Curated edges (``source = 'curated'``) are excluded from the
        comparison entirely, in both directions: they come from
        ``data/requirements.json`` / ``data/document_edges.json``, not from
        this document's text, so "not in ``produced``" says nothing about
        whether they are still correct -- no corpus parse ever regenerates
        them, including the QCVN 08 history edge that predates every
        document this pipeline can parse. Treating a curated edge as stale
        would delete the requirement cluster and the document-edge seed on
        the very next re-ingest.

        A stale edge still ``PENDING`` is deleted outright: nothing has
        judged it, so nothing is lost. A stale edge that is ``APPROVED`` or
        ``REJECTED`` is left untouched and reported as a conflict instead --
        a person looked at that edge and the evidence has since moved, which
        is a disagreement for a person to resolve. Silently deleting their
        decision and silently keeping a now-unsupported edge are both wrong;
        surfacing it (count and rows, via ``ReconcileResult.conflicts``) is
        the one the caller can act on.

        Scope is exactly the edges ``document_id``'s own text produced, and
        that is decided by ``origin_id``, never by ``src_id`` (issue #68). The
        two only coincide for an edge that starts where it is stated. A
        REPLACED_BY edge runs OLD -> NEW, but the repeal is written in the NEW
        document: it starts at the repealed document -- a stub, or a manifest
        document -- and is produced by the repealing one. Scoping by ``src_id``
        got both halves of that wrong. A stub belongs to no ingested document,
        so a stale edge into it was never pruned; and a repealed document that
        was itself in the manifest saw the edge as "not produced by my own
        parse" and deleted it (or reported a false conflict, once approved) on
        every run, while the repealing document re-created it. ``origin_id``
        names the producer directly. ``ReconcileResult.conflicts`` therefore
        never contains an edge another document's text produced, however it is
        directed.

        The caller only calls this for a document it actually just persisted,
        so a document skipped by ``--limit``, or a metadata-only document
        (issue #62, which never reaches this repository at all), is never
        touched by construction. A row with no ``origin_id`` -- curated, or
        seeded straight into the table -- is in no document's scope.

        Args:
            document_id: The document whose edges to reconcile.
            produced: The edges this run's parse built for ``document_id``
                (any source -- curated rows are filtered out internally, so
                the caller need not pre-filter).

        Returns:
            How many stale ``PENDING`` rows were deleted, and which stale
            ``APPROVED``/``REJECTED`` rows were left in place as conflicts.
        """
        produced_triples = {
            (edge.src_id, edge.dst_id, edge.edge_type.value)
            for edge in produced
            if edge.source.value != "curated"
        }
        result = await self._session.execute(
            text(
                """
                SELECT id, src_id, dst_id, edge_type, review_status
                FROM edges
                WHERE (origin_id = :document_id OR origin_id LIKE :document_prefix)
                  AND source != 'curated'
                """
            ),
            {"document_id": document_id, "document_prefix": f"{document_id}#%"},
        )
        stale_pending_ids: list[int] = []
        conflicts: list[StaleEdgeConflict] = []
        for row in result:
            if (row.src_id, row.dst_id, row.edge_type) in produced_triples:
                continue
            if row.review_status == ReviewStatus.PENDING.value:
                stale_pending_ids.append(row.id)
            else:
                conflicts.append(
                    StaleEdgeConflict(
                        id=row.id,
                        src_id=row.src_id,
                        dst_id=row.dst_id,
                        edge_type=row.edge_type,
                        review_status=row.review_status,
                    )
                )
        if stale_pending_ids:
            await self._session.execute(
                text("DELETE FROM edges WHERE id = ANY(:edge_ids)"),
                {"edge_ids": stale_pending_ids},
            )
        return ReconcileResult(removed=len(stale_pending_ids), conflicts=tuple(conflicts))

    async def list_pending(
        self,
        *,
        edge_type: EdgeType | None = None,
        src_document: str | None = None,
        limit: int | None = None,
    ) -> list[ReviewRow]:
        """List pending edges with the text that produced them, for a human to review.

        An edge belongs to the document whose *text* produced it, and that is
        ``COALESCE(origin_id, src_id)``, never ``src_id`` alone (issue #83).
        The two only coincide for an edge that starts where it is stated. A
        REPLACED_BY edge runs OLD -> NEW but the repeal is written in the NEW
        document, so it starts at the repealed one -- usually a stub with no
        text. Joining and filtering by ``src_id`` showed the reviewer no
        sentence at all ("document-level citation; no unit body") and left the
        repealing document's own edges out of ``src_document``, though
        ``evidence`` and ``origin_id`` (issue #68) already say what justifies
        the edge and where it was read. It is the same attribution
        ``reconcile_document_edges`` scopes by. A row with no ``origin_id``
        (a curated seed, which no text produces) falls back to ``src_id``, so
        it lists exactly as it always did.

        ``units.id`` has no foreign key from ``edges`` (a node can be a
        document, a unit, or an entity -- see ``migrations/002_graph.sql``),
        so this is a ``LEFT JOIN``: a preamble-level citation, whose producer
        is a bare document id, still comes back with ``source_body=None``
        rather than being silently dropped from the list.

        Args:
            edge_type: Restrict to one edge type, or every type when ``None``.
            src_document: Restrict to edges this document's text produced --
                attributed by ``COALESCE(origin_id, src_id)``, so the document
                itself or a unit under it (``"<src_document>#..."``) -- or
                every document when ``None``.
            limit: Maximum rows to return, or every matching row when ``None``.

        Returns:
            Matching rows, ordered by edge id.
        """
        conditions = ["e.review_status = :review_status"]
        params: dict[str, object] = {"review_status": ReviewStatus.PENDING.value}
        if edge_type is not None:
            conditions.append("e.edge_type = :edge_type")
            params["edge_type"] = edge_type.value
        if src_document is not None:
            conditions.append(
                "(COALESCE(e.origin_id, e.src_id) = :src_document "
                "OR COALESCE(e.origin_id, e.src_id) LIKE :src_document_prefix)"
            )
            params["src_document"] = src_document
            params["src_document_prefix"] = f"{src_document}#%"
        sql = f"""
            SELECT e.id, e.src_id, e.dst_id, e.edge_type, e.evidence, e.confidence,
                   e.review_status, e.source, e.origin_id, u.body AS source_body
            FROM edges e
            LEFT JOIN units u ON u.id = COALESCE(e.origin_id, e.src_id)
            WHERE {" AND ".join(conditions)}
            ORDER BY e.id
        """
        if limit is not None:
            sql += " LIMIT :limit"
            params["limit"] = limit
        result = await self._session.execute(text(sql), params)
        return [ReviewRow(**row._mapping) for row in result]

    async def set_review_status(self, edge_ids: Sequence[int], review_status: ReviewStatus) -> int:
        """Set ``review_status`` on the given edges, by id -- the only way it ever changes.

        Args:
            edge_ids: Database row ids to update.
            review_status: The status to set.

        Returns:
            How many rows were actually updated -- lets a caller notice an id
            that named no row.
        """
        if not edge_ids:
            return 0
        result = await self._session.execute(
            _SET_REVIEW_STATUS_SQL,
            {"edge_ids": list(edge_ids), "review_status": review_status.value},
        )
        assert isinstance(result, CursorResult)
        return int(result.rowcount)

    async def stats(self) -> list[EdgeStatsRow]:
        """Return edge counts grouped by ``(edge_type, review_status, source)``."""
        result = await self._session.execute(_STATS_SQL)
        return [EdgeStatsRow(**row._mapping) for row in result]

    async def count(self) -> int:
        """Return how many rows ``edges`` currently holds.

        The stats table's TOTAL row uses this directly (issue #75) rather
        than summing every row's own ``edges`` count. Two passes can declare
        the same edge in one run (a regex-produced reference and a curated
        document edge sharing endpoints), and a repeat ingest upserts rows
        that already existed -- neither is "newly built," but summing counts
        both as if they were, so the summed total silently drifts from what
        the table actually holds, in either direction depending on which
        rows happened to overlap. The row count the table actually holds is
        unambiguous no matter how many passes touched it this run.
        """
        result = await self._session.execute(text("SELECT count(*) FROM edges"))
        return int(result.scalar_one())
