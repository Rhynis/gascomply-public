"""Recursive-CTE graph traversal over approved edges (build plan A6 Phase 2.5).

The citation graph (Phase 2.1's structural ``BELONGS_TO`` edges today, Phase
2.2's reference edges once Phase 2.3 approves them) is only useful if
something actually walks it instead of relying on vector similarity alone --
that is this project's whole architectural claim. This module is that walk.

Only ``review_status = 'approved'`` edges are ever traversed, and that is not
a parameter a caller can override (see ``GraphTraversalService.traverse``'s
signature) -- an inferred edge is not trusted until a human approves it, and
a wrong legal graph is dangerous. Approving an edge is what makes it walkable:
nothing here changes when one is approved; it simply starts showing up in the
same walk, including across documents.

A walk follows an edge from its ``src_id`` to its ``dst_id`` by default. Some of
the graph's most useful edges point the other way from the question being asked
(issue #86): a requirement ``REFERENCES`` the clause that states it, so from a
retrieved clause the obligations are *upstream*, and ``APPLIES_TO`` runs
requirement -> concept, so from a concept the requirements are upstream too. A
call names the edge types it wants followed backwards (``reverse_types``); every
safety property below holds identically in that direction, and a call that names
none walks exactly as it always did.
"""

import logging
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ids import is_entity_id
from app.schemas.edge import EdgeType

logger = logging.getLogger(__name__)


class TraversalHit(BaseModel):
    """One node reached by a graph walk, at its shortest distance from a seed.

    Attributes:
        unit_id: The id reached -- a unit (``doc#dieu-6``), a document
            (``doc``), or an entity (``req:...``, ``concept:...``, A5.1/A6
            Phase 2.4's scheme), whichever the followed edges lead to --
            forwards, or backwards for an edge type named in
            ``reverse_types``. Never a seed id itself; the walk only ever
            returns nodes reached by following at least one edge.
        hop: Fewest edges from any seed to reach this id.
        possibly_superseded: ``True`` when this hit's document carries an
            unreviewed (``pending``) ``REPLACED_BY`` edge. Not excluded --
            only an *approved* ``REPLACED_BY`` excludes -- but the caller
            should warn a reader before treating this hit as current law
            (Phase 3.3). Always ``False`` when ``include_history=True`` (the
            in-force check that produces this flag does not run) and always
            ``False`` for an entity hit (issue #58) -- an entity has no
            document of its own for the flag to describe.
    """

    model_config = ConfigDict(frozen=True)

    unit_id: str
    hop: int
    possibly_superseded: bool = False


_WALK_SQL = text(
    """
    WITH RECURSIVE walk AS (
      SELECT e.src_id, e.dst_id, e.edge_type, 1 AS hop, ARRAY[e.src_id, e.dst_id] AS path
      FROM edges e
      WHERE e.src_id = ANY(:seed_ids) AND e.edge_type = ANY(:follow_types)
        AND e.review_status='approved'
      UNION ALL
      SELECT e.src_id, e.dst_id, e.edge_type, w.hop+1, w.path || e.dst_id
      FROM edges e JOIN walk w ON e.src_id = w.dst_id
      WHERE w.hop < :max_hops AND e.edge_type = ANY(:follow_types)
        AND e.review_status='approved' AND NOT e.dst_id = ANY(w.path)   -- cycle guard
    )
    SELECT DISTINCT dst_id AS unit_id, MIN(hop) AS hop FROM walk GROUP BY dst_id
    """
)
"""The exact query from BUILD_PLAN.md § A6 Phase 2.5, unmodified -- including
the path-array cycle guard. ``BELONGS_TO`` cannot cycle (it is a tree), but a
reference edge can and will: a decree citing a circular that cites it back is
ordinary in this corpus, and ``NOT e.dst_id = ANY(w.path)`` is what stops the
recursion from looping forever once that data arrives. ``review_status =
'approved'`` is a literal in the SQL text, never a bind parameter -- there is
no argument on this module's public function that could reach it.

Used only when a call follows no edge type in reverse: a forward-only walk runs
this query byte for byte as it always has (issue #86). A walk that names
``reverse_types`` runs ``_DIRECTED_WALK_SQL`` instead."""

_DIRECTED_WALK_SQL = text(
    """
    WITH RECURSIVE walk AS (
      SELECT seed.id AS node, 0 AS hop, ARRAY[seed.id] AS path
      FROM unnest(CAST(:seed_ids AS text[])) AS seed(id)
      UNION ALL
      SELECT step.node, w.hop+1, w.path || step.node
      FROM walk w
      CROSS JOIN LATERAL (
        SELECT e.dst_id AS node FROM edges e
        WHERE e.src_id = w.node AND e.edge_type = ANY(:follow_types)
          AND e.review_status='approved'
        UNION ALL
        SELECT e.src_id AS node FROM edges e
        WHERE e.dst_id = w.node AND e.edge_type = ANY(:reverse_types)
          AND e.review_status='approved'
      ) step
      WHERE w.hop < :max_hops AND NOT step.node = ANY(w.path)   -- cycle guard
    )
    SELECT node AS unit_id, MIN(hop) AS hop FROM walk WHERE hop > 0 GROUP BY node
    """
)
"""``_WALK_SQL`` with a direction per edge type (issue #86), built to read as the
same query: one step of the walk is either an approved edge of a ``follow_types``
type taken ``src_id`` -> ``dst_id``, or an approved edge of a ``reverse_types``
type taken ``dst_id`` -> ``src_id``, and everything around that step is what the
forward query already does.

* ``review_status='approved'`` is a literal in *both* branches, never a bind
  parameter, so no argument can reach it in either direction.
* The cycle guard is the same path array, and it matters as much backwards:
  reference edges genuinely cycle in this corpus, and so does their reverse.
* Each branch is a lookup by ``src_id`` or ``dst_id`` under ``edges_src_type`` /
  ``edges_dst_type`` (``migrations/002_graph.sql``), per node reached, so a walk
  costs its degree and not a scan of ``edges``.
* The seeds are the ``hop = 0`` rows of the walk, and only rows with ``hop > 0``
  are returned, so a seed is never a hit of its own path -- including through a
  self-loop.

A type in both lists is followed both ways, which is the one way to ask for a
bidirectional walk: deliberately, per call, per type."""

_IN_FORCE_SQL = text(
    """
    SELECT d.id AS document_id,
           d.status AS status,
           COALESCE(bool_or(r.review_status = 'approved'), false) AS approved_replaced,
           COALESCE(bool_or(r.review_status = 'pending'), false) AS pending_replaced
    FROM documents d
    LEFT JOIN edges r ON r.src_id = d.id AND r.edge_type = 'replaced_by'
    WHERE d.id = ANY(:document_ids)
    GROUP BY d.id, d.status
    """
)
"""One batched lookup for every hit's owning document, run at most once per
``traverse`` call regardless of how many hits there are -- never one query
per hit. A5.2's in-force rule: a document is superseded when its own
``status`` is not ``in_force``, or an *approved* ``REPLACED_BY`` edge names
it as the source. A *pending* ``REPLACED_BY`` does neither -- ``status`` is
human-set data and an unapproved edge is only a machine guess, and silently
hiding a law that is actually still in force costs more than a warning --
but it is reported back so the caller can flag it."""


class GraphTraversalService:
    """Walks the citation graph from a set of seed nodes, approved edges only."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the service.

        Args:
            session: Session to run queries on. The caller owns its
                transaction (see ``app.core.db.session_scope``).
        """
        self._session = session

    async def traverse(
        self,
        seed_ids: Sequence[str],
        *,
        max_hops: int,
        follow_types: Sequence[EdgeType],
        include_history: bool = False,
        reverse_types: Sequence[EdgeType] = (),
    ) -> tuple[TraversalHit, ...]:
        """Walk the graph from ``seed_ids``, following only approved edges.

        **Direction is chosen per edge type, per call** (issue #86):
        ``follow_types`` are followed ``src_id`` -> ``dst_id``, as ever, and
        ``reverse_types`` are followed ``dst_id`` -> ``src_id``. The two lists
        together are the whole request, so one call can walk both ways:
        clause -> requirement -> concept is ``REFERENCES`` in reverse (a
        requirement references the clause that states it) and then
        ``APPLIES_TO`` forwards (a requirement applies to a concept), in one
        walk of two hops.

        Why a second parameter, and not a ``direction`` flag or a second
        method. A single flag for the whole call cannot express that two-hop
        walk, which is the shape of the product's central question, so the
        direction has to belong to the edge type. A separate method would put
        the safety-critical parts -- the approved-only literal, the cycle
        guard, the in-force filter, the entity-id handling -- behind two public
        entry points that have to be kept identical forever; here they are one
        code path, and only the SQL that takes a step differs. And the default
        is empty, so a walk that does not ask for reverse is exactly the walk it
        was: making every traversal bidirectional would pull in unrelated
        material and inflate hop counts on a graph where ``belongs_to`` alone
        is most of the edges (a reverse ``belongs_to`` from a document is the
        entire document).

        Exactly two queries run, regardless of ``max_hops`` or how many
        nodes the walk reaches: the recursive walk itself, and -- unless
        ``include_history`` skips it -- one batched in-force lookup for
        every distinct document the walk landed in.

        Args:
            seed_ids: Starting node ids (units or documents, A5.1 scheme).
            max_hops: Maximum edges to follow from a seed, enforced inside
                the recursive query itself, not by truncating results here.
            follow_types: Edge types to follow forwards. Must be ``EdgeType``
                members -- A5.2 prints edge types uppercase, the enum stores
                them lowercase, and a hand-written raw string would silently
                match zero rows instead of raising.
            reverse_types: Edge types to follow backwards -- from an edge's
                ``dst_id`` to its ``src_id`` -- under the same rule. A type
                listed in both is followed both ways. Every restriction below
                applies to what a reverse step reaches exactly as to a
                forward one: only ``approved`` edges are taken, the cycle
                guard covers the reverse path, and a reached node is judged by
                the in-force filter.
            include_history: When ``True``, skips the in-force filter
                entirely and returns every reached node, including one whose
                document is superseded -- for questions about superseded
                law. Every hit's ``possibly_superseded`` is ``False`` in
                this mode, since the check that sets it does not run.

        Returns:
            Every distinct node reached, at its shortest hop count, ordered
            by hop then id. Never includes a seed id itself. An entity hit
            (``req:...``, ``concept:...``, decided from the id's own prefix,
            never from a failed lookup) always passes the in-force filter
            unchanged, with ``include_history`` at either setting: a
            ``req:``/``concept:`` node has no ``status`` or effective date
            of its own for the filter to check -- that question belongs to
            the unit its own ``REFERENCES`` edge points at -- and dropping
            it instead would silently truncate a walk one hop short of the
            requirement cluster (issue #58). A non-entity hit whose document
            has no row in ``documents`` -- a dangling edge, since ``edges``
            carries no foreign key on either endpoint -- is excluded and
            logged as a warning, with ``include_history=False``: its status
            is unknown, and reporting an unknown-status id as current law is
            worse than a gap in the walk. ``include_history=True`` skips
            this check like every other, so a dangling edge still surfaces
            there for whoever is looking at history, not ordinary answers.
        """
        if not seed_ids or not (follow_types or reverse_types):
            return ()
        params: dict[str, object] = {
            "seed_ids": list(seed_ids),
            "follow_types": [edge_type.value for edge_type in follow_types],
            "max_hops": max_hops,
        }
        if reverse_types:
            params["reverse_types"] = [edge_type.value for edge_type in reverse_types]
            result = await self._session.execute(_DIRECTED_WALK_SQL, params)
        else:
            result = await self._session.execute(_WALK_SQL, params)
        hops_by_id = {row.unit_id: row.hop for row in result}
        if not hops_by_id:
            return ()
        if include_history:
            hits = [TraversalHit(unit_id=unit_id, hop=hop) for unit_id, hop in hops_by_id.items()]
        else:
            hits = await self._apply_in_force_filter(hops_by_id)
        return tuple(sorted(hits, key=lambda hit: (hit.hop, hit.unit_id)))

    async def _apply_in_force_filter(self, hops_by_id: dict[str, int]) -> list[TraversalHit]:
        """Drop a hit whose document is superseded; flag one pending review.

        Args:
            hops_by_id: Every reached node's shortest hop count, by id.

        Returns:
            Every hit whose document is currently in force, each carrying
            ``possibly_superseded`` when that document also has an
            unreviewed ``REPLACED_BY`` edge pointed at it. An entity hit
            (``req:...``, ``concept:...``, A6 Phase 2.4), identified by its
            own prefix via ``is_entity_id`` rather than by a failed lookup,
            passes through unchanged: it has no status or effective date of
            its own to check, so the in-force question does not apply to
            it, and dropping it would silently truncate the walk one hop
            short of the requirement cluster (issue #58). A non-entity hit
            whose document has no row in ``documents`` at all is a dangling
            edge -- ``edges`` carries no foreign key on either endpoint, so
            one can exist -- and its status is unknown, not in force;
            excluded, with a warning naming the id, rather than either
            raised (one bad edge must not take down every other answer the
            walk reaches) or returned (an unknown-status id must never be
            reported as current law).
        """
        document_ids = {
            unit_id.split("#", 1)[0] for unit_id in hops_by_id if not is_entity_id(unit_id)
        }
        result = await self._session.execute(_IN_FORCE_SQL, {"document_ids": list(document_ids)})
        document_rows = {row.document_id: row for row in result}

        hits: list[TraversalHit] = []
        for unit_id, hop in hops_by_id.items():
            if is_entity_id(unit_id):
                hits.append(TraversalHit(unit_id=unit_id, hop=hop))
                continue
            document = document_rows.get(unit_id.split("#", 1)[0])
            if document is None:
                logger.warning(
                    "traverse: %s names no document in the graph -- excluded as a "
                    "dangling edge, not reported as current law",
                    unit_id,
                )
                continue
            if document.status != "in_force" or document.approved_replaced:
                continue
            hits.append(
                TraversalHit(
                    unit_id=unit_id, hop=hop, possibly_superseded=document.pending_replaced
                )
            )
        return hits
