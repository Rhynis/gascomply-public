"""Hybrid retrieval: vector search, graph expansion, effective-date filter, rerank (issue #89).

Vector search alone finds the clause the question resembles. The graph finds what surrounds it:
the requirements a clause states, the other requirements of the same kind of business, the
article it cites. ``Retriever.retrieve`` does both and hands the generator one ranked context.

The steps, in order:

1. **Vector** -- the ``k`` chunks nearest the question (``ChunkRepository.search``), with the
   effective-date filter applied inside the query so a superseded chunk never takes one of the
   ``k`` places.
2. **Graph expansion** -- ``GraphTraversalService`` walks from those chunks, one walk per kind of
   relationship (below), so each walk's result is labelled with the reason it was reached without
   changing what a ``TraversalHit`` holds.
3. **Back to chunks and entities** -- a reached unit is mapped to the chunk that contains it and
   de-duplicated; a reached entity (``req:...``, ``concept:...``) has no chunk, so it is returned
   as an entity together with the units it cites as its legal basis, and those units' chunks join
   the passages so the fact can be quoted.
4. **Supersession** -- a returned unit that cites a retired document yields a ``Supersession``
   notice, and the in-force replacement's own text is searched for the same question.
5. **Focus expansion** -- every real unit the ``units`` table has nested under a returned
   chunk becomes a ``Passage.focus_unit``, whether or not any edge pointed at it (issue #106,
   below).
6. **Rerank** -- ``rerank.score``: similarity discounted by graph distance from a hit.
7. **Cross-encoder** -- optional (issue #114, below): the passages that survived step 6 are
   scored again, this time by reading each one *together with* the question, and only the best
   ``rerank_top_n`` are kept.

## Fewer, better passages: the cross-encoder (issue #114)

Retrieval already finds the right text (recall@k 0.900); what fails is the generator's model
choosing a different Điều out of too many (issue #108: 9 of 12 wrong citations). A cross-encoder
(``app.rerankers``) scores each (question, passage) pair jointly, which the embedding distance of
step 1 cannot, and everything below its top-N is removed *before* the model sees it -- along with
every ``focus_unit`` line that passage would have carried, which is most of the prompt.

It runs on the ``passage_limit`` survivors of step 6, after the graph has already chosen them:
widening that pool to thirty passages found no recall the ten had missed (a question has 13.5
candidates on average), so there is nothing to gain from scoring more of them and each one costs a
pair to score. Focus expansion (step 5) then runs on the survivors only, so a
dropped passage's clauses stop being citable, which is the point.

It is optional twice over. No reranker is configured by default. And a reranker that fails
(``RerankError``: sidecar down, timed out, a body that is not scores) is logged and skipped, so
``/ask`` answers exactly as it did before this stage existed instead of failing.

## A clause inside a returned chunk is citable even if no edge points at it (issue #106)

Before ``_expand_focus``, ``focus_units`` held only what the graph walk reached
(``_add_unit``), so a clause no requirement or reference edge happened to point at never
became a ``focus_unit`` even when its text was already part of a returned chunk -- the model
would see the clause, cite it correctly, and ``generator._validate_citations`` would reject
that citation as fabricated, because ``generator._build_sources`` never offered it.

The fix is here, not in ``generator.py``: ``eval/compare.py``'s own measurement of recall
(``_citable_unit_ids``) reads a ``RetrievalResult``'s ``focus_units``/entity ``sources``
directly and never calls into ``generator`` at all, so a fix confined to citation-time
validation would leave that number exactly where it was -- correct behaviour with no visible
effect on the trusted measurement. Expanding ``focus_units`` here means the eval harness sees
the same improvement the product does, with no change to ``eval/`` itself, and it gives the
model a clearly labelled, addressable source for the clause rather than an unstated allowance
buried in a validator.

The anti-fabrication guarantee (#93) is unweakened by construction: ``_expand_focus`` only
ever adds an id it read back from the ``units`` table, so an id that merely has the right
*shape* (e.g. a clause number past the end of the real article) is never added -- there is no
row to find. And it only adds ids nested under a chunk that is *already* a candidate, i.e.
already a passage whose text the model is shown -- it never makes a new chunk citable, only a
chunk's own real substructure. ``eval.metrics._unit_contains`` decides "nested" here, the
identical function ``eval``'s own containment recall is measured by, imported rather than
reimplemented so the two definitions of "the text was in the context" cannot drift apart.

## Descending into a reached container with no chunk of its own (issue #136)

The graph can reach a unit that has no chunk of its own but does have chunked descendants --
the real case is an appendix: ``Điều 35 Khoản 1`` cites ``Phụ lục VII`` (forward ``REFERENCES``),
``ingest/chunk.py`` chunks each item of an appendix and never the appendix container itself, so
``#phu-luc-7`` is reached but ``_containing_chunk`` finds nothing to show -- before this, the
unit was simply dropped and its 44 items never reached the prompt, no matter how decisive one of
them was (``#phu-luc-7#muc-15``'s 500 kg threshold, left out of the insurance-requirement
answer).

``_chunk_pool`` answers this *before* ``_assemble`` runs: for every unit the graph walk reached
that names no chunk anywhere in its own-and-ancestors chain, it looks up every real descendant
``units`` has (any depth, by id prefix) that does have a current chunk, scores those descendants
against the question's embedding -- already computed, so no extra model call -- and keeps them
in similarity order until ``config.descend_char_budget`` characters of their text would be
exceeded (issue #140; #138's fixed top-3 cut a 44-item appendix down to 3 and lost the item that
answered the question -- a character budget lets a container of many short items come in nearly
whole while still bounding a container of few long ones). ``_add_unit`` then offers each kept
descendant chunk as its own passage candidate, at the container's hop and why, instead of the
container itself (which still has nothing to show).

This only ever fires for a container reached *through the graph*: the vector-hit path
(``_assemble``'s loop over ``hits``) never calls ``_add_unit`` at all, so an appendix that is
itself a vector hit -- one of the container chunks #134 introduced for a form appendix -- behaves
exactly as before. And it only ever fires when the container truly has no chunk of its own
(``_containing_chunk`` found nothing): a container #134 gave its own chunk is shown at that
chunk, same as any other reached unit, descent never runs for it.

## Direction, edge type by edge type

``traverse`` follows a type forwards (``follow_types``), backwards (``reverse_types``), or both
(named in each). The direction is chosen here, deliberately, because the wrong choice is the
difference between 11 relevant hits and 83 irrelevant ones.

* ``BELONGS_TO`` (child -> parent): **backwards only**, to descend from a hit to the clauses and
  points inside it. Never forwards. A requirement points at a *clause*, while the vector search
  returns a whole *article*, so the walk must descend to meet it. Named in both lists, the same
  walk would also climb to the chapter and come back down to every sibling article -- the whole
  chapter, measured at 83 hits from Điều 53 -- text that shares a parent with the hit and nothing
  else. Descending returns only what is inside the hit, which is already in the hit's own text.
  It is walked first and counted as no distance at all: a chunk *is* the text of its clauses.
* ``REFERENCES`` (referrer -> referenced): **forwards**, up to ``cite_hops``, for what a hit cites
  (the legal basis, the article it sends the reader to: the multi-hop case). **Backwards**, one
  hop only, for what cites a hit -- a requirement points at the clause that states it, so this
  is how a clause reaches its requirements. One hop because it fans out: the next step backwards
  would reach units that merely cite the same thing as the hit.
* ``APPLIES_TO`` (requirement -> concept): **both ways**, from the requirements found, two hops.
  Forwards reaches the concept, backwards from there reaches its other requirements: the whole
  obligation set of a kind of business, including requirements whose clauses lie in articles the
  vector search never returned. A concept is a curated hub with a small membership, unlike a
  ``BELONGS_TO`` parent; walking it both ways is what it is for.
* ``BASED_ON``, ``GUIDED_BY``, ``AMENDED_BY``: **not walked**. Their sources are documents, and a
  chunk seed is a unit, so from a unit they lead nowhere without a unit-to-document step this
  service does not take (see the issue for the measured effect on document-level questions).
* ``REPLACED_BY``: not walked as expansion. It is read as supersession (step 4), where what matters
  is which clause declares the repeal, which a walk cannot report.

Only approved edges are ever followed: every walk goes through ``GraphTraversalService``, whose
approved-only rule is a literal in its SQL, and the direct edge lookups here carry the same literal.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ids import EntityType, is_entity_id, unit_and_ancestors
from app.embedders.base import Embedder
from app.repositories.chunks import ChunkRepository
from app.repositories.edges import EdgeRepository, RetiredCitation
from app.repositories.entities import EntityRepository
from app.repositories.units import UnitRepository
from app.rerankers.base import Reranker, RerankError
from app.schemas.chunk import ChunkMatch
from app.schemas.document import DocStatus
from app.schemas.edge import EdgeType
from app.schemas.entity import Entity
from app.schemas.retrieval import (
    Passage,
    RetrievalConfig,
    RetrievalResult,
    RetrievedEntity,
    Supersession,
    Why,
)
from app.services.graph_traversal import GraphTraversalService, TraversalHit
from app.services.rerank import Candidate, Ranked, rank
from eval.metrics import _unit_contains

logger = logging.getLogger(__name__)

_CONTAINMENT_HOPS = 2
"""Article -> clause -> point: the deepest containment the unit tree has."""

_REQUIREMENT_HOP = 1
"""A requirement is one edge (a backwards ``REFERENCES``) from the clause that states it."""

_CONCEPT_HOPS = 2
"""Requirement -> concept -> the concept's other requirements."""

_REPLACEMENT_CHAIN_HOPS = 5
"""How many ``REPLACED_BY`` steps are followed to reach the replacement that is in force."""

_SUCCESSOR_HOPS = 2
"""Citing unit -> retired document (``REFERENCES``) -> its replacement (``REPLACED_BY``)."""


@dataclass
class _Found:
    """A node a walk reached: at what distance from a hit, and by which routes."""

    hop: int
    why: set[Why] = field(default_factory=set)


@dataclass
class _Walk:
    """Everything graph expansion reached, kept apart by what it is.

    Attributes:
        frontier: The hit units and every unit inside them -- what the walks start from.
        units: Reached units (not the frontier) by id.
        entities: Reached entities by id.
    """

    frontier: list[str]
    units: dict[str, _Found] = field(default_factory=dict)
    entities: dict[str, _Found] = field(default_factory=dict)

    def add(self, hit: TraversalHit, why: Why, *, base: int = 0) -> None:
        """Record ``hit``, ``base`` edges further than the walk that found it started.

        A bare document id (no ``#``, no entity prefix) is dropped: a document has no text to
        return, and a stub is excluded by the walk's own in-force filter anyway.
        """
        if is_entity_id(hit.unit_id):
            bucket = self.entities
        elif "#" in hit.unit_id:
            bucket = self.units
        else:
            return
        found = bucket.setdefault(hit.unit_id, _Found(hop=base + hit.hop))
        found.hop = min(found.hop, base + hit.hop)
        found.why.add(why)


@dataclass(frozen=True)
class _Retired:
    """A retired document a returned unit cites, and where its replacement sits in the graph.

    Attributes:
        document: The retired document.
        replacements: The in-force documents at the end of its ``REPLACED_BY`` chain.
        hop: Distance from a hit to the replacement: the citing unit's, plus the two edges.
    """

    document: str
    replacements: tuple[str, ...]
    hop: int


@dataclass
class _Assembly:
    """The candidates and side information the ranker and the result builder need.

    Attributes:
        candidates: Every passage candidate, one per route to a chunk.
        focus: Chunk id -> the units inside it the graph pointed at.
        replaces: Chunk id -> the retired documents its document replaces.
        unit_hops: Every unit considered -> its distance from a hit, for the supersession check.
    """

    candidates: list[Candidate] = field(default_factory=list)
    focus: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    replaces: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    unit_hops: dict[str, int] = field(default_factory=dict)

    def note_unit(self, unit_id: str, hop: int) -> None:
        """Remember that ``unit_id`` is in play, at its nearest distance."""
        self.unit_hops[unit_id] = min(hop, self.unit_hops.get(unit_id, hop))


class Retriever:
    """Answers a question with passages, entities and supersession notices.

    Attributes:
        rerank_ok: How many retrievals the cross-encoder reranked successfully.
        rerank_fallbacks: How many retrievals kept the graph-proximity order because the
            cross-encoder failed. A reader of a measurement must be able to tell a run that
            reranked from a run that silently did not -- ``eval.compare`` prints both.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            session: Session to read through. The caller owns its transaction. Retrieval only
                ever reads.
            embedder: Embeds the question. ``MockEmbedder`` in tests, ``BgeM3Embedder`` for real.
            config: Tunables; the defaults if omitted.
            reranker: Cross-encoder that reorders and cuts the passages (issue #114). ``None``,
                the default, skips the stage.
        """
        self._traversal = GraphTraversalService(session)
        self._chunks = ChunkRepository(session)
        self._entities = EntityRepository(session)
        self._edges = EdgeRepository(session)
        self._units = UnitRepository(session)
        self._embedder = embedder
        self._config = config or RetrievalConfig()
        self._reranker = reranker
        self.rerank_ok = 0
        self.rerank_fallbacks = 0

    async def vector_search(
        self, question: str, *, include_history: bool = False
    ) -> list[ChunkMatch]:
        """Return the ``k`` chunks nearest ``question`` -- vector retrieval alone (Phase 3.1).

        What ``retrieve`` starts from, exposed so the value of the graph can be measured against
        it: the same ``k``, the same filter, none of the expansion.

        Args:
            question: The question, in Vietnamese.
            include_history: Keep chunks of superseded documents, flagged with ``replaced_by``.

        Returns:
            Matches, most similar first, at most ``config.k``.

        Raises:
            ValueError: If ``question`` is blank.
        """
        vector = await self._embed(question)
        return await self._chunks.search(
            vector, limit=self._config.k, include_history=include_history
        )

    async def retrieve(self, question: str, *, include_history: bool = False) -> RetrievalResult:
        """Retrieve passages, entities and supersession notices for ``question``.

        Args:
            question: The question, in Vietnamese.
            include_history: When ``True``, chunks of superseded documents are kept and flagged
                (``Passage.superseded``, ``superseded_by``) instead of dropped, for a question
                about law that has been replaced. Supersession notices are produced either way.

        Returns:
            The ranked context. Empty when the corpus has nothing current to match.

        Raises:
            ValueError: If ``question`` is blank.
        """
        vector = await self._embed(question)
        hits = await self._chunks.search(
            vector, limit=self._config.k, include_history=include_history
        )
        if not hits:
            return RetrievalResult(question=question, passages=())
        walk = await self._expand([hit.unit_id for hit in hits], include_history)
        entities = {entity.id: entity for entity in await self._load_entities(walk)}
        sources = await self._edges.approved_targets(list(entities), EdgeType.REFERENCES)
        pool = await self._chunk_pool(vector, hits, walk, sources, include_history)

        assembly = self._assemble(hits, walk, pool)
        entity_candidates = self._entity_candidates(walk, entities, sources, pool, assembly)
        supersessions = await self._supersessions(vector, assembly, pool)

        ranked_passages = rank(assembly.candidates, self._config.decay)[
            : self._config.passage_limit
        ]
        ranked_entities = rank(entity_candidates, self._config.decay)[: self._config.entity_limit]
        ranked_passages, rerank_scores = await self._rerank_passages(
            question, ranked_passages, pool
        )
        await self._expand_focus(assembly, pool, [item.candidate.key for item in ranked_passages])
        return RetrievalResult(
            question=question,
            passages=tuple(
                self._passage(
                    item, pool[item.candidate.key], assembly, rerank_scores.get(item.candidate.key)
                )
                for item in ranked_passages
            ),
            entities=tuple(
                self._entity(item, entities[item.candidate.key], sources)
                for item in ranked_entities
            ),
            supersessions=tuple(supersessions),
        )

    async def _rerank_passages(
        self, question: str, ranked: list[Ranked], pool: dict[str, ChunkMatch]
    ) -> tuple[list[Ranked], dict[str, float]]:
        """Reorder ``ranked`` by cross-encoder score and keep the best ``rerank_top_n``.

        The sort is stable, so passages the cross-encoder scores alike keep the order the graph
        gave them. A failed reranker (``RerankError``) leaves ``ranked`` exactly as it came in:
        the caller then behaves as it did before this stage existed. One passage has nothing to
        be ranked against, so it is not sent.

        Args:
            question: The question the passages were retrieved for.
            ranked: The passages that survived graph-proximity ranking, best first.
            pool: Every chunk in play, for each passage's text.

        Returns:
            The passages to keep, best first, and every scored passage's score by unit id
            (empty when nothing was reranked).
        """
        if self._reranker is None or len(ranked) < 2:
            return ranked, {}
        texts = [pool[item.candidate.key].body for item in ranked]
        try:
            scores = await self._reranker.score(question, texts)
        except RerankError as error:
            self.rerank_fallbacks += 1
            logger.warning(
                "retrieve: reranking failed, keeping the graph-proximity order: %s", error
            )
            return ranked, {}
        self.rerank_ok += 1
        by_key = {item.candidate.key: score for item, score in zip(ranked, scores, strict=True)}
        best_first = sorted(ranked, key=lambda item: -by_key[item.candidate.key])
        return best_first[: self._config.rerank_top_n], by_key

    async def _embed(self, question: str) -> list[float]:
        """Embed ``question`` off the event loop's critical path (the embedder is a coroutine)."""
        if not question.strip():
            msg = "question must not be blank"
            raise ValueError(msg)
        [vector] = await self._embedder.embed([question])
        return vector

    # --- graph expansion --------------------------------------------------------------------

    async def _expand(self, hit_ids: Sequence[str], include_history: bool) -> _Walk:
        """Walk the graph from the vector hits, one walk per kind of relationship.

        The direction of each walk, and why, is the module docstring's table; each walk's reason
        is what it is named for.
        """
        inside = await self._walk(
            hit_ids, _CONTAINMENT_HOPS, include_history, reverse=(EdgeType.BELONGS_TO,)
        )
        frontier = list(dict.fromkeys([*hit_ids, *(hit.unit_id for hit in inside)]))
        walk = _Walk(frontier=frontier)
        cited = await self._walk(
            frontier, self._config.cite_hops, include_history, follow=(EdgeType.REFERENCES,)
        )
        citing = await self._walk(frontier, 1, include_history, reverse=(EdgeType.REFERENCES,))
        for hit in cited:
            walk.add(hit, Why.CITED_BY_HIT)
        for hit in citing:
            walk.add(hit, Why.CITES_HIT)
        requirements = list(walk.entities)
        related = await self._walk(
            requirements,
            _CONCEPT_HOPS,
            include_history,
            follow=(EdgeType.APPLIES_TO,),
            reverse=(EdgeType.APPLIES_TO,),
        )
        for hit in related:
            walk.add(hit, Why.SAME_CONCEPT, base=_REQUIREMENT_HOP)
        return walk

    async def _walk(
        self,
        seeds: Sequence[str],
        max_hops: int,
        include_history: bool,
        *,
        follow: Sequence[EdgeType] = (),
        reverse: Sequence[EdgeType] = (),
    ) -> tuple[TraversalHit, ...]:
        """One ``GraphTraversalService`` walk; approved edges only, and only in these directions."""
        return await self._traversal.traverse(
            seeds,
            max_hops=max_hops,
            follow_types=follow,
            reverse_types=reverse,
            include_history=include_history,
        )

    # --- entities and chunks ------------------------------------------------------------------

    async def _load_entities(self, walk: _Walk) -> list[Entity]:
        """Load the entities the walk reached, warning about an id with no row.

        ``edges`` has no foreign key to ``entities``, so a walk can reach an id that was never
        inserted. Such an id has no label to show and is left out rather than invented.
        """
        found = await self._entities.get_many(list(walk.entities))
        missing = set(walk.entities) - {entity.id for entity in found}
        for entity_id in sorted(missing):
            logger.warning("retrieve: %s names no row in entities -- left out", entity_id)
        return found

    async def _chunk_pool(
        self,
        vector: list[float],
        hits: Sequence[ChunkMatch],
        walk: _Walk,
        sources: dict[str, tuple[str, ...]],
        include_history: bool,
    ) -> dict[str, ChunkMatch]:
        """Return every chunk in play by unit id: the hits, and the chunks holding what was reached.

        A reached unit rarely has a chunk of its own (a clause of a short article does not), so
        every unit is looked up under itself *and* each unit enclosing it, and
        ``_containing_chunk`` takes the deepest that exists. The lookup measures each chunk against
        the question and applies the effective-date filter, so a graph-reached chunk of a superseded
        document is not in the pool.

        A reached unit with no chunk anywhere in that chain (a container -- issue #136) gets its
        best chunked descendants fetched by ``_descend`` instead, so it is never silently dropped
        just because its own text was never chunked.
        """
        wanted = set(chain(walk.units, *sources.values()))
        known = {hit.unit_id for hit in hits}
        ids = {ancestor for unit in wanted for ancestor in unit_and_ancestors(unit)} - known
        scored = await self._chunks.score(vector, sorted(ids), include_history=include_history)
        pool = {match.unit_id: match for match in [*hits, *scored]}
        containers = [
            unit_id
            for unit_id in walk.units
            if not any(ancestor in pool for ancestor in unit_and_ancestors(unit_id))
        ]
        if containers:
            for unit_id, match in (
                await self._descend(vector, containers, include_history)
            ).items():
                pool.setdefault(unit_id, match)
        return pool

    @staticmethod
    def _containing_chunk(unit_id: str, pool: dict[str, ChunkMatch]) -> ChunkMatch | None:
        """Return the chunk that holds ``unit_id``'s text: the deepest enclosing one in ``pool``."""
        return next(
            (pool[ancestor] for ancestor in unit_and_ancestors(unit_id) if ancestor in pool), None
        )

    async def _descend(
        self, vector: list[float], containers: Sequence[str], include_history: bool
    ) -> dict[str, ChunkMatch]:
        """Offer each ``containers`` id's chunked descendants, up to a character budget (#140).

        Called only for a unit the graph reached that has no chunk anywhere in its own-and-
        ancestors chain -- the caller (``_chunk_pool``) has already checked that. Ranked by each
        descendant's own chunk embedding against ``vector``, the question's embedding, which is
        already computed: no extra model call. A container with no chunked descendant at all (a
        truly empty reach) contributes nothing.

        Args:
            vector: The question's embedding.
            containers: Unit ids the graph reached that have no chunk of their own.
            include_history: As for ``_chunk_pool``.

        Returns:
            Chunk by unit id, for every descendant kept, across every container.
        """
        by_document = await self._units.list_ids_by_documents(
            sorted({container.split("#", 1)[0] for container in containers})
        )
        found: dict[str, ChunkMatch] = {}
        for container in containers:
            document_id = container.split("#", 1)[0]
            descendants = sorted(
                unit_id
                for unit_id in by_document.get(document_id, ())
                if unit_id != container and _unit_contains(container, unit_id)
            )
            if not descendants:
                continue
            scored = await self._chunks.score(vector, descendants, include_history=include_history)
            ranked = sorted(scored, key=lambda match: -match.similarity)
            for match in self._within_budget(ranked):
                found.setdefault(match.unit_id, match)
        return found

    def _within_budget(self, ranked: Sequence[ChunkMatch]) -> list[ChunkMatch]:
        """Take ``ranked`` (best first) until ``config.descend_char_budget`` would be exceeded.

        The first match is always kept, even alone over budget, so a container with one long
        descendant is never emptied just because that descendant does not fit -- the budget
        bounds what is added *after* it, not whether anything is offered at all.
        """
        kept: list[ChunkMatch] = []
        spent = 0
        for match in ranked:
            if kept and spent + len(match.body) > self._config.descend_char_budget:
                break
            kept.append(match)
            spent += len(match.body)
        return kept

    # --- candidates -----------------------------------------------------------------------------

    def _assemble(
        self, hits: Sequence[ChunkMatch], walk: _Walk, pool: dict[str, ChunkMatch]
    ) -> _Assembly:
        """Turn the hits and the units the walk reached into passage candidates."""
        assembly = _Assembly()
        for hit in hits:
            assembly.candidates.append(
                Candidate(hit.unit_id, hit.similarity, 0, frozenset({Why.SIMILAR}))
            )
        for unit_id in walk.frontier:
            assembly.note_unit(unit_id, 0)
        for unit_id, found in walk.units.items():
            assembly.note_unit(unit_id, found.hop)
            self._add_unit(assembly, pool, unit_id, found.hop, frozenset(found.why))
        return assembly

    def _add_unit(
        self,
        assembly: _Assembly,
        pool: dict[str, ChunkMatch],
        unit_id: str,
        hop: int,
        why: frozenset[Why],
    ) -> None:
        """Add the chunk holding ``unit_id`` as a candidate; note the unit if it is inside it.

        ``unit_id`` names no chunk anywhere in its own-and-ancestors chain when it is a
        container the graph reached with no chunk of its own (issue #136) -- an appendix, most
        often. ``_chunk_pool`` has already fetched its best chunked descendants into ``pool`` in
        that case, and each of those is offered here as its own candidate, at this same hop and
        why, instead of the container (which still has nothing to show).
        """
        chunk = self._containing_chunk(unit_id, pool)
        if chunk is not None:
            assembly.candidates.append(Candidate(chunk.unit_id, chunk.similarity, hop, why))
            if unit_id != chunk.unit_id:
                assembly.focus[chunk.unit_id].add(unit_id)
            return
        for descendant in self._descended_chunks(unit_id, pool):
            assembly.candidates.append(
                Candidate(descendant.unit_id, descendant.similarity, hop, why)
            )

    @staticmethod
    def _descended_chunks(container_id: str, pool: dict[str, ChunkMatch]) -> list[ChunkMatch]:
        """Return every chunk in ``pool`` nested under ``container_id`` -- issue #136's descent.

        ``pool`` only ever holds a descendant of a containerless unit because ``_chunk_pool``
        put it there (``_descend``), already cut to ``config.descend_char_budget`` and already
        filtered to real, chunked descendants -- so no further ranking or limiting happens here.
        """
        prefix = container_id + "#"
        return [chunk for key, chunk in pool.items() if key.startswith(prefix)]

    async def _expand_focus(
        self, assembly: _Assembly, pool: dict[str, ChunkMatch], chunk_ids: Sequence[str]
    ) -> None:
        """Add every real unit ``units`` has nested under each of ``chunk_ids`` to its focus set.

        Run once, on the chunks that survived ranking and are about to be returned as passages
        (never on every candidate considered along the way, which would just be wasted queries
        on chunks nobody will see). See the module docstring's "issue #106" section for why this
        step exists and why it cannot make a citation to a non-existent or unseen unit valid:
        an id is only ever added here because ``units`` itself named it, under a chunk that was
        already going to be shown to the model.

        Args:
            assembly: Mutated in place -- ``assembly.focus`` gains the ids this finds.
            pool: Every chunk in play, keyed by unit id, for each candidate's document id.
            chunk_ids: The final, ranked passage unit ids to expand focus for.
        """
        documents = {pool[chunk_id].document_id for chunk_id in chunk_ids if chunk_id in pool}
        if not documents:
            return
        by_document = await self._units.list_ids_by_documents(sorted(documents))
        for chunk_id in chunk_ids:
            chunk = pool.get(chunk_id)
            if chunk is None:
                continue
            for unit_id in by_document.get(chunk.document_id, ()):
                if unit_id != chunk_id and _unit_contains(chunk_id, unit_id):
                    assembly.focus[chunk_id].add(unit_id)

    def _entity_candidates(
        self,
        walk: _Walk,
        entities: dict[str, Entity],
        sources: dict[str, tuple[str, ...]],
        pool: dict[str, ChunkMatch],
        assembly: _Assembly,
    ) -> list[Candidate]:
        """Score each reached entity by its legal basis, and add that basis' chunks to the passages.

        An entity has no embedding, so its similarity is its source's: an obligation is as relevant
        as the clause that states it. One whose every source is gone -- the source's document is
        superseded, or names no chunk -- is dropped, and so is a requirement that cites nothing:
        with no clause to cite it cannot be grounded. A concept cites nothing by nature; it takes
        the best similarity of the requirements found with it, and is dropped if none survived.
        """
        candidates: list[Candidate] = []
        concepts: list[tuple[str, _Found]] = []
        for entity_id, found in walk.entities.items():
            entity = entities.get(entity_id)
            if entity is None:
                continue
            if entity.entity_type is EntityType.CONCEPT:
                concepts.append((entity_id, found))
                continue
            similarity = self._cite_sources(
                assembly, pool, sources.get(entity_id, ()), found.hop + 1
            )
            if similarity is not None:
                candidates.append(Candidate(entity_id, similarity, found.hop, frozenset(found.why)))
        if candidates:
            best = max(candidate.similarity for candidate in candidates)
            candidates.extend(
                Candidate(entity_id, best, found.hop, frozenset(found.why))
                for entity_id, found in concepts
            )
        return candidates

    def _cite_sources(
        self,
        assembly: _Assembly,
        pool: dict[str, ChunkMatch],
        sources: Sequence[str],
        hop: int,
    ) -> float | None:
        """Add the chunks holding an entity's sources as passages; return their best similarity.

        Returns ``None`` when no source has a current chunk, which is what drops the entity.
        """
        similarities: list[float] = []
        for source in sources:
            chunk = self._containing_chunk(source, pool)
            if chunk is None:
                continue
            assembly.note_unit(source, hop)
            self._add_unit(assembly, pool, source, hop, frozenset({Why.SOURCE_OF_REQUIREMENT}))
            similarities.append(chunk.similarity)
        return max(similarities, default=None)

    # --- supersession ---------------------------------------------------------------------------

    async def _supersessions(
        self, vector: list[float], assembly: _Assembly, pool: dict[str, ChunkMatch]
    ) -> list[Supersession]:
        """Report each retired document a returned unit cites, and fetch what replaced it.

        For every approved ``REFERENCES`` edge from a unit in play to a document that was retired,
        the notice names the document, its in-force replacement (``REPLACED_BY`` followed to its
        end, so a chain of replacements resolves), and the clause that declares the repeal. The
        replacement's own text is searched for the same question and added to the passages,
        flagged ``Why.SUCCESSOR`` -- so a reader who cited a repealed decree is handed the decree
        in force, not just told there is one.
        """
        citations = await self._edges.retired_citations(sorted(assembly.unit_hops))
        by_document: dict[str, list[RetiredCitation]] = defaultdict(list)
        for citation in citations:
            by_document[citation.retired_document].append(citation)
        notices: list[Supersession] = []
        for retired, rows in sorted(by_document.items()):
            current = await self._in_force_replacements(retired)
            declaring = next((row for row in rows if row.declared_in is not None), rows[0])
            notices.append(
                Supersession(
                    retired_document=retired,
                    replaced_by=current,
                    declared_in=declaring.declared_in,
                    evidence=declaring.evidence,
                    cited_by=tuple(sorted({row.citing_unit for row in rows})),
                )
            )
            hop = min(assembly.unit_hops[row.citing_unit] for row in rows) + _SUCCESSOR_HOPS
            await self._add_successors(vector, assembly, pool, _Retired(retired, current, hop))
        return notices

    async def _in_force_replacements(self, retired: str) -> tuple[str, ...]:
        """Return the documents in force at the end of ``retired``'s ``REPLACED_BY`` chain."""
        reached = await self._walk(
            [retired], _REPLACEMENT_CHAIN_HOPS, False, follow=(EdgeType.REPLACED_BY,)
        )
        return tuple(sorted(hit.unit_id for hit in reached if "#" not in hit.unit_id))

    async def _add_successors(
        self,
        vector: list[float],
        assembly: _Assembly,
        pool: dict[str, ChunkMatch],
        retired: _Retired,
    ) -> None:
        """Search the replacements for the question and add what is found as successor passages."""
        if not retired.replacements:
            return
        found = await self._chunks.search(
            vector, limit=self._config.successor_k, document_ids=retired.replacements
        )
        for match in found:
            pool.setdefault(match.unit_id, match)
            assembly.candidates.append(
                Candidate(match.unit_id, match.similarity, retired.hop, frozenset({Why.SUCCESSOR}))
            )
            assembly.replaces[match.unit_id].add(retired.document)

    # --- result ---------------------------------------------------------------------------

    @staticmethod
    def _passage(
        item: Ranked, chunk: ChunkMatch, assembly: _Assembly, rerank_score: float | None
    ) -> Passage:
        """Build the ``Passage`` for one ranked chunk, with its cross-encoder score if any."""
        candidate = item.candidate
        return Passage(
            unit_id=chunk.unit_id,
            document_id=chunk.document_id,
            dieu_no=chunk.dieu_no,
            khoan_no=chunk.khoan_no,
            effective=chunk.effective,
            text=chunk.body,
            similarity=chunk.similarity,
            hop=candidate.hop,
            score=item.score,
            rerank_score=rerank_score,
            why=_ordered(candidate.why),
            focus_units=tuple(sorted(assembly.focus.get(chunk.unit_id, ()))),
            superseded=chunk.status is not DocStatus.IN_FORCE or bool(chunk.replaced_by),
            superseded_by=chunk.replaced_by,
            possibly_superseded=chunk.possibly_superseded,
            replaces=tuple(sorted(assembly.replaces.get(chunk.unit_id, ()))),
        )

    @staticmethod
    def _entity(
        item: Ranked, entity: Entity, sources: dict[str, tuple[str, ...]]
    ) -> RetrievedEntity:
        """Build the ``RetrievedEntity`` for one ranked entity."""
        candidate = item.candidate
        return RetrievedEntity(
            entity_id=entity.id,
            entity_type=entity.entity_type,
            label_vi=entity.label_vi,
            meta=entity.meta,
            similarity=candidate.similarity,
            hop=candidate.hop,
            score=item.score,
            why=_ordered(candidate.why),
            sources=sources.get(entity.id, ()),
        )


def _ordered(why: frozenset[Why]) -> tuple[Why, ...]:
    """Return ``why`` in the order ``Why`` declares, so the same routes always print alike."""
    return tuple(reason for reason in Why if reason in why)
