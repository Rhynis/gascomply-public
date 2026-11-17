"""Rerank: vector similarity combined with graph proximity (build plan A6 Phase 3.4, issue #89).

    score = similarity * decay ** hop

``similarity`` is how near the item's text is to the question -- the cosine similarity of a
chunk's embedding, which every chunk has a real value of (``ChunkRepository.score`` measures
the ones the vector search did not itself return). ``hop`` is how many graph edges separate the
item from a chunk the vector search chose, counting no edge for containment: an article chunk
*is* the text of its clauses, so a clause inside a hit is the hit, at hop 0. ``decay`` is the price
of indirection.

The formula is small on purpose, and what it does is fixed by a few properties, each tested:

* A hit is scored by its similarity alone (``hop = 0``). The graph never penalises a hit and,
  because every other item scores ``similarity * decay ** hop`` with ``hop >= 1``, the graph can
  never lift an item above a hit it is not at least as similar to.
* **The vector top-1 always stays first.** Anything that is not a hit ranked below the k-th
  hit's similarity, which is at most the top-1's; discounted by ``decay < 1`` it is strictly
  lower. This is what "so the on-point clause stays first" needs, and it holds for every
  ``decay`` in ``(0, 1)``, so no weight can be tuned into breaking it.
* Items *derived* from a hit -- a requirement stated by a clause inside it, whose similarity is
  its source chunk's, so equal to the hit's -- score ``decay ** hop`` of the hit. That is what
  lets the obligation set of the best hit outrank the weakest hits: with ``decay = 0.9`` a
  requirement one edge from a hit of similarity 0.74 scores 0.67, above hits scoring less.
  A chunk that only the graph reached scores below every hit (its similarity is at most the
  k-th hit's, and the discount applies), so it never displaces text the question itself found.

What ``decay`` does not do: change the order among items at the same hop. It sets how far down a
list the graph's contribution sits, not which of it is on point; what is on point is
``similarity``, which is why a graph-reached chunk is scored by its own embedding rather than
inheriting its neighbour's.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from app.schemas.retrieval import Why


@dataclass(frozen=True)
class Candidate:
    """One thing that may be returned, with what is known about how relevant it is.

    Attributes:
        key: What identifies it: a chunk's unit id, or an entity id. Candidates with the same
            key are one item reached by several routes.
        similarity: Cosine similarity to the question, or the similarity an entity inherits from
            its source.
        hop: Fewest edges from a vector hit, containment not counted.
        why: Every route that reached it.
    """

    key: str
    similarity: float
    hop: int
    why: frozenset[Why]


@dataclass(frozen=True)
class Ranked:
    """A ``Candidate`` with its score."""

    candidate: Candidate
    score: float


def score(similarity: float, hop: int, decay: float) -> float:
    """Return ``similarity * decay ** hop``, with a negative similarity counted as zero.

    Args:
        similarity: Cosine similarity of the item to the question.
        hop: Graph distance from a vector hit; ``0`` for a hit itself.
        decay: Factor kept per hop, in ``(0, 1]``.

    Returns:
        The score. A chunk whose embedding points *away* from the question has nothing to
        discount -- a negative number times a decay would rise with distance, the opposite of
        the point -- so it scores ``0.0``.
    """
    return max(similarity, 0.0) * decay**hop


def merge(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Collapse candidates with the same key into one, keeping every route and the nearest hop.

    Args:
        candidates: Candidates, possibly several per key.

    Returns:
        One candidate per key: its best (lowest) hop, the union of every ``why``, and the largest
        similarity offered (a chunk's is one number, so they agree; an entity's may be offered
        by several sources).
    """
    merged: dict[str, Candidate] = {}
    for candidate in candidates:
        seen = merged.get(candidate.key)
        merged[candidate.key] = (
            candidate
            if seen is None
            else Candidate(
                key=candidate.key,
                similarity=max(seen.similarity, candidate.similarity),
                hop=min(seen.hop, candidate.hop),
                why=seen.why | candidate.why,
            )
        )
    return list(merged.values())


def rank(candidates: Iterable[Candidate], decay: float) -> list[Ranked]:
    """Merge duplicates, score every candidate, and order best first.

    Ties -- equal scores are the norm for the requirements of one clause, which inherit the same
    similarity -- break by nearer hop, then by key, so the same inputs always give the same list.

    Args:
        candidates: Candidates, possibly several per key.
        decay: The proximity factor, in ``(0, 1]``.

    Returns:
        Every distinct candidate with its score, best first.
    """
    ranked = [
        Ranked(candidate, score(candidate.similarity, candidate.hop, decay))
        for candidate in merge(candidates)
    ]
    return sorted(ranked, key=lambda item: (-item.score, item.candidate.hop, item.candidate.key))
