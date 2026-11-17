"""Tests for the rerank formula (build plan A6 Phase 3.4, issue #89).

Pure functions, no database: ``app/services/rerank.py``'s own docstring states three
properties the formula must have (a hit is never penalised, the vector top-1 always stays
first, and an item derived from a hit inherits a discounted share of it) and this file is
what proves each one, rather than trusting the docstring's arithmetic by eye.
"""

import math
from itertools import product

import pytest

from app.schemas.retrieval import Why
from app.services.rerank import Candidate, merge, rank, score

SIMILAR = frozenset({Why.SIMILAR})
CITED = frozenset({Why.CITED_BY_HIT})


def _candidate(key: str, similarity: float, hop: int, why: frozenset[Why] = SIMILAR) -> Candidate:
    return Candidate(key=key, similarity=similarity, hop=hop, why=why)


# --- score() ------------------------------------------------------------------------------


def test_a_hit_scores_its_own_similarity_unchanged():
    assert score(0.74, hop=0, decay=0.9) == 0.74


def test_score_is_similarity_times_decay_to_the_hop():
    assert math.isclose(score(0.8, hop=2, decay=0.5), 0.8 * 0.25)


def test_a_smaller_decay_discounts_a_distant_item_more():
    far = score(0.8, hop=3, decay=0.5)
    near_decay = score(0.8, hop=3, decay=0.95)
    assert far < near_decay


@pytest.mark.parametrize("similarity", [-1.0, -0.5, -1e-9])
def test_a_negative_similarity_scores_zero_at_any_hop(similarity: float):
    assert score(similarity, hop=0, decay=0.9) == 0.0
    assert score(similarity, hop=5, decay=0.5) == 0.0


@pytest.mark.parametrize(
    ("similarity", "hop", "decay"),
    list(
        product(
            [-1.0, -0.3, 0.0, 0.1, 0.5, 0.99, 1.0],
            [0, 1, 2, 5, 20],
            [1e-6, 0.1, 0.5, 0.9, 1.0],
        )
    ),
)
def test_score_never_exceeds_the_bare_similarity(similarity: float, hop: int, decay: float):
    # decay <= 1 and hop >= 0: discounting can only shrink a score toward zero, never grow it.
    assert score(similarity, hop, decay) <= max(similarity, 0.0) + 1e-9


@pytest.mark.parametrize(
    ("similarity", "hop", "decay"),
    list(product([0.01, 0.3, 0.74, 1.0], [0, 1, 4, 9], [1e-6, 0.1, 0.5, 0.9, 0.999999])),
)
def test_score_strictly_decreases_with_one_more_hop_when_similarity_is_positive(
    similarity: float, hop: int, decay: float
):
    assert score(similarity, hop + 1, decay) < score(similarity, hop, decay)


# --- merge() --------------------------------------------------------------------------------


def test_merge_is_the_identity_on_distinct_keys():
    candidates = [_candidate("a", 0.5, 1), _candidate("b", 0.6, 2)]
    assert {c.key for c in merge(candidates)} == {"a", "b"}


def test_merging_the_same_key_keeps_the_nearer_hop():
    merged = merge([_candidate("x", 0.5, 3), _candidate("x", 0.5, 1)])
    assert [c.hop for c in merged] == [1]


def test_merging_the_same_key_unions_every_why():
    merged = merge([_candidate("x", 0.5, 1, SIMILAR), _candidate("x", 0.5, 1, CITED)])
    [candidate] = merged
    assert candidate.why == SIMILAR | CITED


def test_merging_the_same_key_keeps_the_larger_similarity():
    # An entity offered by two sources: the more relevant source should win.
    merged = merge([_candidate("req:x", 0.3, 1), _candidate("req:x", 0.9, 1)])
    assert [c.similarity for c in merged] == [0.9]


def test_merge_of_nothing_is_nothing():
    assert merge([]) == []


# --- rank(): the properties the module promises ----------------------------------------------


def test_a_hit_outranks_everything_it_is_at_least_as_similar_to():
    # rank()'s central promise: the graph can only ever push an item BELOW a hit, never above one
    # it is not at least as similar to -- true for hop 0 vs any hop >= 1, at any decay in (0, 1).
    hit = _candidate("hit", 0.74, 0)
    same_similarity_far = _candidate("derived", 0.74, 1)
    ranked = rank([hit, same_similarity_far], decay=0.9)
    assert [item.candidate.key for item in ranked] == ["hit", "derived"]


def test_the_vector_top_1_always_ranks_first_regardless_of_decay():
    top1 = _candidate("top1", 0.90, 0)
    kth_hit = _candidate("kth", 0.60, 0)
    graph_reached = _candidate("graph", 0.60, 1)  # at most as similar as the weakest hit
    for decay in (0.99, 0.9, 0.5, 0.1, 1e-6):
        ranked = rank([top1, kth_hit, graph_reached], decay=decay)
        assert ranked[0].candidate.key == "top1", f"decay={decay}"


def test_an_item_one_hop_from_a_hit_can_outrank_a_weaker_hit():
    # The concrete case from rerank.py's own docstring: a requirement one edge from a 0.74
    # hit, at decay=0.9, scores 0.666, which beats a second hit similar at only 0.6.
    strong_hit = _candidate("gas-decree-87-2018#dieu-53", 0.74, 0)
    weak_hit = _candidate("lpg-station-qcvn-10-2012#dieu-7", 0.60, 0)
    requirement = _candidate("req:min-store-area", 0.74, 1, CITED)

    ranked = rank([strong_hit, weak_hit, requirement], decay=0.9)

    assert [item.candidate.key for item in ranked] == [
        "gas-decree-87-2018#dieu-53",
        "req:min-store-area",
        "lpg-station-qcvn-10-2012#dieu-7",
    ]
    assert math.isclose(ranked[1].score, 0.74 * 0.9, rel_tol=1e-9)


def test_rank_deduplicates_by_key_before_scoring():
    ranked = rank([_candidate("x", 0.5, 2), _candidate("x", 0.5, 1)], decay=0.9)
    assert len(ranked) == 1
    assert ranked[0].candidate.hop == 1


def test_ties_break_by_hop_then_key_for_a_deterministic_order():
    a = _candidate("b-key", 0.5, 1)
    b = _candidate("a-key", 0.5, 0)
    c = _candidate("c-key", 0.5, 0)
    ranked = rank([a, b, c], decay=0.9)
    # b and c tie at hop 0 (same score, lower hop than a): key breaks the tie between them.
    assert [item.candidate.key for item in ranked] == ["a-key", "c-key", "b-key"]


def test_rank_of_nothing_is_nothing():
    assert rank([], decay=0.9) == []


@pytest.mark.parametrize(
    ("similarities", "decay"),
    [
        ([0.9, 0.5, 0.5, 0.1], 0.9),
        ([0.74, 0.6, 0.62, 0.3], 0.5),
        ([0.1], 0.9),
        ([0.5, 0.5, 0.5], 0.1),
    ],
)
def test_rank_never_reorders_hits_among_themselves(similarities: list[float], decay: float):
    # At hop 0 the discount is 1 for everyone, so rank() must reduce to a plain sort by
    # similarity among the hits -- the graph has nothing to say about their relative order.
    hits = [_candidate(f"hit-{i}", value, 0) for i, value in enumerate(similarities)]
    ranked = rank(hits, decay)
    assert [item.score for item in ranked] == sorted((c.similarity for c in hits), reverse=True)
