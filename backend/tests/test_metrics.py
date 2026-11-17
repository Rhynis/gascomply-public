"""Tests for the pure metric functions (build plan A8, issue #96).

Every function ``eval.metrics`` exports is pure -- ids and outcomes in, a number out --
so this file needs no database, embedder or model. The fixture below is five gold-shaped
rows (two ``compliance``, one ``multi_hop``, one ``refusal``, one ``emergency`` -- the same
category mix the real gold set has) with hand-computed expected values in the comments next
to each one, so a change to the arithmetic itself, not just a typo, is what would make this
fail.
"""

import re
import unicodedata

import pytest

from app.schemas.ask import AskOutcome
from eval.metrics import (
    CITATION_ACCURACY_THRESHOLD,
    EFFECTIVE_ANSWER_RATE_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    RECALL_AT_K_THRESHOLD,
    SAFETY_RECALL_THRESHOLD,
    answer_check,
    citation_containment_precision,
    citation_precision,
    containment_recall_at_k,
    effective_answer_rate,
    exit_code,
    make_verdict,
    mean,
    recall_at_k,
    refusal_correctness,
    safety_recall,
)

# --- the five-question fixture, hand-computed ------------------------------------------------
#
# Row 1 (compliance): naive finds the gold unit plus one irrelevant one; hybrid finds only
#   the gold unit. Both cite it correctly. recall@k: naive 1/1=1.0, hybrid 1/1=1.0.
# Row 2 (compliance): naive misses the gold unit entirely and (correctly) refuses rather
#   than cite something else; hybrid's graph expansion finds it. recall@k: naive 0/1=0.0,
#   hybrid 1/1=1.0. This is the row where hybrid beats naive.
# Row 3 (multi_hop): two gold units. naive finds one of the two (and cites exactly that
#   one, correctly); hybrid finds both (and cites both). recall@k: naive 1/2=0.5, hybrid
#   2/2=1.0.
# Row 4 (refusal): the system correctly refuses -- scored by refusal_correctness, not
#   effective_answer_rate.
# Row 5 (emergency): the safety check correctly escalates -- scored by safety_recall.

_ROW1_GOLD = ("u1",)
_ROW1_NAIVE_RETRIEVED = ("u1", "u9")
_ROW1_HYBRID_RETRIEVED = ("u1",)
_ROW1_NAIVE_CITED = ("u1",)
_ROW1_HYBRID_CITED = ("u1",)

_ROW2_GOLD = ("u2",)
_ROW2_NAIVE_RETRIEVED = ("u8",)
_ROW2_HYBRID_RETRIEVED = ("u2", "u3")
_ROW2_NAIVE_CITED: tuple[str, ...] = ()  # refused: nothing to cite
_ROW2_HYBRID_CITED = ("u2",)

_ROW3_GOLD = ("u4", "u5")
_ROW3_NAIVE_RETRIEVED = ("u4",)
_ROW3_HYBRID_RETRIEVED = ("u4", "u5", "u6")
_ROW3_NAIVE_CITED = ("u4",)
_ROW3_HYBRID_CITED = ("u4", "u5")

_ANSWERABLE_NAIVE_OUTCOMES = (AskOutcome.ANSWER, AskOutcome.REFUSAL, AskOutcome.ANSWER)
_ANSWERABLE_HYBRID_OUTCOMES = (AskOutcome.ANSWER, AskOutcome.ANSWER, AskOutcome.ANSWER)

_ROW4_REFUSAL_OUTCOMES = (AskOutcome.REFUSAL,)
_ROW5_ESCALATED = (True,)


# --- recall@k ----------------------------------------------------------------------------


def test_recall_at_k_row_by_row():
    assert recall_at_k(_ROW1_NAIVE_RETRIEVED, _ROW1_GOLD) == 1.0
    assert recall_at_k(_ROW1_HYBRID_RETRIEVED, _ROW1_GOLD) == 1.0
    assert recall_at_k(_ROW2_NAIVE_RETRIEVED, _ROW2_GOLD) == 0.0
    assert recall_at_k(_ROW2_HYBRID_RETRIEVED, _ROW2_GOLD) == 1.0
    assert recall_at_k(_ROW3_NAIVE_RETRIEVED, _ROW3_GOLD) == pytest.approx(0.5)
    assert recall_at_k(_ROW3_HYBRID_RETRIEVED, _ROW3_GOLD) == 1.0


def test_recall_at_k_mean_across_the_three_answerable_rows():
    naive = mean(
        [
            recall_at_k(_ROW1_NAIVE_RETRIEVED, _ROW1_GOLD),
            recall_at_k(_ROW2_NAIVE_RETRIEVED, _ROW2_GOLD),
            recall_at_k(_ROW3_NAIVE_RETRIEVED, _ROW3_GOLD),
        ]
    )
    hybrid = mean(
        [
            recall_at_k(_ROW1_HYBRID_RETRIEVED, _ROW1_GOLD),
            recall_at_k(_ROW2_HYBRID_RETRIEVED, _ROW2_GOLD),
            recall_at_k(_ROW3_HYBRID_RETRIEVED, _ROW3_GOLD),
        ]
    )
    assert naive == pytest.approx(0.5)  # (1.0 + 0.0 + 0.5) / 3
    assert hybrid == 1.0  # (1.0 + 1.0 + 1.0) / 3
    assert hybrid > naive  # the headline claim, on this fixture


def test_recall_at_k_raises_on_a_row_with_no_gold_ids():
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(("u1",), ())


# --- containment recall@k (issue #103) ----------------------------------------------------
#
# The gold set's own case: source ids are clause-level, the corpus chunks at article level.
# A retrieved article-level chunk contains the gold clause's text even though its own id is
# not the gold id -- recall_at_k (exact match) misses this on purpose; containment_recall_at_k
# is what credits it.

_DOC = "gas-decree-87-2018"
_ARTICLE = f"{_DOC}#dieu-53"
_CLAUSE = f"{_DOC}#dieu-53#khoan-1"
_POINT = f"{_DOC}#dieu-53#khoan-1#diem-a"
_OTHER_ARTICLE = f"{_DOC}#dieu-5"  # shares the "#dieu-5" *prefix* of #dieu-53, must not match


def test_containment_recall_credits_an_article_level_chunk_for_a_clause_level_gold_id():
    # This is the exact shape of the issue's own finding: naive retrieval returns the whole
    # article; the gold id names one clause inside it.
    assert containment_recall_at_k((_ARTICLE,), (_CLAUSE,)) == 1.0
    assert recall_at_k((_ARTICLE,), (_CLAUSE,)) == 0.0  # strict recall still misses it


def test_containment_recall_credits_an_article_level_chunk_for_a_point_level_gold_id():
    assert containment_recall_at_k((_ARTICLE,), (_POINT,)) == 1.0


def test_containment_recall_exact_match_still_counts():
    assert containment_recall_at_k((_CLAUSE,), (_CLAUSE,)) == 1.0


def test_containment_recall_does_not_credit_a_finer_id_for_a_coarser_gold_id():
    # Containment only runs coarse -> fine: a retrieved clause does not "contain" the whole
    # article it belongs to.
    assert containment_recall_at_k((_CLAUSE,), (_ARTICLE,)) == 0.0


def test_containment_recall_does_not_false_positive_on_a_shared_numeric_prefix():
    # "#dieu-5" is a *string* prefix of "#dieu-53" but not a real ancestor -- the "#" boundary
    # in _unit_contains is what tells them apart.
    assert containment_recall_at_k((_OTHER_ARTICLE,), (_ARTICLE,)) == 0.0
    assert containment_recall_at_k((_OTHER_ARTICLE,), (_CLAUSE,)) == 0.0


def test_containment_recall_mixed_hits_and_misses():
    gold = (_CLAUSE, "unrelated-doc#dieu-1")
    assert containment_recall_at_k((_ARTICLE,), gold) == pytest.approx(0.5)


def test_containment_recall_raises_on_a_row_with_no_gold_ids():
    with pytest.raises(ValueError, match="undefined"):
        containment_recall_at_k((_ARTICLE,), ())


def test_containment_recall_is_never_below_strict_recall_on_the_same_inputs():
    # Containment is a superset of strict matches (equality counts), so it can only credit
    # more, never less, than exact-id recall on the same retrieved/gold pair.
    retrieved = (_ARTICLE,)
    gold = (_CLAUSE, _ARTICLE)
    assert containment_recall_at_k(retrieved, gold) >= recall_at_k(retrieved, gold)


# --- citation precision -----------------------------------------------------------------


def test_citation_precision_row_by_row():
    assert citation_precision(_ROW1_NAIVE_CITED, _ROW1_GOLD) == 1.0
    assert citation_precision(_ROW1_HYBRID_CITED, _ROW1_GOLD) == 1.0
    assert citation_precision(_ROW2_NAIVE_CITED, _ROW2_GOLD) is None  # refused: nothing emitted
    assert citation_precision(_ROW2_HYBRID_CITED, _ROW2_GOLD) == 1.0
    assert (
        citation_precision(_ROW3_NAIVE_CITED, _ROW3_GOLD) == 1.0
    )  # cited what it found, correctly
    assert citation_precision(_ROW3_HYBRID_CITED, _ROW3_GOLD) == 1.0


def test_citation_precision_penalises_a_wrong_citation():
    # Half of what was cited names a real gold unit, half does not.
    assert citation_precision(("u1", "u999"), ("u1",)) == pytest.approx(0.5)


def test_citation_precision_mean_ignores_rows_that_refused():
    # Row 2's naive side emitted nothing (a refusal), so it contributes no term at all --
    # excluded, not scored as 0.0, since there was no citation to be wrong.
    values = [
        precision
        for precision in (
            citation_precision(_ROW1_NAIVE_CITED, _ROW1_GOLD),
            citation_precision(_ROW2_NAIVE_CITED, _ROW2_GOLD),
            citation_precision(_ROW3_NAIVE_CITED, _ROW3_GOLD),
        )
        if precision is not None
    ]
    assert values == [1.0, 1.0]
    assert mean(values) == 1.0


# --- citation precision, under containment (issue #118's own follow-up review) -----------


def test_citation_containment_precision_credits_an_article_for_its_own_clause():
    # Gold names a clause; the citation is only the whole article -- a strict miss, but the
    # article's text does contain the clause, so this diagnostic credits it.
    assert (
        citation_precision(("gas-decree-87-2018#dieu-10",), ("gas-decree-87-2018#dieu-10#khoan-2",))
        == 0.0
    )
    assert (
        citation_containment_precision(
            ("gas-decree-87-2018#dieu-10",), ("gas-decree-87-2018#dieu-10#khoan-2",)
        )
        == 1.0
    )


def test_citation_containment_precision_does_not_false_positive_on_a_shared_numeric_prefix():
    # Mirrors containment_recall_at_k's own guard against this: #dieu-5 must never contain
    # #dieu-53's text just because "5" is a string prefix of "53".
    assert citation_containment_precision(("doc#dieu-5",), ("doc#dieu-53#khoan-1",)) == 0.0


def test_citation_containment_precision_still_penalises_a_genuinely_wrong_citation():
    assert citation_containment_precision(("doc#dieu-99",), ("doc#dieu-1#khoan-1",)) == 0.0


def test_citation_containment_precision_is_never_below_strict_on_the_same_inputs():
    emitted, gold = _ROW1_HYBRID_CITED, _ROW1_GOLD
    assert citation_containment_precision(emitted, gold) >= citation_precision(emitted, gold)


def test_citation_containment_precision_none_when_nothing_was_emitted():
    assert citation_containment_precision((), ("doc#dieu-1",)) is None


# --- effective-answer rate ----------------------------------------------------------------


def test_effective_answer_rate_naive_vs_hybrid():
    assert effective_answer_rate(_ANSWERABLE_NAIVE_OUTCOMES) == pytest.approx(2 / 3)
    assert effective_answer_rate(_ANSWERABLE_HYBRID_OUTCOMES) == 1.0


def test_effective_answer_rate_raises_on_empty():
    with pytest.raises(ValueError, match="undefined"):
        effective_answer_rate(())


# --- refusal correctness ------------------------------------------------------------------


def test_refusal_correctness_row4():
    assert refusal_correctness(_ROW4_REFUSAL_OUTCOMES) == 1.0


def test_refusal_correctness_catches_over_answering():
    # A system that fabricates a citation for a question it should have refused --
    # invisible to effective_answer_rate, since that metric never sees refusal rows.
    assert refusal_correctness((AskOutcome.ANSWER,)) == 0.0


def test_refusal_correctness_raises_on_empty():
    with pytest.raises(ValueError, match="undefined"):
        refusal_correctness(())


# --- safety recall -------------------------------------------------------------------------


def test_safety_recall_row5():
    assert safety_recall(_ROW5_ESCALATED) == 1.0


def test_safety_recall_below_one_when_one_emergency_is_missed():
    assert safety_recall((True, True, False)) == pytest.approx(2 / 3)


def test_safety_recall_raises_on_empty():
    with pytest.raises(ValueError, match="undefined"):
        safety_recall(())


# --- mean --------------------------------------------------------------------------------


def test_mean_raises_on_empty():
    with pytest.raises(ValueError, match="undefined"):
        mean([])


# --- verdicts and the process exit code ---------------------------------------------------


def test_every_threshold_passes_on_the_hybrid_side_of_the_fixture():
    hybrid_recall = mean(
        [
            recall_at_k(_ROW1_HYBRID_RETRIEVED, _ROW1_GOLD),
            recall_at_k(_ROW2_HYBRID_RETRIEVED, _ROW2_GOLD),
            recall_at_k(_ROW3_HYBRID_RETRIEVED, _ROW3_GOLD),
        ]
    )
    hybrid_citation = mean(
        [
            citation_precision(_ROW1_HYBRID_CITED, _ROW1_GOLD),
            citation_precision(_ROW2_HYBRID_CITED, _ROW2_GOLD),
            citation_precision(_ROW3_HYBRID_CITED, _ROW3_GOLD),
        ]
    )
    verdicts = [
        make_verdict("recall@k", hybrid_recall, RECALL_AT_K_THRESHOLD),
        make_verdict("citation accuracy", hybrid_citation, CITATION_ACCURACY_THRESHOLD),
        make_verdict(
            "effective-answer rate",
            effective_answer_rate(_ANSWERABLE_HYBRID_OUTCOMES),
            EFFECTIVE_ANSWER_RATE_THRESHOLD,
        ),
        make_verdict("safety detection", safety_recall(_ROW5_ESCALATED), SAFETY_RECALL_THRESHOLD),
        make_verdict("faithfulness", None, FAITHFULNESS_THRESHOLD, note="not measured"),
    ]

    assert [verdict.passed for verdict in verdicts] == [True, True, True, True, None]
    assert exit_code(verdicts) == 0


def test_the_naive_recall_on_this_fixture_would_fail_the_threshold():
    # Not a claim that naive retrieval is broken in general -- just this fixture,
    # deliberately built so the comparison has a losing side to show.
    naive_recall = mean(
        [
            recall_at_k(_ROW1_NAIVE_RETRIEVED, _ROW1_GOLD),
            recall_at_k(_ROW2_NAIVE_RETRIEVED, _ROW2_GOLD),
            recall_at_k(_ROW3_NAIVE_RETRIEVED, _ROW3_GOLD),
        ]
    )
    verdict = make_verdict("recall@k (naive, for comparison)", naive_recall, RECALL_AT_K_THRESHOLD)

    assert verdict.passed is False
    assert exit_code([verdict]) == 1


def test_exit_code_is_zero_when_nothing_failed():
    assert exit_code([make_verdict("x", 1.0, 0.8), make_verdict("y", None, None)]) == 0


def test_exit_code_is_one_when_one_metric_failed_even_if_others_passed():
    assert (
        exit_code(
            [make_verdict("x", 1.0, 0.8), make_verdict("y", 0.5, EFFECTIVE_ANSWER_RATE_THRESHOLD)]
        )
        == 1
    )


def test_a_verdict_with_no_threshold_never_fails_the_run():
    # Faithfulness, reported but deliberately unmeasured (see eval/compare.py): its
    # verdict must never be conflated with a pass, but it must not fail the run either.
    verdict = make_verdict("faithfulness", None, FAITHFULNESS_THRESHOLD, note="not measured")
    assert verdict.passed is None
    assert exit_code([verdict]) == 0


# --- answer_check: the deciding-fact check (issue #167) -------------------------------------

_MUST_MATCH_500KG = (re.compile(r"500\s*kg", re.IGNORECASE),)
_MUST_NOT_MATCH_035 = (re.compile(r"0,35", re.IGNORECASE),)


def test_answer_check_is_none_with_no_patterns_at_all():
    assert answer_check(AskOutcome.ANSWER, "bất kỳ câu trả lời nào", (), ()) is None


def test_answer_check_fails_a_refusal():
    # A row with at least one pattern always fails a refusal -- there is no answer text to
    # have stated the deciding fact at all.
    assert answer_check(AskOutcome.REFUSAL, None, _MUST_MATCH_500KG, ()) is False


def test_answer_check_fails_when_answer_text_is_none_despite_an_answer_outcome():
    # Defensive: should never happen for a real ANSWER outcome, but must not crash or
    # silently pass if it ever does.
    assert answer_check(AskOutcome.ANSWER, None, _MUST_MATCH_500KG, ()) is False


def test_answer_check_passes_when_every_must_match_pattern_matches():
    answer = "Cơ sở tồn chứa trên 500 kg khí đốt thuộc danh mục phải mua bảo hiểm."
    assert answer_check(AskOutcome.ANSWER, answer, _MUST_MATCH_500KG, ()) is True


def test_answer_check_fails_when_a_must_match_pattern_is_absent():
    answer = "Cửa hàng kinh doanh gas có phải mua bảo hiểm cháy, nổ bắt buộc."
    assert answer_check(AskOutcome.ANSWER, answer, _MUST_MATCH_500KG, ()) is False


def test_answer_check_fails_when_a_must_not_match_pattern_is_present():
    answer = "Tỷ lệ phí bảo hiểm cháy, nổ bắt buộc tối thiểu là 0,35%."
    assert answer_check(AskOutcome.ANSWER, answer, (), _MUST_NOT_MATCH_035) is False


def test_answer_check_passes_when_must_not_match_pattern_is_absent():
    answer = "Tỷ lệ phí bảo hiểm cháy, nổ bắt buộc tối thiểu là 0,3%."
    assert answer_check(AskOutcome.ANSWER, answer, (), _MUST_NOT_MATCH_035) is True


def test_answer_check_is_case_insensitive():
    pattern = (re.compile(r"trên 500 kg", re.IGNORECASE),)
    answer = "Cơ sở TỒN CHỨA TRÊN 500 KG khí đốt."
    assert answer_check(AskOutcome.ANSWER, answer, pattern, ()) is True


def test_answer_check_nfc_normalises_the_answer_before_matching():
    # A combining-character (NFD) answer must still match an NFC pattern -- composed vs.
    # decomposed diacritics are a real possibility from upstream text processing, not a
    # contrived edge case.
    nfc_answer = "Cơ sở tồn chứa trên 500 kg khí đốt."
    nfd_answer = unicodedata.normalize("NFD", nfc_answer)
    assert nfd_answer != nfc_answer  # the decomposition actually changed the bytes

    pattern = (re.compile(unicodedata.normalize("NFC", "trên 500\\s*kg"), re.IGNORECASE),)
    assert answer_check(AskOutcome.ANSWER, nfd_answer, pattern, ()) is True


def test_answer_check_requires_every_must_match_pattern_not_just_one():
    patterns = (re.compile(r"không", re.IGNORECASE), re.compile(r"500\s*kg", re.IGNORECASE))
    answer_missing_second = "Không bắt buộc mua bảo hiểm cho cơ sở này."
    assert answer_check(AskOutcome.ANSWER, answer_missing_second, patterns, ()) is False
