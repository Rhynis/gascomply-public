"""Metrics with pass/fail thresholds (build plan A8, issue #96).

Every function here is pure: given the ids or outcomes a run already produced, it returns
a number. None of them touch a database, an embedder or a model -- that is deliberate, so
``tests/test_metrics.py`` can pin every one of them against hand-computed values with no
fixture heavier than a handful of literal strings. ``eval/compare.py`` is what actually runs
retrieval and generation and hands their output to these functions.

Thresholds, from the build plan (Part A8):

    recall@k             >= 0.8   -- gold source ids present in the top-k retrieved units
    citation accuracy    >= 0.85  -- emitted citations that name a real gold source id
    faithfulness         >= 0.9   -- answer claims entailed by the retrieved context
    effective-answer rate >= 0.9  -- of answerable questions, the fraction actually answered
    safety detection     == 1.0   -- recall on the emergency category

A threshold nobody can fail is not a measurement, so every one of these is checked, and
``exit_code`` is non-zero the moment one is missed -- see ``eval/compare.py`` for where that
number becomes a process exit status.

Two recall functions, not one (issue #103)
-------------------------------------------
``recall_at_k`` (strict) and ``containment_recall_at_k`` answer different questions, and the
gold set's own granularity is why both are needed. Gold source ids name a specific clause or
point (``gas-decree-87-2018#dieu-53#khoan-1``); the corpus is chunked at article granularity
(``gas-decree-87-2018#dieu-53``, see ``app/services/retriever.py``'s own module docstring), so
an exact-id comparison against a bare article-level chunk is structurally unwinnable -- it
scores "the right article came back, but not the graph's clause-level pointer to it" the same
as "nothing relevant came back at all", which are not the same failure.

    recall_at_k              -- "can the system name the specific clause?" (exact id match)
    containment_recall_at_k  -- "did retrieval surface the text the answer needs?" (the gold
                                 clause's text is inside *some* retrieved unit, coarser or not)

Neither replaces the other: ``eval/compare.py`` reports both, for both retrieval paths, and
states in the printed table which one the build-plan threshold above actually gates on.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.ask import AskOutcome

RECALL_AT_K_THRESHOLD = 0.8
CITATION_ACCURACY_THRESHOLD = 0.85
FAITHFULNESS_THRESHOLD = 0.9
EFFECTIVE_ANSWER_RATE_THRESHOLD = 0.9
SAFETY_RECALL_THRESHOLD = 1.0


def recall_at_k(retrieved_unit_ids: Sequence[str], gold_source_ids: Sequence[str]) -> float:
    """Fraction of ``gold_source_ids`` present in ``retrieved_unit_ids``.

    Args:
        retrieved_unit_ids: The unit ids a retrieval run returned, in any order --
            only membership is checked, never rank.
        gold_source_ids: The unit ids the gold row says the answer is grounded in.

    Returns:
        ``len(hits) / len(gold_source_ids)``, where a hit is a gold id that appears
        anywhere in ``retrieved_unit_ids``.

    Raises:
        ValueError: If ``gold_source_ids`` is empty -- recall is undefined with
            nothing to find. Callers filter to rows that have at least one source
            id (``compliance``/``multi_hop``) before calling this.
    """
    if not gold_source_ids:
        raise ValueError("recall@k is undefined for a row with no gold source ids")
    retrieved = set(retrieved_unit_ids)
    hits = sum(1 for gold_id in gold_source_ids if gold_id in retrieved)
    return hits / len(gold_source_ids)


def _unit_contains(retrieved_id: str, gold_id: str) -> bool:
    """Whether the unit ``retrieved_id`` names would, if read, include ``gold_id``'s text.

    Unit ids nest by appending a ``#chuong-N``/``#dieu-N``/``#khoan-N``/``#diem-x`` segment at
    a time (``app/core/ids.py``'s own frozen scheme: document -> chapter -> article -> clause
    -> point), so a coarser id is always an exact string prefix of every finer id nested under
    it, one ``#``-segment at a time -- ``gas-decree-87-2018#dieu-53`` is a prefix of
    ``gas-decree-87-2018#dieu-53#khoan-1``, but not of ``gas-decree-87-2018#dieu-5`` (the
    ``#`` boundary is what stops ``#dieu-5`` from falsely matching as a prefix of
    ``#dieu-53``). Exact equality counts as containment too, so a retrieved id already at gold
    granularity is credited the same way strict ``recall_at_k`` credits it.
    """
    return gold_id == retrieved_id or gold_id.startswith(retrieved_id + "#")


def containment_recall_at_k(
    retrieved_unit_ids: Sequence[str], gold_source_ids: Sequence[str]
) -> float:
    """Fraction of ``gold_source_ids`` whose text is contained in some retrieved unit.

    Answers "did retrieval surface the text the answer needs?" -- looser than
    ``recall_at_k``'s exact-id match on purpose: a retrieved article-level chunk that contains
    the gold clause counts as a hit here even though its own id is not the gold id (see
    ``_unit_contains``, and the module docstring for why this matters on this corpus).

    Args:
        retrieved_unit_ids: The unit ids a retrieval run returned, in any order.
        gold_source_ids: The unit ids the gold row says the answer is grounded in.

    Returns:
        ``len(hits) / len(gold_source_ids)``, where a hit is a gold id contained in (equal to,
        or nested under) at least one retrieved id.

    Raises:
        ValueError: If ``gold_source_ids`` is empty -- undefined with nothing to find, same as
            ``recall_at_k``.
    """
    if not gold_source_ids:
        raise ValueError("containment recall@k is undefined for a row with no gold source ids")
    hits = sum(
        1
        for gold_id in gold_source_ids
        if any(_unit_contains(retrieved_id, gold_id) for retrieved_id in retrieved_unit_ids)
    )
    return hits / len(gold_source_ids)


def citation_precision(
    emitted_unit_ids: Sequence[str], gold_source_ids: Sequence[str]
) -> float | None:
    """Fraction of ``emitted_unit_ids`` that name a real gold source id (right Điều/Khoản).

    This is precision, not recall: a citation the model did not need to give is not
    penalised, but every citation it did give must be one of the ones the gold row
    actually names.

    Args:
        emitted_unit_ids: Unit ids from the citations a generated answer actually
            gave, in any order.
        gold_source_ids: The unit ids the gold row says the answer is grounded in.

    Returns:
        ``len(correct) / len(emitted_unit_ids)``, or ``None`` when nothing was
        emitted at all -- undefined, not zero. A row with zero citations is either
        a refusal (scored by ``effective_answer_rate``) or a malformed answer that
        ``generate_answer`` itself already turns into a refusal (``no_citations_given``
        / ``no_valid_citations``), so it never reaches this function with citations
        to judge.
    """
    if not emitted_unit_ids:
        return None
    gold = set(gold_source_ids)
    correct = sum(1 for unit_id in emitted_unit_ids if unit_id in gold)
    return correct / len(emitted_unit_ids)


def citation_containment_precision(
    emitted_unit_ids: Sequence[str], gold_source_ids: Sequence[str]
) -> float | None:
    """Fraction of ``emitted_unit_ids`` whose text would include some gold clause's text.

    The same strict-vs-containment split ``recall_at_k``/``containment_recall_at_k`` make for
    recall (issue #103), made here for citation precision instead (issue #118's own follow-up
    review): a citation at article granularity is not a strict gold match when gold names one
    of its clauses, even when fixing what the grounding check is actually shown (the review's
    item 2) makes the *answer* better -- it may still make the *id* the model cites coarser,
    which this diagnostic credits and ``citation_precision`` above never will. Reported beside
    strict precision, never in place of it; strict stays the build-plan threshold metric.

    Args:
        emitted_unit_ids: Unit ids from the citations a generated answer actually gave.
        gold_source_ids: The unit ids the gold row says the answer is grounded in.

    Returns:
        ``len(correct) / len(emitted_unit_ids)``, where a hit is an emitted id equal to, or an
        exact ``#``-segment prefix of, some gold id (``_unit_contains``); ``None`` when nothing
        was emitted, same as ``citation_precision``.
    """
    if not emitted_unit_ids:
        return None
    correct = sum(
        1
        for unit_id in emitted_unit_ids
        if any(_unit_contains(unit_id, gold_id) for gold_id in gold_source_ids)
    )
    return correct / len(emitted_unit_ids)


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of ``values``.

    Raises:
        ValueError: If ``values`` is empty -- an empty-set average is undefined,
            never silently ``0.0``.
    """
    if not values:
        raise ValueError("mean of an empty sequence is undefined")
    return sum(values) / len(values)


def effective_answer_rate(outcomes: Sequence[AskOutcome]) -> float:
    """Fraction of *answerable* questions actually answered.

    Args:
        outcomes: One outcome per answerable (``compliance``/``multi_hop``) gold
            row -- never include a ``refusal``/``emergency`` row's outcome here,
            since "answerable" is the whole premise of this metric.

    Returns:
        ``count(ANSWER) / len(outcomes)``.

    Raises:
        ValueError: If ``outcomes`` is empty.
    """
    if not outcomes:
        raise ValueError("effective-answer rate is undefined with no answerable rows")
    answered = sum(1 for outcome in outcomes if outcome is AskOutcome.ANSWER)
    return answered / len(outcomes)


def refusal_correctness(outcomes: Sequence[AskOutcome]) -> float:
    """Fraction of *expected-refusal* questions the system actually refused.

    Kept apart from ``effective_answer_rate`` on purpose: that metric is scoped to
    answerable rows, so a system that over-answers (fabricates a citation for a
    question it should have refused) would not show up there at all. This metric
    is what makes over-answering visible, the same way ``effective_answer_rate``
    makes over-refusal visible -- one row set is scored as "answering is correct",
    the other as "refusing is correct", and neither metric can hide the other's
    kind of mistake.

    Args:
        outcomes: One outcome per ``refusal``-category gold row.

    Returns:
        ``count(REFUSAL) / len(outcomes)``.

    Raises:
        ValueError: If ``outcomes`` is empty.
    """
    if not outcomes:
        raise ValueError("refusal correctness is undefined with no expected-refusal rows")
    refused = sum(1 for outcome in outcomes if outcome is AskOutcome.REFUSAL)
    return refused / len(outcomes)


def safety_recall(escalated: Sequence[bool]) -> float:
    """Fraction of emergency-category questions the safety check actually escalated.

    Args:
        escalated: One ``SafetyCheck.escalated`` value per ``emergency``-category
            gold row.

    Returns:
        ``count(True) / len(escalated)``.

    Raises:
        ValueError: If ``escalated`` is empty.
    """
    if not escalated:
        raise ValueError("safety recall is undefined with no emergency rows")
    caught = sum(1 for value in escalated if value)
    return caught / len(escalated)


def answer_check(
    outcome: AskOutcome,
    answer_text: str | None,
    must_match: Sequence[re.Pattern[str]],
    must_not_match: Sequence[re.Pattern[str]],
) -> bool | None:
    """Check whether a generated answer states a gold row's own deciding fact (issue #167).

    Citation precision (above) scores which units an answer cites, never what the answer
    actually says -- a wrong conclusion can still name a gold unit and score full marks (the
    issue's own flagship example: a model that echoes the question as "Cửa hàng kinh doanh
    gas có phải mua bảo hiểm cháy, nổ bắt buộc" -- a bare affirmative restatement, with no
    "trên 500 kg" threshold anywhere in it -- while citing the correct unit). This checks the
    text of the answer itself against the row's own ``answer_must_match``/
    ``answer_must_not_match`` patterns (``eval.gold.GoldRow``).

    Args:
        outcome: What ``generate_answer`` returned for this call.
        answer_text: The model's own answer text, matched *before* issue #146's deterministic
            verbatim-quote step appends anything from the corpus (``GroundingTrace.
            pre_quote_answer``) -- a quoted clause's own words are not the model's claim, and
            crediting a match found only inside that quote would check the corpus, not the
            model's reasoning. ``None`` when nothing was generated at all.
        must_match: Every pattern here must match somewhere in the answer.
        must_not_match: No pattern here may match anywhere in the answer.

    Returns:
        ``None`` when the row carries no pattern at all (nothing to check -- most rows).
        Otherwise ``True`` only when ``outcome`` is ``AskOutcome.ANSWER``, ``answer_text`` is
        not ``None``, every ``must_match`` pattern matches the NFC-normalised answer, and no
        ``must_not_match`` pattern does. ``False`` for a refusal (which, by definition, states
        no fact at all) or any pattern violation.
    """
    if not must_match and not must_not_match:
        return None
    if outcome is not AskOutcome.ANSWER or answer_text is None:
        return False
    normalized = unicodedata.normalize("NFC", answer_text)
    if any(pattern.search(normalized) is None for pattern in must_match):
        return False
    return not any(pattern.search(normalized) is not None for pattern in must_not_match)


@dataclass(frozen=True)
class MetricVerdict:
    """One measured (or deliberately unmeasured) metric, against its threshold.

    Attributes:
        name: Short label, e.g. ``"recall@k (multi_hop)"``.
        value: The measured number, or ``None`` when this metric was not measured
            at all (see ``eval/compare.py``'s faithfulness decision).
        threshold: The pass bar from the build plan, or ``None`` for a metric
            reported without one (faithfulness, when unmeasured).
        passed: ``True``/``False`` when both ``value`` and ``threshold`` are
            known, ``None`` when there is nothing to verdict on -- ``None`` is
            never treated as a pass; see ``exit_code``.
        note: Free text shown alongside the number, e.g. what the number means or
            why it could not be measured. Printed, never buried in a docstring.
    """

    name: str
    value: float | None
    threshold: float | None
    passed: bool | None
    note: str = ""


def make_verdict(
    name: str, value: float | None, threshold: float | None, note: str = ""
) -> MetricVerdict:
    """Build a ``MetricVerdict``, computing ``passed`` from ``value``/``threshold``.

    ``passed`` is ``value >= threshold`` when both are known, and ``None`` -- not
    a pass -- when either is missing.
    """
    passed = None if value is None or threshold is None else value >= threshold
    return MetricVerdict(name=name, value=value, threshold=threshold, passed=passed, note=note)


def exit_code(verdicts: Sequence[MetricVerdict]) -> int:
    """Return ``1`` if any verdict explicitly failed its threshold, else ``0``.

    A verdict with ``passed is None`` (not measured, or no threshold to check)
    never causes a non-zero exit on its own -- only a measured, failing number
    does. This is the number ``eval/compare.py``'s ``main`` returns to the shell.
    """
    return 1 if any(verdict.passed is False for verdict in verdicts) else 0
