"""``python -m eval.compare`` -- GraphRAG vs naive vector search on the gold set (issue #96).

The build plan calls this table "the CV number -- do not skip it", and the headline claim it
exists to test is *GraphRAG >= naive on recall@k and citation accuracy for multi-hop
questions*. That is a claim to be tested, not a result to be produced: both retrieval paths
run on the exact same gold set, the same ``k``, the same embeddings, the same effective-date
filter, and every question's own naive-vs-hybrid numbers are printed, not only the average --
a table where the new method wins every row is evidence of a rigged comparison, not of a good
method (see ``_row_lines``).

The two paths, identical apart from graph expansion and rerank (``app/services/retriever.py``):

    Retriever.vector_search(q)   # naive
    Retriever.retrieve(q)        # hybrid

Reranking (issue #114) is measured the same way: ``RERANKER=llama-server RERANK_TOP_N=<n>`` in
the environment makes the *hybrid* path's ``Retriever`` cross-encode its passages and keep the best
``n`` (the naive path is never reranked), and the table's first lines say which configuration
produced it and how many retrievals actually reranked. A configured reranker that fails falls
back silently for ``/ask`` -- right for a request, wrong for a measurement -- so a run that fell
back is counted and flagged in the table rather than passed off as a reranked one. Each ``n`` is
its own run; recall@k and citation accuracy have to be read together, since fewer passages can
cut the gold clause (see ``docs/adr/0002-cross-encoder-reranker-runtime.md``).

Citation accuracy is measured the same honest way: both a naive-only ``RetrievalResult``
(``_naive_retrieval_result``, built straight from ``vector_search``'s chunks with no graph
expansion) and the real hybrid one are handed to the unmodified ``generate_answer`` -- so the
only thing that differs between the two numbers is retrieval, never generation.

Faithfulness (build plan A8: "answer claims entailed by the retrieved context") is not
measured here. The only model on this machine is the one that generates the answers
(``qwen3:8b`` via Ollama, the default chat model since issue #156, or ``MockLLM`` offline);
a model judging its own
output is not independent evidence, and reporting a number from that would be worse than
reporting none (the same reasoning issue #90 applied to citations: a plausible-looking number
that is not actually checked is more dangerous than an honest gap). ``FAITHFULNESS_NOTE``
below is printed in the table itself, not left in this docstring, so nobody mistakes the
absence of a number for a passing one -- see ``eval.metrics.MetricVerdict``.

Improving retrieval to pass a threshold here is explicitly out of scope (the issue's own
words: tuning the system to its own benchmark inside the benchmarking PR destroys the
benchmark). A failing number is reported as a failing number.

Generating the way ``/ask`` does (issue #167)
-----------------------------------------------
Before this issue, every ``generate_answer`` call in this module ran retrieval through the
same ``EVAL_RETRIEVAL_CONFIG`` (``k=10``, ``passage_limit=10``) that recall@k is scored
against. That is not simply "narrower than a live ``/ask`` request ever uses" -- ``/ask``
seeds graph expansion with *fewer* vector hits than recall@k's own width
(``GENERATION_RETRIEVAL_CONFIG.k``, pinned explicitly since issue #169) but then expands to
*more* passages (``passage_limit=15`` vs. 10). Both differences were invisible but real
before this issue: for the flagship multi-hop question, the gold unit sits at position 12 of
the wider passage pool and is simply absent from the narrower one, so this eval's own
generated answer lacked the deciding clause while the deployed ``/ask`` endpoint had it --
conversely, #168's own PR review found two compliance rows where ``/ask``'s narrower seed set
(``k=5`` at the time) pushed the gold unit out of the pool entirely before graph expansion
ever got a chance to add it back, a loss this eval could not have shown before this issue
made generation retrieve the way ``/ask`` does. Retrieval now runs twice per answerable row
-- once at ``EVAL_RETRIEVAL_CONFIG`` purely to score recall@k (``_citable_unit_ids``,
unchanged), and once at this run's generation config (``GENERATION_RETRIEVAL_CONFIG``, with
``rerank_top_n`` from settings and, since issue #169, ``k``/``passage_limit`` overridable from
the command line for a sweep -- see ``--generation-k``/``--generation-passage-limit`` below)
to build what every ``generate_answer`` call actually sees -- on both the naive and the
hybrid path alike, so the naive-vs-hybrid comparison stays like for like. recall@k's own
meaning and width are completely unchanged by this; see ``EVAL_RETRIEVAL_CONFIG``'s own
docstring.

Choosing ``/ask``'s own seed ``k`` and ``passage_limit`` by measurement (issue #169)
----------------------------------------------------------------------------------------
``GENERATION_RETRIEVAL_CONFIG`` (``app/services/generator.py``) is a fixed module constant --
the config a real ``/ask`` request always runs with. ``--generation-k`` and
``--generation-passage-limit`` exist only so a sweep across candidate values can be measured
without editing that constant for each point: they default to ``GENERATION_RETRIEVAL_CONFIG``'s
own ``k``/``passage_limit`` (so an ordinary run, with neither flag given, measures exactly what
``/ask`` ships with), and they affect only the generation retriever built in ``_amain`` below --
``EVAL_RETRIEVAL_CONFIG`` and recall@k's own retriever never see them. The report's header line
prints both configs' ``k`` (previously only the recall one's), since that is exactly the number
issue #169 exists to choose.

Checking the deciding fact, not just the citation (issue #167)
-------------------------------------------------------------------
A citation can be in the gold set and the answer can still be wrong: the flagship question's
own failure mode is a model that echoes the question as a bare affirmative ("Cửa hàng kinh
doanh gas có phải mua bảo hiểm cháy, nổ bắt buộc") while correctly citing Phụ lục VII mục 15 --
which names a 500 kg threshold the answer never actually states. Citation precision cannot see
this; it only ever checks *which* unit was cited. ``eval.gold.GoldRow.answer_must_match``/
``answer_must_not_match`` (optional, regex, case-insensitive) check the answer's own text
instead, matched against the model's own words *before* issue #146's verbatim quote is
appended (``GroundingTrace.pre_quote_answer``) -- a quote can state a fact (that 500 kg
threshold is inside mục 15's own quoted text) the model itself never reasoned about, and
crediting a match found only there would check the corpus, not the model. See
``eval.metrics.answer_check``. Printed as its own "answer check" line, the way
``FAITHFULNESS_NOTE`` is printed -- not gated by any threshold yet (a threshold is premature
before the generation fix the next issue makes; see that note's own comment at its call site).

Two recall numbers, not one (issue #103)
-----------------------------------------
The gold set's source ids are clause-level (``gas-decree-87-2018#dieu-53#khoan-1``); the
corpus is chunked at article level. Scoring naive retrieval by exact id alone is therefore
structurally unwinnable for it, and reporting only that number under the name "recall@k"
made naive retrieval look far worse than it is (0.022 exact-match, against ~0.8 once credited
for surfacing the right article). Both paths are now scored two ways -- see
``eval.metrics.recall_at_k`` (strict: exact id) and ``eval.metrics.containment_recall_at_k``
(containment: the gold clause's text is inside *some* retrieved unit) -- and ``format_report``
states which one the build-plan threshold (``RECALL_AT_K_THRESHOLD``) actually gates on:
strict recall, on the hybrid path, unchanged from before this issue. Containment recall is
reported alongside for both paths, not gated by any threshold.
"""

import argparse
import asyncio
import dataclasses
import logging
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx2

from app.core.config import EmbedderProvider, RerankerProvider, Settings, get_settings
from app.core.config import LLMProvider as LLMProviderKind
from app.core.db import create_engine, create_session_factory, session_scope
from app.core.logging import configure_logging
from app.embedders.base import EmbedderUnreachableError
from app.embedders.factory import create_embedder
from app.llm.base import LLMProvider, LLMUnreachableError
from app.llm.mock import MockLLM
from app.llm.ollama import OllamaProvider
from app.repositories.documents import DocumentRepository
from app.rerankers.factory import create_reranker
from app.schemas.ask import AskOutcome, AskResponse
from app.schemas.chunk import ChunkMatch
from app.schemas.retrieval import Passage, RetrievalConfig, RetrievalResult, Why
from app.services.generator import GENERATION_RETRIEVAL_CONFIG, generate_answer
from app.services.grounding import GroundingTrace
from app.services.retriever import Retriever
from app.services.safety import check_emergency
from eval.gold import GOLD_PATH, GoldCategory, GoldRow, load_gold_set
from eval.metrics import (
    CITATION_ACCURACY_THRESHOLD,
    EFFECTIVE_ANSWER_RATE_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    RECALL_AT_K_THRESHOLD,
    SAFETY_RECALL_THRESHOLD,
    MetricVerdict,
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

logger = logging.getLogger(__name__)

EVAL_RETRIEVAL_CONFIG = RetrievalConfig(k=10, passage_limit=10, entity_limit=10)
"""Both retrieval paths share this for recall@k ONLY: ``vector_search`` returns ``config.k``
chunks, and ``retrieve`` returns up to ``config.passage_limit`` passages -- the same 10, so
"top-k" means the same thing on both sides of the comparison. Deliberately not
``GENERATION_RETRIEVAL_CONFIG``: recall@k measures retrieval quality at a fixed, equal width,
not the product's own generation headroom, and issue #103's own strict/containment split is
defined against this width -- changing it here would silently move every past recall number
(see the module docstring's "Two recall numbers" section). ``GENERATION_RETRIEVAL_CONFIG``
below is what every ``generate_answer`` call in this module runs retrieval with instead
(issue #167)."""

EVAL_LLM_TIMEOUT_SECONDS = 300.0
"""Longer than ``OllamaProvider``'s own interactive-request default (120s,
``app/llm/ollama.py``): a batch run over dozens of questions, back to back with no pause,
can tolerate a slow reply far better than a live ``/ask`` request can, and this local model
was observed taking over two minutes for a handful of calls under that sustained load, once
past the point a single interactive request ever runs. Set here, in this script alone --
never as a change to ``OllamaProvider``'s own shared default, which stays right-sized for the
interactive endpoint."""

OLLAMA_WAIT_POLL_SECONDS = 5.0
"""Issue #185: how often ``_build_ollama_waiter``'s probe re-checks ``/api/version`` once a
row has raised ``LLMUnreachableError``/``EmbedderUnreachableError``. 5s keeps a real outage
(measured: Ollama restarting under another project's LaunchAgent, back within roughly a
second) from hammering the daemon while it is mid-restart, without making a run that recovers
quickly wait any longer than it has to."""

OLLAMA_WAIT_TIMEOUT_SECONDS = 300.0
"""Issue #185: the bound on ``_build_ollama_waiter``'s probe loop -- the same 300s
``EVAL_LLM_TIMEOUT_SECONDS`` already gives one slow reply, since an outage this run cannot
wait out is exactly as fatal to the report as a single reply that never arrives. Past this,
``run_evaluation`` raises ``ProviderStillUnreachableError`` rather than keep polling forever."""

EXIT_PROVIDER_UNREACHABLE = 2
"""Process exit code for "the provider never came back -- no report was produced", the same
code ``eval.holdout``'s own ``EXIT_FAILED`` already uses for that kind of failure (issue
#185). Distinct from ``1`` (a finished report with a failing threshold)."""


class ProviderStillUnreachableError(Exception):
    """Raised by ``run_evaluation`` when a wait for the provider ran past its bound (issue #185).

    ``eval.compare``'s own ``_amain`` catches this one exception type, prints its message (a
    fixed sentence naming the row id and the seconds waited -- safe to print as-is, since a
    gold row id and an elapsed time are never the holdout's own content) instead of a report
    computed over a batch missing every row from this point on, and exits
    ``EXIT_PROVIDER_UNREACHABLE``. ``eval.holdout`` never prints this message: its own outer
    handler (``eval/holdout.py``'s ``main``) already prints only an uncaught exception's type
    name, whatever that exception is, so no privacy guarantee depends on this message's text.
    """


def _build_ollama_waiter(
    base_url: str,
    *,
    poll_seconds: float = OLLAMA_WAIT_POLL_SECONDS,
    timeout_seconds: float = OLLAMA_WAIT_TIMEOUT_SECONDS,
) -> Callable[[], Awaitable[bool]]:
    """Build a waiter that polls ``{base_url}/api/version`` until Ollama answers again.

    Issue #185, Scope 2. Returns a zero-argument async callable (``run_evaluation``'s
    ``wait_until_reachable``) rather than a bound method, so a caller with nothing to probe
    (``MockLLM``, every existing test double) can pass ``None`` instead.

    Args:
        base_url: Ollama's base URL -- the same ``settings.ollama_base_url`` the LLM provider
            and/or embedder were built with.
        poll_seconds: Seconds between probes. Defaults to the module constant; a test may pass
            a smaller value to keep a real-socket wait short (trap 5) without changing what a
            real run does.
        timeout_seconds: Total seconds to keep polling before giving up. Same reasoning as
            ``poll_seconds``.

    Returns:
        An async callable with no arguments, returning ``True`` the moment ``/api/version``
        answers with a 2xx status, or ``False`` once ``timeout_seconds`` has elapsed. Never
        raises: a transport error while probing is treated as "still unreachable", exactly
        like a probe that got no response at all. Issues no request other than this one GET
        -- no warm-up call against ``/api/chat`` (trap 3): the next row's own request reloads
        the model within ``EVAL_LLM_TIMEOUT_SECONDS``.
    """

    async def wait_until_reachable() -> bool:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                async with httpx2.AsyncClient(timeout=poll_seconds) as client:
                    response = await client.get(f"{base_url}/api/version")
                if response.status_code < 300:
                    return True
            except httpx2.TransportError:
                pass
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(poll_seconds)

    return wait_until_reachable


# --- issue #109: three gold rows (one of them the flagship compliance-min-store-area
# question) were timing out at exactly EVAL_LLM_TIMEOUT_SECONDS and dropping out of every
# metric. Confirmed against a real Ollama daemon before choosing a fix, not assumed:
#
#   * Re-running the three rows in isolation (no batch at all) reproduces the same timeout --
#     this is not "40 minutes of sustained load exhausts something", since it happens on the
#     very first call of a fresh process.
#   * The local model (this deployment) runs with a 4096-token context window
#     (``n_ctx=4096``, printed by the Ollama daemon at load), and this eval's own naive-path
#     prompts (raw, unranked whole-article chunks, unlike the graph-narrowed hybrid path a
#     live ``/ask`` request builds -- see ``_naive_retrieval_result``) plus some multi-hop
#     hybrid prompts routinely land at or past that ceiling. A prompt at the ceiling forces
#     llama.cpp to discard part of its own context mid-generation ("context shift") just to
#     keep decoding.
#   * Most of the time, decoding after a context shift still reaches a stop token quickly
#     (observed: a handful of tokens, well under a second's worth of throughput). Occasionally
#     -- observed directly, reproduced more than once against the real daemon -- it does not,
#     and generation runs on for thousands of tokens across repeated context shifts until the
#     client's timeout cuts it off. Temperature is low (``_TEMPERATURE = 0.1`` in
#     ``app/llm/ollama.py``) but not zero, so this is a real per-call coin flip, not a
#     deterministic property of the question.
#
# That evidence rules out "the batch exhausts something" as the mechanism, and it rules out
# "raise the timeout" as a fix: a run that is not converging toward a stop token does not
# reliably start converging at 600s just because it did not converge by 300s, and every row
# genuinely on that path would now cost twice as long for no confirmed benefit. It also rules
# out retrying a row immediately inline, in place, as the *only* fallback -- immediate retry
# is not wrong (this failure is a per-call coin flip, not a function of position in the batch,
# so retrying right away is exactly as sound as retrying later), but folding the retry into
# the main loop would mean a second 300s wait lands in the middle of the batch, and it
# would make the two attempts harder to keep straight in the code from the exclusion path
# below. Retrying failed rows as one clearly separated second pass, once each, after the
# first full pass over every row -- gets the same benefit (a fresh, independent sample from
# the same low-but-nonzero-temperature model, which is what actually resolves a coin-flip
# failure) with a simpler ``run_evaluation`` (one dispatch helper, called from two places) and
# an honest, visible accounting: a row that needed the second attempt is named as such in the
# report (``EvaluationReport.retried_row_ids``), not folded silently back into a normal pass,
# and a row that fails *both* attempts is still excluded and still named
# (``EvaluationReport.failed_row_ids``) -- the exclusion behaviour this issue requires is kept
# exactly as it was, just given one genuine second chance first.


def _create_llm(settings: Settings) -> LLMProvider:
    """Build the ``LLMProvider`` this eval run uses.

    Mirrors ``app.llm.factory.create_llm_provider``'s own choice of concrete provider, but
    gives ``OllamaProvider`` ``EVAL_LLM_TIMEOUT_SECONDS`` instead of its interactive default
    -- seconds this batch script can afford to wait, since it is not a live request.
    """
    if settings.llm_provider is LLMProviderKind.OLLAMA:
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=EVAL_LLM_TIMEOUT_SECONDS,
            num_ctx=settings.ollama_num_ctx,
        )
    return MockLLM()


FAITHFULNESS_NOTE = (
    "not measured: the only model available (qwen3:8b / MockLLM) is the same "
    "model that generates the answers, so it cannot judge them independently. A stronger, "
    "separate judge is required before this number means anything -- see the module docstring."
)


# --- naive retrieval, wrapped as a RetrievalResult, so generate_answer is unmodified --------


def _naive_passage(match: ChunkMatch) -> Passage:
    """Wrap one ``vector_search`` hit as a ``Passage`` with no graph information at all."""
    return Passage(
        unit_id=match.unit_id,
        document_id=match.document_id,
        dieu_no=match.dieu_no,
        khoan_no=match.khoan_no,
        effective=match.effective,
        text=match.body,
        similarity=match.similarity,
        hop=0,
        score=match.similarity,
        why=(Why.SIMILAR,),
        possibly_superseded=match.possibly_superseded,
    )


def _naive_retrieval_result(question: str, matches: Sequence[ChunkMatch]) -> RetrievalResult:
    """Build the naive-path ``RetrievalResult``: ``vector_search``'s chunks, nothing else.

    Handing this to the same ``generate_answer`` the hybrid path uses is what makes the
    citation-accuracy comparison honest -- the only difference from the hybrid call is which
    passages were offered, never how they are turned into an answer.
    """
    return RetrievalResult(question=question, passages=tuple(_naive_passage(m) for m in matches))


def _citable_unit_ids(result: RetrievalResult) -> list[str]:
    """Every unit id ``result`` actually makes citable -- not just each passage's own id.

    A passage is chunked at article granularity (``app/services/retriever.py``'s own module
    docstring: vector search returns a whole article, a requirement points at one clause
    inside it), so the clause the graph actually pointed at travels as ``Passage.focus_units``,
    not as the passage's own ``unit_id``. ``app.services.generator._build_sources`` treats
    both, plus every entity's ``sources``, as citable and validates a model's citation against
    that combined set (``_validate_citations``) -- recall@k is measured against the identical
    set, or it would silently score a real, correctly-surfaced clause-level citation as a miss.
    Naive retrieval has no graph and so no ``focus_units``/entities at all: this asymmetry is
    real, not a thumb on the scale -- clause-level attribution from a whole-article chunk is
    exactly the kind of thing only the graph can do.
    """
    ids: list[str] = []
    for passage in result.passages:
        ids.append(passage.unit_id)
        ids.extend(passage.focus_units)
    for entity in result.entities:
        ids.extend(entity.sources)
    return ids


# --- per-row scoring ------------------------------------------------------------------------


@dataclass(frozen=True)
class RowScore:
    """One answerable (``compliance``/``multi_hop``) row's naive-vs-hybrid comparison.

    Attributes:
        row_id: The gold row's own id.
        category: ``compliance`` or ``multi_hop``.
        naive_strict_recall: Exact-id recall@k of ``vector_search`` alone -- "can the system
            name the specific gold clause?" (``eval.metrics.recall_at_k``).
        hybrid_strict_recall: The same, for ``retrieve`` (graph expansion + rerank). This is
            the number the build-plan threshold gates on (see ``build_verdicts``).
        naive_containment_recall: "Did retrieval surface the text the answer needs?" -- the
            gold clause's text found inside *any* retrieved unit, article-level chunks
            included (``eval.metrics.containment_recall_at_k``). Reported for comparison, not
            gated by any threshold.
        hybrid_containment_recall: The same, for the hybrid path.
        naive_citation_precision: Precision of the citations ``generate_answer`` gave when
            handed only the naive passages; ``None`` when that path refused (nothing emitted).
        hybrid_citation_precision: The same, for the hybrid path.
        naive_citation_containment_precision: The same citations, scored under containment
            (``eval.metrics.citation_containment_precision``) instead of exact id match -- a
            diagnostic beside strict precision, never a replacement for it (issue #118's own
            follow-up review, the same strict/containment split #103 made for recall).
        hybrid_citation_containment_precision: The same, for the hybrid path.
        naive_outcome: What the naive path's ``generate_answer`` call returned.
        hybrid_outcome: What the hybrid path's ``generate_answer`` call returned.
        layer1_refused: Whether the hybrid path's deterministic document-number guard (layer 1)
            refused this row -- previously invisible (issue #118's own follow-up review).
        hybrid_prompt_eval_count: The hybrid path's own generation ``prompt_eval_count``
            (issue #129), read off ``GroundingTrace`` right after that call -- ``None`` for
            ``MockLLM`` or any provider that does not expose it.
        context_overflow: Whether either path's ``generate_answer`` call refused with
            ``refusal_reason == "context_overflow"`` (issue #129) -- a regression here must
            be visible in this table, not just in the provider's own error log.
        llm_unavailable: Whether either path's ``generate_answer`` call refused with
            ``refusal_reason == "llm_unavailable"`` (issue #112) -- the provider timed out or
            could not be reached. A model that needs this on many rows is slower than this
            batch's own ``EVAL_LLM_TIMEOUT_SECONDS`` tolerates, which is a property of that
            candidate to report, not something to paper over by raising the timeout further.
        naive_answer_check: ``eval.metrics.answer_check`` on the naive path's own answer text
            against this row's ``answer_must_match``/``answer_must_not_match`` patterns
            (issue #167) -- ``None`` when the row has no pattern at all (most rows).
        hybrid_answer_check: The same, for the hybrid path. Not gated by any threshold yet.
        verdict_contradicts_clause: Whether the hybrid path's ``generate_answer`` call refused
            with ``refusal_reason == "verdict_contradicts_clause"`` (issue #195) -- the
            deterministic check (``app.services.verdict``) found the answer's own opening
            "Có"/"Không" contradicts the threshold its own kept citation's text states.
        retried: Whether the hybrid path's ``generate_answer`` call ran its one allowed retry
            (issue #202, ``GroundingTrace.retried``) -- the per-citation verifier rejected
            every citation of the first attempt, and at least one rejected id was a passage
            that had actually been offered. Hybrid-only, the same way ``layer1_refused`` and
            ``verdict_contradicts_clause`` above are: the naive path's own trace is discarded
            right after this row is scored.
    """

    row_id: str
    category: GoldCategory
    naive_strict_recall: float
    hybrid_strict_recall: float
    naive_containment_recall: float
    hybrid_containment_recall: float
    naive_citation_precision: float | None
    hybrid_citation_precision: float | None
    naive_outcome: AskOutcome
    hybrid_outcome: AskOutcome
    grounding_table: tuple[int, int, int, int] = (0, 0, 0, 0)
    wrong_cited: bool = False
    naive_citation_containment_precision: float | None = None
    hybrid_citation_containment_precision: float | None = None
    layer1_refused: bool = False
    hybrid_prompt_eval_count: int | None = None
    context_overflow: bool = False
    llm_unavailable: bool = False
    naive_answer_check: bool | None = None
    hybrid_answer_check: bool | None = None
    verdict_contradicts_clause: bool = False
    retried: bool = False


@dataclass(frozen=True)
class RefusalScore:
    """One refusal-row outcome, retaining its reason for deterministic-guard reporting."""

    row_id: str
    outcome: AskOutcome
    refusal_reason: str | None


@dataclass(frozen=True)
class RowRetrieval:
    """One answerable row's retrieval at both widths ``eval.compare`` now needs (issue #167).

    Bundled into one argument so ``score_answerable_row`` takes it, ``llm`` and ``documents``
    as its own separate parameters rather than up to seven positional ones -- the same
    PLR0913 reasoning ``SupportCheck``/``_GroundingRequest`` already apply in
    ``app/services/generator.py``.

    Attributes:
        naive_matches: What ``Retriever.vector_search(row.question)`` returned at
            ``EVAL_RETRIEVAL_CONFIG`` -- scores recall@k only (``naive_strict_recall``/
            ``naive_containment_recall``), never fed to ``generate_answer``.
        hybrid_result: The same, for ``Retriever.retrieve`` -- scores recall@k only.
        generation_naive_matches: What ``Retriever.vector_search(row.question)`` returned at
            ``GENERATION_RETRIEVAL_CONFIG`` -- what the naive ``generate_answer`` call
            actually sees.
        generation_hybrid_result: The same, for the hybrid call.
    """

    naive_matches: Sequence[ChunkMatch]
    hybrid_result: RetrievalResult
    generation_naive_matches: Sequence[ChunkMatch]
    generation_hybrid_result: RetrievalResult

    @classmethod
    def single(
        cls, naive_matches: Sequence[ChunkMatch], hybrid_result: RetrievalResult
    ) -> "RowRetrieval":
        """Build a ``RowRetrieval`` that uses one retrieval for both recall@k and generation.

        For a caller with only one retrieval available (a hand-built test fixture, or a
        double that does not model the recall/generation split at all) -- recall@k and
        generation then see exactly the same passages, which was this module's only
        behaviour before issue #167.
        """
        return cls(naive_matches, hybrid_result, naive_matches, hybrid_result)


def _raise_if_llm_unreachable(row_id: str, response: AskResponse) -> None:
    """Re-raise an unreachable-provider refusal, so ``run_evaluation``'s retry pass sees it.

    Issue #183: ``generate_answer`` now catches ``LLMUnreachableError`` (a connection failure,
    not a timeout) and turns it into a ``refusal_reason == "llm_unreachable"`` response rather
    than letting it propagate -- the right thing for ``/ask``, but wrong for this module, which
    must keep treating a connection failure as infrastructure (excluded via the existing retry
    pass, exactly as the raw ``httpx2.ConnectError`` it replaces was excluded before this issue)
    rather than score it as a refusal. ``refusal_reason == "llm_unavailable"`` (a timeout) is
    untouched: issue #112 already scores that one as a real, listed refusal.
    """
    if response.refusal_reason == "llm_unreachable":
        msg = f"row {row_id}: generate_answer refused with llm_unreachable -- raising for retry"
        raise LLMUnreachableError(msg)


async def score_answerable_row(
    row: GoldRow, retrieval: RowRetrieval, llm: LLMProvider, documents: DocumentRepository
) -> RowScore:
    """Score one ``compliance``/``multi_hop`` row on both retrieval paths.

    Args:
        row: The gold row; ``row.source_unit_ids`` must be non-empty.
        retrieval: This row's retrieval at both widths -- recall@k's and generation's
            (``RowRetrieval``, issue #167).
        llm: The provider both ``generate_answer`` calls use -- the same one, so
            generation itself is not a variable between the two sides.
        documents: Looks up each cited unit's owning document (``generate_answer``'s own
            dependency).

    Returns:
        The recall (strict and containment) and citation precision of each path, plus each
        path's outcome and the deciding-fact answer check (issue #167).
    """
    naive_ids = [match.unit_id for match in retrieval.naive_matches]
    hybrid_ids = _citable_unit_ids(retrieval.hybrid_result)

    naive_trace = GroundingTrace()
    naive_response = await generate_answer(
        row.question,
        _naive_retrieval_result(row.question, retrieval.generation_naive_matches),
        llm,
        documents,
        naive_trace,
    )
    _raise_if_llm_unreachable(row.id, naive_response)
    grounding_trace = GroundingTrace()
    _latency_start = time.perf_counter()
    hybrid_response = await generate_answer(
        row.question, retrieval.generation_hybrid_result, llm, documents, grounding_trace
    )
    _raise_if_llm_unreachable(row.id, hybrid_response)
    # Total /ask latency for this row (generation included) -- observability only, never read
    # back by this module, never affects a scored metric. Not the grounding check's own added
    # latency: see "verifier_call_latency_ms=" below for that (issue #118's own follow-up
    # review -- this number alone was previously reported, wrongly, as "added latency").
    logger.info(
        "row %s: ask_latency_ms=%.1f",
        row.id,
        (time.perf_counter() - _latency_start) * 1000,
    )
    logger.info(
        "row %s: generation_prompt_eval_count=%s", row.id, grounding_trace.prompt_eval_count
    )
    for call_ms in grounding_trace.verifier_latencies_ms:
        logger.info("row %s: verifier_call_latency_ms=%.1f", row.id, call_ms)
    if grounding_trace.verifier_latencies_ms or grounding_trace.layer1_refusals:
        logger.info("row %s: citations_per_answer=%d", row.id, len(hybrid_response.citations))
    grounding_table = [0, 0, 0, 0]
    gold_ids = set(row.source_unit_ids)
    for unit_id, kept in grounding_trace.decisions:
        gold_correct = unit_id in gold_ids
        index = (0 if kept else 2) + (0 if gold_correct else 1)
        grounding_table[index] += 1
    hybrid_precision = citation_precision(
        [c.unit_id for c in hybrid_response.citations], row.source_unit_ids
    )
    naive_answer_check = answer_check(
        naive_response.outcome,
        naive_trace.pre_quote_answer,
        row.answer_must_match,
        row.answer_must_not_match,
    )
    hybrid_answer_check = answer_check(
        hybrid_response.outcome,
        grounding_trace.pre_quote_answer,
        row.answer_must_match,
        row.answer_must_not_match,
    )

    return RowScore(
        row_id=row.id,
        category=row.category,
        naive_strict_recall=recall_at_k(naive_ids, row.source_unit_ids),
        hybrid_strict_recall=recall_at_k(hybrid_ids, row.source_unit_ids),
        naive_containment_recall=containment_recall_at_k(naive_ids, row.source_unit_ids),
        hybrid_containment_recall=containment_recall_at_k(hybrid_ids, row.source_unit_ids),
        naive_citation_precision=citation_precision(
            [c.unit_id for c in naive_response.citations], row.source_unit_ids
        ),
        hybrid_citation_precision=hybrid_precision,
        naive_citation_containment_precision=citation_containment_precision(
            [c.unit_id for c in naive_response.citations], row.source_unit_ids
        ),
        hybrid_citation_containment_precision=citation_containment_precision(
            [c.unit_id for c in hybrid_response.citations], row.source_unit_ids
        ),
        naive_outcome=naive_response.outcome,
        hybrid_outcome=hybrid_response.outcome,
        grounding_table=tuple(grounding_table),
        wrong_cited=hybrid_response.outcome is AskOutcome.ANSWER and hybrid_precision == 0.0,
        layer1_refused=grounding_trace.layer1_refusals > 0,
        hybrid_prompt_eval_count=grounding_trace.prompt_eval_count,
        context_overflow=(
            naive_response.refusal_reason == "context_overflow"
            or hybrid_response.refusal_reason == "context_overflow"
        ),
        llm_unavailable=(
            naive_response.refusal_reason == "llm_unavailable"
            or hybrid_response.refusal_reason == "llm_unavailable"
        ),
        naive_answer_check=naive_answer_check,
        hybrid_answer_check=hybrid_answer_check,
        verdict_contradicts_clause=hybrid_response.refusal_reason == "verdict_contradicts_clause",
        retried=grounding_trace.retried,
    )


async def score_refusal_row(
    row: GoldRow, hybrid_result: RetrievalResult, llm: LLMProvider, documents: DocumentRepository
) -> RefusalScore:
    """Return the outcome of the deployed (hybrid) path on one ``refusal``-category row.

    Naive retrieval is not scored for refusal rows: the claim under test is whether the
    deployed system correctly declines, not a naive-vs-hybrid comparison (a refusal row cites
    nothing, so there is no recall or citation precision to compare).
    """
    _latency_start = time.perf_counter()
    response = await generate_answer(row.question, hybrid_result, llm, documents)
    logger.info(
        "row %s: ask_latency_ms=%.1f",
        row.id,
        (time.perf_counter() - _latency_start) * 1000,
    )
    _raise_if_llm_unreachable(row.id, response)
    return RefusalScore(row.id, response.outcome, response.refusal_reason)


def score_emergency_row(row: GoldRow) -> bool:
    """Return whether ``app.services.safety.check_emergency`` escalates this row's question.

    Calls ``check_emergency`` directly -- the exact function ``app.api.ask.ask`` calls before
    any other dependency is resolved -- and nothing else. Its signature takes a question and
    returns a verdict; it has no parameter an ``LLMProvider`` could even be passed through, so
    zero LLM calls on this path is true by construction, not by counting.
    """
    return check_emergency(row.question).escalated


# --- the whole run ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationReport:
    """Everything one full run over the gold set produced.

    Attributes:
        answerable: One ``RowScore`` per ``compliance``/``multi_hop`` row that was scored
            successfully, in gold-set order.
        refusal_outcomes: One outcome per ``refusal``-category row that was scored
            successfully.
        emergency_escalated: One ``bool`` per ``emergency``-category row.
        verdicts: The threshold verdicts this report gates the process exit code on,
            computed only from the rows that were scored successfully.
        retried_row_ids: Gold row ids that raised on their first attempt but were scored
            successfully on the one retry ``run_evaluation`` gives a failed row at the end of
            the batch (issue #109). Counted normally everywhere above -- ``answerable``,
            ``refusal_outcomes``, ``emergency_escalated`` and every verdict all include these
            rows exactly as if they had succeeded the first time -- this field exists only so
            the report states plainly that a second attempt was needed, rather than the retry
            being an invisible implementation detail.
        failed_row_ids: Gold row ids that raised on both the first attempt and the retry (a
            real model, run for real, occasionally times out or loses termination control near
            its context limit -- see ``run_evaluation``) rather than returning a score.
            Excluded from every metric above rather than counted as a pass or a fail, since
            neither would be true; named here so a run is never silently short a row.
        reranker: What reranked the hybrid path's passages (issue #114), as
            ``"<provider>, top-N <n>"``, or ``"off"``. Printed at the top of the table so a
            pasted table says which configuration produced it.
        rerank_ok: Retrievals the cross-encoder reranked successfully.
        rerank_fallbacks: Retrievals that kept the graph-proximity order because the reranker
            failed (sidecar down, timed out). Non-zero means the run is *not* a fully reranked
            measurement, and ``format_report`` says so instead of leaving it to be inferred.
        layer1_refusals: How many answerable rows the deterministic document-number guard
            (layer 1) refused -- previously invisible (issue #118's own follow-up review): a
            layer-1 refusal short-circuits before the per-citation verifier runs, so it never
            touched ``grounding_table`` either.
        verdict_contradicts_clause_refusals: How many answerable rows the deterministic
            verdict check (``app.services.verdict``, issue #195) refused -- a
            ``refusal_reason == "verdict_contradicts_clause"`` row, named here next to
            ``layer1_refusals`` since both are deterministic guards reported the same way.
        verdict_contradicts_clause_row_ids: The gold ids of those rows, so a reviewer can
            match each one against its own log line and the row's right verdict from gold.
        grounding_retry_row_ids: The gold ids of every answerable row whose hybrid call ran
            issue #202's one retry (``RowScore.retried``) -- named here, next to the other
            deterministic-guard ids, so a reviewer can match each one against its own
            ``generator: retrying once without rejected passage(s): ...`` log line.
        max_prompt_eval_count: The largest generation ``prompt_eval_count`` observed across
            every answerable row's hybrid call (issue #129) -- ``None`` when nothing was
            measured (``MockLLM``, or no answerable rows). Printed so a regression that
            silently widens the generation prompt again shows up here before it shows up as a
            truncation.
        context_overflow_row_ids: Gold row ids where either path's ``generate_answer`` call
            refused with ``refusal_reason == "context_overflow"`` (issue #129). Expected to be
            empty at ``num_ctx=16384`` against the current gold set and corpus -- a non-empty
            tuple here is the exact regression this field exists to catch.
        llm_unavailable_row_ids: Gold row ids where either path's ``generate_answer`` call
            refused with ``refusal_reason == "llm_unavailable"`` (issue #112) -- the provider
            timed out or could not be reached. Reported the same way as
            ``context_overflow_row_ids`` so a candidate model's own timeout rate is visible in
            the table, never folded silently into ``failed_row_ids`` (a different mechanism --
            see ``run_evaluation``'s own retry pass, which never sees this case at all, since
            ``generate_answer`` always returns a refusal here rather than raising).
        generation_k: This run's own generation ``k`` (issue #169) -- ``GENERATION_RETRIEVAL_
            CONFIG.k`` by default, or ``--generation-k`` when a sweep overrode it. Printed in
            the header so a pasted table says which seed width produced it.
        generation_passage_limit: The same, for ``passage_limit``/``--generation-passage-limit``.
        over_answered_row_ids: Gold ids of ``refusal`` rows the system answered instead of
            refusing (issue #178) -- the rows ``refusal_outcomes`` counts against it, named so
            ``eval.holdout --show`` can find them.
        unescalated_row_ids: Gold ids of ``emergency`` rows ``check_emergency`` did not
            escalate (issue #178), for the same reason.
        provider_wait_count: How many times ``run_evaluation`` waited for the provider to
            become reachable again (issue #185) -- once per row whose attempt raised
            ``LLMUnreachableError``/``EmbedderUnreachableError``, in either pass. ``0`` when
            ``wait_until_reachable`` was never given (``MockLLM`` runs, every test double) or
            when it was given but never needed.
        provider_wait_seconds: Total wall-clock seconds spent inside every
            ``wait_until_reachable`` call this run made. ``0.0`` alongside
            ``provider_wait_count == 0``.
    """

    answerable: tuple[RowScore, ...]
    refusal_outcomes: tuple[AskOutcome, ...]
    emergency_escalated: tuple[bool, ...]
    verdicts: tuple[MetricVerdict, ...]
    retried_row_ids: tuple[str, ...] = ()
    failed_row_ids: tuple[str, ...] = ()
    reranker: str = "off"
    rerank_ok: int = 0
    rerank_fallbacks: int = 0
    grounding_table: tuple[int, int, int, int] = (0, 0, 0, 0)
    wrong_cited_answers: int = 0
    layer1_refusals: int = 0
    verdict_contradicts_clause_refusals: int = 0
    verdict_contradicts_clause_row_ids: tuple[str, ...] = ()
    grounding_retry_row_ids: tuple[str, ...] = ()
    max_prompt_eval_count: int | None = None
    context_overflow_row_ids: tuple[str, ...] = ()
    llm_unavailable_row_ids: tuple[str, ...] = ()
    status_unknown_refusal_row_ids: tuple[str, ...] = ()
    generation_k: int = GENERATION_RETRIEVAL_CONFIG.k
    generation_passage_limit: int = GENERATION_RETRIEVAL_CONFIG.passage_limit
    over_answered_row_ids: tuple[str, ...] = ()
    unescalated_row_ids: tuple[str, ...] = ()
    provider_wait_count: int = 0
    provider_wait_seconds: float = 0.0


def _by_category(rows: Sequence[RowScore], category: GoldCategory) -> list[RowScore]:
    return [row for row in rows if row.category is category]


def _mean_strict_recall(rows: Sequence[RowScore], *, hybrid: bool) -> float | None:
    if not rows:
        return None
    return mean([row.hybrid_strict_recall if hybrid else row.naive_strict_recall for row in rows])


def _mean_containment_recall(rows: Sequence[RowScore], *, hybrid: bool) -> float | None:
    if not rows:
        return None
    values = [
        row.hybrid_containment_recall if hybrid else row.naive_containment_recall for row in rows
    ]
    return mean(values)


def _mean_citation_precision(rows: Sequence[RowScore], *, hybrid: bool) -> float | None:
    values = [
        (row.hybrid_citation_precision if hybrid else row.naive_citation_precision) for row in rows
    ]
    measured = [value for value in values if value is not None]
    return mean(measured) if measured else None


def _mean_citation_containment_precision(rows: Sequence[RowScore], *, hybrid: bool) -> float | None:
    values = [
        (
            row.hybrid_citation_containment_precision
            if hybrid
            else row.naive_citation_containment_precision
        )
        for row in rows
    ]
    measured = [value for value in values if value is not None]
    return mean(measured) if measured else None


def build_verdicts(
    answerable: Sequence[RowScore],
    refusal_outcomes: Sequence[AskOutcome],
    emergency_escalated: Sequence[bool],
) -> tuple[MetricVerdict, ...]:
    """Build the threshold-gated verdicts from one run's raw scores.

    Every gated verdict is computed on the hybrid (deployed) path only -- the naive path is
    reported for comparison but is not what ships, so it is never what the exit code gates on.

    The recall@k verdict below is strict recall (``eval.metrics.recall_at_k``, exact id
    match) -- unchanged from before issue #103. Containment recall
    (``eval.metrics.containment_recall_at_k``) is reported in the by-category table for both
    paths but is diagnostic only and never gates the exit code; see the module docstring.
    """
    hybrid_recall = _mean_strict_recall(answerable, hybrid=True)
    hybrid_citation = _mean_citation_precision(answerable, hybrid=True)
    hybrid_outcomes = tuple(row.hybrid_outcome for row in answerable)

    verdicts = [
        make_verdict(
            "recall@k (hybrid, all answerable, strict)",
            hybrid_recall,
            RECALL_AT_K_THRESHOLD,
            note=(
                "HEADLINE NUMBER -- exact id match (can the system name the specific gold "
                "clause?). Containment recall (did retrieval surface the gold clause's text "
                "at all, article-level chunks included) is reported per-category below for "
                "both paths, but is diagnostic only and does not gate this threshold"
            ),
        ),
        make_verdict(
            "citation accuracy (hybrid, all answerable)",
            hybrid_citation,
            CITATION_ACCURACY_THRESHOLD,
        ),
        make_verdict("faithfulness", None, FAITHFULNESS_THRESHOLD, note=FAITHFULNESS_NOTE),
        make_verdict(
            "effective-answer rate",
            effective_answer_rate(hybrid_outcomes) if hybrid_outcomes else None,
            EFFECTIVE_ANSWER_RATE_THRESHOLD,
        ),
        make_verdict(
            "safety detection",
            safety_recall(emergency_escalated) if emergency_escalated else None,
            SAFETY_RECALL_THRESHOLD,
            note="0 LLM calls by construction -- check_emergency takes no LLM at all",
        ),
    ]
    if refusal_outcomes:
        verdicts.append(
            make_verdict(
                "refusal correctness (diagnostic, no build-plan threshold)",
                refusal_correctness(refusal_outcomes),
                None,
                note="distinguishes over-answering from the effective-answer rate above",
            )
        )
    return tuple(verdicts)


async def _score_one_row(
    row: GoldRow,
    retriever: Retriever,
    generation_retriever: Retriever,
    documents: DocumentRepository,
    llm: LLMProvider,
) -> RowScore | RefusalScore | bool:
    """Score one gold row through the path its category calls for, and return its raw result.

    One attempt, no exception handling -- ``run_evaluation`` below is what catches a raise and
    decides whether this was the first attempt (queue a retry) or the retry itself (exclude for
    good). Factored out so both of ``run_evaluation``'s passes call the exact same dispatch
    logic; the two passes must never be able to drift into scoring a row differently depending
    on which pass caught it.

    Args:
        row: The gold row to score.
        retriever: Retrieves at ``EVAL_RETRIEVAL_CONFIG`` -- feeds recall@k only.
        generation_retriever: Retrieves at ``GENERATION_RETRIEVAL_CONFIG`` (issue #167) --
            feeds every ``generate_answer`` call, on an answerable row's both paths and on a
            refusal row's single (hybrid) path alike, so generation always runs the way
            ``/ask`` does.
        documents: Looks up each cited unit's owning document (``generate_answer``'s own
            dependency).
        llm: The provider every ``generate_answer`` call uses.

    Returns:
        A ``RowScore`` for a ``compliance``/``multi_hop`` row, a ``RefusalScore`` for a
        ``refusal`` row, or a ``bool`` for an ``emergency`` row -- ``_record_outcome`` below
        knows which accumulator each type belongs in.
    """
    if row.category in (GoldCategory.COMPLIANCE, GoldCategory.MULTI_HOP):
        retrieval = RowRetrieval(
            naive_matches=await retriever.vector_search(row.question),
            hybrid_result=await retriever.retrieve(row.question),
            generation_naive_matches=await generation_retriever.vector_search(row.question),
            generation_hybrid_result=await generation_retriever.retrieve(row.question),
        )
        return await score_answerable_row(row, retrieval, llm, documents)
    if row.category is GoldCategory.REFUSAL:
        hybrid_result = await generation_retriever.retrieve(row.question)
        return await score_refusal_row(row, hybrid_result, llm, documents)
    return score_emergency_row(row)


def _record_outcome(
    outcome: RowScore | RefusalScore | bool,
    answerable: list[RowScore],
    refusal_scores: list[RefusalScore],
    emergency_escalated: list[bool],
) -> None:
    """Append one row's ``_score_one_row`` result to whichever accumulator its type identifies.

    A plain type switch, not a category switch: ``_score_one_row`` already picked the right
    path, and this only needs to know where the *shape* of what came back belongs, so it never
    needs ``row.category`` at all.
    """
    if isinstance(outcome, RowScore):
        answerable.append(outcome)
    elif isinstance(outcome, RefusalScore):
        refusal_scores.append(outcome)
    else:
        emergency_escalated.append(outcome)


_UNREACHABLE_ERRORS = (LLMUnreachableError, EmbedderUnreachableError)
"""The only exceptions ``run_evaluation``'s wait reacts to (issue #185, trap 1). Anything
else -- a timeout (``LLMUnavailableError`` alone, which ``generate_answer`` already turns
into a scored ``llm_unavailable`` refusal rather than raising at all -- issue #112) or a
non-2xx status (a real bug, issue #183) -- keeps #109's plain immediate-retry-at-the-end
behaviour, with no wait: waiting on every exception would hide a slow model behind the wait,
the exact failure mode this split exists to avoid."""


async def _wait_if_unreachable(
    row: GoldRow,
    error: Exception,
    wait_until_reachable: Callable[[], Awaitable[bool]] | None,
) -> float | None:
    """Wait for the provider after an outage, before the caller scores the next row.

    A module-level function, not a closure inside ``run_evaluation``, to keep that function's
    own statement count under this repository's C10 cap (``pyproject.toml``'s
    ``PLR0915``/``max-complexity``) -- the logic itself is exactly what issue #185 Scope 1
    asks for: a no-op for anything that is not ``_UNREACHABLE_ERRORS``, or when no waiter was
    given at all. The two log lines carry only the row id and the exception's type name
    (issue #185, trap 2: ``eval.holdout`` prints ``eval.compare``'s own log lines, so neither
    one may carry a question, a reference answer or ``str(error)``, which could quote a URL
    with a payload).

    Returns:
        ``None`` if no wait was attempted at all (not an unreachable error, or no waiter).
        Otherwise, the seconds actually spent inside ``wait_until_reachable`` -- the caller
        adds this into ``EvaluationReport.provider_wait_seconds`` and counts one wait into
        ``provider_wait_count``, whatever the elapsed value (even ``0.0``, in a test with an
        instant fake waiter).

    Raises:
        ProviderStillUnreachableError: If ``wait_until_reachable`` returned ``False`` -- the
            provider never came back within its own bound.
    """
    if wait_until_reachable is None or not isinstance(error, _UNREACHABLE_ERRORS):
        return None
    logger.info(
        "waiting for Ollama after row %s (%s), up to %d s",
        row.id,
        type(error).__name__,
        int(OLLAMA_WAIT_TIMEOUT_SECONDS),
    )
    started = time.monotonic()
    reachable = await wait_until_reachable()
    elapsed = time.monotonic() - started
    if not reachable:
        msg = (
            f"gave up waiting for the provider after row {row.id}: still unreachable "
            f"after {elapsed:.0f}s"
        )
        raise ProviderStillUnreachableError(msg)
    logger.info("Ollama answered again after %.0f s", elapsed)
    return elapsed


@dataclass(frozen=True)
class RunOptions:
    """``run_evaluation``'s less-often-overridden arguments, bundled (issue #185).

    Keeps ``run_evaluation`` itself at this repository's own five-argument cap (C10,
    ``pyproject.toml``'s ``max-args = 5``) rather than a ``# noqa`` -- the same PLR0913
    reasoning ``RowRetrieval``/``SupportCheck``/``_GroundingRequest`` already apply, now that
    issue #185 adds a second optional argument where issue #167's ``generation_retriever``
    had already used the cap's last slot.

    Attributes:
        generation_retriever: Retrieves at ``GENERATION_RETRIEVAL_CONFIG`` (issue #167) --
            feeds every ``generate_answer`` call. Defaults to ``retriever`` itself when left
            ``None``, so a caller with only one retriever (e.g. an existing test double)
            keeps its old, single-config behaviour unchanged.
        wait_until_reachable: Probes the provider and returns once it answers again, or
            ``False`` once it gives up (``_build_ollama_waiter``). ``None`` (the default --
            every ``MockLLM`` run, every existing test) means an outage is queued/excluded
            exactly as it was before this issue, with no wait at all.
    """

    generation_retriever: Retriever | None = None
    wait_until_reachable: Callable[[], Awaitable[bool]] | None = None


async def run_evaluation(
    rows: Sequence[GoldRow],
    retriever: Retriever,
    documents: DocumentRepository,
    llm: LLMProvider,
    options: RunOptions | None = None,
) -> EvaluationReport:
    """Run every gold row through the path its category calls for, and score the result.

    Two passes (issue #109). The first tries every row once; a row whose retrieval or
    generation call raises (a real network timeout, or a real local model that loses
    termination control near its context limit -- both observed running this against a real
    Ollama daemon, and confirmed to reproduce outside a long batch too -- see
    ``EVAL_LLM_TIMEOUT_SECONDS``'s own comment) is logged by id and queued for the second pass
    rather than excluded immediately. The second pass retries each queued row exactly once,
    with the same retriever/LLM and no special-casing; a row that succeeds this time is scored
    completely normally (``EvaluationReport.retried_row_ids`` only records that it needed the
    second attempt, it does not change how the row counts). A row that raises again on the
    retry is excluded from every metric, the same as before this issue -- never counted as a
    pass or a fail, since neither would be a real measurement. One row's infrastructure hiccup
    must not discard the other thirty-odd rows' real results, and it must not silently vanish
    either -- see ``EvaluationReport.failed_row_ids``.

    Issue #185: when a row's attempt raises ``LLMUnreachableError`` or
    ``EmbedderUnreachableError`` (an outage, not a timeout -- see ``_UNREACHABLE_ERRORS``),
    and ``wait_until_reachable`` is given, this waits for the provider to come back before
    scoring the next row, in either pass -- an outage that excludes every row from this point
    on (the problem this issue measured, #182's "Outage during the candidate run") rather than
    just the one row the existing retry pass already tolerates. If the wait exhausts its own
    bound, ``ProviderStillUnreachableError`` is raised straight out of this function: a report
    computed over a batch missing every remaining row would be worse than no report at all.

    Args:
        rows: The gold rows to score.
        retriever: Retrieves at ``EVAL_RETRIEVAL_CONFIG`` -- feeds recall@k only.
        documents: Looks up each cited unit's owning document.
        llm: The provider every ``generate_answer`` call uses.
        options: ``generation_retriever`` and ``wait_until_reachable`` (``RunOptions``,
            bundled to stay under this repository's five-argument cap). Defaults to
            ``RunOptions()`` -- every field's own default -- when omitted entirely.

    Raises:
        ProviderStillUnreachableError: If ``options.wait_until_reachable`` was given and
            returned ``False`` -- the provider never came back within its own bound.
    """
    options = options or RunOptions()
    generation_retriever = (
        retriever if options.generation_retriever is None else options.generation_retriever
    )
    wait_until_reachable = options.wait_until_reachable
    answerable: list[RowScore] = []
    refusal_scores: list[RefusalScore] = []
    emergency_escalated: list[bool] = []
    unescalated: list[str] = []
    wait_count = 0
    wait_seconds = 0.0

    to_retry: list[GoldRow] = []
    for row in rows:
        try:
            outcome = await _score_one_row(row, retriever, generation_retriever, documents, llm)
        except Exception as error:
            logger.exception(
                "run_evaluation: %s raised on its first attempt -- retrying once, at the end "
                "of the batch, before deciding whether to exclude it",
                row.id,
            )
            to_retry.append(row)
            elapsed = await _wait_if_unreachable(row, error, wait_until_reachable)
            if elapsed is not None:
                wait_count += 1
                wait_seconds += elapsed
            continue
        _record_outcome(outcome, answerable, refusal_scores, emergency_escalated)
        if outcome is False:
            unescalated.append(row.id)

    retried_row_ids: list[str] = []
    failed_row_ids: list[str] = []
    for row in to_retry:
        try:
            outcome = await _score_one_row(row, retriever, generation_retriever, documents, llm)
        except Exception as error:
            logger.exception(
                "run_evaluation: %s raised again on its retry -- excluded from every metric",
                row.id,
            )
            failed_row_ids.append(row.id)
            elapsed = await _wait_if_unreachable(row, error, wait_until_reachable)
            if elapsed is not None:
                wait_count += 1
                wait_seconds += elapsed
            continue
        _record_outcome(outcome, answerable, refusal_scores, emergency_escalated)
        if outcome is False:
            unescalated.append(row.id)
        retried_row_ids.append(row.id)

    refusal_outcomes = [score.outcome for score in refusal_scores]
    verdicts = build_verdicts(answerable, refusal_outcomes, emergency_escalated)
    grounding_table = tuple(
        sum(row.grounding_table[index] for row in answerable) for index in range(4)
    )
    prompt_eval_counts = [
        row.hybrid_prompt_eval_count
        for row in answerable
        if row.hybrid_prompt_eval_count is not None
    ]
    return EvaluationReport(
        answerable=tuple(answerable),
        refusal_outcomes=tuple(refusal_outcomes),
        emergency_escalated=tuple(emergency_escalated),
        verdicts=verdicts,
        retried_row_ids=tuple(retried_row_ids),
        failed_row_ids=tuple(failed_row_ids),
        grounding_table=grounding_table,
        wrong_cited_answers=sum(row.wrong_cited for row in answerable),
        layer1_refusals=sum(row.layer1_refused for row in answerable),
        verdict_contradicts_clause_refusals=sum(
            row.verdict_contradicts_clause for row in answerable
        ),
        verdict_contradicts_clause_row_ids=tuple(
            row.row_id for row in answerable if row.verdict_contradicts_clause
        ),
        grounding_retry_row_ids=tuple(row.row_id for row in answerable if row.retried),
        max_prompt_eval_count=max(prompt_eval_counts) if prompt_eval_counts else None,
        context_overflow_row_ids=tuple(row.row_id for row in answerable if row.context_overflow),
        llm_unavailable_row_ids=tuple(row.row_id for row in answerable if row.llm_unavailable),
        status_unknown_refusal_row_ids=tuple(
            score.row_id for score in refusal_scores if score.refusal_reason == "status_unknown"
        ),
        over_answered_row_ids=tuple(
            score.row_id for score in refusal_scores if score.outcome is not AskOutcome.REFUSAL
        ),
        unescalated_row_ids=tuple(unescalated),
        provider_wait_count=wait_count,
        provider_wait_seconds=wait_seconds,
    )


# --- printing ----------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _verdict_line(verdict: MetricVerdict) -> str:
    status = "PASS" if verdict.passed else ("FAIL" if verdict.passed is False else "UNMEASURED")
    threshold = "" if verdict.threshold is None else f" (threshold {_fmt(verdict.threshold)})"
    note = f" -- {verdict.note}" if verdict.note else ""
    return f"  [{status}] {verdict.name}: {_fmt(verdict.value)}{threshold}{note}"


def _category_line(label: str, rows: Sequence[RowScore]) -> str:
    naive_strict = _fmt(_mean_strict_recall(rows, hybrid=False))
    hybrid_strict = _fmt(_mean_strict_recall(rows, hybrid=True))
    naive_contain = _fmt(_mean_containment_recall(rows, hybrid=False))
    hybrid_contain = _fmt(_mean_containment_recall(rows, hybrid=True))
    naive_citation = _fmt(_mean_citation_precision(rows, hybrid=False))
    hybrid_citation = _fmt(_mean_citation_precision(rows, hybrid=True))
    naive_citation_contain = _fmt(_mean_citation_containment_precision(rows, hybrid=False))
    hybrid_citation_contain = _fmt(_mean_citation_containment_precision(rows, hybrid=True))
    return (
        f"  {label:<14} n={len(rows):<3} "
        f"recall strict naive={naive_strict} hybrid={hybrid_strict}  "
        f"recall containment naive={naive_contain} hybrid={hybrid_contain}  "
        f"citation strict naive={naive_citation} hybrid={hybrid_citation}  "
        f"citation containment naive={naive_citation_contain} hybrid={hybrid_citation_contain}"
    )


def _row_lines(rows: Sequence[RowScore]) -> list[str]:
    """One line per answerable row, so a row where hybrid does not beat naive is visible.

    Printed for every row, not only the losing ones -- a table that only shows its losses (or
    only its wins) is as misleading as one that hides them. Shows strict recall (the headline
    metric); ``_category_line`` above is where containment recall's aggregate is reported.

    Each line also carries that row's hybrid citation precision and outcome (issue #114): two
    runs of a local model at temperature 0.1 do not score the same rows (a timeout drops one, a
    coin-flip refuses another), so comparing the aggregate of one run with another's says little
    until the reader can see which rows moved.
    """
    lines = ["  per-question recall@k, strict (naive -> hybrid):"]
    for row in rows:
        if row.hybrid_strict_recall > row.naive_strict_recall:
            flag = "better"
        elif row.hybrid_strict_recall == row.naive_strict_recall:
            flag = "same"
        else:
            flag = "WORSE -- hybrid did not beat naive"
        # Issue #167: the hybrid path's deciding-fact check, only for a row that has a
        # pattern to check -- most rows print nothing extra here.
        answer_check_suffix = ""
        if row.hybrid_answer_check is not None:
            answer_check_suffix = f"  answer check {'PASS' if row.hybrid_answer_check else 'FAIL'}"
        lines.append(
            f"    {row.row_id:<40} [{row.category.value:<10}] "
            f"{row.naive_strict_recall:.2f} -> {row.hybrid_strict_recall:.2f}  ({flag})  "
            f"hybrid citation {_fmt(row.hybrid_citation_precision)} ({row.hybrid_outcome.value})"
            f"{answer_check_suffix}"
        )
    return lines


def _answer_check_lines(rows: Sequence[RowScore]) -> list[str]:
    """Report the "answer check" line for each path (issue #167), the way ``FAITHFULNESS_NOTE`` is.

    Not gated by any threshold yet -- see the module docstring's "Checking the deciding fact"
    section -- so this is printed plainly, never folded into ``Thresholds`` below, where a
    reader could mistake the absence of a gate for a pass.
    """

    def _tally(*, hybrid: bool) -> tuple[int, int]:
        checks = [(row.hybrid_answer_check if hybrid else row.naive_answer_check) for row in rows]
        measured = [check for check in checks if check is not None]
        return sum(measured), len(measured)

    naive_passed, naive_total = _tally(hybrid=False)
    hybrid_passed, hybrid_total = _tally(hybrid=True)
    return [
        "Answer check (issue #167, not gated by any threshold yet -- see GoldRow."
        "answer_must_match/answer_must_not_match): does the answer's OWN text state the "
        "row's deciding fact, checked on rows that define a pattern only:",
        f"  naive:  {naive_passed}/{naive_total} rows with patterns passed",
        f"  hybrid: {hybrid_passed}/{hybrid_total} rows with patterns passed",
    ]


def _reranker_lines(report: EvaluationReport) -> list[str]:
    """Say what reranked the hybrid path, and shout if a configured reranker did not do its job.

    Reranking degrades silently by design (``Retriever`` falls back when the sidecar is down),
    which is right for ``/ask`` and wrong for a measurement: a run whose sidecar died would print
    a table that looks reranked and is not. So the fallbacks are counted and printed here.
    """
    if report.reranker == "off":
        return [
            "Reranker: off (hybrid path is graph-proximity order only; naive is never reranked)",
            "",
        ]
    lines = [
        f"Reranker: {report.reranker} -- hybrid path only, naive is never reranked; "
        f"{report.rerank_ok} retrieval(s) reranked, {report.rerank_fallbacks} fell back",
    ]
    if report.rerank_fallbacks:
        lines.append(
            f"WARNING: {report.rerank_fallbacks} retrieval(s) kept the un-reranked order because "
            "the reranker failed -- this is NOT a fully reranked measurement."
        )
    return [*lines, ""]


def _grounding_retry_line(report: EvaluationReport) -> str:
    """Return issue #202's own report line, naming retried rows and each one's outcome.

    Answered/refused is read off each retried row's own ``hybrid_outcome`` (``report.
    answerable`` is in gold-set order; ``grounding_retry_row_ids`` names the subset that
    retried), never a separate counter kept alongside it that could drift from the rows
    themselves.
    """
    retried = [row for row in report.answerable if row.row_id in report.grounding_retry_row_ids]
    answered = sum(1 for row in retried if row.hybrid_outcome is AskOutcome.ANSWER)
    refused = len(retried) - answered
    line = (
        f"grounding retries (issue #202): {len(retried)} rows -- "
        f"{answered} answered, {refused} refused"
    )
    if report.grounding_retry_row_ids:
        line += f" -- {', '.join(report.grounding_retry_row_ids)}"
    return line


def format_report(report: EvaluationReport) -> str:
    """Render ``report`` as the printed table -- pure text formatting, no I/O of its own."""
    compliance = _by_category(report.answerable, GoldCategory.COMPLIANCE)
    multi_hop = _by_category(report.answerable, GoldCategory.MULTI_HOP)

    lines = [
        "GraphRAG vs naive vector search -- gold-set evaluation (issue #96)",
        "",
        "Retrieval configs (issue #167; both configs' k printed since issue #169): recall@k is "
        "scored at EVAL_RETRIEVAL_CONFIG "
        f"(k={EVAL_RETRIEVAL_CONFIG.k}, passage_limit={EVAL_RETRIEVAL_CONFIG.passage_limit}, "
        f"entity_limit={EVAL_RETRIEVAL_CONFIG.entity_limit}); every generate_answer call runs "
        "retrieval at this run's generation config "
        f"(k={report.generation_k}, passage_limit={report.generation_passage_limit}, "
        f"entity_limit={GENERATION_RETRIEVAL_CONFIG.entity_limit}) -- GENERATION_RETRIEVAL_"
        "CONFIG's own values by default, overridable for a sweep via --generation-k/"
        "--generation-passage-limit, on both the naive and the hybrid path alike.",
        "",
        *_reranker_lines(report),
        "Two recall numbers below, because the gold set is clause-level and the corpus is",
        "chunked at article level (issue #103):",
        "  recall strict       -- exact id match: can the system name the specific clause?",
        "                          This is the HEADLINE number the build-plan threshold gates",
        "                          on (hybrid path -- see the 'strict' verdict below).",
        "  recall containment  -- the gold clause's text found inside ANY retrieved unit,",
        "                          article-level chunks included: did retrieval surface the",
        "                          text the answer needs? Reported for comparison, not gated.",
        "Citation precision gets the same strict/containment split (issue #118's own follow-up",
        "review): strict is the exact-id match the build-plan threshold gates on; containment",
        "credits a citation at article granularity when gold names one of its clauses, so a",
        "fix that makes the grounding check see an article's full text (rather than a bare,",
        "possibly-empty heading) can be judged on whether answers got better even where it",
        "makes the cited id coarser than an exact gold match.",
        "",
        "By category (recall strict/containment, citation strict/containment -- naive vs hybrid):",
        _category_line("compliance", compliance),
        _category_line("multi_hop", multi_hop),  # the separate multi-hop row the issue asks for
        _category_line("all answerable", report.answerable),
        "",
        *_row_lines(report.answerable),
        "",
        *_answer_check_lines(report.answerable),
        "",
        f"refusal rows: {len(report.refusal_outcomes)}, "
        f"emergency rows: {len(report.emergency_escalated)}",
        "",
        "Grounding check (kept/dropped x gold-correct/not gold-correct):",
        f"  kept, gold-correct={report.grounding_table[0]}  "
        f"kept, not gold-correct={report.grounding_table[1]}",
        f"  dropped, gold-correct={report.grounding_table[2]}  "
        f"dropped, not gold-correct={report.grounding_table[3]}",
        f"wrong-cited answers: {report.wrong_cited_answers}",
        f"layer-1 (document-number guard) refusals: {report.layer1_refusals}",
        f"verdict-check (issue #195) refusals: {report.verdict_contradicts_clause_refusals}"
        + (
            f" -- {', '.join(report.verdict_contradicts_clause_row_ids)}"
            if report.verdict_contradicts_clause_row_ids
            else ""
        ),
        _grounding_retry_line(report),
        "",
        "Context window (issue #129):",
        f"  max generation prompt_eval_count observed: {_fmt_int(report.max_prompt_eval_count)}",
        f"  context_overflow rows: {len(report.context_overflow_row_ids)}"
        + (
            f" -- {', '.join(report.context_overflow_row_ids)}"
            if report.context_overflow_row_ids
            else ""
        ),
        f"  llm_unavailable rows: {len(report.llm_unavailable_row_ids)}"
        + (
            f" -- {', '.join(report.llm_unavailable_row_ids)}"
            if report.llm_unavailable_row_ids
            else ""
        ),
        f"  status_unknown refusal rows: {len(report.status_unknown_refusal_row_ids)}"
        + (
            f" -- {', '.join(report.status_unknown_refusal_row_ids)}"
            if report.status_unknown_refusal_row_ids
            else ""
        ),
        f"  provider waits: {report.provider_wait_count}, {report.provider_wait_seconds:.0f} s",
        "",
        "Thresholds (build plan A8):",
        *[_verdict_line(v) for v in report.verdicts],
    ]
    if report.retried_row_ids:
        lines += [
            "",
            f"RETRIED (raised once, succeeded on the one retry at the end of the batch -- "
            f"issue #109 -- scored normally above, listed here only so the retry stays "
            f"visible): {', '.join(report.retried_row_ids)}",
        ]
    if report.failed_row_ids:
        lines += [
            "",
            f"FAILED TO EVALUATE (excluded from every metric above, not scored either way, "
            f"even after a retry at the end of the batch): {', '.join(report.failed_row_ids)}",
        ]
    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gold-set evaluation and print its table.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        ``0`` when every measured threshold passed, ``1`` when at least one failed, or
        ``EXIT_PROVIDER_UNREACHABLE`` (``2``) if a wait for the provider (issue #185) ran past
        its bound -- no report is printed in that case, since it would be computed over a
        batch missing every row from that point on. A verdict with nothing measured
        (faithfulness) never contributes to a non-zero exit.
    """
    return asyncio.run(_amain(argv))


async def _amain(argv: Sequence[str] | None) -> int:
    configure_logging()
    args = _parse_args(argv)

    rows = load_gold_set(Path(args.gold_path))
    if args.limit:
        rows = rows[: args.limit]

    settings = get_settings()
    try:
        report = await run_gold_evaluation(
            rows,
            settings,
            args.database_url or settings.database_url,
            _resolve_generation_config(args, settings),
        )
    except ProviderStillUnreachableError as error:
        # Trap 4: never print a report computed over a batch that lost every row from the
        # point of the outage on. The message is a fixed sentence naming only the row id and
        # the seconds waited (see the exception's own docstring).
        print(str(error), file=sys.stderr)
        return EXIT_PROVIDER_UNREACHABLE
    if report is None:
        return 1

    print(format_report(report))
    return exit_code(report.verdicts)


async def run_gold_evaluation(
    rows: Sequence[GoldRow],
    settings: Settings,
    database_url: str | None,
    generation_config: RetrievalConfig,
) -> EvaluationReport | None:
    """Build the engine, embedder, LLM, reranker and both retrievers, and score ``rows``.

    The run setup ``eval.compare`` and ``eval.holdout`` (issue #178) share, so the holdout is
    scored by exactly the code path the dev set is -- same scorers, same ``/ask`` generation
    config -- and the two can never drift apart.

    Args:
        rows: The gold rows to score.
        settings: Process settings: embedder, LLM, reranker and ``rerank_top_n``.
        database_url: Postgres URL to read from; ``None`` or empty means none is configured.
        generation_config: The retrieval config every ``generate_answer`` call runs with
            (``_resolve_generation_config``).

    Returns:
        The finished report, or ``None`` (after logging why) when no database is configured.

    Raises:
        ProviderStillUnreachableError: See ``run_evaluation``. Only possible when
            ``settings.llm_provider`` is ``ollama`` or ``settings.embedder`` is ``bge-m3`` --
            the only cases a waiter is built for at all (issue #185, Scope 2).
    """
    embedder = create_embedder(settings)
    llm = _create_llm(settings)
    reranker = create_reranker(settings)
    logger.info(
        "Embedder: %s, LLM: %s, reranker: %s",
        settings.embedder.value,
        settings.llm_provider.value,
        settings.reranker.value,
    )
    # Issue #185, Scope 2: a probe is only meaningful when something in this run actually
    # talks to the local Ollama daemon -- MockLLM/MockEmbedder reach no network at all, and
    # waiting for a daemon neither side calls would just slow down every offline test run.
    wait_until_reachable = (
        _build_ollama_waiter(settings.ollama_base_url)
        if settings.llm_provider is LLMProviderKind.OLLAMA
        or settings.embedder is EmbedderProvider.BGE_M3
        else None
    )

    if not database_url:
        logger.error("no database configured -- pass --database-url or set DATABASE_URL")
        return None

    engine = create_engine(database_url)
    try:
        async with session_scope(create_session_factory(engine)) as session:
            # Issue #167: recall@k keeps measuring at EVAL_RETRIEVAL_CONFIG's own fixed
            # width (unchanged by this issue), while every generate_answer call runs
            # retrieval the way /ask does -- GENERATION_RETRIEVAL_CONFIG, with the same
            # rerank_top_n app/main.py gives it. Two Retriever instances, one retrieval
            # each per row, rather than one retriever trying to serve both widths at once.
            recall_config = EVAL_RETRIEVAL_CONFIG.model_copy(
                update={"rerank_top_n": settings.rerank_top_n}
            )
            retriever = Retriever(session, embedder, config=recall_config, reranker=reranker)
            generation_retriever = Retriever(
                session, embedder, config=generation_config, reranker=reranker
            )
            documents = DocumentRepository(session)
            report = await run_evaluation(
                rows,
                retriever,
                documents,
                llm,
                RunOptions(
                    generation_retriever=generation_retriever,
                    wait_until_reachable=wait_until_reachable,
                ),
            )
            return dataclasses.replace(
                report,
                reranker=_reranker_label(settings),
                rerank_ok=retriever.rerank_ok + generation_retriever.rerank_ok,
                rerank_fallbacks=retriever.rerank_fallbacks + generation_retriever.rerank_fallbacks,
                generation_k=generation_config.k,
                generation_passage_limit=generation_config.passage_limit,
            )
    finally:
        await engine.dispose()


def _reranker_label(settings: Settings) -> str:
    """Name the reranker configuration a run used, for the table's header (``"off"`` if none)."""
    if settings.reranker is RerankerProvider.NONE:
        return "off"
    return f"{settings.reranker.value}, top-N {settings.rerank_top_n}"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.compare",
        description="Compare GraphRAG (hybrid) against naive vector search on the gold set.",
    )
    parser.add_argument(
        "--gold-path",
        default=str(GOLD_PATH),
        help=f"Gold set YAML file (default: {GOLD_PATH}).",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL to read from (default: the DATABASE_URL setting/env var).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N gold rows."
    )
    parser.add_argument(
        "--generation-k",
        type=int,
        default=None,
        help=(
            "Override the generation retriever's k (vector-search seed count) for this run "
            "only (issue #169 sweep). Defaults to GENERATION_RETRIEVAL_CONFIG.k -- the value "
            "/ask itself runs with. Never affects EVAL_RETRIEVAL_CONFIG or recall@k."
        ),
    )
    parser.add_argument(
        "--generation-passage-limit",
        type=int,
        default=None,
        help=(
            "Override the generation retriever's passage_limit for this run only (issue #169 "
            "sweep). Defaults to GENERATION_RETRIEVAL_CONFIG.passage_limit -- the value /ask "
            "itself runs with. Never affects EVAL_RETRIEVAL_CONFIG or recall@k."
        ),
    )
    return parser.parse_args(argv)


def _resolve_generation_config(args: argparse.Namespace, settings: Settings) -> RetrievalConfig:
    """Build this run's generation retrieval config (issue #169).

    ``GENERATION_RETRIEVAL_CONFIG`` is the fixed module constant /ask always runs with;
    ``--generation-k``/``--generation-passage-limit`` let a sweep measure a different point
    without editing that constant. Neither flag given reproduces ``/ask``'s own config exactly
    (plus ``rerank_top_n``, as ``app/main.py`` already builds it) -- an ordinary run is
    unaffected by this function's existence.

    Args:
        args: Parsed CLI arguments (``_parse_args``).
        settings: Process settings, for ``rerank_top_n``.

    Returns:
        ``GENERATION_RETRIEVAL_CONFIG``, with ``rerank_top_n`` always applied and ``k``/
        ``passage_limit`` replaced only where the corresponding flag was actually given.
    """
    return ask_generation_config(settings, args.generation_k, args.generation_passage_limit)


def ask_generation_config(
    settings: Settings, k: int | None = None, passage_limit: int | None = None
) -> RetrievalConfig:
    """``/ask``'s own generation retrieval config, with ``k``/``passage_limit`` optionally swept.

    Shared by ``_resolve_generation_config`` and ``eval.holdout`` (issue #178), which never
    sweeps and so calls it with neither override.
    """
    updates: dict[str, int] = {"rerank_top_n": settings.rerank_top_n}
    if k is not None:
        updates["k"] = k
    if passage_limit is not None:
        updates["passage_limit"] = passage_limit
    return GENERATION_RETRIEVAL_CONFIG.model_copy(update=updates)


if __name__ == "__main__":
    sys.exit(main())
