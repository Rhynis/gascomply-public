"""Tests for the GraphRAG-vs-naive comparison CLI (build plan A8, issue #96).

Three layers, cheapest first:

1. Pure formatting/aggregation (``_naive_passage``, ``build_verdicts``, ``format_report``,
   ``_row_lines``) -- no database.
2. Row-scoring (``score_answerable_row``, ``score_refusal_row``, ``score_emergency_row``) --
   needs ``migrated_db`` the same way ``tests/test_generator.py`` does (``generate_answer``
   looks up each cited unit's document), but the retrieved context is hand-built, exactly
   like that file's own ``_passage``/``RetrievalResult`` literals -- no real vector search.
3. The real CLI entry point, ``main``, against a small real corpus ingested through
   ``ingest.run.main`` (the fixture corpus ``tests/test_gold.py`` already built and proved
   resolves, reused here rather than inventing a second one) -- this is what proves the
   script's own exit code goes non-zero on degraded data (the issue's own acceptance
   criterion), not just that the ``eval.metrics`` arithmetic can.
"""

import asyncio
import http.server
import importlib.util
import json
import socket
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import RerankerProvider, Settings, get_settings
from app.core.db import create_engine, create_session_factory, session_scope
from app.llm.base import LLMProvider, LLMUnreachableError
from app.llm.ollama import OllamaProvider
from app.repositories.documents import DocumentRepository
from app.schemas.ask import AskOutcome
from app.schemas.chunk import ChunkMatch
from app.schemas.document import DocStatus, DocType, Document
from app.schemas.entity import EntityType
from app.schemas.retrieval import Passage, RetrievalResult, RetrievedEntity, Why
from eval import compare
from eval.gold import GoldCategory, GoldRow
from eval.metrics import recall_at_k

DOC_A = "gas-decree-87-2018"


def _document(doc_id: str, number: str, effective: date) -> Document:
    return Document(
        id=doc_id,
        number=number,
        title_vi=f"Văn bản {number}",
        doc_type=DocType.DECREE,
        status=DocStatus.IN_FORCE,
        effective=effective,
        path=f"corpus/{doc_id}.html",
    )


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


def _seed_documents(database_url: str, *documents: Document) -> None:
    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            repository = DocumentRepository(session)
            for document in documents:
                await repository.upsert(document)

    _run(database_url, body)


def _chunk_match(
    unit_id: str, body: str = "Cửa hàng phải có diện tích tối thiểu 12 m2."
) -> ChunkMatch:
    return ChunkMatch(
        unit_id=unit_id,
        document_id=unit_id.split("#", 1)[0],
        dieu_no=53,
        status=DocStatus.IN_FORCE,
        body=body,
        similarity=0.9,
    )


def _passage(unit_id: str) -> Passage:
    return Passage(
        unit_id=unit_id,
        document_id=unit_id.split("#", 1)[0],
        dieu_no=53,
        text="Cửa hàng phải có diện tích tối thiểu 12 m2.",
        similarity=0.9,
        hop=0,
        score=0.9,
        why=(Why.SIMILAR,),
    )


class _StubLLM(LLMProvider):
    """A double that always returns the same fixed, schema-shaped reply.

    Not ``app.llm.mock.MockLLM``: that stand-in reads back this codebase's own
    ``[unit_id: ...]`` marker from the prompt, but ``generator._build_user_prompt``
    always appends the source's own label text on the same line (``"[unit_id: X]
    Văn bản ..., Điều 53"``), which ``MockLLM``'s regex (anchored to end-of-line right
    after the closing bracket) never matches -- a real, previously-undiscovered defect
    this file's own row-scoring tests surfaced while writing them (see the PR: it means
    the default ``LLM_PROVIDER=mock`` path silently refuses every real question, not
    just this eval's). Out of scope to fix here (``app/llm/mock.py`` is neither
    ``eval/`` nor ``tests/``); reported separately. A fixed, hand-written reply sidesteps
    it entirely and is what ``tests/test_generator.py``/``tests/test_ask_api.py``
    already use for exactly this reason.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        if json_schema is not None and "supported" in json_schema.get("properties", {}):
            # #118's verifier fails closed on a reply it cannot parse. This double always
            # returns the generation schema, which is not the verifier's schema (identified
            # by its json_schema, not a marker in the prompt -- issue #118's own follow-up
            # review), so without this branch every row below would have its citation
            # dropped by the grounding check for a reason unrelated to what each test
            # actually exercises.
            return json.dumps({"supported": True})
        return self.reply


def _citing_llm(unit_id: str) -> _StubLLM:
    return _StubLLM(
        json.dumps({"refusal": False, "answer": "câu trả lời", "citations": [{"unit_id": unit_id}]})
    )


class _StubLLMWithPromptEvalCount(_StubLLM):
    """Mirrors ``OllamaProvider.last_prompt_eval_count`` (issue #129).

    So ``score_answerable_row``'s own threading of it into ``RowScore`` can be tested with no
    real Ollama -- ``_StubLLM`` above has no such attribute at all, matching ``MockLLM``.
    """

    def __init__(self, reply: str, prompt_eval_count: int) -> None:
        super().__init__(reply)
        self.last_prompt_eval_count = prompt_eval_count


def _citing_llm_with_prompt_eval_count(unit_id: str, prompt_eval_count: int) -> _StubLLM:
    return _StubLLMWithPromptEvalCount(
        json.dumps(
            {"refusal": False, "answer": "câu trả lời", "citations": [{"unit_id": unit_id}]}
        ),
        prompt_eval_count,
    )


def _gold_row(
    row_id: str, category: GoldCategory, source_unit_ids: tuple[str, ...] = ()
) -> GoldRow:
    kwargs: dict[str, Any] = {
        "id": row_id,
        "category": category,
        "question": f"question for {row_id}",
        "reference_answer": "câu trả lời tham khảo",
        "source_unit_ids": source_unit_ids,
    }
    if category is GoldCategory.MULTI_HOP:
        kwargs["multi_hop"] = True
    if category is GoldCategory.REFUSAL:
        kwargs["expected_refusal"] = True
    return GoldRow(**kwargs)


# --- pure: naive-passage wrapping ------------------------------------------------------------


def test_naive_passage_carries_no_graph_information():
    match = _chunk_match(f"{DOC_A}#dieu-53#khoan-1")

    passage = compare._naive_passage(match)

    assert passage.unit_id == match.unit_id
    assert passage.hop == 0
    assert passage.why == (Why.SIMILAR,)
    assert passage.focus_units == ()


def test_naive_retrieval_result_has_no_entities():
    result = compare._naive_retrieval_result("q", [_chunk_match(f"{DOC_A}#dieu-53")])
    assert result.entities == ()
    assert len(result.passages) == 1


def _entity(entity_id: str, sources: tuple[str, ...]) -> RetrievedEntity:
    return RetrievedEntity(
        entity_id=entity_id,
        entity_type=EntityType.REQUIREMENT,
        label_vi="label",
        meta={},
        similarity=0.8,
        hop=1,
        score=0.7,
        why=(Why.CITES_HIT,),
        sources=sources,
    )


def test_citable_unit_ids_includes_focus_units_and_entity_sources():
    """The bug this eval's own real acceptance run against the dev DB surfaced (see the PR).

    The corpus chunks at article granularity, so the clause the graph actually points at
    travels as ``Passage.focus_units``, never as the passage's own ``unit_id`` -- exactly
    what ``app.services.generator._build_sources`` treats as citable. Measuring recall@k
    against bare ``passage.unit_id`` alone silently scored a real, correctly-surfaced
    clause-level match as a miss.
    """
    article_chunk = f"{DOC_A}#dieu-53"
    clause = f"{DOC_A}#dieu-53#khoan-1"
    other_source = f"{DOC_A}#dieu-10#khoan-2"
    result = RetrievalResult(
        question="q",
        passages=(
            Passage(
                unit_id=article_chunk,
                document_id=DOC_A,
                text="x",
                similarity=0.9,
                hop=0,
                score=0.9,
                why=(Why.SIMILAR,),
                focus_units=(clause,),
            ),
        ),
        entities=(_entity("req:supply-contract", (other_source,)),),
    )

    ids = compare._citable_unit_ids(result)

    assert ids == [article_chunk, clause, other_source]


def test_recall_at_k_finds_a_clause_only_present_as_a_focus_unit():
    article_chunk = f"{DOC_A}#dieu-53"
    clause = f"{DOC_A}#dieu-53#khoan-1"
    result = RetrievalResult(
        question="q",
        passages=(
            Passage(
                unit_id=article_chunk,
                document_id=DOC_A,
                text="x",
                similarity=0.9,
                hop=0,
                score=0.9,
                why=(Why.SIMILAR,),
                focus_units=(clause,),
            ),
        ),
    )

    # The gold id names the clause specifically; a naive check of bare passage.unit_id
    # values would miss it entirely (article_chunk != clause).
    assert recall_at_k(compare._citable_unit_ids(result), (clause,)) == 1.0


# --- pure: verdicts and formatting -----------------------------------------------------------


def _row_score(
    row_id: str, category: GoldCategory, naive_recall: float, hybrid_recall: float
) -> compare.RowScore:
    """Build a ``RowScore`` fixture where containment recall equals strict recall.

    Good enough for every test that does not care about the strict/containment split itself;
    a test that does (e.g. ``test_format_report_prints_both_recall_numbers_and_names_the_headline``)
    builds ``compare.RowScore`` directly instead of going through this helper.
    """
    return compare.RowScore(
        row_id=row_id,
        category=category,
        naive_strict_recall=naive_recall,
        hybrid_strict_recall=hybrid_recall,
        naive_containment_recall=naive_recall,
        hybrid_containment_recall=hybrid_recall,
        naive_citation_precision=naive_recall,
        hybrid_citation_precision=hybrid_recall,
        naive_outcome=AskOutcome.ANSWER,
        hybrid_outcome=AskOutcome.ANSWER,
    )


def test_build_verdicts_passes_on_a_clean_fixture():
    answerable = [
        _row_score("c1", GoldCategory.COMPLIANCE, 1.0, 1.0),
        _row_score("m1", GoldCategory.MULTI_HOP, 1.0, 1.0),
    ]
    verdicts = compare.build_verdicts(answerable, [AskOutcome.REFUSAL], [True])

    by_name = {v.name: v for v in verdicts}
    assert by_name["recall@k (hybrid, all answerable, strict)"].passed is True
    assert by_name["citation accuracy (hybrid, all answerable)"].passed is True
    assert by_name["faithfulness"].passed is None
    assert by_name["faithfulness"].note  # stated, not silent
    assert by_name["effective-answer rate"].passed is True
    assert by_name["safety detection"].passed is True
    assert compare.exit_code(verdicts) == 0


def test_build_verdicts_fails_when_recall_is_degraded():
    answerable = [
        _row_score("c1", GoldCategory.COMPLIANCE, 0.0, 0.0),
        _row_score("m1", GoldCategory.MULTI_HOP, 0.0, 0.0),
    ]
    verdicts = compare.build_verdicts(answerable, [], [])

    by_name = {v.name: v for v in verdicts}
    assert by_name["recall@k (hybrid, all answerable, strict)"].passed is False
    assert compare.exit_code(verdicts) == 1


def test_build_verdicts_omits_refusal_diagnostic_when_there_are_no_refusal_rows():
    verdicts = compare.build_verdicts([_row_score("c1", GoldCategory.COMPLIANCE, 1.0, 1.0)], [], [])
    assert not any("refusal correctness" in v.name for v in verdicts)


def test_build_verdicts_safety_is_unmeasured_with_no_emergency_rows():
    verdicts = compare.build_verdicts([_row_score("c1", GoldCategory.COMPLIANCE, 1.0, 1.0)], [], [])
    safety = next(v for v in verdicts if v.name == "safety detection")
    assert safety.passed is None
    assert safety.value is None


def test_row_lines_flags_a_row_where_hybrid_does_not_beat_naive():
    rows = [
        _row_score("wins", GoldCategory.COMPLIANCE, 0.0, 1.0),
        _row_score("loses", GoldCategory.COMPLIANCE, 1.0, 0.0),
        _row_score("ties", GoldCategory.COMPLIANCE, 1.0, 1.0),
    ]
    lines = "\n".join(compare._row_lines(rows))

    assert "wins" in lines and "better" in lines
    assert "loses" in lines and "WORSE" in lines
    assert "ties" in lines and "same" in lines


def test_format_report_names_the_faithfulness_method_in_the_output_itself():
    report = compare.EvaluationReport(
        answerable=(_row_score("c1", GoldCategory.COMPLIANCE, 1.0, 1.0),),
        refusal_outcomes=(AskOutcome.REFUSAL,),
        emergency_escalated=(True,),
        verdicts=compare.build_verdicts(
            [_row_score("c1", GoldCategory.COMPLIANCE, 1.0, 1.0)], [AskOutcome.REFUSAL], [True]
        ),
    )
    text = compare.format_report(report)

    assert "multi_hop" in text  # the separate multi-hop row the issue asks for
    assert "not measured" in text  # faithfulness method, stated in the output
    assert "qwen" in text.lower() or "judg" in text.lower()


def test_format_report_prints_both_recall_numbers_and_names_the_headline():
    """Issue #103's own acceptance criteria: strict and containment recall for both paths.

    Both are printed, and the output states which one the build-plan threshold gates on.
    """
    row = compare.RowScore(
        row_id="c1",
        category=GoldCategory.COMPLIANCE,
        naive_strict_recall=0.0,
        hybrid_strict_recall=1.0,
        naive_containment_recall=0.8,
        hybrid_containment_recall=1.0,
        naive_citation_precision=None,
        hybrid_citation_precision=1.0,
        naive_outcome=AskOutcome.REFUSAL,
        hybrid_outcome=AskOutcome.ANSWER,
    )
    report = compare.EvaluationReport(
        answerable=(row,),
        refusal_outcomes=(),
        emergency_escalated=(),
        verdicts=compare.build_verdicts([row], [], []),
    )
    text = compare.format_report(report)

    assert "recall strict" in text
    assert "recall containment" in text
    assert "0.800" in text  # naive containment recall for this fixture, printed verbatim
    assert "HEADLINE" in text  # states which number the headline claim uses
    headline_verdict_line = next(
        line for line in text.splitlines() if "recall@k" in line and "[" in line
    )
    assert "strict" in headline_verdict_line  # the gated verdict is explicitly named "strict"


def test_format_report_prints_both_configs_k_not_just_the_recall_ones():
    """Issue #169: the header used to print k for EVAL_RETRIEVAL_CONFIG only.

    That hid the exact gap issue #169 exists to fix (``/ask`` seeds with a different k than
    recall@k is scored at) -- the header must now name both.
    """
    report = compare.EvaluationReport(
        answerable=(),
        refusal_outcomes=(),
        emergency_escalated=(),
        verdicts=(),
        generation_k=8,
        generation_passage_limit=20,
    )
    text = compare.format_report(report)

    assert f"k={compare.EVAL_RETRIEVAL_CONFIG.k}" in text
    assert "k=8" in text
    assert "passage_limit=20" in text


# --- the generation-only CLI flags (issue #169) ----------------------------------------------


def test_resolve_generation_config_applies_cli_overrides():
    args = compare._parse_args(["--generation-k", "8", "--generation-passage-limit", "20"])
    settings = Settings(_env_file=None)

    generation_config = compare._resolve_generation_config(args, settings)

    assert generation_config.k == 8
    assert generation_config.passage_limit == 20
    assert generation_config.entity_limit == compare.GENERATION_RETRIEVAL_CONFIG.entity_limit
    assert generation_config.rerank_top_n == settings.rerank_top_n


def test_resolve_generation_config_defaults_to_generation_retrieval_config_unchanged():
    args = compare._parse_args([])
    settings = Settings(_env_file=None)

    generation_config = compare._resolve_generation_config(args, settings)

    assert generation_config.k == compare.GENERATION_RETRIEVAL_CONFIG.k
    assert generation_config.passage_limit == compare.GENERATION_RETRIEVAL_CONFIG.passage_limit


def test_generation_cli_flags_never_reach_the_recall_config():
    """The flags are generation-only (issue #169's own acceptance criterion).

    Builds both configs exactly as ``_amain`` does, with a generation override that differs
    from ``EVAL_RETRIEVAL_CONFIG``'s own k -- the recall config (what feeds recall@k) must
    come out with the committed k/passage_limit untouched regardless of what the generation
    flags said.
    """
    settings = Settings(_env_file=None)
    args = compare._parse_args(["--generation-k", "999", "--generation-passage-limit", "999"])

    generation_config = compare._resolve_generation_config(args, settings)
    recall_config = compare.EVAL_RETRIEVAL_CONFIG.model_copy(
        update={"rerank_top_n": settings.rerank_top_n}
    )

    assert generation_config.k == 999
    assert recall_config.k == compare.EVAL_RETRIEVAL_CONFIG.k != 999
    assert recall_config.passage_limit == compare.EVAL_RETRIEVAL_CONFIG.passage_limit != 999


# --- database-backed: score_answerable_row / score_refusal_row / score_emergency_row --------


def test_score_answerable_row_naive_misses_hybrid_finds(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    naive_matches: list[ChunkMatch] = []  # nothing retrieved at all -> refuses, no_context
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(unit_id),))

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval.single(naive_matches, hybrid_result),
                _citing_llm(unit_id),
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.naive_strict_recall == 0.0
    assert score.hybrid_strict_recall == 1.0
    assert score.naive_containment_recall == 0.0  # nothing retrieved at all
    assert score.hybrid_containment_recall == 1.0
    assert score.naive_citation_precision is None  # refused: nothing emitted
    assert score.hybrid_citation_precision == 1.0
    assert score.naive_outcome is AskOutcome.REFUSAL
    assert score.hybrid_outcome is AskOutcome.ANSWER


def test_score_answerable_row_records_the_hybrid_calls_prompt_eval_count(migrated_db: str):
    """Issue #129.

    ``RowScore.hybrid_prompt_eval_count`` must come from the hybrid path's own generation
    call -- see ``GroundingTrace.prompt_eval_count``'s own capture point.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(unit_id),))

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval.single([], hybrid_result),
                _citing_llm_with_prompt_eval_count(unit_id, 6053),
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.hybrid_prompt_eval_count == 6053
    assert score.context_overflow is False


class _ScriptedRetryLLM(LLMProvider):
    """One scripted main-generation reply per non-verifier call, in order (issue #202).

    One scripted verifier verdict per verifier call, in order. Mirrors ``tests/test_
    generator.py``'s own double of the same name -- this file keeps its own copy since its
    verifier-schema check is against ``_StubLLM``'s own convention, not that module's.
    """

    def __init__(self, replies: list[str], verdicts: list[bool]) -> None:
        self._replies = iter(replies)
        self._verdicts = iter(verdicts)

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        if json_schema is not None and "supported" in json_schema.get("properties", {}):
            return json.dumps({"supported": next(self._verdicts)})
        return next(self._replies)


def test_score_answerable_row_records_whether_the_hybrid_path_retried(migrated_db: str):
    """Issue #202: ``RowScore.retried`` mirrors the hybrid call's own ``GroundingTrace.retried``.

    The naive path uses an empty retrieval (``RowRetrieval.single([], hybrid_result)``, as
    ``test_score_answerable_row_records_the_hybrid_calls_prompt_eval_count`` above already
    does) so it refuses on ``no_context`` without calling ``llm`` at all -- otherwise it would
    consume this double's own scripted replies/verdicts before the hybrid call ever ran.
    """
    unit_a = f"{DOC_A}#dieu-53#khoan-1"
    unit_b = f"{DOC_A}#dieu-60"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_b,))
    hybrid_result = RetrievalResult(
        question=row.question, passages=(_passage(unit_a), _passage(unit_b))
    )
    llm = _ScriptedRetryLLM(
        replies=[
            json.dumps({"refusal": False, "answer": "a", "citations": [{"unit_id": unit_a}]}),
            json.dumps({"refusal": False, "answer": "b", "citations": [{"unit_id": unit_b}]}),
        ],
        verdicts=[False, True],
    )

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval.single([], hybrid_result),
                llm,
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.retried is True
    assert score.hybrid_outcome is AskOutcome.ANSWER


def test_score_answerable_row_flags_context_overflow(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(unit_id),))

    class _OverflowingLLM(LLMProvider):
        async def generate(
            self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
        ) -> str:
            from app.llm.base import ContextOverflowError

            raise ContextOverflowError("prompt_eval_count=4095 is within budget")

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval.single([], hybrid_result),
                _OverflowingLLM(),
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.hybrid_outcome is AskOutcome.REFUSAL
    assert score.context_overflow is True
    assert score.llm_unavailable is False


def test_score_answerable_row_flags_llm_unavailable(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(unit_id),))

    class _UnavailableLLM(LLMProvider):
        async def generate(
            self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
        ) -> str:
            from app.llm.base import LLMUnavailableError

            raise LLMUnavailableError("simulated: Ollama request timed out after 300.0s")

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval.single([], hybrid_result),
                _UnavailableLLM(),
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.hybrid_outcome is AskOutcome.REFUSAL
    assert score.llm_unavailable is True
    assert score.context_overflow is False


def test_score_answerable_row_raises_on_llm_unreachable_for_the_retry_pass(migrated_db: str):
    """Issue #183, Scope 5 and the eval trap.

    A connection failure must behave like the raw ``ConnectError`` it replaces: it raises from
    the row's scoring (never scored as a refusal), so ``run_evaluation`` queues it for the
    existing retry pass rather than letting an Ollama outage silently lower the effective-
    answer rate.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(unit_id),))

    class _UnreachableLLM(LLMProvider):
        async def generate(
            self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
        ) -> str:
            raise LLMUnreachableError("simulated: Ollama unreachable: connection refused")

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            with pytest.raises(LLMUnreachableError):
                await compare.score_answerable_row(
                    row,
                    compare.RowRetrieval.single([], hybrid_result),
                    _UnreachableLLM(),
                    DocumentRepository(session),
                )

    _run(migrated_db, body)


def test_score_refusal_row_raises_on_llm_unreachable_for_the_retry_pass(migrated_db: str):
    # A passage is seeded so generate_answer actually reaches the LLM call rather than
    # refusing immediately with "no_context" (test_score_refusal_row_with_no_context_refuses
    # already covers that path, where the LLM is never called at all).
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("refusal-x", GoldCategory.REFUSAL)
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(unit_id),))

    class _UnreachableLLM(LLMProvider):
        async def generate(
            self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
        ) -> str:
            raise LLMUnreachableError("simulated: Ollama unreachable: connection refused")

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            with pytest.raises(LLMUnreachableError):
                await compare.score_refusal_row(
                    row, hybrid_result, _UnreachableLLM(), DocumentRepository(session)
                )

    _run(migrated_db, body)


def test_run_evaluation_queues_an_llm_unreachable_row_for_the_retry_pass_then_recovers(
    migrated_db: str,
):
    """The full path: an Ollama outage on a row's first attempt is retried, not scored.

    Mirrors ``test_run_evaluation_recovers_a_row_that_fails_once_then_succeeds_on_retry``, but
    through the real ``llm_unreachable`` refusal conversion rather than a raw exception stand-in
    -- proving the conversion in ``generate_answer`` (issue #183) does not defeat issue #109's
    retry pass.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-flaky", GoldCategory.COMPLIANCE, (unit_id,))
    retriever = _FixedRetriever(
        {row.question: ([], RetrievalResult(question=row.question, passages=(_passage(unit_id),)))}
    )

    class _UnreachableOnceLLM(LLMProvider):
        """Raises ``LLMUnreachableError`` on its first call, then answers normally."""

        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
        ) -> str:
            self.calls += 1
            if self.calls == 1:
                raise LLMUnreachableError("simulated: Ollama unreachable: connection refused")
            if "supported" in (json_schema or {}).get("properties", {}):
                return json.dumps({"supported": True})
            return json.dumps(
                {"refusal": False, "answer": "ok", "citations": [{"unit_id": unit_id}]}
            )

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.run_evaluation(
                (row,),
                retriever,  # type: ignore[arg-type]
                DocumentRepository(session),
                _UnreachableOnceLLM(),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    report: compare.EvaluationReport = result["value"]

    assert report.failed_row_ids == ()  # the retry succeeded -- never excluded
    assert report.retried_row_ids == ("compliance-flaky",)
    assert len(report.answerable) == 1
    assert report.answerable[0].row_id == "compliance-flaky"
    assert report.answerable[0].llm_unavailable is False  # recovered cleanly, not scored at all


class _FakeWaiter:
    """A ``wait_until_reachable`` double: counts its own calls, answers ``reachable`` every time.

    Never sleeps -- these tests are about ``run_evaluation``'s own dispatch logic (does it
    call the waiter at all, does it stop the run when the waiter gives up), not about a real
    probe's timing. ``test_run_evaluation_recovers_through_a_real_socket_once_ollama_comes_up``
    below is what proves the real probe itself (``compare._build_ollama_waiter``) works
    against a real socket.
    """

    def __init__(self, *, reachable: bool) -> None:
        self.calls = 0
        self._reachable = reachable

    async def __call__(self) -> bool:
        self.calls += 1
        return self._reachable


def test_run_evaluation_waits_for_the_provider_then_recovers(migrated_db: str):
    """Issue #185, Scope 1 and 6: a scripted unreachable failure, followed by recovery.

    The fake waiter is called exactly once -- between the first attempt (which raises) and
    the retry (which succeeds) -- and the row is scored normally, not excluded.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-flaky", GoldCategory.COMPLIANCE, (unit_id,))
    retriever = _FixedRetriever(
        {row.question: ([], RetrievalResult(question=row.question, passages=(_passage(unit_id),)))}
    )
    waiter = _FakeWaiter(reachable=True)

    class _UnreachableOnceLLM(LLMProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
        ) -> str:
            self.calls += 1
            if self.calls == 1:
                raise LLMUnreachableError("simulated: Ollama unreachable: connection refused")
            if "supported" in (json_schema or {}).get("properties", {}):
                return json.dumps({"supported": True})
            return json.dumps(
                {"refusal": False, "answer": "ok", "citations": [{"unit_id": unit_id}]}
            )

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.run_evaluation(
                (row,),
                retriever,  # type: ignore[arg-type]
                DocumentRepository(session),
                _UnreachableOnceLLM(),
                compare.RunOptions(wait_until_reachable=waiter),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    report: compare.EvaluationReport = result["value"]

    assert waiter.calls == 1
    assert report.failed_row_ids == ()
    assert report.retried_row_ids == ("compliance-flaky",)
    assert len(report.answerable) == 1
    assert report.provider_wait_count == 1
    assert "provider waits: 1," in compare.format_report(report)


def test_run_evaluation_raises_when_the_provider_never_comes_back(migrated_db: str):
    """Issue #185, Scope 3 and 6: a waiter that gives up stops the run instead of excluding a row.

    No report with thresholds computed over a batch missing every row past this point --
    ``ProviderStillUnreachableError`` propagates straight out of ``run_evaluation``.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    retriever = _FixedRetriever(
        {row.question: ([], RetrievalResult(question=row.question, passages=(_passage(unit_id),)))}
    )
    waiter = _FakeWaiter(reachable=False)

    class _AlwaysUnreachableLLM(LLMProvider):
        async def generate(
            self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
        ) -> str:
            raise LLMUnreachableError("simulated: Ollama unreachable: connection refused")

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            with pytest.raises(compare.ProviderStillUnreachableError) as excinfo:
                await compare.run_evaluation(
                    (row,),
                    retriever,  # type: ignore[arg-type]
                    DocumentRepository(session),
                    _AlwaysUnreachableLLM(),
                    compare.RunOptions(wait_until_reachable=waiter),
                )
            assert row.id in str(excinfo.value)

    _run(migrated_db, body)


def test_run_evaluation_never_waits_for_a_timeout(migrated_db: str):
    """Issue #185, trap 1 and Scope 6: ``llm_unavailable`` (a timeout) never calls the waiter.

    ``generate_answer`` already turns this into a scored refusal rather than raising at all
    (issue #112), so the row is never queued/excluded either -- the waiter's presence changes
    nothing about this path.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    retriever = _FixedRetriever(
        {row.question: ([], RetrievalResult(question=row.question, passages=(_passage(unit_id),)))}
    )
    waiter = _FakeWaiter(reachable=True)

    class _UnavailableLLM(LLMProvider):
        async def generate(
            self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
        ) -> str:
            from app.llm.base import LLMUnavailableError

            raise LLMUnavailableError("simulated: Ollama request timed out after 300.0s")

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.run_evaluation(
                (row,),
                retriever,  # type: ignore[arg-type]
                DocumentRepository(session),
                _UnavailableLLM(),
                compare.RunOptions(wait_until_reachable=waiter),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    report: compare.EvaluationReport = result["value"]

    assert waiter.calls == 0
    assert report.provider_wait_count == 0
    assert len(report.answerable) == 1
    assert report.answerable[0].llm_unavailable is True


def test_run_evaluation_never_waits_for_a_non_unreachable_exception(migrated_db: str):
    """Issue #185, trap 1 and Scope 6: any other exception keeps #109's plain retry, no wait.

    Reuses ``test_run_evaluation_isolates_one_rows_failure_from_the_rest``'s own timeout
    stand-in, with a waiter added that would make this test fail if it were ever called.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-times-out", GoldCategory.COMPLIANCE, (unit_id,))

    class _AlwaysRaisingRetriever:
        async def vector_search(
            self, question: str, *, include_history: bool = False
        ) -> list[ChunkMatch]:
            raise TimeoutError("simulated: a real network/model timeout, standing in")

        async def retrieve(
            self, question: str, *, include_history: bool = False
        ) -> RetrievalResult:
            raise TimeoutError("simulated: a real network/model timeout, standing in")

    waiter = _FakeWaiter(reachable=True)

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.run_evaluation(
                (row,),
                _AlwaysRaisingRetriever(),  # type: ignore[arg-type]
                DocumentRepository(session),
                _citing_llm(unit_id),
                compare.RunOptions(wait_until_reachable=waiter),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    report: compare.EvaluationReport = result["value"]

    assert waiter.calls == 0
    assert report.provider_wait_count == 0
    assert report.failed_row_ids == ("compliance-times-out",)  # #109's plain retry, unchanged


def _free_port() -> int:
    """Return a TCP port nothing is listening on yet (bind on an ephemeral port, then close)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _FakeOllamaHandler(http.server.BaseHTTPRequestHandler):
    """Answers ``/api/version`` and one ``/api/chat`` call -- just enough for a retry to work.

    Real HTTP over a real socket (issue #185, Scope 6's own real-socket requirement) --
    ``tests/test_llm_ollama.py`` stubs ``httpx2.MockTransport`` instead for every other
    ``OllamaProvider`` test, which proves the request/response *shape*; this test instead
    proves ``compare._build_ollama_waiter`` and ``OllamaProvider`` cooperate correctly over an
    actual connection that is refused, then succeeds.
    """

    def log_message(self, log_format: str, *args: object) -> None:
        pass  # silence BaseHTTPRequestHandler's own per-request stderr logging

    def do_GET(self) -> None:
        if self.path == "/api/version":
            self._reply({"version": "0.0.0-test"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain the request body; its content is not needed here
        content = json.dumps({"refusal": True, "answer": None, "citations": []})
        self._reply({"message": {"content": content}})

    def _reply(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_run_evaluation_recovers_through_a_real_socket_once_ollama_comes_up(migrated_db: str):
    """Issue #185, Scope 6: the real probe and the real ``OllamaProvider``, no doubles at all.

    Nothing listens on the chosen port at first, so the first attempt's real
    ``httpx2.ConnectError`` becomes a real ``LLMUnreachableError``
    (``OllamaProvider.generate``). A minimal HTTP server starts on that same port about 2s
    later (trap 5: this is the one test allowed to wait for real); the real
    ``_build_ollama_waiter`` probe picks that up on its next poll, and the retry pass's own
    real ``/api/chat`` call then succeeds -- proving the row is scored once Ollama is back,
    with no warm-up request beyond the probe's own ``GET /api/version`` (trap 3).
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("refusal-x", GoldCategory.REFUSAL)
    retriever = _FixedRetriever(
        {row.question: ([], RetrievalResult(question=row.question, passages=(_passage(unit_id),)))}
    )

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    # The server itself (socket bind + listen) must not exist until the 2s mark either --
    # constructing ``HTTPServer`` starts it listening immediately, which would let the first
    # connection attempt succeed right away (the kernel completes the handshake before
    # ``accept()`` is ever called) and defeat the "nothing listens at first" premise.
    server_holder: list[http.server.HTTPServer] = []

    def _start_late() -> None:
        time.sleep(2.0)
        server = http.server.HTTPServer(("127.0.0.1", port), _FakeOllamaHandler)
        server_holder.append(server)
        server.serve_forever(poll_interval=0.05)

    thread = threading.Thread(target=_start_late, daemon=True)
    thread.start()
    try:
        llm = OllamaProvider(base_url=base_url, model="test-model", timeout=5.0)
        waiter = compare._build_ollama_waiter(base_url, poll_seconds=0.3, timeout_seconds=10.0)

        async def body(engine: AsyncEngine) -> None:
            async with session_scope(create_session_factory(engine)) as session:
                result["value"] = await compare.run_evaluation(
                    (row,),
                    retriever,  # type: ignore[arg-type]
                    DocumentRepository(session),
                    llm,
                    compare.RunOptions(wait_until_reachable=waiter),
                )

        result: dict[str, Any] = {}
        _run(migrated_db, body)
    finally:
        if server_holder:
            server_holder[0].shutdown()
            server_holder[0].server_close()
        thread.join(timeout=5.0)

    report: compare.EvaluationReport = result["value"]

    assert report.failed_row_ids == ()
    assert report.retried_row_ids == ("refusal-x",)
    assert report.refusal_outcomes == (AskOutcome.REFUSAL,)
    assert report.provider_wait_count == 1
    assert report.provider_wait_seconds > 0


def test_score_answerable_row_naive_gets_containment_credit_for_a_whole_article_chunk(
    migrated_db: str,
):
    """The issue's own finding (#103): naive returns a whole article; gold names one clause in it.

    Strict recall must still miss (the ids are not equal); containment recall must credit the
    article for containing the clause's text -- this is what makes naive's real recall ~0.8
    rather than the ~0 an exact-id comparison alone reports.
    """
    article_id = f"{DOC_A}#dieu-53"
    clause_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-y", GoldCategory.COMPLIANCE, (clause_id,))
    naive_matches = [_chunk_match(article_id)]
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(clause_id),))

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval.single(naive_matches, hybrid_result),
                _citing_llm(article_id),
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.naive_strict_recall == 0.0  # article_id != clause_id
    assert score.naive_containment_recall == 1.0  # the article contains the clause's text
    assert score.hybrid_strict_recall == 1.0
    assert score.hybrid_containment_recall == 1.0


def test_score_answerable_row_hybrid_citation_containment_credits_a_whole_article_citation(
    migrated_db: str,
):
    """A hybrid citation at article granularity must score 0 on strict precision, 1 on containment.

    Issue #118's own follow-up review: when gold names a clause inside the cited article,
    strict citation precision must score 0 but containment must score 1 -- proving
    ``score_answerable_row``'s own wiring credits it (not just ``eval.metrics``'s pure
    function in isolation, which ``tests/test_metrics.py`` already covers).
    """
    article_id = f"{DOC_A}#dieu-53"
    clause_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-w", GoldCategory.COMPLIANCE, (clause_id,))
    naive_matches: list[ChunkMatch] = []
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(article_id),))

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval.single(naive_matches, hybrid_result),
                _citing_llm(article_id),
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.hybrid_outcome is AskOutcome.ANSWER
    assert score.hybrid_citation_precision == 0.0  # article_id != the clause gold names
    assert score.hybrid_citation_containment_precision == 1.0  # but the article contains it


# --- generation uses its own retrieval, separate from recall@k's (issue #167) ---------------


def test_score_answerable_row_generation_uses_the_separate_generation_retrieval(
    migrated_db: str,
):
    """Generation must use the separate ``generation_*`` arguments, not the recall ones.

    The recall arguments here carry nothing at all (recall@k measured against an empty
    result -> 0.0 on both strict and containment); the ``generation_*`` arguments carry the
    real passage -- proving ``score_answerable_row`` actually generates from the separate
    argument, which is how ``_score_one_row`` now calls it with a dedicated
    ``GENERATION_RETRIEVAL_CONFIG`` retriever (issue #167's own acceptance criterion: "naive
    and hybrid generation receive the /ask config while recall@k still sees k=10").
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    recall_hybrid_result = RetrievalResult(question=row.question, passages=())
    generation_hybrid_result = RetrievalResult(question=row.question, passages=(_passage(unit_id),))

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval(
                    naive_matches=[],
                    hybrid_result=recall_hybrid_result,
                    generation_naive_matches=[],
                    generation_hybrid_result=generation_hybrid_result,
                ),
                _citing_llm(unit_id),
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.hybrid_strict_recall == 0.0  # recall measured against the empty recall result
    assert score.hybrid_containment_recall == 0.0
    assert score.hybrid_outcome is AskOutcome.ANSWER  # but generation saw the real passage
    assert score.hybrid_citation_precision == 1.0


def test_row_retrieval_single_uses_the_one_retrieval_for_both_recall_and_generation(
    migrated_db: str,
):
    """``RowRetrieval.single`` is the pre-#167 behaviour: one retrieval, both purposes.

    Every test of ``score_answerable_row`` that pre-dates issue #167 builds its
    ``RowRetrieval`` this way, so they keep passing unchanged with the split added.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(unit_id),))

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval.single([], hybrid_result),
                _citing_llm(unit_id),
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.hybrid_strict_recall == 1.0
    assert score.hybrid_outcome is AskOutcome.ANSWER
    assert score.hybrid_citation_precision == 1.0


def test_run_evaluation_uses_the_separate_generation_retriever(migrated_db: str):
    """End to end through ``run_evaluation``/``_score_one_row``.

    A distinct generation retriever is what actually feeds ``generate_answer``, even when
    the recall retriever found nothing.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    recall_retriever = _FixedRetriever(
        {row.question: ([], RetrievalResult(question=row.question, passages=()))}
    )
    generation_retriever = _FixedRetriever(
        {
            row.question: (
                [],
                RetrievalResult(question=row.question, passages=(_passage(unit_id),)),
            )
        }
    )

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.run_evaluation(
                (row,),
                recall_retriever,  # type: ignore[arg-type]
                DocumentRepository(session),
                _citing_llm(unit_id),
                compare.RunOptions(generation_retriever=generation_retriever),  # type: ignore[arg-type]
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    report: compare.EvaluationReport = result["value"]

    assert len(report.answerable) == 1
    assert report.answerable[0].hybrid_strict_recall == 0.0  # the recall retriever found nothing
    assert report.answerable[0].hybrid_outcome is AskOutcome.ANSWER  # generation found the passage
    assert report.answerable[0].hybrid_citation_precision == 1.0


# --- the deciding-fact answer check (issue #167) ---------------------------------------------


def test_score_answerable_row_hybrid_answer_check_passes_when_the_pattern_matches(
    migrated_db: str,
):
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = GoldRow(
        id="compliance-check",
        category=GoldCategory.COMPLIANCE,
        question="q",
        reference_answer="a",
        source_unit_ids=(unit_id,),
        answer_must_match=["500\\s*kg"],
    )
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(unit_id),))
    llm = _StubLLM(
        json.dumps(
            {
                "refusal": False,
                "answer": "Cơ sở tồn chứa trên 500 kg thuộc diện này.",
                "citations": [{"unit_id": unit_id}],
            }
        )
    )

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval.single([], hybrid_result),
                llm,
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.hybrid_answer_check is True
    assert score.naive_answer_check is False  # naive refused: no_context -> fails the check


def test_score_answerable_row_hybrid_answer_check_fails_when_the_pattern_is_missing(
    migrated_db: str,
):
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = GoldRow(
        id="compliance-check",
        category=GoldCategory.COMPLIANCE,
        question="q",
        reference_answer="a",
        source_unit_ids=(unit_id,),
        answer_must_match=["500\\s*kg"],
    )
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(unit_id),))
    llm = _citing_llm(unit_id)  # answers "câu trả lời" -- no "500 kg" anywhere

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval.single([], hybrid_result),
                llm,
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.hybrid_answer_check is False


def test_score_answerable_row_answer_check_is_none_when_the_row_has_no_pattern(
    migrated_db: str,
):
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    hybrid_result = RetrievalResult(question=row.question, passages=(_passage(unit_id),))

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_answerable_row(
                row,
                compare.RowRetrieval.single([], hybrid_result),
                _citing_llm(unit_id),
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RowScore = result["value"]

    assert score.hybrid_answer_check is None
    assert score.naive_answer_check is None


def test_answer_check_lines_report_both_paths_separately():
    checked_pass = compare.RowScore(
        row_id="checked-pass",
        category=GoldCategory.MULTI_HOP,
        naive_strict_recall=0.0,
        hybrid_strict_recall=1.0,
        naive_containment_recall=0.0,
        hybrid_containment_recall=1.0,
        naive_citation_precision=None,
        hybrid_citation_precision=1.0,
        naive_outcome=AskOutcome.REFUSAL,
        hybrid_outcome=AskOutcome.ANSWER,
        naive_answer_check=False,
        hybrid_answer_check=True,
    )
    unchecked = _row_score("unchecked", GoldCategory.COMPLIANCE, 1.0, 1.0)
    report = compare.EvaluationReport(
        answerable=(checked_pass, unchecked),
        refusal_outcomes=(),
        emergency_escalated=(),
        verdicts=compare.build_verdicts([checked_pass, unchecked], [], []),
    )

    text = compare.format_report(report)

    assert "Answer check" in text
    assert "naive:  0/1 rows with patterns passed" in text
    assert "hybrid: 1/1 rows with patterns passed" in text
    assert "answer check PASS" in text  # the per-row hybrid flag


def test_score_refusal_row_with_no_context_refuses(migrated_db: str):
    row = _gold_row("refusal-x", GoldCategory.REFUSAL)
    hybrid_result = RetrievalResult(question=row.question, passages=())

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.score_refusal_row(
                row,
                hybrid_result,
                _StubLLM("unused: no context means no LLM call"),
                DocumentRepository(session),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    score: compare.RefusalScore = result["value"]
    assert score.outcome is AskOutcome.REFUSAL
    assert score.refusal_reason == "no_context"


def test_score_emergency_row_true_for_a_real_emergency_phrase():
    row = _gold_row("emergency-x", GoldCategory.EMERGENCY)
    row = row.model_copy(update={"question": "ngửi thấy mùi gas"})
    assert compare.score_emergency_row(row) is True


def test_score_emergency_row_false_for_an_ordinary_question():
    row = _gold_row("emergency-y", GoldCategory.EMERGENCY)
    row = row.model_copy(update={"question": "Điều kiện mở cửa hàng gas là gì?"})
    assert compare.score_emergency_row(row) is False


# --- run_evaluation end-to-end, with a fixed (non-DB) retriever double -----------------------


class _FixedRetriever:
    """A retriever double whose answers are pre-baked per question -- no vector search at all.

    Standing in for a real ``Retriever`` here is deliberate: this test is about
    ``run_evaluation``'s own dispatch-by-category logic (does a ``refusal`` row skip the naive
    side, does an ``emergency`` row skip retrieval altogether), not about retrieval quality --
    that is what ``tests/test_retriever.py`` and the real ``python -m eval.compare`` run
    against the dev database are for.
    """

    def __init__(self, by_question: dict[str, tuple[list[ChunkMatch], RetrievalResult]]) -> None:
        self._by_question = by_question

    async def vector_search(
        self, question: str, *, include_history: bool = False
    ) -> list[ChunkMatch]:
        return self._by_question[question][0]

    async def retrieve(self, question: str, *, include_history: bool = False) -> RetrievalResult:
        return self._by_question[question][1]


def test_run_evaluation_dispatches_every_category_correctly(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))

    compliance = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    refusal = _gold_row("refusal-x", GoldCategory.REFUSAL)
    emergency = _gold_row("emergency-x", GoldCategory.EMERGENCY).model_copy(
        update={"question": "cháy nhà"}
    )
    rows = (compliance, refusal, emergency)

    retriever = _FixedRetriever(
        {
            compliance.question: (
                [],
                RetrievalResult(question=compliance.question, passages=(_passage(unit_id),)),
            ),
            refusal.question: ([], RetrievalResult(question=refusal.question, passages=())),
        }
    )

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.run_evaluation(
                rows,
                retriever,  # type: ignore[arg-type]
                DocumentRepository(session),
                _citing_llm(unit_id),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    report: compare.EvaluationReport = result["value"]

    assert len(report.answerable) == 1
    assert report.answerable[0].hybrid_strict_recall == 1.0
    assert report.refusal_outcomes == (AskOutcome.REFUSAL,)
    assert report.emergency_escalated == (True,)
    assert compare.exit_code(report.verdicts) == 0
    assert "multi_hop" in compare.format_report(report)
    assert report.failed_row_ids == ()


def test_run_evaluation_reports_the_maximum_prompt_eval_count_and_no_overflow(
    migrated_db: str,
):
    """Issue #129.

    A clean run (no overflow at the configured ``num_ctx``) must still report the maximum
    prompt size it saw, and an empty overflow list -- not just when something went wrong.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-x", GoldCategory.COMPLIANCE, (unit_id,))
    retriever = _FixedRetriever(
        {row.question: ([], RetrievalResult(question=row.question, passages=(_passage(unit_id),)))}
    )

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.run_evaluation(
                (row,),
                retriever,  # type: ignore[arg-type]
                DocumentRepository(session),
                _citing_llm_with_prompt_eval_count(unit_id, 6053),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    report: compare.EvaluationReport = result["value"]

    assert report.max_prompt_eval_count == 6053
    assert report.context_overflow_row_ids == ()


def test_run_evaluation_isolates_one_rows_failure_from_the_rest(migrated_db: str):
    """A row that raises on both attempts must not sink the run, and stays excluded (#109).

    Observed for real running this evaluation's own acceptance check against a real Ollama
    daemon (see the PR): a question that pushes near the model's context limit occasionally
    lost termination control and the HTTP call timed out well past a generous batch timeout.
    One row's infrastructure hiccup must never discard the other thirty-odd rows' real results
    -- and a row that genuinely cannot be scored, even after a second chance, must still be
    named and still never counted as a pass or a fail (the issue's own required test of this
    path).
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    good = _gold_row("compliance-good", GoldCategory.COMPLIANCE, (unit_id,))
    bad = _gold_row("compliance-times-out", GoldCategory.COMPLIANCE, (unit_id,))

    class _MixedRetriever:
        async def vector_search(
            self, question: str, *, include_history: bool = False
        ) -> list[ChunkMatch]:
            if question == bad.question:
                raise TimeoutError("simulated: a real network/model timeout, standing in")
            return []

        async def retrieve(
            self, question: str, *, include_history: bool = False
        ) -> RetrievalResult:
            if question == bad.question:
                raise TimeoutError("simulated: a real network/model timeout, standing in")
            return RetrievalResult(question=question, passages=(_passage(unit_id),))

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.run_evaluation(
                (good, bad),
                _MixedRetriever(),  # type: ignore[arg-type]
                DocumentRepository(session),
                _citing_llm(unit_id),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    report: compare.EvaluationReport = result["value"]

    assert len(report.answerable) == 1
    assert report.answerable[0].row_id == "compliance-good"
    assert report.retried_row_ids == ()  # never claimed a recovery that did not happen
    assert report.failed_row_ids == ("compliance-times-out",)
    assert "compliance-times-out" in compare.format_report(report)


def test_run_evaluation_recovers_a_row_that_fails_once_then_succeeds_on_retry(
    migrated_db: str,
):
    """The retry itself (#109): a row that raises once must get a second, independent attempt.

    The real failure this reproduces is a per-call coin flip, not a property of the question
    (confirmed against a real Ollama daemon: the same near-context-limit prompt sometimes
    finishes in seconds and sometimes runs on for minutes -- see ``EVAL_LLM_TIMEOUT_SECONDS``'s
    own comment) -- so a fresh retry, not a longer wait on the same attempt, is what the
    evidence supports, and this is the row-scored-normally-after-recovering half of that fix.
    ``test_run_evaluation_isolates_one_rows_failure_from_the_rest`` above is the row that fails
    both attempts, still excluded.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    row = _gold_row("compliance-flaky", GoldCategory.COMPLIANCE, (unit_id,))

    class _FlakyOnceRetriever:
        """Raises on its very first call, then behaves normally for every call after."""

        def __init__(self) -> None:
            self.vector_search_calls = 0

        async def vector_search(
            self, question: str, *, include_history: bool = False
        ) -> list[ChunkMatch]:
            self.vector_search_calls += 1
            if self.vector_search_calls == 1:
                raise TimeoutError("simulated: fails once, standing in for issue #109's timeout")
            return []

        async def retrieve(
            self, question: str, *, include_history: bool = False
        ) -> RetrievalResult:
            return RetrievalResult(question=question, passages=(_passage(unit_id),))

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await compare.run_evaluation(
                (row,),
                _FlakyOnceRetriever(),  # type: ignore[arg-type]
                DocumentRepository(session),
                _citing_llm(unit_id),
            )

    result: dict[str, Any] = {}
    _run(migrated_db, body)
    report: compare.EvaluationReport = result["value"]

    assert report.failed_row_ids == ()  # the retry succeeded -- never excluded
    assert report.retried_row_ids == ("compliance-flaky",)
    assert len(report.answerable) == 1
    assert report.answerable[0].row_id == "compliance-flaky"
    assert report.answerable[0].hybrid_strict_recall == 1.0  # scored completely normally
    text = compare.format_report(report)
    assert "compliance-flaky" in text  # the retry stays visible, not a silent recovery
    assert "RETRIED" in text


def test_format_report_names_a_row_that_needed_a_retry_without_touching_the_failed_section():
    """Pure formatting: a recovered row is reported distinctly from an excluded one (#109)."""
    row = _row_score("c1", GoldCategory.COMPLIANCE, 1.0, 1.0)
    report = compare.EvaluationReport(
        answerable=(row,),
        refusal_outcomes=(),
        emergency_escalated=(),
        verdicts=compare.build_verdicts([row], [], []),
        retried_row_ids=("c1",),
    )
    text = compare.format_report(report)

    assert "RETRIED" in text
    assert "c1" in text
    assert "FAILED TO EVALUATE" not in text  # recovered, never excluded


def test_row_lines_show_each_rows_hybrid_citation_precision_and_outcome():
    """Issue #114: so two runs can be compared row by row, not only by their aggregates."""
    answered = _row_score("answered", GoldCategory.COMPLIANCE, 0.0, 1.0)
    refused = compare.RowScore(
        row_id="refused",
        category=GoldCategory.COMPLIANCE,
        naive_strict_recall=0.0,
        hybrid_strict_recall=1.0,
        naive_containment_recall=0.0,
        hybrid_containment_recall=1.0,
        naive_citation_precision=None,
        hybrid_citation_precision=None,
        naive_outcome=AskOutcome.REFUSAL,
        hybrid_outcome=AskOutcome.REFUSAL,
    )

    lines = compare._row_lines([answered, refused])

    assert "hybrid citation 1.000 (answer)" in next(line for line in lines if "answered" in line)
    assert "hybrid citation n/a (refusal)" in next(line for line in lines if "refused" in line)


# --- what reranked the hybrid path (issue #114) ------------------------------------------------


def _report(**overrides: Any) -> compare.EvaluationReport:
    row = _row_score("c1", GoldCategory.COMPLIANCE, 1.0, 1.0)
    return compare.EvaluationReport(
        answerable=(row,),
        refusal_outcomes=(),
        emergency_escalated=(),
        verdicts=compare.build_verdicts([row], [], []),
        **overrides,
    )


def test_the_report_says_reranking_is_off_when_it_is():
    text = compare.format_report(_report())

    assert "Reranker: off" in text
    assert "WARNING" not in text


def test_the_report_names_the_reranker_and_top_n_that_produced_it():
    text = compare.format_report(
        _report(reranker="llama-server, top-N 5", rerank_ok=31, rerank_fallbacks=0)
    )

    assert "Reranker: llama-server, top-N 5" in text
    assert "31 retrieval(s) reranked, 0 fell back" in text
    assert "WARNING" not in text


# --- the context window, printed per issue #129 --------------------------------------------


def test_the_report_prints_the_maximum_prompt_eval_count():
    text = compare.format_report(_report(max_prompt_eval_count=9549))

    assert "max generation prompt_eval_count observed: 9549" in text


def test_the_report_says_n_a_when_prompt_eval_count_was_never_measured():
    text = compare.format_report(_report(max_prompt_eval_count=None))

    assert "max generation prompt_eval_count observed: n/a" in text


def test_the_report_names_every_row_that_hit_context_overflow():
    text = compare.format_report(
        _report(context_overflow_row_ids=("compliance-pccc-design-appraisal-authority",))
    )

    assert "context_overflow rows: 1" in text
    assert "compliance-pccc-design-appraisal-authority" in text


def test_the_report_shows_zero_overflow_rows_with_no_names_when_none_occurred():
    text = compare.format_report(_report(context_overflow_row_ids=()))

    assert "context_overflow rows: 0" in text


def test_the_report_names_every_row_that_hit_llm_unavailable():
    text = compare.format_report(_report(llm_unavailable_row_ids=("compliance-slow-model",)))

    assert "llm_unavailable rows: 1" in text
    assert "compliance-slow-model" in text


def test_the_report_shows_zero_llm_unavailable_rows_with_no_names_when_none_occurred():
    text = compare.format_report(_report(llm_unavailable_row_ids=()))

    assert "llm_unavailable rows: 0" in text


def test_the_report_names_status_unknown_refusal_rows():
    text = compare.format_report(
        _report(status_unknown_refusal_row_ids=("refusal-tt-41-2011-status",))
    )

    assert "status_unknown refusal rows: 1" in text
    assert "refusal-tt-41-2011-status" in text


def test_the_report_shows_zero_grounding_retries_by_default():
    """Issue #202's own report line, next to the verdict-check line."""
    text = compare.format_report(_report())

    assert "grounding retries (issue #202): 0 rows -- 0 answered, 0 refused" in text


def test_the_report_names_grounding_retries_and_splits_them_by_outcome():
    answered = compare.RowScore(
        row_id="m-retry-answered",
        category=GoldCategory.MULTI_HOP,
        naive_strict_recall=1.0,
        hybrid_strict_recall=1.0,
        naive_containment_recall=1.0,
        hybrid_containment_recall=1.0,
        naive_citation_precision=1.0,
        hybrid_citation_precision=1.0,
        naive_outcome=AskOutcome.ANSWER,
        hybrid_outcome=AskOutcome.ANSWER,
        retried=True,
    )
    refused = compare.RowScore(
        row_id="m-retry-refused",
        category=GoldCategory.MULTI_HOP,
        naive_strict_recall=0.0,
        hybrid_strict_recall=0.0,
        naive_containment_recall=0.0,
        hybrid_containment_recall=0.0,
        naive_citation_precision=None,
        hybrid_citation_precision=None,
        naive_outcome=AskOutcome.REFUSAL,
        hybrid_outcome=AskOutcome.REFUSAL,
        retried=True,
    )
    not_retried = _row_score("m-no-retry", GoldCategory.MULTI_HOP, 1.0, 1.0)
    answerable = [answered, refused, not_retried]
    report = compare.EvaluationReport(
        answerable=tuple(answerable),
        refusal_outcomes=(),
        emergency_escalated=(),
        verdicts=compare.build_verdicts(answerable, [], []),
        grounding_retry_row_ids=("m-retry-answered", "m-retry-refused"),
    )

    text = compare.format_report(report)

    assert "grounding retries (issue #202): 2 rows -- 1 answered, 1 refused" in text
    assert "m-retry-answered" in text
    assert "m-retry-refused" in text
    assert "m-no-retry" not in text.split("grounding retries")[1].split("\n")[0]


def test_the_report_shows_zero_provider_waits_by_default():
    """Issue #185, Scope 5: printed ``0`` when there were none, like the lines above it."""
    text = compare.format_report(_report())

    assert "provider waits: 0, 0 s" in text


def test_the_report_names_the_provider_wait_count_and_total_seconds():
    text = compare.format_report(_report(provider_wait_count=2, provider_wait_seconds=7.4))

    assert "provider waits: 2, 7 s" in text


def test_a_run_whose_reranker_failed_says_it_is_not_a_reranked_measurement():
    # Reranking degrades silently by design; a measurement must not. A run whose sidecar died
    # would otherwise print a table that looks reranked and is the un-reranked baseline.
    text = compare.format_report(
        _report(reranker="llama-server, top-N 5", rerank_ok=20, rerank_fallbacks=11)
    )

    assert "11 fell back" in text
    assert "WARNING: 11 retrieval(s) kept the un-reranked order" in text
    assert "NOT a fully reranked measurement" in text


def test_the_reranker_label_names_provider_and_top_n_or_off():
    off = Settings(_env_file=None)
    on = Settings(reranker=RerankerProvider.LLAMA_SERVER, rerank_top_n=4, _env_file=None)

    assert compare._reranker_label(off) == "off"
    assert compare._reranker_label(on) == "llama-server, top-N 4"


# --- the real CLI entry point: proving a non-zero exit by degrading data, not thresholds -----


def _load_sibling_module(filename: str) -> ModuleType:
    """Load a sibling test module by file path (mirrors ``tests/test_gold.py``'s own helper).

    Reused here, not reimplemented differently, so this file's fixture corpus can never
    silently drift from the one ``tests/test_gold.py`` already proved resolves against the
    real parser -- see that file's own docstring for why a bare package import is unsafe
    under CI's invocation.
    """
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(f"compare_sibling_{path.stem}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive only
        raise ImportError(f"cannot load sibling test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_gold_test_module = _load_sibling_module("test_gold.py")


@pytest.fixture(autouse=True)
def _no_ambient_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``eval.compare.main`` deterministic and offline regardless of the environment.

    ``get_settings()`` is process-wide cached (``lru_cache``), the same reason
    ``tests/test_ingest_cli.py`` clears it -- without this, a test file that ran earlier in
    the same session and set ``EMBEDDER``/``LLM_PROVIDER`` could leak into this file's calls.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RERANKER", raising=False)
    monkeypatch.delenv("RERANK_TOP_N", raising=False)
    monkeypatch.delenv("RERANKER_BASE_URL", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_main_exits_non_zero_when_the_gold_set_cites_units_that_do_not_exist(
    migrated_db: str, tmp_path: Path
):
    """Prove the exit code by degrading the *data*, never the threshold (the issue's own rule).

    The fake unit ids below are well-formed (they pass ``GoldRow``'s own id-shape check) but
    were never ingested anywhere, so no retrieval of any quality -- naive or hybrid, mock
    embeddings or real -- can ever find them. recall@k is therefore exactly ``0.0`` for both
    rows regardless of how good or bad ``MockEmbedder``'s hash-based similarity happens to be
    on this corpus, which is what makes this proof robust rather than a fluke of one run.
    """
    corpus = _gold_test_module._build_fixture_corpus(tmp_path)
    from ingest.run import main as ingest_main

    exit_code = ingest_main(
        [
            "--corpus",
            str(corpus),
            "--database-url",
            migrated_db,
            "--references",
            str(tmp_path / "references.csv"),
        ]
    )
    assert exit_code == 0, "fixture corpus failed to ingest -- cannot run the CLI test at all"

    degraded_gold = tmp_path / "degraded_gold.yaml"
    degraded_gold.write_text(
        "- id: compliance-nonexistent-one\n"
        "  category: compliance\n"
        '  question: "Câu hỏi không thể trả lời được, đơn vị trích dẫn không tồn tại?"\n'
        '  reference_answer: "không có thật"\n'
        '  source_unit_ids: ["totally-fake-document#dieu-999"]\n'
        "- id: compliance-nonexistent-two\n"
        "  category: compliance\n"
        '  question: "Một câu hỏi khác, cũng không có nguồn thật?"\n'
        '  reference_answer: "không có thật"\n'
        '  source_unit_ids: ["another-fake-document#dieu-1"]\n',
        encoding="utf-8",
    )

    exit_code = compare.main(["--gold-path", str(degraded_gold), "--database-url", migrated_db])

    assert exit_code == 1, "recall@k must fail when every gold source id is unreachable"


def test_main_reports_the_configured_reranker_and_that_nothing_fell_back(
    migrated_db: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """The CLI builds the reranker from settings and prints what reranked the run (issue #114)."""
    corpus = _gold_test_module._build_fixture_corpus(tmp_path)
    from ingest.run import main as ingest_main

    assert (
        ingest_main(
            [
                "--corpus",
                str(corpus),
                "--database-url",
                migrated_db,
                "--references",
                str(tmp_path / "references.csv"),
            ]
        )
        == 0
    )
    gold = tmp_path / "gold.yaml"
    gold.write_text(
        "- id: compliance-nonexistent-one\n"
        "  category: compliance\n"
        '  question: "Câu hỏi không thể trả lời được, đơn vị trích dẫn không tồn tại?"\n'
        '  reference_answer: "không có thật"\n'
        '  source_unit_ids: ["totally-fake-document#dieu-999"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("RERANKER", "mock")
    monkeypatch.setenv("RERANK_TOP_N", "3")
    get_settings.cache_clear()

    compare.main(["--gold-path", str(gold), "--database-url", migrated_db])

    output = capsys.readouterr().out
    assert "Reranker: mock, top-N 3" in output
    assert "0 fell back" in output
