"""Tests for generation with enforced citations (issue #90).

Needs a live Postgres (``migrated_db``): ``generate_answer`` looks up each cited
unit's owning document (number, effective date) via ``DocumentRepository``, so even
a purely offline stub ``LLMProvider`` still needs a real database underneath it.
"""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import get_settings
from app.core.db import create_engine, create_session_factory, session_scope
from app.core.ids import (
    appendix_section_clause_id,
    appendix_section_id,
    appendix_section_point_id,
    article_id,
    clause_id,
    point_id,
)
from app.llm.base import ContextOverflowError, LLMProvider, LLMUnavailableError, LLMUnreachableError
from app.llm.mock import MockLLM
from app.repositories.documents import AppendixItemRow, DocumentRepository
from app.repositories.units import UnitRepository
from app.schemas.ask import AskOutcome, AskResponse, Citation
from app.schemas.document import DocStatus, DocType, Document
from app.schemas.entity import EntityType
from app.schemas.retrieval import Passage, RetrievalResult, RetrievedEntity, Why
from app.schemas.unit import Unit, UnitType
from app.services import generator
from eval.gold import load_gold_set
from eval.metrics import answer_check

DOC_A = "gas-decree-87-2018"
DOC_B = "other-decree-2020"

_CHOSEN_GENERATION_K = 5
"""Issue #169's own sweep result -- kept here, not computed, so this test fails loudly if
``GENERATION_RETRIEVAL_CONFIG.k`` ever drifts from what the module docstring's measurement
section says was chosen. Updated at the end of issue #169's own PR, from the 4-point sweep
table; see ``app/services/generator.py``'s module docstring for the measured numbers."""


def _document(
    doc_id: str, number: str, effective: date, status: DocStatus = DocStatus.IN_FORCE
) -> Document:
    return Document(
        id=doc_id,
        number=number,
        title_vi=f"Văn bản {number}",
        doc_type=DocType.DECREE,
        status=status,
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


def _seed_units(database_url: str, *units: Unit) -> None:
    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            await UnitRepository(session).upsert_many(list(units))

    _run(database_url, body)


def _passage(
    unit_id: str, *, focus_units: tuple[str, ...] = (), possibly_superseded: bool = False
) -> Passage:
    document_id = unit_id.split("#", 1)[0]
    return Passage(
        unit_id=unit_id,
        document_id=document_id,
        dieu_no=53,
        text="Cửa hàng bán lẻ LPG chai phải có diện tích tối thiểu là 12 m2.",
        similarity=0.8,
        hop=0,
        score=0.8,
        why=(Why.SIMILAR,),
        focus_units=focus_units,
        possibly_superseded=possibly_superseded,
    )


def _entity(entity_id: str, sources: tuple[str, ...]) -> RetrievedEntity:
    return RetrievedEntity(
        entity_id=entity_id,
        entity_type=EntityType.REQUIREMENT,
        label_vi="Diện tích tối thiểu",
        meta={"value": "≥ 12 m²"},
        similarity=0.8,
        hop=1,
        score=0.7,
        why=(Why.CITES_HIT,),
        sources=sources,
    )


def _is_verifier_schema(json_schema: dict[str, Any] | None) -> bool:
    """Whether a call's ``json_schema`` is the grounding verifier's own (not the generator's).

    Every double below keys on this instead of a marker in the prompt, matching how the real
    ``MockLLM`` identifies a verifier call since issue #118's own follow-up review: the product
    prompt itself carries no test-only hook.
    """
    return json_schema is not None and "supported" in json_schema.get("properties", {})


class _StubLLM(LLMProvider):
    """A double that returns a fixed reply and records every call, for issue #90's own test.

    Answers the grounding verifier's own call (#118) with a well-formed ``{"supported": true}``
    reply -- the same thing ``MockLLM`` does -- rather than replaying ``reply`` (the generator's
    own schema) into the verifier's strict schema. Since #118's verifier fails closed on an
    unparseable reply, replaying the wrong shape here would now drop every citation and turn
    these tests' answers into refusals for a reason unrelated to what each test actually
    exercises.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        if _is_verifier_schema(json_schema):
            return json.dumps({"supported": True})
        return self.reply


class _FailingLLM(LLMProvider):
    """A double that fails the test outright if it is ever invoked (mirrors issue #91's rule)."""

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        pytest.fail("LLMProvider.generate must not be called with no retrieved context")


class _RaisingLLM(LLMProvider):
    """Raises a fixed exception on its first call, optionally answering normally after that.

    Used to prove issue #129's refusal conversion applies both to the main generation call
    (``after=None``: every call raises) and to the grounding verifier's own call (``after``
    set to the generator's own well-formed reply, returned once before the exception fires on
    the first verifier call).
    """

    def __init__(self, error: Exception, after: str | None = None) -> None:
        self._error = error
        self._after = after
        self.calls = 0

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        self.calls += 1
        if self._after is not None and not _is_verifier_schema(json_schema):
            return self._after
        raise self._error


class _StubLLMWithPromptEvalCount(LLMProvider):
    """Mirrors ``OllamaProvider``'s own ``last_prompt_eval_count`` attribute (issue #129).

    So ``GroundingTrace.prompt_eval_count`` capture can be tested with no real Ollama.
    """

    def __init__(self, reply: str, prompt_eval_count: int) -> None:
        self.reply = reply
        self.last_prompt_eval_count = prompt_eval_count
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        if _is_verifier_schema(json_schema):
            # A later verifier call would overwrite the attribute in a real provider too --
            # a different, deliberately distinguishable value proves the generator reads it
            # right after the main call, not after this one.
            self.last_prompt_eval_count = 1
            return json.dumps({"supported": True})
        return self.reply


def _generate(
    question: str,
    retrieval: RetrievalResult,
    llm: LLMProvider,
    database_url: str,
    trace: generator.GroundingTrace | None = None,
) -> AskResponse:
    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await generator.generate_answer(
                question, retrieval, llm, DocumentRepository(session), trace
            )

    result: dict[str, Any] = {}
    _run(database_url, body)
    return result["value"]


# --- no context: refuse without calling the model -----------------------------------------


def test_no_context_at_all_refuses_without_calling_the_llm(migrated_db: str):
    retrieval = RetrievalResult(question="x", passages=(), entities=())

    response = _generate("x", retrieval, _FailingLLM(), migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "no_context"
    assert response.answer is None
    assert response.citations == ()


# --- a grounded answer -----------------------------------------------------------------------


def test_a_grounded_answer_carries_verified_citations(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(
        json.dumps(
            {"refusal": False, "answer": "Tối thiểu 12 m2.", "citations": [{"unit_id": unit_id}]}
        )
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.ANSWER
    assert response.answer == "Tối thiểu 12 m2."
    assert response.disclaimer == generator.DISCLAIMER_VI
    (citation,) = response.citations
    assert citation.unit_id == unit_id
    assert citation.document_id == DOC_A
    assert citation.document_number == "87/2018/NĐ-CP"
    assert citation.dieu_no == 53
    assert citation.khoan_no == 1
    assert citation.effective == date(2018, 8, 1)
    assert len(llm.calls) == 2  # generator plus one per-citation grounding check


def test_grounding_check_adds_one_verifier_call_when_context_exists(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": unit_id}]})
    )

    _generate("x", retrieval, llm, migrated_db)

    assert len(llm.calls) == 2


def test_grounding_check_false_preserves_the_single_generation_call(
    migrated_db: str, monkeypatch: pytest.MonkeyPatch
):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": unit_id}]})
    )
    monkeypatch.setenv("GROUNDING_CHECK", "false")
    get_settings.cache_clear()
    try:
        response = _generate("x", retrieval, llm, migrated_db)
    finally:
        get_settings.cache_clear()
    assert response.outcome is AskOutcome.ANSWER
    assert len(llm.calls) == 1


class _ScriptedGroundingLLM(LLMProvider):
    """Returns a fixed generation reply, then one scripted verdict per verifier call, in order.

    Distinct from ``_StubLLM``, which answers every verifier call the same way: this double
    drives a *different* verdict per citation, so it can prove the per-citation,
    order-preserving drop end to end through the real ``generate_answer`` -- not only through
    ``filter_supported_citations`` directly (``tests/test_grounding.py`` already covers that).
    """

    def __init__(self, reply: str, verdicts: list[bool]) -> None:
        self.reply = reply
        self._verdicts = iter(verdicts)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        if _is_verifier_schema(json_schema):
            return json.dumps({"supported": next(self._verdicts)})
        return self.reply


def test_grounding_check_drops_one_citation_and_keeps_the_other_end_to_end(migrated_db: str):
    unit_a = f"{DOC_A}#dieu-53"
    unit_b = f"{DOC_A}#dieu-54"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_a), _passage(unit_b)))
    llm = _ScriptedGroundingLLM(
        json.dumps(
            {
                "refusal": False,
                "answer": "ok",
                "citations": [{"unit_id": unit_a}, {"unit_id": unit_b}],
            }
        ),
        verdicts=[True, False],
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.ANSWER
    assert [c.unit_id for c in response.citations] == [unit_a]


def test_grounding_check_refuses_with_unsupported_citations_when_the_verifier_drops_all(
    migrated_db: str,
):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    # Only one passage and no entity: issue #202's retry would have nothing left to answer
    # from if it removed it, so this stays a plain refusal with no retry attempted at all.
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _ScriptedGroundingLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": unit_id}]}),
        verdicts=[False],
    )
    trace = generator.GroundingTrace()

    response = _generate("x", retrieval, llm, migrated_db, trace)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "unsupported_citations"
    assert response.answer is None
    assert trace.retried is False
    assert len(llm.calls) == 2  # one generation call, one verifier call -- no retry


# --- retrying once without the rejected passages (issue #202) ------------------------------


class _ScriptedRetryLLM(LLMProvider):
    """One scripted main-generation reply per non-verifier call, in order (issue #202).

    One scripted verifier verdict per verifier call, in order, as ``_ScriptedGroundingLLM``
    already does. Distinct from it, which always replays the same main-generation
    reply: a retry test needs the SECOND generation call to answer differently from the
    first (citing the passage that survived, not the one the first call's citation was
    rejected for), so each call needs its own scripted reply.
    """

    def __init__(self, replies: list[str], verdicts: list[bool]) -> None:
        self._replies = iter(replies)
        self._verdicts = iter(verdicts)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        if _is_verifier_schema(json_schema):
            return json.dumps({"supported": next(self._verdicts)})
        return next(self._replies)


def _generation_calls(
    llm: "_StubLLM | _ScriptedGroundingLLM | _ScriptedRetryLLM",
) -> list[dict[str, Any]]:
    """Return ``llm.calls`` with the grounding verifier's own calls filtered out.

    Shared by every #202 test below that needs the generation-call count alone (never the
    verifier's), regardless of which scripted double recorded the calls -- each already
    stores them in the same ``{"system", "user", "json_schema"}`` shape.
    """
    return [call for call in llm.calls if not _is_verifier_schema(call["json_schema"])]


def test_a_retry_recovers_when_the_second_passages_citation_is_supported(migrated_db: str):
    unit_a = f"{DOC_A}#dieu-53"
    unit_b = f"{DOC_A}#dieu-54"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_a), _passage(unit_b)))
    llm = _ScriptedRetryLLM(
        replies=[
            json.dumps(
                {"refusal": False, "answer": "trả lời a", "citations": [{"unit_id": unit_a}]}
            ),
            json.dumps(
                {"refusal": False, "answer": "trả lời b", "citations": [{"unit_id": unit_b}]}
            ),
        ],
        verdicts=[False, True],
    )
    trace = generator.GroundingTrace()

    response = _generate("x", retrieval, llm, migrated_db, trace)

    assert response.outcome is AskOutcome.ANSWER
    assert [c.unit_id for c in response.citations] == [unit_b]
    assert trace.retried is True
    generation_calls = _generation_calls(llm)
    assert len(generation_calls) == 2
    assert f"[unit_id: {unit_a}]" not in generation_calls[1]["user"]
    assert f"[unit_id: {unit_b}]" in generation_calls[1]["user"]


def test_a_retry_rejected_again_refuses_after_exactly_two_generation_calls(migrated_db: str):
    unit_a = f"{DOC_A}#dieu-53"
    unit_b = f"{DOC_A}#dieu-54"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_a), _passage(unit_b)))
    llm = _ScriptedRetryLLM(
        replies=[
            json.dumps(
                {"refusal": False, "answer": "trả lời a", "citations": [{"unit_id": unit_a}]}
            ),
            json.dumps(
                {"refusal": False, "answer": "trả lời b", "citations": [{"unit_id": unit_b}]}
            ),
        ],
        verdicts=[False, False],
    )
    trace = generator.GroundingTrace()

    response = _generate("x", retrieval, llm, migrated_db, trace)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "unsupported_citations"
    assert trace.retried is True
    assert len(_generation_calls(llm)) == 2  # exactly one retry, never a second one


def test_no_retry_when_the_only_rejected_citation_is_a_focus_unit(migrated_db: str):
    """Trap: a rejected citation that is only a focus unit never removes its own passage."""
    chunk_unit = f"{DOC_A}#dieu-53"
    focus_unit = f"{DOC_A}#dieu-53#khoan-2"
    other_unit = f"{DOC_A}#dieu-60"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(
        question="x",
        passages=(_passage(chunk_unit, focus_units=(focus_unit,)), _passage(other_unit)),
    )
    llm = _ScriptedGroundingLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": focus_unit}]}),
        verdicts=[False],
    )
    trace = generator.GroundingTrace()

    response = _generate("x", retrieval, llm, migrated_db, trace)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "unsupported_citations"
    assert trace.retried is False
    assert len(llm.calls) == 2  # one generation call, one verifier call -- no retry


def test_a_layer_1_refusal_never_retries(migrated_db: str):
    unit_a = f"{DOC_A}#dieu-53"
    unit_b = f"{DOC_A}#dieu-54"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_a), _passage(unit_b)))
    llm = _StubLLM(
        json.dumps(
            {
                "refusal": False,
                "answer": "Nghị định 157/2022/NĐ-CP thay thế văn bản cũ.",
                "citations": [{"unit_id": unit_a}],
            }
        )
    )
    trace = generator.GroundingTrace()

    response = _generate("x", retrieval, llm, migrated_db, trace)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "unsupported_citations"
    assert trace.retried is False
    assert len(llm.calls) == 1  # the document-number guard refused before any verifier call


def test_no_retry_when_the_first_attempt_already_answers(migrated_db: str):
    """Scope item 3's own enumerated case (issue #202).

    The ordinary path, the great majority of requests, never reaches the retry decision at
    all -- this is the exact fixture and expected ``AskResponse`` of ``test_a_grounded_
    answer_carries_verified_citations`` above, now also checked for ``trace.retried`` and the
    generation call count, to prove issue #202 left the byte-for-byte response of a first
    attempt that answers completely unchanged.
    """
    unit_id = f"{DOC_A}#dieu-53#khoan-1"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(
        json.dumps(
            {"refusal": False, "answer": "Tối thiểu 12 m2.", "citations": [{"unit_id": unit_id}]}
        )
    )
    trace = generator.GroundingTrace()

    response = _generate("x", retrieval, llm, migrated_db, trace)

    assert response.outcome is AskOutcome.ANSWER
    assert response.answer == "Tối thiểu 12 m2."
    assert response.citations == (
        Citation(
            unit_id=unit_id,
            document_id=DOC_A,
            document_number="87/2018/NĐ-CP",
            dieu_no=53,
            khoan_no=1,
            effective=date(2018, 8, 1),
        ),
    )
    assert response.possibly_superseded is False
    assert trace.retried is False
    assert len(_generation_calls(llm)) == 1


def test_document_number_guard_refuses_without_calling_the_verifier(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(
        json.dumps(
            {
                "refusal": False,
                "answer": "Nghị định 157/2022/NĐ-CP thay thế văn bản cũ.",
                "citations": [{"unit_id": unit_id}],
            }
        )
    )
    trace = generator.GroundingTrace()

    response = _generate("x", retrieval, llm, migrated_db, trace)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "unsupported_citations"
    assert len(llm.calls) == 1  # the document-number guard refused before any verifier call
    # Issue #118's own follow-up review (item 4): a layer-1 refusal must be visible in the
    # trace, not just the response's reason code -- and must not record a layer-2 decision,
    # since the verifier was never called.
    assert trace.layer1_refusals == 1
    assert trace.decisions == []


def test_status_guard_refuses_unknown_document_without_calling_the_verifier(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(
        migrated_db,
        _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)),
        _document("stub-41", "41/2011/TT-BCT", date(2011, 1, 1), DocStatus.UNKNOWN),
    )
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(
        json.dumps(
            {
                "refusal": False,
                "answer": "Thông tư 41/2011/TT-BCT vẫn còn áp dụng.",
                "citations": [{"unit_id": unit_id}],
            }
        )
    )
    trace = generator.GroundingTrace()

    response = _generate("x", retrieval, llm, migrated_db, trace)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "status_unknown"
    assert len(llm.calls) == 1
    assert trace.layer1_refusals == 1


# --- the real MockLLM end to end (issue #101) -----------------------------------------------
#
# Every test above stubs the LLM with a fixed reply, which never exercises whether MockLLM
# itself can read the prompt generate_answer actually builds. #101 was exactly that: MockLLM's
# marker regex never matched a real, generator-built prompt, so LLM_PROVIDER=mock (the app's
# own default) refused every question. These two tests run the real MockLLM through the real
# generate_answer, so a regression here is caught here rather than only in production.


def test_the_real_mock_llm_answers_a_real_retrieval_with_a_verified_citation(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))

    response = _generate("x", retrieval, MockLLM(), migrated_db)

    assert response.outcome is AskOutcome.ANSWER
    assert response.citations  # at least one -- and only ever one _validate_citations verified
    assert {c.unit_id for c in response.citations} <= {unit_id}


def test_the_real_mock_llm_still_refuses_with_no_retrieved_context(migrated_db: str):
    # A genuinely unanswerable question (nothing retrieved) must still refuse -- the fix must
    # not turn MockLLM into something that always answers regardless of context.
    retrieval = RetrievalResult(question="x", passages=(), entities=())

    response = _generate("x", retrieval, MockLLM(), migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "no_context"
    assert response.citations == ()


# --- fabricated citations: the test that matters more than the happy path -----------------


def test_a_fabricated_citation_is_dropped_but_a_real_one_survives(
    migrated_db: str, caplog: pytest.LogCaptureFixture
):
    real_unit = f"{DOC_A}#dieu-53"
    fake_unit = f"{DOC_A}#dieu-61#khoan-3"  # a plausible id, never retrieved
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(real_unit),))
    llm = _StubLLM(
        json.dumps(
            {
                "refusal": False,
                "answer": "ok",
                "citations": [{"unit_id": real_unit}, {"unit_id": fake_unit}],
            }
        )
    )

    with caplog.at_level("WARNING", logger="app.services.generator"):
        response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.ANSWER
    assert [c.unit_id for c in response.citations] == [real_unit]
    assert any(fake_unit in record.getMessage() for record in caplog.records)
    assert any("fabricat" in record.getMessage() for record in caplog.records)


def test_every_citation_fabricated_refuses(migrated_db: str):
    real_unit = f"{DOC_A}#dieu-53"
    fake_unit = f"{DOC_A}#dieu-99"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(real_unit),))
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": fake_unit}]})
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "no_valid_citations"
    assert response.answer is None


# --- the model's own outcomes -----------------------------------------------------------------


def test_the_model_declining_is_a_refusal(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(json.dumps({"refusal": True, "answer": None, "citations": []}))

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "model_declined"


def test_an_answer_with_no_citations_at_all_refuses(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(json.dumps({"refusal": False, "answer": "ok, trust me", "citations": []}))

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "no_citations_given"


def test_malformed_json_refuses(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM("khong phai JSON dau")

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "invalid_model_output"


def test_json_that_does_not_match_the_schema_refuses(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(json.dumps({"some_other_shape": True}))

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "invalid_model_output"


# --- possibly_superseded propagates from the cited passage --------------------------------


def test_possibly_superseded_is_true_only_when_a_cited_unit_carries_it(migrated_db: str):
    flagged_unit = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(
        question="x", passages=(_passage(flagged_unit, possibly_superseded=True),)
    )
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": flagged_unit}]})
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.possibly_superseded is True


def test_possibly_superseded_stays_false_when_nothing_cited_carries_it(migrated_db: str):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(
        question="x", passages=(_passage(unit_id, possibly_superseded=False),)
    )
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": unit_id}]})
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.possibly_superseded is False


# --- focus units are citable at their own, finer precision -----------------------------------


def test_a_focus_unit_inside_a_passage_is_citable_with_its_own_dieu_and_khoan(migrated_db: str):
    chunk_unit = f"{DOC_A}#dieu-53"  # the chunk: an article-level unit, no khoan of its own
    focus_unit = f"{DOC_A}#dieu-53#khoan-2"  # the specific clause the graph pointed at
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(
        question="x", passages=(_passage(chunk_unit, focus_units=(focus_unit,)),)
    )
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": focus_unit}]})
    )

    response = _generate("x", retrieval, llm, migrated_db)

    (citation,) = response.citations
    assert citation.unit_id == focus_unit
    assert citation.dieu_no == 53
    assert citation.khoan_no == 2  # not None, even though the containing chunk has no khoan_no


# --- an entity's own source unit is citable ---------------------------------------------------


def test_an_entitys_source_unit_is_citable_even_with_no_passage_of_its_own(migrated_db: str):
    source_unit = f"{DOC_B}#dieu-5#khoan-1"
    _seed_documents(migrated_db, _document(DOC_B, "5/2020/NĐ-CP", date(2020, 1, 1)))
    retrieval = RetrievalResult(
        question="x", passages=(), entities=(_entity("req:min-store-area", (source_unit,)),)
    )
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": source_unit}]})
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.ANSWER
    (citation,) = response.citations
    assert citation.unit_id == source_unit
    assert citation.document_number == "5/2020/NĐ-CP"
    assert citation.dieu_no == 5
    assert citation.khoan_no == 1


def test_citing_the_entity_id_itself_rather_than_its_source_is_a_fabrication(migrated_db: str):
    source_unit = f"{DOC_B}#dieu-5#khoan-1"
    _seed_documents(migrated_db, _document(DOC_B, "5/2020/NĐ-CP", date(2020, 1, 1)))
    retrieval = RetrievalResult(
        question="x", passages=(), entities=(_entity("req:min-store-area", (source_unit,)),)
    )
    # The prompt tells the model never to cite "req:...", but a model that does anyway must
    # still be caught by validation, not merely discouraged by instruction.
    llm = _StubLLM(
        json.dumps(
            {"refusal": False, "answer": "ok", "citations": [{"unit_id": "req:min-store-area"}]}
        )
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "no_valid_citations"


# --- generation's own retrieval config is a deliberate, pinned choice ---------------------


def test_generation_retrieval_config_is_wider_than_the_bare_default():
    from app.schemas.retrieval import RetrievalConfig

    assert generator.GENERATION_RETRIEVAL_CONFIG.passage_limit > RetrievalConfig().passage_limit
    assert generator.GENERATION_RETRIEVAL_CONFIG.passage_limit == 15
    assert generator.GENERATION_RETRIEVAL_CONFIG.entity_limit == 12


def test_generation_retrieval_config_k_is_pinned_explicitly_by_measurement():
    """Issue #169's own sweep result, pinned.

    Nobody had chosen k=5 for generation before this -- it only ever came from
    ``RetrievalConfig``'s own bare default, silently inherited. ``GENERATION_RETRIEVAL_CONFIG``
    now states its own k explicitly (see the module docstring's measurement section), chosen
    from the 4-point sweep; this value must match whatever that section says was chosen.
    """
    assert generator.GENERATION_RETRIEVAL_CONFIG.k == _CHOSEN_GENERATION_K


# --- the [unit_id: ...] marker never reaches the returned answer (issue #94) --------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        pytest.param(
            f"...Điều 53 Nghị định 87/2018/NĐ-CP (unit_id: {DOC_A}#dieu-53):",
            "...Điều 53 Nghị định 87/2018/NĐ-CP:",
            id="parens-the-exact-shape-observed-on-the-real-leak",
        ),
        pytest.param(
            f"Đây là câu trả lời [unit_id: {DOC_A}#dieu-53].",
            "Đây là câu trả lời.",
            id="square-brackets-the-prompts-own-marker-form",
        ),
        pytest.param(
            f"{{unit_id: {DOC_A}#dieu-53}} là căn cứ.",
            "là căn cứ.",
            id="curly-braces-at-the-very-start",
        ),
        pytest.param(
            f"Căn cứ pháp lý là unit_id: {DOC_A}#dieu-53",
            "Căn cứ pháp lý là",
            id="no-brackets-at-all-mid-sentence-and-at-the-very-end",
        ),
        pytest.param(
            f"Xem(unit_id: {DOC_A}#dieu-53#khoan-1)ngay.",
            "Xemngay.",
            id="no-surrounding-space-either-side",
        ),
        pytest.param(
            f" (unit_id: {DOC_A}#dieu-53) ",
            "",
            id="the-whole-answer-is-nothing-but-the-marker",
        ),
    ],
)
def test_a_leaked_marker_in_any_plausible_shape_is_stripped(raw: str, expected: str):
    assert generator._strip_leaked_unit_id_markers(raw) == expected


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(
            "Không có thay đổi gì (trừ trường hợp có tường ngăn cháy).",
            id="starts-with-an-ordinary-vietnamese-word",
        ),
        pytest.param(
            "Cần tuân thủ quy định (theo quy định hiện hành) trước khi mở cửa hàng.",
            id="ordinary-clause-no-unit_id-substring",
        ),
        pytest.param(
            "Diện tích tối thiểu (≥ 12 m², theo Điều 53) là bắt buộc.",
            id="nested-punctuation-comma-and-a-non-ascii-symbol",
        ),
        pytest.param(
            "Phải có (biển báo, bình chữa cháy) tại cửa hàng.",
            id="comma-inside-the-parens",
        ),
        pytest.param(
            "Một câu có dấu ngoặc lồng nhau ((xem thêm Điều 10) và Điều 11) là hợp lệ.",
            id="nested-parens",
        ),
    ],
)
def test_ordinary_parenthesised_vietnamese_text_survives_untouched(answer: str):
    assert generator._strip_leaked_unit_id_markers(answer) == answer


def test_a_stub_model_that_deliberately_leaks_the_marker_is_handled(migrated_db: str):
    """The test that matters most.

    Rule 5 of ``_SYSTEM_PROMPT_VI`` is a request, not a guarantee (#90's own reasoning
    about citations), so a stub that disobeys it outright must still come back with a
    clean ``answer`` and an unaffected ``citations``.
    """
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    leaked_answer = (
        "Để mở cửa hàng bán lẻ LPG chai, cần đáp ứng các điều kiện sau theo quy định tại "
        f"Điều 53 Nghị định 87/2018/NĐ-CP (unit_id: {unit_id}):"
    )
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": leaked_answer, "citations": [{"unit_id": unit_id}]})
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.ANSWER
    assert response.answer is not None
    assert "unit_id" not in response.answer
    assert response.answer == (
        "Để mở cửa hàng bán lẻ LPG chai, cần đáp ứng các điều kiện sau theo quy định tại "
        "Điều 53 Nghị định 87/2018/NĐ-CP:"
    )
    # citations is unchanged in content -- same unit, same document number, same date --
    # even though the answer text that named it had to be cleaned up.
    (citation,) = response.citations
    assert citation.unit_id == unit_id
    assert citation.document_id == DOC_A
    assert citation.document_number == "87/2018/NĐ-CP"
    assert citation.dieu_no == 53
    assert citation.effective == date(2018, 8, 1)


def test_a_model_that_follows_the_no_leak_instruction_has_its_answer_left_alone(
    migrated_db: str,
):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(
        json.dumps(
            {
                "refusal": False,
                "answer": "Diện tích tối thiểu là 12 m2 (trừ trường hợp có tường ngăn cháy).",
                "citations": [{"unit_id": unit_id}],
            }
        )
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.answer == "Diện tích tối thiểu là 12 m2 (trừ trường hợp có tường ngăn cháy)."


def test_passages_beyond_the_generation_limit_are_never_offered_to_the_model(migrated_db: str):
    documents = [_document(f"doc-{i}", f"{i}/2020/NĐ-CP", date(2020, 1, 1)) for i in range(20)]
    _seed_documents(migrated_db, *documents)
    passages = tuple(_passage(f"doc-{i}#dieu-1") for i in range(20))
    retrieval = RetrievalResult(question="x", passages=passages)
    llm = _StubLLM(json.dumps({"refusal": True, "answer": None, "citations": []}))

    _generate("x", retrieval, llm, migrated_db)

    (call,) = llm.calls
    offered = call["user"].count("[unit_id: ")
    assert offered == generator.GENERATION_RETRIEVAL_CONFIG.passage_limit == 15


# --- issue #108: each focus unit is its own marked, labelled line -------------------------


def test_each_focus_unit_is_shown_with_its_own_marked_line_and_label(migrated_db: str):
    chunk_unit = f"{DOC_A}#dieu-53"
    focus_a = f"{DOC_A}#dieu-53#khoan-2"
    focus_b = f"{DOC_A}#dieu-53#khoan-3"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(
        question="x", passages=(_passage(chunk_unit, focus_units=(focus_a, focus_b)),)
    )
    llm = _StubLLM(json.dumps({"refusal": True, "answer": None, "citations": []}))

    _generate("x", retrieval, llm, migrated_db)

    (call,) = llm.calls
    prompt = call["user"]
    assert f"[unit_id: {focus_a}]" in prompt
    assert f"[unit_id: {focus_b}]" in prompt
    # Each focus unit carries its own Khoản number, not just the parent passage's bare id --
    # proof a model reading the prompt can tell the two candidates apart, the #108 fix.
    assert "Khoản 2" in prompt
    assert "Khoản 3" in prompt


# --- issue #187: an article/clause passage with focus units renders them structured --------

_DIEU_8_HEADING = "Điều 8. Điều kiện đối với thương nhân kinh doanh mua bán khí"
_DIEU_8_FLATTENED_TEXT = (
    f"{_DIEU_8_HEADING}\n"
    "Điều kiện đối với thương nhân kinh doanh mua bán khí gồm:\n\n"
    "Là thương nhân được thành lập theo quy định của pháp luật;\n\n"
    "Có bồn chứa khí đáp ứng các quy định về an toàn."
)
_KHOAN_1_TRADER_BODY = "Điều kiện đối với thương nhân kinh doanh mua bán khí gồm:"
_DIEM_A_TRADER_BODY = "Là thương nhân được thành lập theo quy định của pháp luật."
_DIEM_C_TRADER_BODY = "Có bồn chứa khí đáp ứng các quy định về an toàn."


def _dieu_8_units(document_id: str = DOC_A) -> tuple[Unit, ...]:
    """Build Điều 8's own unit tree: article (empty body), khoản 1, điểm a and điểm c.

    Shaped like the issue's own example: the article's own ``body`` is empty (its
    real content lives entirely in its clause), so the structured render's heading
    line must come from the passage's flattened ``text``, never from ``units.body``
    (issue #187's own trap 4). điểm b is deliberately absent -- these tests are about
    order and labels, not completeness, same discipline ``_deductible_rule_seed_units``
    uses for a genuine parse gap.
    """
    article = article_id(document_id, 8)
    khoan_1 = clause_id(document_id, 8, 1)
    return (
        Unit(
            id=article,
            document_id=document_id,
            unit_type=UnitType.ARTICLE,
            dieu_no=8,
            heading=_DIEU_8_HEADING,
            ordinal=0,
        ),
        Unit(
            id=khoan_1,
            document_id=document_id,
            unit_type=UnitType.CLAUSE,
            dieu_no=8,
            khoan_no=1,
            body=_KHOAN_1_TRADER_BODY,
            parent_id=article,
            ordinal=1,
        ),
        Unit(
            id=point_id(document_id, 8, 1, "a"),
            document_id=document_id,
            unit_type=UnitType.POINT,
            dieu_no=8,
            khoan_no=1,
            diem="a",
            body=_DIEM_A_TRADER_BODY,
            parent_id=khoan_1,
            ordinal=1,
        ),
        Unit(
            id=point_id(document_id, 8, 1, "c"),
            document_id=document_id,
            unit_type=UnitType.POINT,
            dieu_no=8,
            khoan_no=1,
            diem="c",
            body=_DIEM_C_TRADER_BODY,
            parent_id=khoan_1,
            ordinal=3,
        ),
    )


def _dieu_8_passage(focus_units: tuple[str, ...], document_id: str = DOC_A) -> Passage:
    return Passage(
        unit_id=article_id(document_id, 8),
        document_id=document_id,
        dieu_no=8,
        text=_DIEU_8_FLATTENED_TEXT,
        similarity=0.8,
        hop=0,
        score=0.8,
        why=(Why.SIMILAR,),
        focus_units=focus_units,
    )


def test_article_passage_renders_heading_plus_tree_ordered_child_lines_each_body_once(
    migrated_db: str,
):
    """The issue's own fix, end to end.

    The flattened text and its "Các đơn vị cụ thể hơn" block are gone; the article's
    heading (from ``text``, since its own ``body`` is empty) is followed by one line
    per focus unit, in tree order, each showing that unit's own body exactly once --
    even though ``focus_units`` itself names them out of that order.
    """
    document_id = DOC_A
    _seed_documents(migrated_db, _document(document_id, "87/2018/NĐ-CP", date(2018, 8, 1)))
    _seed_units(migrated_db, *_dieu_8_units(document_id))
    khoan_1 = clause_id(document_id, 8, 1)
    diem_a = point_id(document_id, 8, 1, "a")
    diem_c = point_id(document_id, 8, 1, "c")
    passage = _dieu_8_passage(focus_units=(diem_c, diem_a, khoan_1), document_id=document_id)
    retrieval = RetrievalResult(question="x", passages=(passage,))
    llm = _StubLLM(json.dumps({"refusal": True, "answer": None, "citations": []}))

    _generate("x", retrieval, llm, migrated_db)

    (call,) = llm.calls
    prompt = call["user"]
    assert "Các đơn vị cụ thể hơn" not in prompt
    assert _DIEU_8_HEADING in prompt
    # The flattened text's own body lines, as a block, never reappear.
    assert (
        "Điều kiện đối với thương nhân kinh doanh mua bán khí gồm:\n\nLà thương nhân" not in prompt
    )
    assert prompt.count(_KHOAN_1_TRADER_BODY) == 1
    assert prompt.count(_DIEM_A_TRADER_BODY) == 1
    assert prompt.count(_DIEM_C_TRADER_BODY) == 1
    # Tree order: khoản 1 (the article's only direct child) first, then its điểm in
    # their own ordinal order -- regardless of the order focus_units named them in.
    assert prompt.index(f"[unit_id: {khoan_1}]") < prompt.index(f"[unit_id: {diem_a}]")
    assert prompt.index(f"[unit_id: {diem_a}]") < prompt.index(f"[unit_id: {diem_c}]")


def test_a_passage_with_its_own_body_and_children_renders_that_body_once_before_the_children(
    migrated_db: str,
):
    """Fix round on #187: the passage unit's own ``body`` is not always empty.

    The first round's render assumed an article passage's own ``body`` is always
    empty (the issue's own, since-corrected trap 4). That holds for an article the
    corpus subdivided entirely into clauses, but a clause-level chunk's own body (its
    intro text, before its lettered points) is real legal text, dropped by the first
    round. Here the passage is the clause ``khoan_1`` itself (not the article), with
    its own điểm as children -- the exact shape the review's probe found missing.
    """
    document_id = DOC_A
    khoan_1 = clause_id(document_id, 8, 1)
    diem_a = point_id(document_id, 8, 1, "a")
    diem_c = point_id(document_id, 8, 1, "c")
    _seed_documents(migrated_db, _document(document_id, "87/2018/NĐ-CP", date(2018, 8, 1)))
    _seed_units(migrated_db, *_dieu_8_units(document_id))
    passage = Passage(
        unit_id=khoan_1,
        document_id=document_id,
        dieu_no=8,
        khoan_no=1,
        text=f"Điều 8. Khoản 1\n{_KHOAN_1_TRADER_BODY}",
        similarity=0.8,
        hop=0,
        score=0.8,
        why=(Why.SIMILAR,),
        focus_units=(diem_c, diem_a),
    )
    retrieval = RetrievalResult(question="x", passages=(passage,))
    llm = _StubLLM(json.dumps({"refusal": True, "answer": None, "citations": []}))

    _generate("x", retrieval, llm, migrated_db)

    (call,) = llm.calls
    prompt = call["user"]
    # The own body appears exactly once, between the heading line and the first
    # child's own line -- never duplicated, never dropped.
    assert prompt.count(_KHOAN_1_TRADER_BODY) == 1
    heading_index = prompt.index(f"[unit_id: {khoan_1}]")
    own_body_index = prompt.index(_KHOAN_1_TRADER_BODY)
    first_child_index = prompt.index(f"[unit_id: {diem_a}]")
    assert heading_index < own_body_index < first_child_index


def test_child_lines_are_ordered_by_ordinal_not_by_sorting_the_id_strings(migrated_db: str):
    """``khoan-10`` sorts before ``khoan-2`` as text -- the issue's own trap 3.

    Seeds two clauses whose ordinals disagree with their id strings' own sort order
    and proves the render follows ``ordinal``, never the id.
    """
    document_id = DOC_A
    article = article_id(document_id, 20)
    khoan_2 = clause_id(document_id, 20, 2)
    khoan_10 = clause_id(document_id, 20, 10)
    _seed_documents(migrated_db, _document(document_id, "87/2018/NĐ-CP", date(2018, 8, 1)))
    _seed_units(
        migrated_db,
        Unit(
            id=article,
            document_id=document_id,
            unit_type=UnitType.ARTICLE,
            dieu_no=20,
            heading="Điều 20. Quy định khác.",
            ordinal=0,
        ),
        Unit(
            id=khoan_2,
            document_id=document_id,
            unit_type=UnitType.CLAUSE,
            dieu_no=20,
            khoan_no=2,
            body="Nội dung khoản 2.",
            parent_id=article,
            ordinal=1,
        ),
        Unit(
            id=khoan_10,
            document_id=document_id,
            unit_type=UnitType.CLAUSE,
            dieu_no=20,
            khoan_no=10,
            body="Nội dung khoản 10.",
            parent_id=article,
            ordinal=2,
        ),
    )
    passage = Passage(
        unit_id=article,
        document_id=document_id,
        dieu_no=20,
        text="Điều 20. Quy định khác.",
        similarity=0.8,
        hop=0,
        score=0.8,
        why=(Why.SIMILAR,),
        focus_units=(khoan_10, khoan_2),  # named out of tree order on purpose
    )
    retrieval = RetrievalResult(question="x", passages=(passage,))
    llm = _StubLLM(json.dumps({"refusal": True, "answer": None, "citations": []}))

    _generate("x", retrieval, llm, migrated_db)

    (call,) = llm.calls
    prompt = call["user"]
    assert prompt.index(f"[unit_id: {khoan_2}]") < prompt.index(f"[unit_id: {khoan_10}]")


def test_point_label_carries_diem_after_khoan():
    """Scope item 2: a point unit's label gets ", Điểm x" after its Khoản (issue #187).

    Points a and c of the same clause no longer carry identical labels -- the bug
    the issue opens with.
    """
    unit_id = f"{DOC_A}#dieu-8#khoan-1#diem-c"
    source = generator._Source(
        unit_id=unit_id,
        document_id=DOC_A,
        document_number="87/2018/NĐ-CP",
        doc_type="decree",
        dieu_no=8,
        khoan_no=1,
        effective=date(2018, 8, 1),
        possibly_superseded=False,
    )

    label = generator._label(source)

    assert label == "Văn bản 87/2018/NĐ-CP, Điều 8, Khoản 1, Điểm c, hiệu lực từ 2018-08-01"


def test_label_does_not_add_diem_for_an_appendix_point():
    """#187's fix is scoped to the dieu/khoan branch.

    Issue #165's appendix labelling (an ``appendix_section_point`` has no mục
    number to show) stays untouched.
    """
    unit_id = appendix_section_point_id(_PCCC, 6, "ii", 1, "b")
    source = generator._Source(
        unit_id=unit_id,
        document_id=_PCCC,
        document_number="105/2025/NĐ-CP",
        doc_type="decree",
        dieu_no=None,
        khoan_no=None,
        effective=None,
        possibly_superseded=False,
    )

    label = generator._label(source)

    assert "Điểm" not in label
    assert label == "Phụ lục VI — 105/2025/NĐ-CP"


def test_non_appendix_passage_with_no_focus_units_prompt_is_byte_identical_to_pre_187():
    """A clause-level chunk with no children is untouched by #187's own carve-out.

    Scope item 1's third bullet: the structured rendering only ever replaces a
    passage that *has* focus units.
    """
    unit_id = f"{DOC_A}#dieu-10"
    passage = Passage(
        unit_id=unit_id,
        document_id=DOC_A,
        dieu_no=10,
        text="Điều 10. Quy định chung về an toàn.",
        similarity=0.8,
        hop=0,
        score=0.8,
        why=(Why.SIMILAR,),
        focus_units=(),
    )
    sources = {
        unit_id: generator._Source(
            unit_id=unit_id,
            document_id=DOC_A,
            document_number="87/2018/NĐ-CP",
            doc_type="decree",
            dieu_no=10,
            khoan_no=None,
            effective=None,
            possibly_superseded=False,
        )
    }

    prompt = generator._build_user_prompt("x", (passage,), (), sources)

    expected = (
        "NGỮ CẢNH PHÁP LÝ:\n\n"
        f"[unit_id: {unit_id}] Văn bản 87/2018/NĐ-CP, Điều 10\n"
        "Điều 10. Quy định chung về an toàn.\n\n"
        "CÂU HỎI: x"
    )
    assert prompt == expected


def test_appendix_passage_with_focus_units_prompt_is_byte_identical_to_pre_187():
    """An appendix passage keeps its exact pre-#187 shape, focus units included.

    The literal below is what ``_build_user_prompt`` produced before this issue's
    own change -- the appendix branch is explicitly never touched by it (scope item
    1's third bullet) -- never recomputed from the current code, same discipline as
    ``test_non_yes_no_question_prompt_is_byte_identical_to_the_pre_173_prompt``.
    """
    unit_id = f"{_PCCC}#phu-luc-6#muc-17"
    focus_unit = f"{_PCCC}#phu-luc-6#muc-ii#khoan-1#diem-b"
    passage = Passage(
        unit_id=unit_id,
        document_id=_PCCC,
        dieu_no=None,
        text="Co so kinh doanh khi dot ... N 0,3",
        similarity=0.8,
        hop=0,
        score=0.8,
        why=(Why.SIMILAR,),
        focus_units=(focus_unit,),
        possibly_superseded=False,
    )
    sources = {
        unit_id: _source(unit_id, document_number="105/2025/NĐ-CP"),
        focus_unit: generator._Source(
            unit_id=focus_unit,
            document_id=_PCCC,
            document_number="105/2025/NĐ-CP",
            doc_type="decree",
            dieu_no=None,
            khoan_no=None,
            effective=None,
            possibly_superseded=False,
        ),
    }

    prompt = generator._build_user_prompt(_PREMIUM_QUESTION, (passage,), (), sources)

    expected = (
        "NGỮ CẢNH PHÁP LÝ:\n\n"
        f"[unit_id: {unit_id}] Phụ lục VI, mục 17 — 105/2025/NĐ-CP\n"
        "Co so kinh doanh khi dot ... N 0,3\n"
        "Các đơn vị cụ thể hơn bên trong đoạn trên:\n"
        f"  [unit_id: {focus_unit}] Phụ lục VI — 105/2025/NĐ-CP\n\n"
        f"CÂU HỎI: {_PREMIUM_QUESTION}"
    )
    assert prompt == expected


def test_structured_prompt_is_no_longer_than_the_old_shape_plus_120_chars_per_child_line():
    """Scope item 4's own ceiling, checked against the pre-#187 shape, hand-built here.

    Never against the frozen function itself, which no longer exists -- so a future
    change that starts bloating a child's own line (e.g. repeating the whole passage
    text per child, the issue's own trap 1) fails this even though it only touches
    the new code path.
    """
    document_id = DOC_A
    article = article_id(document_id, 8)
    khoan_1 = clause_id(document_id, 8, 1)
    diem_a = point_id(document_id, 8, 1, "a")
    diem_c = point_id(document_id, 8, 1, "c")
    sources = {
        article: generator._Source(
            unit_id=article,
            document_id=document_id,
            document_number="87/2018/NĐ-CP",
            doc_type="decree",
            dieu_no=8,
            khoan_no=None,
            effective=None,
            possibly_superseded=False,
        ),
        khoan_1: generator._Source(
            unit_id=khoan_1,
            document_id=document_id,
            document_number="87/2018/NĐ-CP",
            doc_type="decree",
            dieu_no=8,
            khoan_no=1,
            effective=None,
            possibly_superseded=False,
            body=_KHOAN_1_TRADER_BODY,
            parent_id=article,
            ordinal=1,
        ),
        diem_a: generator._Source(
            unit_id=diem_a,
            document_id=document_id,
            document_number="87/2018/NĐ-CP",
            doc_type="decree",
            dieu_no=8,
            khoan_no=1,
            effective=None,
            possibly_superseded=False,
            body=_DIEM_A_TRADER_BODY,
            parent_id=khoan_1,
            ordinal=1,
        ),
        diem_c: generator._Source(
            unit_id=diem_c,
            document_id=document_id,
            document_number="87/2018/NĐ-CP",
            doc_type="decree",
            dieu_no=8,
            khoan_no=1,
            effective=None,
            possibly_superseded=False,
            body=_DIEM_C_TRADER_BODY,
            parent_id=khoan_1,
            ordinal=3,
        ),
    }
    focus_units = (khoan_1, diem_a, diem_c)
    passage = _dieu_8_passage(focus_units=focus_units, document_id=document_id)

    new_prompt = generator._build_user_prompt("x", (passage,), (), sources)

    old_lines = [
        f"[unit_id: {article}] {generator._label(sources[article])}",
        passage.text,
        "Các đơn vị cụ thể hơn bên trong đoạn trên:",
        *(
            f"  [unit_id: {focus_unit}] {generator._label(sources[focus_unit])}"
            for focus_unit in focus_units
        ),
    ]
    old_prompt_len = len("\n".join(["NGỮ CẢNH PHÁP LÝ:", "", *old_lines, "", "CÂU HỎI: x"]))

    assert len(new_prompt) <= old_prompt_len + 120 * len(focus_units)


# --- issue #108: a bundled, comma-joined unit_id is split before validation ----------------


def test_a_bundled_multi_id_citation_is_split_and_every_piece_survives(migrated_db: str):
    unit_a = f"{DOC_A}#dieu-53#khoan-2"
    unit_b = f"{DOC_A}#dieu-53#khoan-3"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    chunk_unit = f"{DOC_A}#dieu-53"
    retrieval = RetrievalResult(
        question="x", passages=(_passage(chunk_unit, focus_units=(unit_a, unit_b)),)
    )
    bundled_unit_id = f"{unit_a}, {unit_b}"
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": bundled_unit_id}]})
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.ANSWER
    assert {c.unit_id for c in response.citations} == {unit_a, unit_b}


def test_a_bundle_mixing_a_real_and_a_fabricated_id_keeps_only_the_real_one(
    migrated_db: str, caplog: pytest.LogCaptureFixture
):
    real_unit = f"{DOC_A}#dieu-53"
    fake_unit = f"{DOC_A}#dieu-99"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(real_unit),))
    bundled_unit_id = f"{real_unit}, {fake_unit}"
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": bundled_unit_id}]})
    )

    with caplog.at_level("WARNING", logger="app.services.generator"):
        response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.ANSWER
    assert [c.unit_id for c in response.citations] == [real_unit]
    assert any(fake_unit in record.getMessage() for record in caplog.records)


def test_a_bundle_of_only_fabricated_ids_still_refuses(migrated_db: str):
    real_unit = f"{DOC_A}#dieu-53"
    fake_unit_a = f"{DOC_A}#dieu-98"
    fake_unit_b = f"{DOC_A}#dieu-99"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(real_unit),))
    bundled_unit_id = f"{fake_unit_a}, {fake_unit_b}"
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": bundled_unit_id}]})
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "no_valid_citations"


# --- issue #129: a provider failure is a refusal, never an unhandled exception ------------


def test_a_context_overflow_on_the_main_generation_call_refuses_with_its_own_reason(
    migrated_db: str, caplog: pytest.LogCaptureFixture
):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _RaisingLLM(ContextOverflowError("prompt_eval_count=4095 is within budget"))

    with caplog.at_level("ERROR", logger="app.services.generator"):
        response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "context_overflow"
    assert response.answer is None
    assert any("4095" in record.getMessage() for record in caplog.records)


def test_a_provider_timeout_on_the_main_generation_call_refuses_as_llm_unavailable_not_a_500(
    migrated_db: str,
):
    """The exact bug issue #129 reports.

    A timed-out generation call used to propagate raw, which ``app.api.ask`` had no handler
    for, surfacing as an unhandled HTTP 500.
    """
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _RaisingLLM(LLMUnavailableError("Ollama request timed out after 120.0s"))

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "llm_unavailable"
    assert response.answer is None


def test_a_context_overflow_on_the_verifiers_own_call_also_refuses(migrated_db: str):
    """Trap 3: the grounding verifier's calls go through the same provider and so get it too.

    A short verifier prompt overflowing is exactly as untrustworthy as a long generation
    prompt overflowing, and must refuse the same way.
    """
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _RaisingLLM(
        ContextOverflowError("prompt_eval_count=999 is within budget"),
        after=json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": unit_id}]}),
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "context_overflow"
    assert llm.calls == 2  # the generation call succeeded; the verifier's own call overflowed


def test_a_provider_timeout_on_the_verifiers_own_call_also_refuses_as_llm_unavailable(
    migrated_db: str,
):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _RaisingLLM(
        LLMUnavailableError("timed out"),
        after=json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": unit_id}]}),
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "llm_unavailable"


# --- issue #183: an unreachable provider refuses with its own reason, never a 500 --------


def test_a_connection_failure_on_the_main_generation_call_refuses_as_llm_unreachable(
    migrated_db: str,
):
    """The exact bug issue #183 reports, for the main generation call.

    A connection failure (Ollama restarted mid-request) used to propagate raw, which
    ``app.api.ask`` had no handler for, surfacing as an unhandled HTTP 500.
    """
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _RaisingLLM(LLMUnreachableError("Ollama unreachable at http://x: connection refused"))

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "llm_unreachable"
    assert response.answer is None


def test_a_connection_failure_on_the_verifiers_own_call_also_refuses_as_llm_unreachable(
    migrated_db: str,
):
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _RaisingLLM(
        LLMUnreachableError("Ollama unreachable at http://x: connection refused"),
        after=json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": unit_id}]}),
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "llm_unreachable"


def test_mock_llm_is_unaffected_by_the_context_window_change(migrated_db: str):
    """Acceptance criterion.

    MockLLM, which has no ``last_prompt_eval_count`` attribute at all and never raises
    either new exception, must answer exactly as before this issue.
    """
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    trace = generator.GroundingTrace()

    response = _generate("x", retrieval, MockLLM(), migrated_db, trace)

    assert response.outcome is AskOutcome.ANSWER
    assert trace.prompt_eval_count is None


def test_grounding_trace_records_the_main_calls_own_prompt_eval_count(migrated_db: str):
    """The value must come from the *generation* call.

    Not get overwritten by the verifier's own later call through the same provider -- see
    ``_StubLLMWithPromptEvalCount``.
    """
    unit_id = f"{DOC_A}#dieu-53"
    _seed_documents(migrated_db, _document(DOC_A, "87/2018/NĐ-CP", date(2018, 8, 1)))
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLMWithPromptEvalCount(
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": unit_id}]}),
        prompt_eval_count=6053,
    )
    trace = generator.GroundingTrace()

    response = _generate("x", retrieval, llm, migrated_db, trace)

    assert response.outcome is AskOutcome.ANSWER
    assert trace.prompt_eval_count == 6053  # not 1, the verifier call's own later value


# --- quoting a cited appendix item verbatim (issue #146) ------------------------------------
#
# Deterministic post-processing (``generator._quote_cited_appendix_items``), never an LLM
# call -- these tests exercise the pure function directly with hand-built fixtures, so no
# database or stub model is needed for most of them. One end-to-end test at the bottom proves
# the real wiring (``DocumentRepository.get_appendix_items``) through ``generate_answer``.

_PCCC = "pccc-decree-105-2025"


def _citation(unit_id: str, document_number: str = "105/2025/NĐ-CP") -> Citation:
    return Citation(
        unit_id=unit_id,
        document_id=unit_id.split("#", 1)[0],
        document_number=document_number,
    )


def _source(
    unit_id: str, document_number: str = "105/2025/NĐ-CP", doc_type: str = "decree"
) -> "generator._Source":
    return generator._Source(
        unit_id=unit_id,
        document_id=unit_id.split("#", 1)[0],
        document_number=document_number,
        doc_type=doc_type,
        dieu_no=None,
        khoan_no=None,
        effective=None,
        possibly_superseded=False,
    )


def _appendix_row(
    unit_id: str, unit_type: str, appendix_no: int | None, item_no: int | None, body: str
) -> AppendixItemRow:
    return AppendixItemRow(
        id=unit_id, unit_type=unit_type, appendix_no=appendix_no, item_no=item_no, body=body
    )


def test_an_appendix_item_is_quoted():
    unit_id = f"{_PCCC}#phu-luc-7#muc-15"
    body = "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg."
    citations = (_citation(unit_id),)
    rows = {unit_id: _appendix_row(unit_id, "appendix_item", 7, 15, body)}
    sources = {unit_id: _source(unit_id)}

    result = generator._quote_cited_appendix_items(
        "Có, cơ sở này phải mua bảo hiểm cháy, nổ bắt buộc.", citations, rows, sources
    )

    assert result == (
        "Có, cơ sở này phải mua bảo hiểm cháy, nổ bắt buộc.\n"
        f'Phụ lục VII, mục 15 (Nghị định 105/2025/NĐ-CP): "{body}"'
    )


def test_a_subitem_is_quoted():
    unit_id = f"{_PCCC}#phu-luc-6#muc-35#muc-1"
    body = "Mức phí bảo hiểm: 0,15%."
    citations = (_citation(unit_id),)
    rows = {unit_id: _appendix_row(unit_id, "appendix_subitem", 6, 35, body)}
    sources = {unit_id: _source(unit_id)}

    result = generator._quote_cited_appendix_items("ok", citations, rows, sources)

    assert result == f'ok\nPhụ lục VI, mục 35, mục 1 (Nghị định 105/2025/NĐ-CP): "{body}"'


def test_a_301_character_item_is_not_quoted():
    unit_id = f"{_PCCC}#phu-luc-7#muc-16"
    body = "x" * 301
    citations = (_citation(unit_id),)
    rows = {unit_id: _appendix_row(unit_id, "appendix_item", 7, 16, body)}
    sources = {unit_id: _source(unit_id)}

    result = generator._quote_cited_appendix_items("ok", citations, rows, sources)

    assert result == "ok"


def test_a_300_character_item_is_still_quoted():
    """Issue #146's own wording: "at most 300 characters" -- 300 itself still qualifies."""
    unit_id = f"{_PCCC}#phu-luc-7#muc-17"
    body = "x" * 300
    citations = (_citation(unit_id),)
    rows = {unit_id: _appendix_row(unit_id, "appendix_item", 7, 17, body)}
    sources = {unit_id: _source(unit_id)}

    result = generator._quote_cited_appendix_items("ok", citations, rows, sources)

    assert result == f'ok\nPhụ lục VII, mục 17 (Nghị định 105/2025/NĐ-CP): "{body}"'


@pytest.mark.parametrize(
    "unit_type", ["appendix_form", "article", "clause", "appendix_point", "appendix_section"]
)
def test_a_form_article_or_clause_is_never_quoted(unit_type: str):
    """Defense in depth: never quoted even if a row of the wrong type reached this function.

    The function does not rely solely on its caller (``DocumentRepository.get_appendix_items``'s
    own SQL filter) having done this.
    """
    unit_id = f"{_PCCC}#phu-luc-7#mau-pc01"
    citations = (_citation(unit_id),)
    rows = {unit_id: _appendix_row(unit_id, unit_type, 7, 1, "short body")}
    sources = {unit_id: _source(unit_id)}

    result = generator._quote_cited_appendix_items("ok", citations, rows, sources)

    assert result == "ok"


def test_an_answer_with_no_appendix_citation_is_unchanged_byte_for_byte():
    unit_id = f"{DOC_A}#dieu-53"
    citations = (_citation(unit_id, document_number="87/2018/NĐ-CP"),)
    sources = {unit_id: _source(unit_id, document_number="87/2018/NĐ-CP")}
    answer = "Tối thiểu 12 m2."

    result = generator._quote_cited_appendix_items(answer, citations, {}, sources)

    assert result is answer  # the very same string object, not merely an equal one


def test_duplicate_citations_of_the_same_appendix_item_are_quoted_once():
    unit_id = f"{_PCCC}#phu-luc-7#muc-15"
    body = "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg."
    citations = (_citation(unit_id), _citation(unit_id))
    rows = {unit_id: _appendix_row(unit_id, "appendix_item", 7, 15, body)}
    sources = {unit_id: _source(unit_id)}

    result = generator._quote_cited_appendix_items("ok", citations, rows, sources)

    assert result.count("Phụ lục VII") == 1


def test_quotes_follow_citation_order():
    unit_15 = f"{_PCCC}#phu-luc-7#muc-15"
    unit_13 = f"{_PCCC}#phu-luc-1#muc-13"
    citations = (_citation(unit_15), _citation(unit_13))
    rows = {
        unit_15: _appendix_row(unit_15, "appendix_item", 7, 15, "mười lăm"),
        unit_13: _appendix_row(unit_13, "appendix_item", 1, 13, "mười ba"),
    }
    sources = {unit_15: _source(unit_15), unit_13: _source(unit_13)}

    result = generator._quote_cited_appendix_items("ok", citations, rows, sources)

    assert result.index("mười lăm") < result.index("mười ba")


# --- the real wiring end to end (issue #146) -------------------------------------------------


def test_a_real_appendix_item_citation_is_quoted_end_to_end(migrated_db: str):
    """Proves ``DocumentRepository.get_appendix_items`` round-trips through ``generate_answer``.

    Every test above exercises ``_quote_cited_appendix_items`` directly; this one goes through
    a real seeded ``units`` row and the real database query that supplies it.
    """
    unit_id = f"{_PCCC}#phu-luc-7#muc-15"
    body = "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg."
    _seed_documents(migrated_db, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    _seed_units(
        migrated_db,
        Unit(
            id=f"{_PCCC}#phu-luc-7",
            document_id=_PCCC,
            unit_type=UnitType.APPENDIX,
            appendix_no=7,
            ordinal=0,
        ),
        Unit(
            id=unit_id,
            document_id=_PCCC,
            unit_type=UnitType.APPENDIX_ITEM,
            appendix_no=7,
            item_no=15,
            parent_id=f"{_PCCC}#phu-luc-7",
            body=body,
            ordinal=1,
        ),
    )
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(
        json.dumps(
            {
                "refusal": False,
                "answer": "Có, cơ sở này phải mua bảo hiểm cháy, nổ bắt buộc.",
                "citations": [{"unit_id": unit_id}],
            }
        )
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.ANSWER
    assert response.answer == (
        "Có, cơ sở này phải mua bảo hiểm cháy, nổ bắt buộc.\n"
        f'Phụ lục VII, mục 15 (Nghị định 105/2025/NĐ-CP): "{body}"'
    )
    # The quote never changes citations -- only the answer text gains the quote line.
    (citation,) = response.citations
    assert citation.unit_id == unit_id


# --- GroundingTrace.pre_quote_answer: the model's own text, before the quote (issue #167) ---
#
# eval.compare's deciding-fact check (eval.metrics.answer_check) must match a gold pattern
# against the model's OWN generated text, never against text #146's quote appended from the
# corpus -- otherwise a model that never states a legal threshold would still "pass" a check
# for that exact threshold, for the wrong reason (the quote said it, not the model). This is
# the trap the issue's own acceptance criteria name explicitly.


def test_pre_quote_answer_does_not_contain_text_only_the_appended_quote_adds(migrated_db: str):
    """A pattern found only inside #146's appended quote must not be satisfiable via the trace.

    Mục 15's own stored body contains "trên 500 kg". The model's answer here never says
    anything close to that -- it is a bare affirmative, exactly the issue's own flagship
    failure mode. The *final* ``AskResponse.answer`` does end up containing "trên 500 kg"
    (because of the quote), which is precisely why checking the final string would be wrong;
    ``GroundingTrace.pre_quote_answer`` -- captured before that quote is appended -- must not.
    """
    unit_id = f"{_PCCC}#phu-luc-7#muc-15"
    body = "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg."
    _seed_documents(migrated_db, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    _seed_units(
        migrated_db,
        Unit(
            id=f"{_PCCC}#phu-luc-7",
            document_id=_PCCC,
            unit_type=UnitType.APPENDIX,
            appendix_no=7,
            ordinal=0,
        ),
        Unit(
            id=unit_id,
            document_id=_PCCC,
            unit_type=UnitType.APPENDIX_ITEM,
            appendix_no=7,
            item_no=15,
            parent_id=f"{_PCCC}#phu-luc-7",
            body=body,
            ordinal=1,
        ),
    )
    model_answer = "Cửa hàng kinh doanh gas có phải mua bảo hiểm cháy, nổ bắt buộc."
    retrieval = RetrievalResult(question="x", passages=(_passage(unit_id),))
    llm = _StubLLM(
        json.dumps({"refusal": False, "answer": model_answer, "citations": [{"unit_id": unit_id}]})
    )
    trace = generator.GroundingTrace()

    response = _generate("x", retrieval, llm, migrated_db, trace)

    assert response.outcome is AskOutcome.ANSWER
    # The trap: the final answer DOES contain the threshold -- but only because the quote
    # below added it, not because the model said it.
    assert "500 kg" not in model_answer
    assert "trên 500 kg" in response.answer
    # The fix: pre_quote_answer is the model's own text, unaffected by the quote.
    assert trace.pre_quote_answer == model_answer
    assert "500 kg" not in trace.pre_quote_answer

    pattern = (re.compile(r"500\s*kg", re.IGNORECASE),)
    # Checking the model's own text (correct): fails, as it should.
    assert answer_check(response.outcome, trace.pre_quote_answer, pattern, ()) is False
    # Checking the final, quote-appended string instead (the trap) would wrongly pass --
    # proving why eval.compare must never do that.
    assert answer_check(response.outcome, response.answer, pattern, ()) is True


# --- offering a Phụ lục VI row's own deductible rule as its own passage (issue #165) ---------
#
# Fix round on #166: an earlier version of this fix appended the rule's text into the row's
# own *chunk* at embedding time, which diluted every row's embedding and regressed two
# unrelated, previously-correct answers. This version instead runs at generation time
# (``generator._with_deductible_rule_passages``), after retrieval, and changes nothing about a
# chunk or an embedding -- these tests exercise that function directly, against a real seeded
# ``units`` row for each deductible type's own rule (found by its own text, never a formula on
# the id scheme). One end-to-end test at the bottom pins the design the fix round settled on:
# a 10% claim is grounded only by actually citing the điểm that states it, never through
# mục 17 alone.

_DIEM_A_BODY = (
    "Đối với cơ sở có nguy hiểm về cháy, nổ thuộc loại M quy định tại khoản 1 Mục I "
    "Phụ lục này: mức khấu trừ bảo hiểm tối đa là 1% số tiền bảo hiểm."
)
_DIEM_B_BODY = (
    "Đối với cơ sở có nguy hiểm về cháy, nổ thuộc loại N quy định tại khoản 1 Mục I "
    "Phụ lục này: mức khấu trừ bảo hiểm tối đa là 10% số tiền bảo hiểm."
)


def _diem_a_id(document_id: str = _PCCC) -> str:
    return appendix_section_point_id(document_id, 6, "ii", 1, "a")


def _diem_b_id(document_id: str = _PCCC) -> str:
    return appendix_section_point_id(document_id, 6, "ii", 1, "b")


def _deductible_rule_seed_units(document_id: str = _PCCC) -> tuple[Unit, ...]:
    """Build Phụ lục VI's own Mục II khoản 1 điểm a/b units and their full parent chain.

    Shaped exactly like a real parse: each điểm's own body is what names its
    deductible type ("thuộc loại M"/"thuộc loại N") -- ``get_deductible_rule_units``
    finds them by that text, never by their id. In reading order (appendix, section,
    clause, then both điểm), since ``_seed_units``/``UnitRepository.upsert_many``
    insert a parent before any row that references it as ``parent_id``. A caller that
    wants only one of the two điểm (a genuine parse gap) slices this tuple, e.g.
    ``[:-1]`` to drop điểm b while keeping its full parent chain.
    """
    appendix_unit_id = f"{document_id}#phu-luc-6"
    section_id = appendix_section_id(document_id, 6, "ii")
    clause_id = appendix_section_clause_id(document_id, 6, "ii", 1)
    return (
        Unit(
            id=appendix_unit_id,
            document_id=document_id,
            unit_type=UnitType.APPENDIX,
            appendix_no=6,
            ordinal=0,
        ),
        Unit(
            id=section_id,
            document_id=document_id,
            unit_type=UnitType.APPENDIX_SECTION,
            appendix_no=6,
            appendix_section="ii",
            parent_id=appendix_unit_id,
            ordinal=1,
        ),
        Unit(
            id=clause_id,
            document_id=document_id,
            unit_type=UnitType.APPENDIX_SECTION_CLAUSE,
            appendix_no=6,
            appendix_section="ii",
            khoan_no=1,
            parent_id=section_id,
            ordinal=2,
        ),
        Unit(
            id=_diem_a_id(document_id),
            document_id=document_id,
            unit_type=UnitType.APPENDIX_SECTION_POINT,
            appendix_no=6,
            appendix_section="ii",
            khoan_no=1,
            diem="a",
            parent_id=clause_id,
            body=_DIEM_A_BODY,
            ordinal=3,
        ),
        Unit(
            id=_diem_b_id(document_id),
            document_id=document_id,
            unit_type=UnitType.APPENDIX_SECTION_POINT,
            appendix_no=6,
            appendix_section="ii",
            khoan_no=1,
            diem="b",
            parent_id=clause_id,
            body=_DIEM_B_BODY,
            ordinal=4,
        ),
    )


def _premium_row_passage(
    unit_id: str, cell_text: str, *, document_id: str = _PCCC, focus_units: tuple[str, ...] = ()
) -> Passage:
    """Build a passage shaped like a real Phụ lục VI row/sub-row chunk, cell text included."""
    return Passage(
        unit_id=unit_id,
        document_id=document_id,
        text=f"Phụ lục 6 — Mục 17\n{cell_text}",
        similarity=0.8,
        hop=0,
        score=0.8,
        why=(Why.SIMILAR,),
        focus_units=focus_units,
    )


def _expand(
    database_url: str, passages: tuple[Passage, ...]
) -> tuple[tuple[Passage, ...], dict[str, str]]:
    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            result["value"] = await generator._with_deductible_rule_passages(
                passages, DocumentRepository(session)
            )

    result: dict[str, Any] = {}
    _run(database_url, body)
    return result["value"]


def test_an_m_row_offers_diem_a_as_a_focus_unit(migrated_db: str):
    row_id = f"{_PCCC}#phu-luc-6#muc-18"
    _seed_documents(migrated_db, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    _seed_units(migrated_db, *_deductible_rule_seed_units())
    row = _premium_row_passage(
        row_id, "Trụ sở cơ quan nhà nước có nhà cao từ 3 tầng trở lên M 0,05"
    )

    passages, labels = _expand(migrated_db, (row,))

    # Exactly the one row passage comes back -- issue #165's fix round measured that
    # also adding a brand-new standalone Passage for the rule unit (on top of this
    # focus-units link) intermittently regressed an unrelated question, so this
    # version never adds one; see the module docstring.
    assert len(passages) == 1
    (row_out,) = passages
    assert row_out.unit_id == row_id
    assert row_out.text == row.text  # row's own text is never touched
    assert row_out.focus_units == (_diem_a_id(),)
    assert labels[_diem_a_id()] == "Phụ lục VI, Mục II Khoản 1 -- ký hiệu M, mức khấu trừ tối đa 1%"


def test_an_n_row_offers_diem_b_as_a_focus_unit(migrated_db: str):
    row_id = f"{_PCCC}#phu-luc-6#muc-17"
    _seed_documents(migrated_db, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    _seed_units(migrated_db, *_deductible_rule_seed_units())
    row = _premium_row_passage(
        row_id, "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg N 0,3"
    )

    passages, labels = _expand(migrated_db, (row,))

    assert len(passages) == 1
    (row_out,) = passages
    assert row_out.text == row.text
    assert row_out.focus_units == (_diem_b_id(),)
    assert labels[_diem_b_id()] == (
        "Phụ lục VI, Mục II Khoản 1 -- ký hiệu N, mức khấu trừ tối đa 10%"
    )


def test_a_subrow_offers_its_own_rule_too(migrated_db: str):
    subrow_id = f"{_PCCC}#phu-luc-6#muc-19#muc-1"
    _seed_documents(migrated_db, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    _seed_units(migrated_db, *_deductible_rule_seed_units())
    subrow = _premium_row_passage(subrow_id, "Cơ sở sản xuất công nghiệp hạng A, B N 0,2")

    passages, labels = _expand(migrated_db, (subrow,))

    assert len(passages) == 1
    assert passages[0].text == subrow.text
    assert passages[0].focus_units == (_diem_b_id(),)
    assert _diem_b_id() in labels


def test_a_row_with_no_mn_cell_offers_nothing(migrated_db: str):
    row_id = f"{_PCCC}#phu-luc-6#muc-20"
    row = _premium_row_passage(row_id, "Chợ; trung tâm thương mại.")

    passages, labels = _expand(migrated_db, (row,))

    assert passages == (row,)
    assert passages[0] is row  # unchanged, not merely equal -- nothing was copied
    assert labels == {}


def test_a_missing_rule_unit_is_logged_and_adds_nothing(
    migrated_db: str, caplog: pytest.LogCaptureFixture
):
    # Only điểm a (loại M's own rule) is seeded -- điểm b is a genuine gap. The N row's
    # own passage is returned unchanged, and the loss is logged rather than silently
    # dropped (issue #165's own fallback rule).
    row_id = f"{_PCCC}#phu-luc-6#muc-17"
    _seed_documents(migrated_db, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    _seed_units(migrated_db, *_deductible_rule_seed_units()[:-1])  # điểm b is missing
    row = _premium_row_passage(
        row_id, "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg N 0,3"
    )

    with caplog.at_level("WARNING"):
        passages, labels = _expand(migrated_db, (row,))

    assert passages == (row,)
    assert passages[0] is row
    assert labels == {}
    assert any("no rule unit for it was found" in record.message for record in caplog.records)


def test_an_already_retrieved_rule_unit_is_not_duplicated(migrated_db: str):
    row_id = f"{_PCCC}#phu-luc-6#muc-17"
    _seed_documents(migrated_db, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    _seed_units(migrated_db, *_deductible_rule_seed_units())
    # điểm b is already a focus unit of this very row -- as the graph itself might
    # legitimately produce -- so it must not be added a second time.
    row = _premium_row_passage(
        row_id,
        "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg N 0,3",
        focus_units=(_diem_b_id(),),
    )

    passages, labels = _expand(migrated_db, (row,))

    assert len(passages) == 1
    (row_out,) = passages
    assert row_out.text == row.text  # never touched
    assert row_out.focus_units == (_diem_b_id(),)  # not duplicated
    assert _diem_b_id() in labels


def test_a_phu_luc_vii_passage_is_untouched(migrated_db: str):
    # Carries an "N 0,3"-shaped cell on purpose: this must not be mistaken for a Phụ
    # lục VI premium-table row just because the text looks similar.
    row_id = f"{_PCCC}#phu-luc-7#muc-15"
    _seed_documents(migrated_db, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    _seed_units(migrated_db, *_deductible_rule_seed_units())
    row = Passage(
        unit_id=row_id,
        document_id=_PCCC,
        text="Phụ lục 7 — Mục 15\nCơ sở kinh doanh khí đốt trên 500 kg N 0,3.",
        similarity=0.8,
        hop=0,
        score=0.8,
        why=(Why.SIMILAR,),
    )

    passages, labels = _expand(migrated_db, (row,))

    assert passages == (row,)
    assert passages[0] is row
    assert labels == {}


def test_pinning_the_design_a_10_percent_answer_citing_only_muc_17_is_refused(migrated_db: str):
    """A 10% claim is grounded only by citing the điểm that actually states it.

    Issue #165's own design rule, settled in the fix round on #166: the grounding
    verifier never accepts "mục 17 supports 10%" -- mục 17's own stored text is just
    the row's bare cell ("... N 0,3"), with no "10%" in it at all, even though this
    fix now also offers điểm b (whose own text states the 10% rule) as its own
    citable passage. A model that cites only mục 17 for a 10% claim must still be
    refused; the fix only helps a model that actually cites điểm b.
    """
    muc_17_id = f"{_PCCC}#phu-luc-6#muc-17"
    muc_17_body = "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg N 0,3"
    _seed_documents(migrated_db, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    _seed_units(
        migrated_db,
        *_deductible_rule_seed_units(),
        Unit(
            id=muc_17_id,
            document_id=_PCCC,
            unit_type=UnitType.APPENDIX_ITEM,
            appendix_no=6,
            item_no=17,
            parent_id=f"{_PCCC}#phu-luc-6",
            body=muc_17_body,
            ordinal=5,
        ),
    )
    retrieval = RetrievalResult(
        question="x", passages=(_premium_row_passage(muc_17_id, muc_17_body),)
    )

    clause_marker = "Nội dung của đúng một điều khoản được trích dẫn:\n"

    class _SemanticVerifierLLM(LLMProvider):
        """A double whose verifier calls check one clause's own text for a literal substring.

        Mimics a real grounding judgment without a real model: "supported" only when the
        *cited clause's own text* -- isolated from ``generator._verifier_prompt``'s own
        ``clause_marker``, never the question or the forecast answer that also appear in the
        same prompt -- contains the claimed number.
        """

        def __init__(self, reply: str, required_substring: str) -> None:
            self.reply = reply
            self._required_substring = required_substring
            self.calls: list[dict[str, Any]] = []

        async def generate(
            self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
        ) -> str:
            self.calls.append({"system": system, "user": user, "json_schema": json_schema})
            if _is_verifier_schema(json_schema):
                clause_text = user.split(clause_marker, 1)[1]
                return json.dumps({"supported": self._required_substring in clause_text})
            return self.reply

    llm = _SemanticVerifierLLM(
        json.dumps(
            {
                "refusal": False,
                "answer": "Mức khấu trừ bảo hiểm tối đa là 10% số tiền bảo hiểm.",
                "citations": [{"unit_id": muc_17_id}],
            }
        ),
        required_substring="10%",
    )

    response = _generate("x", retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "unsupported_citations"


# --- issue #173: yes/no instruction, added only for a yes/no question ---------------------

_YES_NO_GOLD_IDS = frozenset(
    {
        "compliance-display-racks",
        "compliance-blending-lab",
        "compliance-blending-registration",
        "compliance-trader-pccc-condition",
        "compliance-old-pccc-law-repealed",
        "compliance-pccc-fire-management-scope",
        "multihop-decree-87-amended",
        "multihop-gas-store-fire-insurance",
        "multihop-gas-store-300kg-insurance",
        "multihop-gas-store-500kg-insurance",
        "multihop-gas-store-800kg-insurance",
        "multihop-gas-shop-insurance-threshold",
    }
)
"""The issue's own 11 yes/no gold rows (issue #173), plus ``multihop-gas-store-
500kg-insurance`` -- the review comment's own new boundary row (the fix round on PR
#175), itself a yes/no question ("... bắt buộc không?"), bringing the total to 12."""

_PREMIUM_QUESTION = (
    "Tỷ lệ phí bảo hiểm cháy, nổ bắt buộc tối thiểu của cơ sở kinh doanh khí đốt là bao nhiêu?"
)
"""``compliance-gas-business-premium-rate``'s own question -- not a yes/no question, and
the row the issue names explicitly as the one whose prompt must stay unchanged (its own
2026-10-30 probe comment: an earlier, unscoped attempt at this same fix regressed exactly
this question)."""


def test_is_yes_no_question_matches_exactly_the_issues_12_rows_and_no_other_gold_row():
    """``_is_yes_no_question`` against the whole real gold set, not a hand-picked sample.

    Proves both directions at once: every one of the issue's 11 yes/no rows plus the fix
    round's own new boundary row (``multihop-gas-store-500kg-insurance``, 12 total) matches,
    and every other answerable, refusal and emergency row in ``data/gold.yaml`` does not --
    including rows that could plausibly trip a cruder check, e.g.
    ``multihop-decree-87-amended`` ("... có bị sửa đổi bởi văn bản nào không?", a yes/no
    question that also contains "nào") and ``refusal-tt-41-2011-status`` ("... còn áp dụng
    không, quy định cụ thể ra sao?", which carries a mid-sentence "không," with more text
    after it, and is correctly not a yes/no question even under the widened pattern).
    """
    gold = load_gold_set()
    matched = {row.id for row in gold if generator._is_yes_no_question(row.question)}

    assert matched == _YES_NO_GOLD_IDS
    assert len(gold) == 46  # sanity: this really checked the whole file, not a subset


_PRE_198_YES_NO_QUESTION_PATTERN = re.compile(
    r"\b(?:có|còn)\b.*\bkhông\s*\?\s*$",
    re.IGNORECASE | re.DOTALL,
)
"""The exact pattern ``_YES_NO_QUESTION_PATTERN`` had before issue #198 widened it --
kept here as a literal, independent test constant (not read from ``generator.py``), so
``test_is_yes_no_question_gold_detection_is_unchanged_from_before_issue_198`` below keeps
comparing against the real pre-#198 behaviour even if the module's own pattern is edited
again later."""


def test_is_yes_no_question_gold_detection_is_unchanged_from_before_issue_198():
    """Issue #198's own gate: no gold row's yes/no detection may move.

    For every row of ``data/gold.yaml``, today's (widened) ``_is_yes_no_question`` must
    agree with the pre-#198 pattern above -- not just the 12 ids
    ``_YES_NO_GOLD_IDS`` hardcodes, but every row, computed from the old pattern's own
    text, the same check the issue's own measurement script ran. Since every gold
    yes/no question ends in a literal "không?", both patterns match it the same way, so
    every gold prompt stays byte-identical and no gold metric can move from this change
    alone.
    """
    gold = load_gold_set()
    for row in gold:
        question = row.question.strip()
        old = _PRE_198_YES_NO_QUESTION_PATTERN.search(question) is not None
        new = generator._is_yes_no_question(row.question)
        assert new == old, row.id


@pytest.mark.parametrize(
    "question",
    [
        "Mở cửa hàng gas cần diện tích tối thiểu bao nhiêu mét vuông?",  # bao nhiêu
        "Nộp hồ sơ xin giấy chứng nhận đủ điều kiện kinh doanh gas bằng cách nào?",  # nào
        "Thiết bị điện trong cửa hàng gas phải là loại gì, đặt cách bình gas bao xa?",  # gì
        "Chai gas trưng bày ngoài cửa hàng và chai đã quá hạn kiểm định thì xử lý thế nào?",
        "Cửa hàng bán gas lúc mấy giờ?",  # a made-up edge case, no "có"/"còn" at all
    ],
)
def test_is_yes_no_question_rejects_open_questions(question: str) -> None:
    assert generator._is_yes_no_question(question) is False


@pytest.mark.parametrize(
    "question",
    [
        "Có được không?",  # minimal case
        "Cửa hàng này có phải đóng cửa không?",
        "Luật này còn hiệu lực không?",
    ],
)
def test_is_yes_no_question_accepts_minimal_and_hand_built_cases(question: str) -> None:
    assert generator._is_yes_no_question(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không",  # no "?"
        "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không ạ?",  # "ạ?"
        "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không ạ",  # "ạ"
        "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không vậy?",  # "vậy?"
        "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không vậy",  # "vậy"
        "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không nhỉ?",  # "nhỉ?"
        "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không nhỉ",  # "nhỉ"
        "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc ko?",  # "ko?"
        "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc ko",  # "ko"
    ],
)
def test_is_yes_no_question_accepts_the_reapplied_widened_forms(
    question: str,
) -> None:
    """The once-withdrawn widened forms are matched again (issue #198).

    PR #175's first fix round widened detection beyond a literal closing "không?" (no
    "?", "ạ", "vậy", "nhỉ", "ko") and a second review comment withdrew it: on the real
    ``/ask`` server, dev DB, `qwen3:8b`, this exact 300 kg question typed as "... không"
    (no "?") or "... không ạ?" got the yes/no instruction and answered a confident,
    wrong "Có, ... phải mua ..." 4/4 times each -- worse than `main`'s own harmless
    echo, which states no verdict at all. Issue #198 re-applies the same widening now
    that ``app.services.verdict`` (issue #195) refuses any answer whose own verdict
    contradicts its cited clause, so a wrong "Có" reached by one of these forms is
    caught in code rather than shipped. All nine forms below must now match, so
    ``_build_user_prompt`` adds the yes/no instruction for every one of them (see the
    end-to-end test below).
    """
    assert generator._is_yes_no_question(question) is True


def test_is_yes_no_question_still_rejects_the_refusal_rows_mid_sentence_khong() -> None:
    """The widened pattern must not start matching ``refusal-tt-41-2011-status``.

    Its question carries a "không," mid-sentence, with more text (a second clause) after
    it -- not the question's own closing particle -- so it must stay unmatched even
    though the widened pattern no longer requires a literal "?" right after "không".
    """
    question = "Thông tư 41/2011/TT-BCT còn áp dụng không, quy định cụ thể ra sao?"
    assert generator._is_yes_no_question(question) is False


def test_yes_no_instruction_no_longer_hardcodes_equal_or_above_as_yes() -> None:
    """The fix round's blocking bug: the old wording ignored "trên"'s own strict meaning.

    ``_YES_NO_INSTRUCTION_VI`` used to say a number BẰNG hoặc CAO HƠN ("equal to or
    higher than") the threshold always concludes "Có" -- wrong for a "trên N" ("strictly
    more than N") clause, where a number equal to N does not meet it. The reviewer's real
    `/ask` run on that wording answered the 500 kg boundary row "Có, ... phải mua ..."
    3/3, a confident, wrong "yes" (PR #175's review comment on issue #173).
    """
    assert "BẰNG hoặc CAO HƠN" not in generator._YES_NO_INSTRUCTION_VI


def test_yes_no_instruction_is_added_right_before_the_question_line() -> None:
    unit_id = f"{_PCCC}#phu-luc-7#muc-15"
    sources = {unit_id: _source(unit_id)}
    passage = _passage(unit_id)

    prompt = generator._build_user_prompt(
        "Cửa hàng kinh doanh gas có phải mua bảo hiểm cháy, nổ bắt buộc không?",
        (passage,),
        (),
        sources,
    )

    assert generator._YES_NO_INSTRUCTION_VI in prompt
    instruction_pos = prompt.index(generator._YES_NO_INSTRUCTION_VI)
    question_pos = prompt.index("CÂU HỎI:")
    assert instruction_pos < question_pos


def test_non_yes_no_question_prompt_is_byte_identical_to_the_pre_173_prompt() -> None:
    """The premium question's prompt is byte-identical to the pre-#173 baseline.

    The baseline literal below was captured from this exact fixture using the real,
    unmodified pre-#173 ``_build_user_prompt`` (run once, by hand, before this issue's
    own change landed) -- never recomputed from the current code, so a future change
    that starts adding the yes/no block unconditionally cannot silently pass this test
    by changing both sides of the comparison at once.
    """
    unit_id = f"{_PCCC}#phu-luc-6#muc-17"
    passage = Passage(
        unit_id=unit_id,
        document_id=_PCCC,
        dieu_no=None,
        text="Co so kinh doanh khi dot ... N 0,3",
        similarity=0.8,
        hop=0,
        score=0.8,
        why=(Why.SIMILAR,),
        focus_units=(),
        possibly_superseded=False,
    )
    sources = {unit_id: _source(unit_id, document_number="105/2025/NĐ-CP")}

    prompt = generator._build_user_prompt(_PREMIUM_QUESTION, (passage,), (), sources)

    pre_173_prompt = (
        "NGỮ CẢNH PHÁP LÝ:\n\n"
        f"[unit_id: {unit_id}] Phụ lục VI, mục 17 — 105/2025/NĐ-CP\n"
        "Co so kinh doanh khi dot ... N 0,3\n\n"
        f"CÂU HỎI: {_PREMIUM_QUESTION}"
    )
    assert generator._is_yes_no_question(_PREMIUM_QUESTION) is False
    assert prompt == pre_173_prompt
    assert generator._YES_NO_INSTRUCTION_VI not in prompt


@pytest.mark.parametrize(
    "question",
    [
        "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không",
        "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không ạ?",
    ],
)
def test_the_reapplied_widened_forms_now_get_the_yes_no_instruction(question: str) -> None:
    """The two forms the withdrawn fix round once regressed now get the instruction.

    "... không" (no "?") and "... không ạ?" are the exact two forms PR #175's review
    comment measured answering a confident, wrong "Có, ... phải mua ..." 4/4 times each
    when the (then withdrawn) fix-round-1 pattern gave them the yes/no instruction.
    Issue #198 re-applies that same widening, now that ``app.services.verdict`` (issue
    #195) refuses any answer whose own verdict contradicts its cited clause -- so these
    forms get the instruction block again, same place as every other yes/no question
    (``test_yes_no_instruction_is_added_right_before_the_question_line`` above), and a
    wrong "Có" from one of them is caught by the verdict check instead of reaching the
    user unchecked.
    """
    unit_id = f"{_PCCC}#phu-luc-7#muc-15"
    passage = _passage(unit_id)
    sources = {unit_id: _source(unit_id)}

    prompt = generator._build_user_prompt(question, (passage,), (), sources)

    assert generator._is_yes_no_question(question) is True
    assert generator._YES_NO_INSTRUCTION_VI in prompt
    instruction_pos = prompt.index(generator._YES_NO_INSTRUCTION_VI)
    question_pos = prompt.index("CÂU HỎI:")
    assert instruction_pos < question_pos


def test_premium_question_end_to_end_prompt_carries_no_yes_no_instruction(
    migrated_db: str,
) -> None:
    """End-to-end proof for the issue's own blocking row.

    Goes through ``generate_answer``, not just ``_build_user_prompt``: the real prompt
    the premium question reaches the model with carries neither the instruction block
    nor its own system-prompt variant.
    """
    _seed_documents(migrated_db, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    unit_id = f"{_PCCC}#phu-luc-6#muc-17"
    retrieval = RetrievalResult(
        question=_PREMIUM_QUESTION,
        passages=(_passage(unit_id),),
    )
    llm = _StubLLM(
        json.dumps(
            {
                "refusal": False,
                "answer": "Tỷ lệ phí bảo hiểm cháy, nổ bắt buộc tối thiểu là 0,3%.",
                "citations": [{"unit_id": unit_id}],
            }
        )
    )

    _generate(_PREMIUM_QUESTION, retrieval, llm, migrated_db)

    call = llm.calls[0]  # the main generation call; a second call is the grounding verifier
    assert call["system"] == generator._SYSTEM_PROMPT_VI
    assert generator._YES_NO_INSTRUCTION_VI not in call["user"]


def test_yes_no_question_end_to_end_prompt_carries_the_instruction(migrated_db: str) -> None:
    """Mirror of the test above, for a real yes/no gold question (the flagship row)."""
    _seed_documents(migrated_db, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    unit_id = f"{_PCCC}#phu-luc-7#muc-15"
    question = "Cửa hàng kinh doanh gas có phải mua bảo hiểm cháy, nổ bắt buộc không?"
    retrieval = RetrievalResult(question=question, passages=(_passage(unit_id),))
    llm = _StubLLM(
        json.dumps(
            {
                "refusal": False,
                "answer": "Chỉ khi tồn chứa trên 500 kg.",
                "citations": [{"unit_id": unit_id}],
            }
        )
    )

    _generate(question, retrieval, llm, migrated_db)

    call = llm.calls[0]  # the main generation call; a second call is the grounding verifier
    assert call["system"] == generator._SYSTEM_PROMPT_VI  # the system prompt itself never moves
    assert generator._YES_NO_INSTRUCTION_VI in call["user"]


# --- the deterministic verdict check: refuse a wrong Có/Không (issue #195) -------------------
#
# PR #194's review measured qwen3:8b opening "Có, ... phải mua ..." for a 300 kg store while
# citing the exact clause stating "... tồn chứa trên 500 kg" -- the same flagship insurance
# clause ``test_a_real_appendix_item_citation_is_quoted_end_to_end`` above seeds, reused here
# with a question that actually states a quantity so ``app.services.verdict`` has something to
# check. That existing test's own question ("x") states no quantity, so it is itself live proof
# this check leaves a no-quantity row answering exactly as before.

_INSURANCE_UNIT_ID = f"{_PCCC}#phu-luc-7#muc-15"
_INSURANCE_CLAUSE_BODY = "Cơ sở kinh doanh khí đốt có tổng lượng khí đốt tồn chứa trên 500 kg."
_INSURANCE_QUESTION = "Cửa hàng gas chứa 300 kg khí có phải mua bảo hiểm cháy, nổ bắt buộc không?"


def _seed_insurance_clause(database_url: str) -> None:
    _seed_documents(database_url, _document(_PCCC, "105/2025/NĐ-CP", date(2025, 1, 1)))
    _seed_units(
        database_url,
        Unit(
            id=f"{_PCCC}#phu-luc-7",
            document_id=_PCCC,
            unit_type=UnitType.APPENDIX,
            appendix_no=7,
            ordinal=0,
        ),
        Unit(
            id=_INSURANCE_UNIT_ID,
            document_id=_PCCC,
            unit_type=UnitType.APPENDIX_ITEM,
            appendix_no=7,
            item_no=15,
            parent_id=f"{_PCCC}#phu-luc-7",
            body=_INSURANCE_CLAUSE_BODY,
            ordinal=1,
        ),
    )


def test_a_wrong_co_verdict_refuses_with_its_own_reason(
    migrated_db: str, caplog: pytest.LogCaptureFixture
):
    """The exact PR #194 shape: "Có, ... phải mua ..." for 300 kg against a "trên 500 kg" clause."""
    _seed_insurance_clause(migrated_db)
    retrieval = RetrievalResult(
        question=_INSURANCE_QUESTION, passages=(_passage(_INSURANCE_UNIT_ID),)
    )
    answer = (
        "Có, cửa hàng gas chứa 300 kg khí phải mua bảo hiểm cháy, nổ bắt buộc vì lượng khí đốt "
        "tồn chứa vượt quá ngưỡng 500 kg."
    )
    llm = _StubLLM(
        json.dumps(
            {"refusal": False, "answer": answer, "citations": [{"unit_id": _INSURANCE_UNIT_ID}]}
        )
    )

    with caplog.at_level("WARNING", logger="app.services.generator"):
        response = _generate(_INSURANCE_QUESTION, retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "verdict_contradicts_clause"
    assert response.answer is None
    assert response.citations == ()
    (message,) = [
        record.getMessage()
        for record in caplog.records
        if "verdict contradicts clause" in record.getMessage()
    ]
    assert "quantity 300 kg" in message
    assert "threshold > 500 kg" in message
    assert "expected Không" in message
    assert "answer opened Có" in message
    # Trap 4: the log line carries only parsed facts, never the row's own question or answer.
    assert _INSURANCE_QUESTION not in message
    assert answer not in message
    assert "bảo hiểm" not in message


def test_no_retry_on_verdict_contradicts_clause(migrated_db: str):
    """Issue #202, Scope item 3's own enumerated case.

    The exact fixture above: the citation passes the grounding verifier (``_StubLLM``
    answers every verifier call ``{"supported": true}``), but the verdict check itself
    refuses with a reason other than ``"unsupported_citations"`` -- so
    ``_retry_without_rejected_passages`` never even considers retrying, regardless of
    whether the verifier kept or dropped anything.
    """
    _seed_insurance_clause(migrated_db)
    retrieval = RetrievalResult(
        question=_INSURANCE_QUESTION, passages=(_passage(_INSURANCE_UNIT_ID),)
    )
    answer = (
        "Có, cửa hàng gas chứa 300 kg khí phải mua bảo hiểm cháy, nổ bắt buộc vì lượng khí đốt "
        "tồn chứa vượt quá ngưỡng 500 kg."
    )
    llm = _StubLLM(
        json.dumps(
            {"refusal": False, "answer": answer, "citations": [{"unit_id": _INSURANCE_UNIT_ID}]}
        )
    )
    trace = generator.GroundingTrace()

    response = _generate(_INSURANCE_QUESTION, retrieval, llm, migrated_db, trace)

    assert response.outcome is AskOutcome.REFUSAL
    assert response.refusal_reason == "verdict_contradicts_clause"
    assert trace.retried is False
    # The citation passed the verifier -- trace.decisions got exactly one (unit_id, True)
    # entry -- which is exactly why this must never be mistaken for "layer 2 kept nothing".
    assert trace.decisions == [(_INSURANCE_UNIT_ID, True)]
    assert len(_generation_calls(llm)) == 1  # the verdict check refused, never a retry


def test_a_correct_khong_verdict_still_answers(migrated_db: str):
    """The mirror, correct answer (`main`'s own reply on this row) is unaffected."""
    _seed_insurance_clause(migrated_db)
    retrieval = RetrievalResult(
        question=_INSURANCE_QUESTION, passages=(_passage(_INSURANCE_UNIT_ID),)
    )
    answer = (
        "Không, cửa hàng gas chứa 300 kg khí không phải mua bảo hiểm cháy, nổ bắt buộc vì 300 kg "
        "nhỏ hơn ngưỡng 500 kg."
    )
    llm = _StubLLM(
        json.dumps(
            {"refusal": False, "answer": answer, "citations": [{"unit_id": _INSURANCE_UNIT_ID}]}
        )
    )

    response = _generate(_INSURANCE_QUESTION, retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.ANSWER
    assert response.answer == (
        f'{answer}\nPhụ lục VII, mục 15 (Nghị định 105/2025/NĐ-CP): "{_INSURANCE_CLAUSE_BODY}"'
    )
    (citation,) = response.citations
    assert citation.unit_id == _INSURANCE_UNIT_ID


def test_a_polarity_question_with_a_quantity_still_answers(migrated_db: str):
    """A question about the opposite of the clause's own condition is never guessed (n/a)."""
    _seed_insurance_clause(migrated_db)
    question = "Cửa hàng gas chứa 300 kg khí có được miễn mua bảo hiểm cháy, nổ bắt buộc không?"
    retrieval = RetrievalResult(question=question, passages=(_passage(_INSURANCE_UNIT_ID),))
    answer = "Có, cửa hàng gas chứa 300 kg khí được miễn mua bảo hiểm cháy, nổ bắt buộc."
    llm = _StubLLM(
        json.dumps(
            {"refusal": False, "answer": answer, "citations": [{"unit_id": _INSURANCE_UNIT_ID}]}
        )
    )

    response = _generate(question, retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.ANSWER


def test_a_non_yes_no_answer_path_is_unaffected(migrated_db: str):
    """An answer that does not open with "Có"/"Không" at all is never touched by the check."""
    _seed_insurance_clause(migrated_db)
    retrieval = RetrievalResult(
        question=_INSURANCE_QUESTION, passages=(_passage(_INSURANCE_UNIT_ID),)
    )
    answer = "Chỉ khi tổng lượng khí đốt tồn chứa trên 500 kg."
    llm = _StubLLM(
        json.dumps(
            {"refusal": False, "answer": answer, "citations": [{"unit_id": _INSURANCE_UNIT_ID}]}
        )
    )

    response = _generate(_INSURANCE_QUESTION, retrieval, llm, migrated_db)

    assert response.outcome is AskOutcome.ANSWER
