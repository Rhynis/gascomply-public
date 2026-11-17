"""Tests for ``POST /ask`` (build plan A6 Phase 3.7, issue #90).

The emergency-path tests need no database at all (``client``, backed by
``Settings(database_url=None)``) -- that is the point being proven. Everything
past the emergency check needs a live Postgres (``db_client``, ``migrated_db``).
"""

import asyncio
import contextlib
import json
import socket
from collections.abc import Awaitable, Callable, Iterator, Sequence
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import AppEnv, RerankerProvider, Settings
from app.core.db import create_engine, create_session_factory, session_scope
from app.embedders.base import Embedder, EmbedderUnreachableError
from app.embedders.mock import MockEmbedder
from app.llm.base import ContextOverflowError, LLMProvider, LLMUnavailableError, LLMUnreachableError
from app.main import create_app
from app.repositories.chunks import CHUNKS_EMBEDDING_DIMENSION, ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.units import UnitRepository
from app.rerankers.mock import MockReranker
from app.schemas.chunk import Chunk
from app.schemas.document import DocStatus, DocType, Document
from app.schemas.unit import Unit, UnitType

DOC_ID = "gas-decree-87-2018"
UNIT_ID = f"{DOC_ID}#dieu-53"
QUESTION = "diện tích tối thiểu cửa hàng LPG là bao nhiêu"


def _is_verifier_schema(json_schema: dict[str, Any] | None) -> bool:
    """Whether a call's ``json_schema`` is the grounding verifier's own (not the generator's).

    Every double below keys on this instead of a marker in the prompt, matching how the real
    ``MockLLM`` identifies a verifier call since issue #118's own follow-up review: the product
    prompt itself carries no test-only hook.
    """
    return json_schema is not None and "supported" in json_schema.get("properties", {})


class _StubLLM(LLMProvider):
    """A double that returns a fixed reply for generation.

    Answers the grounding verifier's own call (#118) with a well-formed
    ``{"supported": true}`` reply rather than replaying ``reply``'s generation schema -- the
    verifier fails closed on an unparseable reply, so replaying the wrong shape would drop
    the citation and turn these tests into refusals for an unrelated reason.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        self.calls += 1
        if _is_verifier_schema(json_schema):
            return json.dumps({"supported": True})
        return self.reply


class _FailingLLM(LLMProvider):
    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        pytest.fail("LLMProvider.generate must not be called on the emergency path")


class _RaisingLLM(LLMProvider):
    """Always raises a fixed exception.

    Used to prove a provider failure never reaches the client as an unhandled HTTP 500
    (issue #129).
    """

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        raise self._error


class _RaisingEmbedder(Embedder):
    """Always raises a fixed exception.

    Used to prove an embedder failure never reaches the client as an unhandled HTTP 500
    (issue #183), mirroring ``_RaisingLLM`` above for the LLM provider's own calls.
    """

    dimension = 8

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise self._error


# --- the emergency path: no database, no LLM, ever -------------------------------------


def test_an_emergency_question_returns_200_with_the_hotline_and_no_database_configured(
    client: TestClient,
):
    response = client.post("/ask", json={"question": "nhà tôi đang cháy, cứu với"})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "emergency"
    assert "114" in body["safety_response"]
    assert body["answer"] is None
    assert body["citations"] == []


def test_the_emergency_response_is_byte_identical_across_calls(client: TestClient):
    first = client.post("/ask", json={"question": "ngui thay mui gas nong nac"}).json()
    second = client.post("/ask", json={"question": "gas ro ri, phai lam sao bay gio"}).json()

    assert first["safety_response"] == second["safety_response"]


def test_an_ordinary_question_gets_503_with_no_database_configured(client: TestClient):
    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 503


def test_the_llm_is_never_called_on_the_emergency_path(db_client: TestClient):
    db_client.app.state.llm_provider = _FailingLLM()  # type: ignore[attr-defined]

    response = db_client.post("/ask", json={"question": "bình gas vừa nổ, cứu với"})

    assert response.status_code == 200
    assert response.json()["outcome"] == "emergency"


def test_a_blank_question_is_rejected_before_anything_else(client: TestClient):
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


# --- rate limiting -----------------------------------------------------------------------


def test_the_31st_request_in_a_minute_is_rejected(client: TestClient):
    for _ in range(30):
        response = client.post("/ask", json={"question": "cháy nhà"})
        assert response.status_code == 200

    response = client.post("/ask", json={"question": "cháy nhà"})
    assert response.status_code == 429
    assert "Retry-After" in response.headers


# --- a grounded answer, through the real endpoint -----------------------------------------


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


def _seed_matching_chunk(database_url: str, question: str) -> None:
    """Seed one document/unit/chunk whose embedding is a perfect match for ``question``.

    ``MockEmbedder`` is deterministic in the text alone, so embedding the question itself
    and storing that as the chunk's vector guarantees this chunk ranks first -- the same
    trick ``tests/test_retrieve_vector.py`` uses, since the mock has no real semantics.
    """

    async def body(engine: AsyncEngine) -> None:
        [vector] = await MockEmbedder(dimension=CHUNKS_EMBEDDING_DIMENSION).embed([question])
        async with session_scope(create_session_factory(engine)) as session:
            await DocumentRepository(session).upsert(
                Document(
                    id=DOC_ID,
                    number="87/2018/NĐ-CP",
                    title_vi="Nghị định về kinh doanh khí",
                    doc_type=DocType.DECREE,
                    status=DocStatus.IN_FORCE,
                    effective=date(2018, 8, 1),
                    path=f"corpus/{DOC_ID}.html",
                )
            )
            await UnitRepository(session).upsert_many(
                [
                    Unit(
                        id=UNIT_ID,
                        document_id=DOC_ID,
                        unit_type=UnitType.ARTICLE,
                        dieu_no=53,
                        body="Cửa hàng bán lẻ LPG chai phải có diện tích tối thiểu là 12 m2.",
                        ordinal=53,
                    )
                ]
            )
            await ChunkRepository(session).upsert_many(
                [
                    Chunk(
                        unit_id=UNIT_ID,
                        document_id=DOC_ID,
                        dieu_no=53,
                        effective=date(2018, 8, 1),
                        status=DocStatus.IN_FORCE,
                        body="Cửa hàng bán lẻ LPG chai phải có diện tích tối thiểu là 12 m2.",
                    )
                ],
                [vector],
            )

    _run(database_url, body)


def test_a_grounded_answer_returns_verified_structured_citations(
    db_client: TestClient, migrated_db: str
):
    _seed_matching_chunk(migrated_db, QUESTION)
    db_client.app.state.llm_provider = _StubLLM(  # type: ignore[attr-defined]
        json.dumps(
            {
                "refusal": False,
                "answer": "Diện tích tối thiểu là 12 m2.",
                "citations": [{"unit_id": UNIT_ID}],
            }
        )
    )

    response = db_client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "answer"
    assert body["answer"] == "Diện tích tối thiểu là 12 m2."
    assert body["disclaimer"]
    (citation,) = body["citations"]
    assert citation["unit_id"] == UNIT_ID
    assert citation["document_id"] == DOC_ID
    assert citation["document_number"] == "87/2018/NĐ-CP"
    assert citation["dieu_no"] == 53
    assert citation["effective"] == "2018-08-01"


def test_a_fabricated_citation_is_dropped_through_the_real_endpoint(
    db_client: TestClient, migrated_db: str
):
    _seed_matching_chunk(migrated_db, QUESTION)
    fake_unit = f"{DOC_ID}#dieu-61#khoan-3"
    db_client.app.state.llm_provider = _StubLLM(  # type: ignore[attr-defined]
        json.dumps({"refusal": False, "answer": "ok", "citations": [{"unit_id": fake_unit}]})
    )

    response = db_client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "refusal"
    assert body["refusal_reason"] == "no_valid_citations"
    assert body["citations"] == []


def test_a_question_with_no_support_in_the_corpus_refuses(db_client: TestClient):
    # Nothing seeded at all: vector search finds no chunk, so no LLM call is even made.
    response = db_client.post("/ask", json={"question": "một câu hỏi bất kỳ"})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "refusal"
    assert body["refusal_reason"] == "no_context"


# --- issue #129: a provider timeout or overflow is a refusal, never an HTTP 500 -----------


def test_a_provider_timeout_refuses_as_llm_unavailable_not_a_500(
    db_client: TestClient, migrated_db: str
):
    """The exact bug the issue reports.

    A 6,972-token prompt filled the truncated window and the generation call ran into the
    client's timeout, which used to surface as HTTP 500.
    """
    _seed_matching_chunk(migrated_db, QUESTION)
    db_client.app.state.llm_provider = _RaisingLLM(  # type: ignore[attr-defined]
        LLMUnavailableError("Ollama request timed out after 120.0s")
    )

    response = db_client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "refusal"
    assert body["refusal_reason"] == "llm_unavailable"


def test_a_context_overflow_refuses_with_its_own_reason_not_a_500(
    db_client: TestClient, migrated_db: str
):
    _seed_matching_chunk(migrated_db, QUESTION)
    db_client.app.state.llm_provider = _RaisingLLM(  # type: ignore[attr-defined]
        ContextOverflowError("prompt_eval_count=4095 is within budget")
    )

    response = db_client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "refusal"
    assert body["refusal_reason"] == "context_overflow"


# --- issue #183: an unreachable provider or embedder refuses, never an HTTP 500 ----------


def test_an_unreachable_llm_refuses_as_llm_unreachable_not_a_500(
    db_client: TestClient, migrated_db: str
):
    """The repro this issue reports for the LLM provider's own call.

    A connection failure (Ollama restarted mid-request, measured in the issue against a
    closed port) used to propagate raw and surface as an unhandled HTTP 500.
    """
    _seed_matching_chunk(migrated_db, QUESTION)
    db_client.app.state.llm_provider = _RaisingLLM(  # type: ignore[attr-defined]
        LLMUnreachableError("Ollama unreachable at http://127.0.0.1:9: connection refused")
    )

    response = db_client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "refusal"
    assert body["refusal_reason"] == "llm_unreachable"


def test_an_unreachable_embedder_refuses_as_embedder_unreachable_not_a_500(
    db_client: TestClient, migrated_db: str
):
    """The repro this issue reports for the embedder's own call.

    A connection failure while embedding the question used to propagate raw -- out of
    ``Retriever.retrieve``, through ``app.api.ask`` -- and surface as an unhandled HTTP 500.
    """
    db_client.app.state.embedder = _RaisingEmbedder(  # type: ignore[attr-defined]
        EmbedderUnreachableError(
            "Ollama embedder unreachable at http://127.0.0.1:9: connection refused"
        )
    )

    response = db_client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "refusal"
    assert body["refusal_reason"] == "embedder_unreachable"


# --- reranking (issue #114) --------------------------------------------------------------------

OTHER_UNIT_ID = f"{DOC_ID}#dieu-16"
OTHER_TEXT = "Chai LPG phải được kiểm định định kỳ theo quy định."


def _seed_a_second_chunk(database_url: str) -> None:
    """Seed a second, off-topic article, so retrieval has two passages for a reranker to order."""

    async def body(engine: AsyncEngine) -> None:
        [vector] = await MockEmbedder(dimension=CHUNKS_EMBEDDING_DIMENSION).embed([OTHER_TEXT])
        async with session_scope(create_session_factory(engine)) as session:
            await UnitRepository(session).upsert_many(
                [
                    Unit(
                        id=OTHER_UNIT_ID,
                        document_id=DOC_ID,
                        unit_type=UnitType.ARTICLE,
                        dieu_no=16,
                        body=OTHER_TEXT,
                        ordinal=16,
                    )
                ]
            )
            await ChunkRepository(session).upsert_many(
                [
                    Chunk(
                        unit_id=OTHER_UNIT_ID,
                        document_id=DOC_ID,
                        dieu_no=16,
                        effective=date(2018, 8, 1),
                        status=DocStatus.IN_FORCE,
                        body=OTHER_TEXT,
                    )
                ],
                [vector],
            )

    _run(database_url, body)


class _RecordingLLM(LLMProvider):
    """Cites ``UNIT_ID`` and remembers the prompt it was shown.

    Answers the grounding verifier's call (#118) with a well-formed ``{"supported": true}``
    reply rather than replaying the generation answer's own schema: since the verifier now
    fails closed on an unparseable reply, replaying the wrong shape would drop the citation and
    turn every test below into a refusal unrelated to what it actually exercises.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        if _is_verifier_schema(json_schema):
            return json.dumps({"supported": True})
        self.prompts.append(user)
        return json.dumps(
            {
                "refusal": False,
                "answer": "Diện tích tối thiểu là 12 m2.",
                "citations": [{"unit_id": UNIT_ID}],
            }
        )


@contextlib.contextmanager
def _client_with(database_url: str, **overrides: Any) -> Iterator[TestClient]:
    settings = Settings(
        app_env=AppEnv.CI,
        log_level="WARNING",
        database_url=database_url,
        _env_file=None,
        **overrides,
    )
    with TestClient(create_app(settings)) as client:
        yield client


def test_the_app_has_no_reranker_by_default_and_carries_the_configured_top_n(
    client: TestClient, settings: Settings
):
    assert client.app.state.reranker is None  # type: ignore[attr-defined]
    assert (
        client.app.state.retrieval_config.rerank_top_n == settings.rerank_top_n  # type: ignore[attr-defined]
    )


def test_the_app_builds_the_configured_reranker_and_top_n(migrated_db: str):
    with _client_with(migrated_db, reranker=RerankerProvider.MOCK, rerank_top_n=3) as client:
        assert isinstance(client.app.state.reranker, MockReranker)  # type: ignore[attr-defined]
        assert client.app.state.retrieval_config.rerank_top_n == 3  # type: ignore[attr-defined]


def test_ask_shows_the_model_only_the_top_n_passages_the_reranker_kept(migrated_db: str):
    _seed_matching_chunk(migrated_db, QUESTION)
    _seed_a_second_chunk(migrated_db)
    llm = _RecordingLLM()

    with _client_with(migrated_db, reranker=RerankerProvider.MOCK, rerank_top_n=1) as client:
        client.app.state.llm_provider = llm  # type: ignore[attr-defined]
        response = client.post("/ask", json={"question": QUESTION})

    # The mock ranks by shared words: the store-area article shares most of the question's words,
    # the cylinder-inspection one almost none, so with top-N 1 only the former reaches the model.
    assert response.status_code == 200
    assert response.json()["outcome"] == "answer"
    (prompt,) = llm.prompts
    assert f"[unit_id: {UNIT_ID}]" in prompt
    assert f"[unit_id: {OTHER_UNIT_ID}]" not in prompt


def test_without_a_reranker_ask_shows_the_model_every_passage(migrated_db: str):
    # The control for the test above: same corpus, same question, reranking off.
    _seed_matching_chunk(migrated_db, QUESTION)
    _seed_a_second_chunk(migrated_db)
    llm = _RecordingLLM()

    with _client_with(migrated_db) as client:
        client.app.state.llm_provider = llm  # type: ignore[attr-defined]
        client.post("/ask", json={"question": QUESTION})

    (prompt,) = llm.prompts
    assert f"[unit_id: {UNIT_ID}]" in prompt
    assert f"[unit_id: {OTHER_UNIT_ID}]" in prompt


def test_ask_still_answers_when_the_rerank_sidecar_is_down(migrated_db: str):
    # A real llama-server reranker configured, with nothing listening at its address: the
    # connection is refused, and /ask must degrade to its behaviour before reranking existed --
    # every passage offered, an answer returned -- not fail.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    _seed_matching_chunk(migrated_db, QUESTION)
    _seed_a_second_chunk(migrated_db)
    llm = _RecordingLLM()

    with _client_with(
        migrated_db,
        reranker=RerankerProvider.LLAMA_SERVER,
        reranker_base_url=f"http://127.0.0.1:{dead_port}",
        rerank_top_n=1,
    ) as client:
        client.app.state.llm_provider = llm  # type: ignore[attr-defined]
        response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json()["outcome"] == "answer"
    (prompt,) = llm.prompts
    assert f"[unit_id: {UNIT_ID}]" in prompt
    assert f"[unit_id: {OTHER_UNIT_ID}]" in prompt  # not cut to top-N: the ranking never happened
