"""Tests for the cross-encoder stage of ``Retriever.retrieve`` (issue #114).

Needs a live Postgres with pgvector (``TEST_DATABASE_URL``) and nothing else: the reranker is a
scripted stand-in that returns preset scores, or the offline ``MockReranker`` -- never a real
``llama-server`` (CI has none, and CI fails on any skipped test). What is under test is the
retriever's own behaviour around a reranker: reordering, the top-N cut, which clauses stay
citable afterwards, and that a reranker that fails leaves retrieval exactly as it was before the
stage existed. Whether a real cross-encoder ranks *well* is measured on the real corpus in the
PR body and ADR, not asserted here.

Every chunk has a hand-built vector so the graph-proximity order the reranker starts from is known
exactly: ``a1 > a2 > a3 > a4`` by cosine similarity to the question, no edges, all at hop 0.

No ``pytest-asyncio`` (C12): each test drives its own ``asyncio.run()``.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.embedders.base import Embedder
from app.repositories.chunks import CHUNKS_EMBEDDING_DIMENSION, ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.units import UnitRepository
from app.rerankers.base import Reranker, RerankError
from app.rerankers.mock import MockReranker
from app.schemas.chunk import Chunk
from app.schemas.document import DocStatus, DocType, Document
from app.schemas.retrieval import RetrievalConfig, RetrievalResult
from app.schemas.unit import Unit, UnitType
from app.services.retriever import Retriever

DOC = "doc-a"
A1, A2, A3, A4 = (f"{DOC}#dieu-{n}" for n in (1, 2, 3, 4))
QUESTION = "diện tích tối thiểu cửa hàng LPG"


def _vector(*components: float) -> list[float]:
    vector = [0.0] * CHUNKS_EMBEDDING_DIMENSION
    for index, value in enumerate(components):
        vector[index] = value
    return vector


class _FixedEmbedder(Embedder):
    dimension = CHUNKS_EMBEDDING_DIMENSION

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        assert list(texts) == [QUESTION], texts
        return [_vector(1.0)]


class _ScriptedReranker(Reranker):
    """Scores each passage by a preset number, keyed by the unit id its text names."""

    def __init__(self, scores: dict[str, float], *, error: Exception | None = None) -> None:
        self._scores = scores
        self._error = error
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if self._error is not None:
            raise self._error
        return [self._scores[document.removeprefix("text of ")] for document in documents]


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


def _seed(database_url: str) -> None:
    """Four articles, one chunk each, and the clauses a1 and a4 contain."""
    document = Document(
        id=DOC,
        number="87/2018/NĐ-CP",
        title_vi=DOC,
        doc_type=DocType.DECREE,
        issued=date(2018, 6, 15),
        effective=date(2018, 8, 1),
        status=DocStatus.IN_FORCE,
        path=f"corpus/{DOC}.html",
    )

    def article(number: int) -> Unit:
        return Unit(
            id=f"{DOC}#dieu-{number}",
            document_id=DOC,
            unit_type=UnitType.ARTICLE,
            dieu_no=number,
            body=f"Điều {number}.",
            ordinal=number,
        )

    def clause(dieu: int, khoan: int) -> Unit:
        return Unit(
            id=f"{DOC}#dieu-{dieu}#khoan-{khoan}",
            document_id=DOC,
            unit_type=UnitType.CLAUSE,
            dieu_no=dieu,
            khoan_no=khoan,
            parent_id=f"{DOC}#dieu-{dieu}",
            body=f"Khoản {khoan}.",
            ordinal=khoan,
        )

    units = [article(n) for n in (1, 2, 3, 4)] + [clause(1, 1), clause(1, 2), clause(4, 1)]
    vectors = {
        A1: _vector(1.0),
        A2: _vector(1.0, 1.0),
        A3: _vector(1.0, 2.0),
        A4: _vector(0.0, 1.0),
    }
    chunks = [
        Chunk(
            unit_id=unit_id,
            document_id=DOC,
            effective=date(2018, 8, 1),
            status=DocStatus.IN_FORCE,
            body=f"text of {unit_id}",
        )
        for unit_id in vectors
    ]

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            await DocumentRepository(session).upsert(document)
            await UnitRepository(session).upsert_many(units)
            await ChunkRepository(session).upsert_many(chunks, list(vectors.values()))

    _run(database_url, body)


@dataclass
class _Outcome:
    result: RetrievalResult
    rerank_ok: int
    rerank_fallbacks: int


def _retrieve(database_url: str, reranker: Reranker | None = None, **config: Any) -> _Outcome:
    holder: list[_Outcome] = []

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            retriever = Retriever(
                session, _FixedEmbedder(), RetrievalConfig(**config), reranker=reranker
            )
            result = await retriever.retrieve(QUESTION)
            holder.append(_Outcome(result, retriever.rerank_ok, retriever.rerank_fallbacks))

    _run(database_url, body)
    return holder[0]


def _ids(result: RetrievalResult) -> list[str]:
    return [passage.unit_id for passage in result.passages]


# --- the baseline everything below is compared to -----------------------------------------------


def test_without_a_reranker_the_passages_are_in_graph_proximity_order(migrated_db: str):
    _seed(migrated_db)

    outcome = _retrieve(migrated_db)

    assert _ids(outcome.result) == [A1, A2, A3, A4]
    assert all(passage.rerank_score is None for passage in outcome.result.passages)
    assert (outcome.rerank_ok, outcome.rerank_fallbacks) == (0, 0)


def test_a_top_n_without_a_reranker_cuts_nothing(migrated_db: str):
    # rerank_top_n only means something to a cross-encoder; with none configured the retriever
    # must behave exactly as before issue #114 no matter what the config says.
    _seed(migrated_db)

    outcome = _retrieve(migrated_db, rerank_top_n=1)

    assert _ids(outcome.result) == [A1, A2, A3, A4]


# --- reordering and the cut --------------------------------------------------------------------


def test_passages_come_back_best_cross_encoder_score_first_with_the_score_attached(
    migrated_db: str,
):
    _seed(migrated_db)
    reranker = _ScriptedReranker({A1: -2.0, A2: 7.5, A3: 0.5, A4: -9.0})

    outcome = _retrieve(migrated_db, reranker)

    assert _ids(outcome.result) == [A2, A3, A1, A4]
    assert [p.rerank_score for p in outcome.result.passages] == [7.5, 0.5, -2.0, -9.0]
    assert (outcome.rerank_ok, outcome.rerank_fallbacks) == (1, 0)


def test_the_reranker_is_shown_the_question_and_the_passage_texts_in_graph_order(
    migrated_db: str,
):
    _seed(migrated_db)
    reranker = _ScriptedReranker({A1: 1.0, A2: 2.0, A3: 3.0, A4: 4.0})

    _retrieve(migrated_db, reranker)

    assert reranker.calls == [(QUESTION, [f"text of {unit}" for unit in (A1, A2, A3, A4)])]


def test_only_the_best_n_passages_survive(migrated_db: str):
    _seed(migrated_db)
    reranker = _ScriptedReranker({A1: -2.0, A2: 7.5, A3: 0.5, A4: -9.0})

    outcome = _retrieve(migrated_db, reranker, rerank_top_n=2)

    assert _ids(outcome.result) == [A2, A3]


def test_no_top_n_reorders_every_passage_without_cutting(migrated_db: str):
    _seed(migrated_db)
    reranker = _ScriptedReranker({A1: -2.0, A2: 7.5, A3: 0.5, A4: -9.0})

    outcome = _retrieve(migrated_db, reranker)

    assert len(outcome.result.passages) == 4


def test_a_top_n_larger_than_the_pool_keeps_the_whole_pool(migrated_db: str):
    _seed(migrated_db)
    reranker = _ScriptedReranker({A1: 1.0, A2: 2.0, A3: 3.0, A4: 4.0})

    outcome = _retrieve(migrated_db, reranker, rerank_top_n=50)

    assert _ids(outcome.result) == [A4, A3, A2, A1]


def test_the_pool_is_the_passage_limit_survivors_not_every_candidate(migrated_db: str):
    # passage_limit=3 already dropped a4 before the cross-encoder could score it, however good
    # it would have scored: the cross-encoder reorders what the graph ranking chose, no more.
    _seed(migrated_db)
    reranker = _ScriptedReranker({A1: 1.0, A2: 2.0, A3: 3.0, A4: 99.0})

    outcome = _retrieve(migrated_db, reranker, passage_limit=3)

    assert _ids(outcome.result) == [A3, A2, A1]
    assert [len(docs) for _, docs in reranker.calls] == [3]


def test_equal_scores_keep_the_graph_order(migrated_db: str):
    _seed(migrated_db)
    reranker = _ScriptedReranker({A1: 0.0, A2: 5.0, A3: 0.0, A4: 0.0})

    outcome = _retrieve(migrated_db, reranker)

    assert _ids(outcome.result) == [A2, A1, A3, A4]


def test_a_single_passage_is_never_sent_to_the_reranker(migrated_db: str):
    # One passage has nothing to be ranked against, so it costs no scoring call.
    _seed(migrated_db)
    reranker = _ScriptedReranker({})

    outcome = _retrieve(migrated_db, reranker, k=1)

    assert _ids(outcome.result) == [A1]
    assert reranker.calls == []
    assert (outcome.rerank_ok, outcome.rerank_fallbacks) == (0, 0)


def test_the_offline_mock_reranker_runs_through_the_retriever(migrated_db: str):
    # The path CI exercises with nothing installed: every passage's text is "text of <id>", so
    # a question made of a passage's own words scores that passage highest.
    _seed(migrated_db)

    outcome = _retrieve(migrated_db, MockReranker(), rerank_top_n=2)

    assert len(outcome.result.passages) == 2
    assert all(isinstance(p.rerank_score, float) for p in outcome.result.passages)
    assert (outcome.rerank_ok, outcome.rerank_fallbacks) == (1, 0)


# --- what stays citable ------------------------------------------------------------------------


def test_a_dropped_passage_takes_its_clauses_with_it(migrated_db: str):
    # The point of the stage: every clause of a passage that is cut stops being offered, which is
    # most of the lines a model would otherwise choose between. a1 holds khoan-1 and khoan-2, a4
    # holds khoan-1; keeping only a4 must leave a4's clause and none of a1's.
    _seed(migrated_db)
    reranker = _ScriptedReranker({A1: -5.0, A2: -4.0, A3: -3.0, A4: 9.0})

    outcome = _retrieve(migrated_db, reranker, rerank_top_n=1)

    [passage] = outcome.result.passages
    assert passage.unit_id == A4
    assert passage.focus_units == (f"{A4}#khoan-1",)


def test_a_kept_passage_keeps_all_its_clauses(migrated_db: str):
    _seed(migrated_db)
    reranker = _ScriptedReranker({A1: 9.0, A2: 1.0, A3: 1.0, A4: 1.0})

    outcome = _retrieve(migrated_db, reranker, rerank_top_n=1)

    [passage] = outcome.result.passages
    assert passage.focus_units == (f"{A1}#khoan-1", f"{A1}#khoan-2")


# --- a reranker that fails degrades to today's behaviour ------------------------------------


@pytest.mark.parametrize("top_n", [None, 2])
def test_a_failing_reranker_leaves_retrieval_exactly_as_it_was_without_one(
    migrated_db: str, top_n: int | None
):
    _seed(migrated_db)
    baseline = _retrieve(migrated_db, None, rerank_top_n=top_n)

    failing = _retrieve(
        migrated_db,
        _ScriptedReranker({}, error=RerankError("the rerank sidecar is down")),
        rerank_top_n=top_n,
    )

    # Passage-for-passage identical, focus units and all: today's behaviour, not a degraded copy of
    # it -- and, notably, not cut to top-N, since the cut only follows a successful ranking.
    assert failing.result == baseline.result
    assert (failing.rerank_ok, failing.rerank_fallbacks) == (0, 1)


def test_a_failing_reranker_is_logged_not_swallowed_silently(
    migrated_db: str, caplog: pytest.LogCaptureFixture
):
    _seed(migrated_db)

    with caplog.at_level("WARNING", logger="app.services.retriever"):
        _retrieve(migrated_db, _ScriptedReranker({}, error=RerankError("sidecar is down")))

    assert "reranking failed" in caplog.text
    assert "sidecar is down" in caplog.text


def test_an_error_that_is_not_a_rerank_error_still_propagates(migrated_db: str):
    # Only RerankError is the documented "degrade" signal. Anything else is a bug in a reranker,
    # and hiding a bug behind a silent fallback would be its own kind of measurement that
    # looks real and is not.
    _seed(migrated_db)

    with pytest.raises(RuntimeError, match="bug"):
        _retrieve(migrated_db, _ScriptedReranker({}, error=RuntimeError("bug in a reranker")))
