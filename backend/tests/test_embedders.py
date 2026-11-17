"""Tests for the embedder interface and its offline mock implementation."""

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Sequence
from typing import TypeVar

import httpx2
import pytest

from app.core.config import EmbedderProvider, Settings
from app.embedders.base import Embedder, EmbedderUnreachableError
from app.embedders.bge_m3 import DIMENSION, BgeM3Embedder, EmbeddingError
from app.embedders.factory import create_embedder
from app.embedders.mock import DEFAULT_DIMENSION, MockEmbedder

T = TypeVar("T")


def _embed(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    """Drive the coroutine ``Embedder.embed`` to completion (issue #89), as one caller would."""
    return asyncio.run(embedder.embed(texts))


# --- interface ---------------------------------------------------------------------


def test_mock_embedder_implements_the_embedder_interface():
    assert isinstance(MockEmbedder(), Embedder)


def test_embedder_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Embedder()


# --- dimension -----------------------------------------------------------------------


def test_default_dimension_matches_the_documented_default():
    assert MockEmbedder().dimension == DEFAULT_DIMENSION


def test_dimension_is_parametrized_not_hard_coded():
    embedder = MockEmbedder(dimension=16)
    assert embedder.dimension == 16
    [vector] = _embed(embedder, ["hello"])
    assert len(vector) == 16


def test_every_produced_vector_has_the_configured_length():
    embedder = MockEmbedder(dimension=32)
    vectors = _embed(embedder, ["a", "bb", "ccc", "dddd"])
    assert all(len(vector) == 32 for vector in vectors)


@pytest.mark.parametrize("dimension", [0, -1, -100])
def test_a_non_positive_dimension_is_rejected(dimension: int):
    with pytest.raises(ValueError, match="dimension must be"):
        MockEmbedder(dimension=dimension)


# --- determinism ---------------------------------------------------------------------


def test_the_same_text_always_yields_the_same_vector():
    embedder = MockEmbedder(dimension=64)
    text = "Cửa hàng bán lẻ LPG chai phải đáp ứng các điều kiện sau"
    assert _embed(embedder, [text]) == _embed(embedder, [text])


def test_embedding_the_same_batch_twice_is_identical():
    embedder = MockEmbedder(dimension=64)
    texts = ["Điều 38", "Điều 39", "Điều 38"]
    assert _embed(embedder, texts) == _embed(embedder, texts)


def test_a_repeated_text_within_one_batch_yields_the_same_vector():
    embedder = MockEmbedder(dimension=64)
    vectors = _embed(embedder, ["Điều 38", "Điều 39", "Điều 38"])
    assert vectors[0] == vectors[2]


def test_different_texts_yield_different_vectors():
    embedder = MockEmbedder(dimension=64)
    vectors = _embed(embedder, ["Điều 38", "Điều 39"])
    assert vectors[0] != vectors[1]


def test_a_new_embedder_instance_reproduces_the_same_vectors():
    text = "Khoảng cách an toàn tối thiểu"
    assert _embed(MockEmbedder(dimension=48), [text]) == _embed(MockEmbedder(dimension=48), [text])


# --- batching --------------------------------------------------------------------------


def test_batch_embedding_matches_embedding_each_text_alone():
    embedder = MockEmbedder(dimension=8)
    texts = ["one", "two", "three"]
    batch = _embed(embedder, texts)
    singles = [_embed(embedder, [text])[0] for text in texts]
    assert batch == singles


def test_an_empty_batch_yields_an_empty_list():
    assert _embed(MockEmbedder(), []) == []


# --- value range -------------------------------------------------------------------------


def test_vector_components_are_bounded_to_unit_range():
    embedder = MockEmbedder(dimension=256)
    [vector] = _embed(embedder, ["Vietnamese LPG retail-store safety requirements"])
    assert all(-1.0 <= component <= 1.0 for component in vector)


# --- BgeM3Embedder (build plan A6 issue #47) ----------------------------------------
#
# Every test here runs against a stub httpx2.MockTransport, never a real Ollama
# daemon: CI has no Ollama, and CI fails on any skipped test (issue #26), so a
# skipif(no ollama) guard would turn CI red. A stub proves request shape,
# batching, and error handling with no model server anywhere -- the one thing
# it cannot prove (that Ollama's real response matches) belongs in the PR's
# acceptance run, not in this suite.


def _embed_response(*, count: int, dimension: int = DIMENSION) -> httpx2.Response:
    return httpx2.Response(
        200, json={"model": "bge-m3", "embeddings": [[0.1] * dimension for _ in range(count)]}
    )


def test_bge_m3_implements_the_embedder_interface():
    assert isinstance(BgeM3Embedder(base_url="http://x"), Embedder)


def test_bge_m3_dimension_is_1024():
    assert BgeM3Embedder(base_url="http://x").dimension == DIMENSION == 1024


def test_bge_m3_sends_exactly_one_request_for_a_whole_batch():
    calls = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return _embed_response(count=3)

    embedder = BgeM3Embedder(
        base_url="http://ollama.local", transport=httpx2.MockTransport(handler)
    )
    _embed(embedder, ["one", "two", "three"])
    assert len(calls) == 1


def test_bge_m3_request_targets_the_embed_endpoint_with_the_right_model_and_input():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return _embed_response(count=2)

    embedder = BgeM3Embedder(
        base_url="http://ollama.local", transport=httpx2.MockTransport(handler)
    )
    _embed(embedder, ["Điều 38", "Điều 39"])
    assert captured["url"] == "http://ollama.local/api/embed"
    assert captured["body"] == {"model": "bge-m3", "input": ["Điều 38", "Điều 39"]}


def test_bge_m3_returns_vectors_in_the_same_order_as_the_response():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "bge-m3",
                "embeddings": [[1.0] * DIMENSION, [2.0] * DIMENSION, [3.0] * DIMENSION],
            },
        )

    embedder = BgeM3Embedder(base_url="http://x", transport=httpx2.MockTransport(handler))
    vectors = _embed(embedder, ["a", "b", "c"])
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0]


def test_bge_m3_empty_batch_never_makes_a_request():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("must not be called for an empty batch")

    embedder = BgeM3Embedder(base_url="http://x", transport=httpx2.MockTransport(handler))
    assert _embed(embedder, []) == []


def test_bge_m3_raises_on_a_vector_count_mismatch():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _embed_response(count=1)  # two texts requested, one vector returned

    embedder = BgeM3Embedder(base_url="http://x", transport=httpx2.MockTransport(handler))
    with pytest.raises(EmbeddingError, match=r"1 embedding.*2 text"):
        _embed(embedder, ["a", "b"])


def test_bge_m3_raises_on_a_dimension_mismatch():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _embed_response(count=1, dimension=768)  # wrong width, e.g. nomic-embed-text

    embedder = BgeM3Embedder(base_url="http://x", transport=httpx2.MockTransport(handler))
    with pytest.raises(EmbeddingError, match=r"768 dimension.*expected 1024"):
        _embed(embedder, ["a"])


def test_bge_m3_raises_on_a_non_2xx_status():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"error": "model not found"})

    embedder = BgeM3Embedder(base_url="http://x", transport=httpx2.MockTransport(handler))
    with pytest.raises(httpx2.HTTPStatusError):
        _embed(embedder, ["a"])


def test_bge_m3_a_connection_failure_raises_embedder_unreachable_error():
    # Issue #183: a connection failure is an operational fact about the deployment (Ollama
    # restarting under another project's LaunchAgent, measured in the issue), not a bug in
    # this codebase, so it is converted rather than left to propagate raw.
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    embedder = BgeM3Embedder(base_url="http://x", transport=httpx2.MockTransport(handler))
    with pytest.raises(EmbedderUnreachableError):
        _embed(embedder, ["a"])


def test_bge_m3_a_timeout_also_raises_embedder_unreachable_error():
    # Unlike the LLM provider, the embedder folds a timeout into the same exception as a
    # connection failure -- there is no analogue here to issue #112's "the model itself is
    # slow" measurement that a separate reason code would protect.
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    embedder = BgeM3Embedder(base_url="http://x", transport=httpx2.MockTransport(handler))
    with pytest.raises(EmbedderUnreachableError):
        _embed(embedder, ["a"])


def test_bge_m3_a_dropped_connection_mid_reply_also_raises_embedder_unreachable_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.RemoteProtocolError("Server disconnected without sending a response")

    embedder = BgeM3Embedder(base_url="http://x", transport=httpx2.MockTransport(handler))
    with pytest.raises(EmbedderUnreachableError):
        _embed(embedder, ["a"])


# --- create_embedder (build plan A6 issue #47) --------------------------------------


def test_create_embedder_defaults_to_mock():
    embedder = create_embedder(Settings(_env_file=None))
    assert isinstance(embedder, MockEmbedder)


def test_create_embedder_returns_bge_m3_when_configured():
    settings = Settings(
        _env_file=None, embedder=EmbedderProvider.BGE_M3, ollama_base_url="http://ollama:11434"
    )
    embedder = create_embedder(settings)
    assert isinstance(embedder, BgeM3Embedder)


# --- the event loop is not blocked (issue #89) ------------------------------------------
#
# ``embed`` is awaited from a request handler, where a blocking call would freeze every
# other request on the loop for a whole Ollama round trip. These tests do not read the
# code and conclude it is fine: they run the embedder against a deliberately slow fake
# server alongside a coroutine that ticks on the same loop, and count how often the loop
# got to run. A blocked loop cannot tick, so the count is the measurement. The control
# test at the end puts the old blocking call back inside a coroutine and shows the same
# harness catching it -- a harness that cannot fail proves nothing.

_SLOW_SECONDS = 0.2
_TICK_SECONDS = 0.005


async def _ticks_during(work: Awaitable[T]) -> tuple[T, int]:
    """Await ``work`` while a task on the same loop counts how often the loop ran it."""
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(_TICK_SECONDS)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        result = await work
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return result, ticks


def _slow_async_transport(in_flight: list[int] | None = None) -> httpx2.MockTransport:
    """A fake Ollama that takes ``_SLOW_SECONDS`` to answer, awaiting rather than sleeping."""
    current = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal current
        current += 1
        if in_flight is not None:
            in_flight.append(current)
        try:
            await asyncio.sleep(_SLOW_SECONDS)
        finally:
            current -= 1
        count = len(json.loads(request.content)["input"])
        return _embed_response(count=count)

    return httpx2.MockTransport(handler)


def test_bge_m3_embed_does_not_block_the_event_loop():
    embedder = BgeM3Embedder(base_url="http://slow", transport=_slow_async_transport())

    vectors, ticks = asyncio.run(_ticks_during(embedder.embed(["một câu hỏi"])))

    assert len(vectors) == 1
    # Ideally ~40 ticks in 0.2 s. A blocked loop manages 0 or 1; the margin is for a busy CI box.
    assert ticks >= 10, f"the event loop ran the ticker only {ticks} time(s) during embed()"


def test_bge_m3_serves_concurrent_requests_at_the_same_time():
    # Two questions in flight at once is exactly what a blocked loop makes impossible: the
    # second request could not even reach the server until the first had been answered.
    in_flight: list[int] = []
    embedder = BgeM3Embedder(base_url="http://slow", transport=_slow_async_transport(in_flight))

    async def two_at_once() -> list[list[list[float]]]:
        return list(await asyncio.gather(embedder.embed(["a"]), embedder.embed(["b"])))

    results = asyncio.run(two_at_once())

    assert [len(result) for result in results] == [1, 1]
    assert max(in_flight) == 2


def test_the_harness_catches_an_embedder_that_blocks_the_loop():
    # The control. This is the shape ``BgeM3Embedder.embed`` had before issue #89 -- a blocking
    # ``httpx2.Client`` called from inside a coroutine -- against a fake server that blocks the
    # way a real socket read does. If this did not fail, the test above would prove nothing.
    def blocking_handler(request: httpx2.Request) -> httpx2.Response:
        time.sleep(_SLOW_SECONDS)
        return _embed_response(count=1)

    class BlockingEmbedder(Embedder):
        dimension = DIMENSION

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            with httpx2.Client(transport=httpx2.MockTransport(blocking_handler)) as client:
                response = client.post("http://slow/api/embed", json={"input": list(texts)})
            embeddings: list[list[float]] = response.json()["embeddings"]
            return embeddings

    _, ticks = asyncio.run(_ticks_during(BlockingEmbedder().embed(["một câu hỏi"])))

    assert ticks <= 2, f"the harness saw {ticks} ticks from a blocked loop; it cannot tell"
