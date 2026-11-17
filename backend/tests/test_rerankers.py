"""Tests for the cross-encoder reranker interface, its offline mock, and the sidecar client (#114).

``LlamaServerReranker`` runs against a stub ``httpx2.MockTransport``, never a real
``llama-server``: CI has none, and CI fails on any skipped test (the rule ``test_embedders.py``
follows for ``BgeM3Embedder``). What a real sidecar returns was checked by hand against a
running ``llama-server --rerank`` while building this, and the request/response shapes below are
that server's own (``POST /v1/rerank`` -> ``{"results": [{"index", "relevance_score"}, ...]}``,
best first).
"""

import asyncio
import contextlib
import http.server
import json
import socket
import socketserver
import threading
from collections.abc import Iterator, Sequence
from typing import Any

import httpx2
import pytest

from app.core.config import RerankerProvider, Settings
from app.rerankers.base import Reranker, RerankError
from app.rerankers.factory import create_reranker
from app.rerankers.llama_server import LlamaServerReranker
from app.rerankers.mock import MockReranker


def _score(reranker: Reranker, query: str, documents: Sequence[str]) -> list[float]:
    """Drive the coroutine ``Reranker.score`` to completion, as one caller would."""
    return asyncio.run(reranker.score(query, documents))


# --- interface ------------------------------------------------------------------------------


def test_the_reranker_interface_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Reranker()  # type: ignore[abstract]


def test_both_implementations_satisfy_the_interface():
    assert isinstance(MockReranker(), Reranker)
    assert isinstance(LlamaServerReranker(base_url="http://x"), Reranker)


# --- MockReranker -----------------------------------------------------------------------------


def test_the_mock_scores_the_share_of_query_words_a_document_contains():
    scores = _score(
        MockReranker(),
        "diện tích cửa hàng",
        ["diện tích cửa hàng là 12 m2", "cửa hàng bán lẻ", "kiểm định chai"],
    )

    assert scores == [1.0, 0.5, 0.0]


def test_the_mock_ranks_the_passage_that_shares_the_most_words_first():
    documents = ["chai LPG kiểm định định kỳ", "diện tích tối thiểu của cửa hàng LPG"]

    scores = _score(MockReranker(), "diện tích tối thiểu cửa hàng", documents)

    assert scores[1] > scores[0]


def test_the_mock_is_deterministic():
    reranker = MockReranker()
    documents = ["a b c", "b c d", "x"]

    assert _score(reranker, "b c", documents) == _score(reranker, "b c", documents)


def test_the_mock_ignores_case_and_repeated_words():
    assert _score(MockReranker(), "Cửa CỬA hàng", ["cửa hàng"]) == [1.0]


def test_the_mock_treats_composed_and_decomposed_vietnamese_as_the_same_word():
    composed = "cửa"  # "ử" as one code point
    decomposed = "cửa"  # "u" + horn + hook above, spelled as combining marks

    assert composed != decomposed
    assert _score(MockReranker(), composed, [decomposed]) == [1.0]


@pytest.mark.parametrize("query", ["", "   ", "?!."])
def test_the_mock_scores_everything_zero_for_a_query_with_no_words(query: str):
    assert _score(MockReranker(), query, ["a", "b"]) == [0.0, 0.0]


def test_the_mock_returns_nothing_for_no_documents():
    assert _score(MockReranker(), "câu hỏi", []) == []


# --- LlamaServerReranker: request shape ----------------------------------------------------------


def _results(*pairs: tuple[int, float]) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "model": "bge-reranker-v2-m3-Q8_0.gguf",
            "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
            "results": [{"index": index, "relevance_score": value} for index, value in pairs],
        },
    )


def _reranker(handler: Any, base_url: str = "http://sidecar.local:8081") -> LlamaServerReranker:
    return LlamaServerReranker(base_url=base_url, transport=httpx2.MockTransport(handler))


def test_one_request_scores_a_whole_batch_against_the_v1_rerank_endpoint():
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return _results((0, 1.5), (1, -2.0), (2, 0.25))

    scores = _score(_reranker(handler), "Điều kiện?", ["một", "hai", "ba"])

    assert scores == [1.5, -2.0, 0.25]
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://sidecar.local:8081/v1/rerank"
    assert json.loads(requests[0].content) == {
        "query": "Điều kiện?",
        "documents": ["một", "hai", "ba"],
    }


def test_the_only_endpoint_ever_requested_is_the_rerank_one_never_an_embeddings_one():
    # The issue's rule: reranker scores must never come from Ollama's /api/embeddings, which
    # exposes a reranker model's embedding layer and not its classification head. Whatever the
    # inputs, this client only ever talks to the sidecar's rerank endpoint.
    paths: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        return _results((0, 1.0), (1, 2.0))

    reranker = _reranker(handler)
    _score(reranker, "q", ["a", "b"])
    _score(reranker, "q2", ["a", "b"])

    assert set(paths) == {"/v1/rerank"}


def test_scores_are_restored_to_input_order_from_a_best_first_response():
    # llama-server answers best first, each tagged with its input index.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _results((2, 9.0), (0, 1.0), (1, -5.0))

    assert _score(_reranker(handler), "q", ["a", "b", "c"]) == [1.0, -5.0, 9.0]


def test_an_integer_score_is_accepted_and_returned_as_a_float():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _results((0, 3), (1, -1))

    scores = _score(_reranker(handler), "q", ["a", "b"])

    assert scores == [3.0, -1.0]
    assert all(isinstance(score, float) for score in scores)


def test_no_documents_means_no_request():
    def handler(request: httpx2.Request) -> httpx2.Response:
        pytest.fail("no request should be made for an empty batch")

    assert _score(_reranker(handler), "q", []) == []


# --- LlamaServerReranker: every failure is a RerankError ------------------------------------


def test_a_connection_failure_is_a_rerank_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(RerankError, match=r"sidecar\.local"):
        _score(_reranker(handler), "q", ["a"])


def test_a_timeout_is_a_rerank_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    with pytest.raises(RerankError):
        _score(_reranker(handler), "q", ["a"])


@pytest.mark.parametrize("status", [400, 500, 503])
def test_a_non_2xx_status_is_a_rerank_error(status: int):
    # e.g. 500 when a passage is longer than the sidecar's batch size, 503 while it loads.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(RerankError):
        _score(_reranker(handler), "q", ["a"])


def test_a_body_that_is_not_json_is_a_rerank_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b"<html>not json</html>")

    with pytest.raises(RerankError, match="not JSON"):
        _score(_reranker(handler), "q", ["a"])


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"results": None},
        {"results": "nope"},
        {"results": [{"index": 0, "relevance_score": 1.0}]},  # one result for two documents
        {"results": [{"index": 0, "relevance_score": 1.0}] * 3},  # three for two
        {"results": [{"index": 0, "relevance_score": 1.0}, {"index": 0, "relevance_score": 2.0}]},
        {"results": [{"index": 0, "relevance_score": 1.0}, {"index": 2, "relevance_score": 2.0}]},
        {"results": [{"index": 0, "relevance_score": 1.0}, {"index": -1, "relevance_score": 2.0}]},
        {"results": [{"index": 0, "relevance_score": 1.0}, {"index": 1}]},
        {"results": [{"index": 0, "relevance_score": 1.0}, {"relevance_score": 2.0}]},
        {"results": [{"index": 0, "relevance_score": 1.0}, {"index": 1, "relevance_score": "9"}]},
        {"results": [{"index": 0, "relevance_score": 1.0}, {"index": 1, "relevance_score": None}]},
        {"results": [{"index": 0, "relevance_score": 1.0}, {"index": 1, "relevance_score": True}]},
        {
            "results": [
                {"index": 0, "relevance_score": 1.0},
                {"index": True, "relevance_score": 2.0},
            ]
        },
        {"results": [{"index": 0, "relevance_score": 1.0}, {"index": "1", "relevance_score": 2.0}]},
        {"results": [{"index": 0, "relevance_score": 1.0}, 7]},
    ],
)
def test_a_body_that_is_not_one_score_per_document_is_a_rerank_error(body: Any):
    # Zipping a partial or re-indexed answer against the documents would attach one passage's
    # score to another -- a wrong ranking that looks like a right one -- so the shape is checked
    # before a single number is trusted.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=body)

    with pytest.raises(RerankError):
        _score(_reranker(handler), "q", ["a", "b"])


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_score_is_a_rerank_error(literal: str):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            content=f'{{"results": [{{"index": 0, "relevance_score": {literal}}}]}}'.encode(),
        )

    with pytest.raises(RerankError):
        _score(_reranker(handler), "q", ["a"])


def test_a_sidecar_that_is_not_running_is_a_rerank_error_over_a_real_socket():
    # No stub at all: bind a port, close it, and ask for scores from where nothing listens -- the
    # real "sidecar is down" case, end to end through httpx2's own connection handling.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    with pytest.raises(RerankError):
        _score(LlamaServerReranker(base_url=f"http://127.0.0.1:{port}"), "q", ["a", "b"])


# --- settings and factory ---------------------------------------------------------------------


def test_reranking_is_off_by_default():
    settings = Settings(_env_file=None)

    assert settings.reranker is RerankerProvider.NONE
    assert create_reranker(settings) is None


def test_the_factory_builds_the_mock_when_asked():
    reranker = create_reranker(Settings(reranker=RerankerProvider.MOCK, _env_file=None))

    assert isinstance(reranker, MockReranker)


class _LocalServer(http.server.ThreadingHTTPServer):
    """``ThreadingHTTPServer`` minus the reverse-DNS lookup ``HTTPServer.server_bind`` does.

    That lookup (``socket.getfqdn``) took 35 seconds on a developer machine whose resolver
    answers reverse queries for 127.0.0.1 slowly; a test of a local socket must not depend on DNS.
    """

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


@contextlib.contextmanager
def _fake_sidecar(paths: list[str]) -> Iterator[str]:
    """Serve one recorded ``llama-server`` rerank answer on a real local socket; yield its URL."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            paths.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            # The shape a real llama-server (build 11178, bge-reranker-v2-m3 Q8_0) returned,
            # best first: index 0 was the on-topic passage of four.
            body = json.dumps(_REAL_RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep the test output clean."""

    server = _LocalServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


_REAL_RESPONSE = {
    "model": "bge-reranker-v2-m3-Q8_0.gguf",
    "object": "list",
    "usage": {"prompt_tokens": 166, "total_tokens": 166},
    "results": [
        {"index": 0, "relevance_score": 0.15698370337486267},
        {"index": 3, "relevance_score": -4.885409832000732},
        {"index": 1, "relevance_score": -7.020950794219971},
        {"index": 2, "relevance_score": -11.018460273742676},
    ],
}


def test_the_factory_builds_a_client_that_talks_to_the_reranker_url_not_ollamas():
    paths: list[str] = []
    with _fake_sidecar(paths) as url:
        settings = Settings(
            reranker=RerankerProvider.LLAMA_SERVER,
            reranker_base_url=url,
            ollama_base_url="http://127.0.0.1:1",  # nothing listens: any use of it would fail
            _env_file=None,
        )
        reranker = create_reranker(settings)
        assert isinstance(reranker, LlamaServerReranker)

        scores = _score(reranker, "Cửa hàng cần diện tích tối thiểu?", ["a", "b", "c", "d"])

    assert paths == ["/v1/rerank"]
    assert scores == [
        0.15698370337486267,
        -7.020950794219971,
        -11.018460273742676,
        -4.885409832000732,
    ]


def test_the_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANKER", "llama-server")
    monkeypatch.setenv("RERANKER_BASE_URL", "http://127.0.0.1:8082")
    monkeypatch.setenv("RERANK_TOP_N", "7")

    settings = Settings(_env_file=None)

    assert settings.reranker is RerankerProvider.LLAMA_SERVER
    assert settings.reranker_base_url == "http://127.0.0.1:8082"
    assert settings.rerank_top_n == 7


@pytest.mark.parametrize("value", [0, -1])
def test_a_non_positive_top_n_is_rejected(value: int):
    with pytest.raises(ValueError, match="rerank_top_n"):
        Settings(rerank_top_n=value, _env_file=None)
