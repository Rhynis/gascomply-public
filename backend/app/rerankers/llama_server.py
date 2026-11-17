"""Real cross-encoder reranker, backed by a local ``llama-server --rerank`` (issue #114).

Ollama has no rerank endpoint (open since March 2024, ollama#17129), and a reranker model
served through its ``/api/embeddings`` exposes only the embedding layer, never the classification
head that turns a (question, passage) pair into a relevance score. Numbers taken from there
would look like scores and not be cross-encoder scores, so this module never talks to Ollama at
all: it talks to a ``llama-server`` process (llama.cpp) started with ``--rerank`` on a
``bge-reranker-v2-m3`` GGUF, whose pooling type is ``rank`` -- the real classification head.
``docs/adr/0002-cross-encoder-reranker-runtime.md`` records why this runtime and not
``sentence-transformers`` (pulls in ``torch``) or a listwise call to the chat model (the very
model whose passage-picking this routes around).

The sidecar must be running, with the model loaded, and started with a batch size at least as
large as the longest passage plus the question -- a non-causal model processes its whole input in
one physical batch, so ``-ub`` below the input length is an error, not a slowdown::

    llama-server -m bge-reranker-v2-m3-Q8_0.gguf --rerank --port 8081 -c 8192 -b 8192 -ub 8192 -np 1

Neither is true by default, which is why ``RERANKER=none`` is the default
(``app/core/config.py``) and why every failure here is a ``RerankError``: ``Retriever`` falls
back to the un-reranked order rather than failing ``/ask``.

Scores are the model's raw logits (``relevance_score`` in the response), not probabilities:
they are unbounded and negative for an irrelevant passage, comparable only within one call.
"""

import logging
import math
from collections.abc import Sequence

import httpx2

from app.rerankers.base import Reranker, RerankError

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 60.0
"""Generous, like ``BgeM3Embedder``'s: a cold model load or a long batch of passages must not be
mistaken for a dead sidecar, but a hung one must surface within a bound and not stall ``/ask``.
Measured: 0.9-7.4 s to score ten to thirty whole-article passages while Ollama held a chat model."""


class LlamaServerReranker(Reranker):
    """Calls a local ``llama-server``'s ``/v1/rerank`` endpoint (Jina/Cohere-compatible)."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize the reranker.

        Args:
            base_url: The sidecar's base URL, e.g. ``http://localhost:8081``.
            timeout: Seconds to wait for one scoring request before giving up.
            transport: Overrides the HTTP transport -- real network I/O by default (``None``),
                or a ``httpx2.MockTransport`` in tests, so request shape and error handling
                are verified with no sidecar anywhere (CI has none, and CI fails on any skipped
                test).
        """
        self._base_url = base_url
        self._timeout = timeout
        self._transport = transport

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Score every document against ``query`` in one request to the sidecar.

        A new client per call, like ``BgeM3Embedder`` and ``OllamaProvider``: one scoring
        request per question does not earn a shared connection's lifecycle.

        Args:
            query: The question.
            documents: Passage texts to score, in order.

        Returns:
            One raw cross-encoder score per document, in the order of ``documents`` (the
            sidecar returns them best first, each tagged with its input ``index``; this
            restores the input order).

        Raises:
            RerankError: If the sidecar is unreachable, times out, answers with a non-2xx
                status, or answers with anything other than exactly one finite score per
                document.
        """
        if not documents:
            return []
        try:
            async with httpx2.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._base_url}/v1/rerank",
                    json={"query": query, "documents": list(documents)},
                )
                response.raise_for_status()
            scores = self._parse(response.json(), len(documents))
        except httpx2.HTTPError as error:
            msg = f"the rerank sidecar at {self._base_url} failed: {error!r}"
            raise RerankError(msg) from error
        except ValueError as error:  # ``response.json()`` on a body that is not JSON.
            msg = f"the rerank sidecar at {self._base_url} sent a body that is not JSON"
            raise RerankError(msg) from error
        logger.debug("Reranked %d document(s) via llama-server", len(documents))
        return scores

    @staticmethod
    def _parse(body: object, expected: int) -> list[float]:
        """Return the scores in input order, or raise ``RerankError`` if ``body`` is not that.

        Checks the shape before trusting a single number: a partial or re-indexed answer that
        was silently zipped against the documents would attach one passage's score to another,
        which is a wrong ranking that looks like a right one.
        """
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list) or len(results) != expected:
            msg = f"expected {expected} rerank result(s), got {body!r:.200}"
            raise RerankError(msg)
        scores: dict[int, float] = {}
        for item in results:
            index = item.get("index") if isinstance(item, dict) else None
            value = item.get("relevance_score") if isinstance(item, dict) else None
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < expected
                or index in scores
                or not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                msg = f"malformed rerank result: {item!r:.200}"
                raise RerankError(msg)
            scores[index] = float(value)
        return [scores[index] for index in range(expected)]
