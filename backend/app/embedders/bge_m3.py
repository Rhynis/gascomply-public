"""Real embedder backed by a local Ollama daemon running bge-m3 (build plan A2/A6 Phase 1.5c).

``ollama serve`` must be running, with the ``bge-m3`` model pulled (``ollama
pull bge-m3``) -- neither is true by default, which is exactly why ``mock``
stays the default ``Embedder`` (``app/core/config.py``): a developer with no
Ollama running must be able to run the whole ingest pipeline unchanged.

One HTTP request per batch, never one per text -- ``Ollama``'s ``/api/embed``
already accepts a list of inputs, and a per-text round trip would make a
692-chunk corpus take minutes instead of the measured ~90 seconds.

The request is made with ``httpx2.AsyncClient`` (issue #89). The blocking
``httpx2.Client`` this module used to open was harmless in the ingest CLI and would
have frozen every concurrent request behind one Ollama round trip once retrieval called
it from a request handler. An async client, rather than a thread around the blocking
one, because the wait is network I/O: it costs no thread while it waits, and cancelling
the awaiting task cancels the request instead of leaving a thread running it to the end.
"""

import logging
from collections.abc import Sequence

import httpx2

from app.embedders.base import Embedder, EmbedderUnreachableError

logger = logging.getLogger(__name__)

DIMENSION = 1024
"""bge-m3's own output width -- matches ``CHUNKS_EMBEDDING_DIMENSION`` and the
``vector(1024)`` column exactly (measured against a real Ollama response, not
assumed from documentation), so no schema change is needed for this model."""

_DEFAULT_TIMEOUT_SECONDS = 60.0
"""Generous, not tuned to the measured ~124 ms/chunk throughput: a cold model
load alone was measured at ~2.2s, and a hung request must never stall a
corpus ingest indefinitely (build plan A6 issue #47) -- a real failure should
surface as a timeout within a reasonable bound, not hang the process."""


class EmbeddingError(Exception):
    """Raised when Ollama's response cannot be trusted as real embeddings.

    Covers a response with a different number of vectors than input texts,
    or a vector whose width does not match ``DIMENSION`` -- either would
    otherwise let a half-ingested, incorrectly-shaped corpus reach the
    database, for ``ChunkRepository`` to catch (or not) much later.
    """


class BgeM3Embedder(Embedder):
    """Calls a local Ollama daemon's ``/api/embed`` endpoint with the ``bge-m3`` model."""

    dimension = DIMENSION

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize the embedder.

        Args:
            base_url: Ollama's base URL, e.g. ``http://localhost:11434``.
            timeout: Seconds to wait for one batch request before raising.
            transport: Overrides the HTTP transport -- real network I/O by
                default (``None``), or a ``httpx2.MockTransport`` in tests, so
                the request shape, batching, and error handling are all
                verified with no model server anywhere (build plan A6 issue
                #47: CI has no Ollama, and CI fails on any skipped test).
                ``MockTransport`` serves the async client as it served the
                blocking one, so those tests needed no new fake.
        """
        self._base_url = base_url
        self._timeout = timeout
        self._transport = transport

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts in one request to Ollama's ``bge-m3`` model.

        Awaits the HTTP round trip, so the event loop keeps serving other tasks while
        Ollama works. A new client (and connection) per call is deliberate: the
        request path embeds one short question, and a shared client would need a
        lifecycle (who closes it, and when) that this call does not yet earn.

        Args:
            texts: Texts to embed, in order.

        Returns:
            One 1024-dimensional vector per input text, in the same order.

        Raises:
            EmbeddingError: If Ollama's response carries a different number
                of vectors than texts given, or any vector is not
                ``DIMENSION`` wide.
            EmbedderUnreachableError: If the request failed on any transport error, timeout
                included (``httpx2.TransportError``) -- issue #183: this used to propagate
                raw, which ``app.api.ask`` had no handler for and surfaced as an unhandled
                HTTP 500. Not caught here for the ingest pipeline: its own per-document error
                handling (``ingest/run.py``) treats this exactly like any other exception and
                reports a document failure, never a silent gap.
            httpx2.HTTPStatusError: If the response is a non-2xx status -- a real bug (the
                model is not pulled, a misconfigured URL), not an outage, so it still
                propagates raw, exactly as before this issue.
        """
        if not texts:
            return []
        async with httpx2.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            try:
                response = await client.post(
                    f"{self._base_url}/api/embed", json={"model": "bge-m3", "input": list(texts)}
                )
            except httpx2.TransportError as error:
                msg = f"Ollama embedder unreachable at {self._base_url}: {error}"
                raise EmbedderUnreachableError(msg) from error
            response.raise_for_status()
        embeddings: list[list[float]] = response.json()["embeddings"]
        self._check_shape(texts, embeddings)
        logger.debug("Embedded %d text(s) via bge-m3", len(texts))
        return embeddings

    def _check_shape(self, texts: Sequence[str], embeddings: list[list[float]]) -> None:
        """Raise before any row is written if Ollama's response is the wrong shape."""
        if len(embeddings) != len(texts):
            msg = f"Ollama returned {len(embeddings)} embedding(s) for {len(texts)} text(s)"
            raise EmbeddingError(msg)
        for index, vector in enumerate(embeddings):
            if len(vector) != self.dimension:
                msg = f"embedding {index} has {len(vector)} dimension(s), expected {self.dimension}"
                raise EmbeddingError(msg)
