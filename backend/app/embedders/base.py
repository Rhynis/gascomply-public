"""Shared interface for embedding providers (build plan A2).

Every provider -- the offline ``mock`` used in tests today, ``bge_m3`` and
others once Phase 1.5b settles on real infrastructure -- implements this same
interface, so the ingest and retrieval code never depends on a specific
provider. ``embed`` takes a batch rather than one text at a time: a per-text
call would be a network round trip for every real (non-mock) implementation,
so the interface is batch-shaped even though the mock does not need it.

``embed`` is a coroutine (issue #89). It began as a plain method because its only
caller was the ingest CLI, where blocking costs nothing. Retrieval calls it from a
request handler, where a blocking HTTP round trip to Ollama would stall every other
request on the same event loop for its whole duration. One async interface, rather
than a blocking method plus an async twin, makes the wrong thing impossible to write by
accident: there is no synchronous ``embed`` left to call from a handler.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbedderUnreachableError(Exception):
    """Raised by an embedder when a transport failure, timeout included, kept it unreached.

    Covers any ``httpx2.TransportError`` -- ``ConnectError`` (nothing listening on the
    configured port), ``RemoteProtocolError`` (the daemon dropped the connection mid-reply) and
    a timeout alike, measured for real against this project's own Ollama daemon restarting
    under another project's LaunchAgent (issue #183, built on #182's "Outage during the
    candidate run"). Unlike ``app.llm.base.LLMUnreachableError``, a timeout is folded in here
    rather than kept as its own case: an embedding call has no analogue to issue #112's "the
    model itself is slow" measurement that a separate code would protect, so there is nothing a
    distinct timeout reason would tell apart. A non-2xx response (``HTTPStatusError``) is not
    covered here and still propagates raw -- that is a real bug, not an outage.

    ``app.api.ask`` catches this (from ``Retriever.retrieve``, which calls the embedder) and
    turns it into ``AskResponse.refusal_reason == "embedder_unreachable"``, never an unhandled
    500. ``ingest/run.py`` does not catch it specially: it is a normal ``Exception`` to that
    module's own per-document error handling, and a failed embed must still fail that document
    loudly, exactly as any other ingest error does.
    """


class Embedder(ABC):
    """Turns text into fixed-length vectors for the ``chunks.embedding`` column.

    Attributes:
        dimension: Length of every vector this embedder produces. Embedder-
            dependent (build plan A2: bge-m3 is 1024-dim, other providers
            differ) -- nothing outside a concrete embedder may assume a
            specific value.
    """

    dimension: int

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, without blocking the event loop.

        Args:
            texts: Texts to embed, in order.

        Returns:
            One vector per input text, in the same order, each of length
            ``dimension``.
        """
        raise NotImplementedError
