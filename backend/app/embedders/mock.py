"""Deterministic, offline embedder for tests and CI (build plan A2 / A6 Phase 1.5).

No real embedding provider is settled yet (Phase 1.5b, once infrastructure is
provisioned), so nothing in the ingest pipeline can be exercised in CI without
a stand-in. ``MockEmbedder`` fills that gap: it makes no network call, needs no
model weights, and always returns the same vector for the same text, which is
enough to test chunking, batching and pipeline plumbing end to end offline.
It is not a semantic embedding and must never be used for real similarity
ranking.

``embed`` is a coroutine only to satisfy ``Embedder`` (issue #89); it never awaits
anything, because it does no I/O. Its work is one SHA-256 hash per vector component
-- measured at 0.5 ms for the single string a retrieval embeds, which is not a stall
worth a thread. A whole document's chunks are embedded in one call only by the ingest
CLI (about 66 ms per hundred 500-character chunks), which has no other request to
starve.
"""

import hashlib
import logging
from collections.abc import Sequence

from app.embedders.base import Embedder

logger = logging.getLogger(__name__)

DEFAULT_DIMENSION = 1024
"""Matches bge-m3's vector width (build plan A5.3) purely so a chunk table
built in manual/CI testing lines up with the real DDL for easy comparison.
This is the mock's own default only -- nothing else in this phase may assume
1024 applies universally; the vector dimension is always parametrized to the
embedder in use (A2)."""

_MAX_UINT64 = 2**64 - 1


class MockEmbedder(Embedder):
    """Hash-based embedder: the same text always yields the same vector.

    Each vector component is derived from a SHA-256 hash of the input text
    combined with the component's index, then scaled into ``[-1.0, 1.0]`` to
    resemble a typical embedding's value range. Using a fresh hash per
    component (rather than repeating a fixed-length digest) lets ``dimension``
    be set to any size while keeping every component independent. Pure
    standard library (``hashlib``) -- no numpy, no ML dependency (C12).
    """

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        """Initialize the embedder.

        Args:
            dimension: Length of every vector this embedder produces. Defaults
                to bge-m3's 1024 for convenience only (see ``DEFAULT_DIMENSION``).

        Raises:
            ValueError: If ``dimension`` is not a positive integer.
        """
        if dimension < 1:
            msg = f"dimension must be >= 1, got {dimension}"
            raise ValueError(msg)
        self.dimension = dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts deterministically, with no network calls.

        Args:
            texts: Texts to embed, in order.

        Returns:
            One vector per input text, in the same order, each of length
            ``dimension``. The same text always produces the same vector;
            different texts produce different vectors.
        """
        vectors = [self._vector_for(text) for text in texts]
        logger.debug("Embedded %d text(s) at dimension %d", len(texts), self.dimension)
        return vectors

    def _vector_for(self, text: str) -> list[float]:
        """Return the deterministic vector for one text."""
        return [self._component(text, index) for index in range(self.dimension)]

    @staticmethod
    def _component(text: str, index: int) -> float:
        """Return one deterministic vector component in ``[-1.0, 1.0]``."""
        digest = hashlib.sha256(f"{index}:{text}".encode()).digest()
        as_uint = int.from_bytes(digest[:8], byteorder="big")
        return (as_uint / _MAX_UINT64) * 2.0 - 1.0
