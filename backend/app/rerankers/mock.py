"""Deterministic, offline reranker for tests and CI (issue #114).

Not a cross-encoder, and it must never be used to judge relevance: it counts how many of the
question's distinct words also occur in a passage. It exists for the same reason
``MockEmbedder`` and ``MockLLM`` do -- so the whole reranking path (scoring, reordering, the
top-N cut, the fallback when a scorer fails) runs and is tested with nothing installed, on CI
runners that have no ``llama-server`` and never will.

It makes no network call, needs no weights, and always returns the same scores for the same
inputs, which is all the wiring tests need. Pure standard library (C12).
"""

import re
import unicodedata
from collections.abc import Sequence

from app.rerankers.base import Reranker

_WORD = re.compile(r"\w+")


def _words(text: str) -> set[str]:
    """Return the distinct lower-cased words of ``text``, NFC-normalised (Vietnamese diacritics)."""
    return set(_WORD.findall(unicodedata.normalize("NFC", text).lower()))


class MockReranker(Reranker):
    """Scores a document by the fraction of the query's distinct words it contains."""

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return, per document, the share of ``query``'s distinct words found in it.

        ``async`` only to satisfy ``Reranker``: this does no I/O and awaits nothing.

        Args:
            query: The question.
            documents: Passage texts, in order.

        Returns:
            One score in ``[0.0, 1.0]`` per document; all ``0.0`` when ``query`` has no words.
        """
        wanted = _words(query)
        if not wanted:
            return [0.0 for _ in documents]
        return [len(wanted & _words(document)) / len(wanted) for document in documents]
