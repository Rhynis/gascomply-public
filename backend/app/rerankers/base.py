"""Shared interface for cross-encoder rerankers (issue #114).

A vector search scores a passage by the distance between two *independent* embeddings, the
question's and the passage's. A cross-encoder reads the question and the passage *together*
and scores that pair, which is what lets it tell "the article about this exact question" from
"an article that merely talks about the same subject" -- the distinction the generator's model
fails to make when a context holds many passages (issue #108: 9 of 12 wrong citations were the
model choosing a different Điều entirely).

Every implementation -- the offline ``MockReranker`` CI uses, ``LlamaServerReranker`` for the
real ``bge-reranker-v2-m3`` -- returns one score per document and nothing else. Which passages
to keep, and how many, is ``app.services.retriever``'s decision, not the reranker's, for the
same reason ``LLMProvider`` knows nothing about citations: this interface only reaches a model.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class RerankError(Exception):
    """Raised when a reranker could not produce trustworthy scores.

    The one exception ``Reranker.score`` is allowed to raise, whatever went wrong underneath
    (the sidecar is down, it timed out, it answered with something that is not a set of scores).
    ``Retriever`` catches exactly this and falls back to the un-reranked order, so a reranker
    that is unavailable degrades ``/ask`` to its behaviour before issue #114 instead of
    breaking it. An implementation must therefore wrap its transport errors in this type
    rather than let them escape.
    """


class Reranker(ABC):
    """Scores how well each of several documents answers one query."""

    @abstractmethod
    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Score every document against ``query``, without blocking the event loop.

        Args:
            query: The question.
            documents: Passage texts to score, in order.

        Returns:
            One score per document, in the same order as ``documents``. Higher means more
            relevant. Scores are only comparable *within* one call -- they are raw model
            outputs, not probabilities, and different queries do not share a scale.

        Raises:
            RerankError: If trustworthy scores could not be produced for any reason.
        """
        raise NotImplementedError
