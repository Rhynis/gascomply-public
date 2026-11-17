"""Construct the ``Reranker`` configured by ``Settings`` (issue #114).

Kept out of ``app/main.py`` for the same reason ``app.llm.factory`` is kept out of
``app/api/ask.py``: deciding which concrete reranker to build is real logic, not wiring.
"""

from app.core.config import RerankerProvider, Settings
from app.rerankers.base import Reranker
from app.rerankers.llama_server import LlamaServerReranker
from app.rerankers.mock import MockReranker


def create_reranker(settings: Settings) -> Reranker | None:
    """Return the ``Reranker`` named by ``settings.reranker``, or ``None`` for no reranking.

    Args:
        settings: Application settings; only ``reranker`` and, for ``LLAMA_SERVER``,
            ``reranker_base_url`` are read.

    Returns:
        ``None`` by default -- ``Retriever`` then skips the stage entirely, which is its
        behaviour before issue #114. A ``MockReranker`` for ``RerankerProvider.MOCK``, or a
        ``LlamaServerReranker`` for ``RerankerProvider.LLAMA_SERVER``.
    """
    if settings.reranker is RerankerProvider.MOCK:
        return MockReranker()
    if settings.reranker is RerankerProvider.LLAMA_SERVER:
        return LlamaServerReranker(base_url=settings.reranker_base_url)
    return None
