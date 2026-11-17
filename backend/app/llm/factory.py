"""Construct the ``LLMProvider`` configured by ``Settings`` (build plan A6 Phase 3.6, issue #90).

Kept out of ``app/api/ask.py`` for the same reason ``app.embedders.factory`` is kept
out of ``ingest/run.py``: deciding which concrete provider to build is real logic,
not request handling.
"""

from app.core.config import LLMProvider as LLMProviderKind
from app.core.config import Settings
from app.llm.base import LLMProvider
from app.llm.mock import MockLLM
from app.llm.ollama import OllamaProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Return the ``LLMProvider`` named by ``settings.llm_provider``.

    Args:
        settings: Application settings; only ``llm_provider`` and, for ``OLLAMA``,
            ``ollama_base_url``/``ollama_model``/``ollama_num_ctx`` are read.

    Returns:
        A ``MockLLM`` by default, or an ``OllamaProvider`` when
        ``settings.llm_provider`` is ``LLMProvider.OLLAMA``.
    """
    if settings.llm_provider is LLMProviderKind.OLLAMA:
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            num_ctx=settings.ollama_num_ctx,
        )
    return MockLLM()
