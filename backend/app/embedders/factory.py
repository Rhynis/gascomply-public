"""Construct the ``Embedder`` configured by ``Settings`` (build plan A6 Phase 1.5c).

Kept out of ``ingest/run.py`` on purpose -- that module is a composition layer
only (its own docstring), and deciding which concrete ``Embedder`` to build is
real logic, not argument parsing or printing.
"""

from app.core.config import EmbedderProvider, Settings
from app.embedders.base import Embedder
from app.embedders.bge_m3 import BgeM3Embedder
from app.embedders.mock import MockEmbedder


def create_embedder(settings: Settings) -> Embedder:
    """Return the ``Embedder`` named by ``settings.embedder``.

    Args:
        settings: Application settings; only ``embedder`` and, for
            ``BGE_M3``, ``ollama_base_url`` are read.

    Returns:
        A ``MockEmbedder`` by default, or a ``BgeM3Embedder`` when
        ``settings.embedder`` is ``EmbedderProvider.BGE_M3``.
    """
    if settings.embedder is EmbedderProvider.BGE_M3:
        return BgeM3Embedder(base_url=settings.ollama_base_url)
    return MockEmbedder()
