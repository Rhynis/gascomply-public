"""Application settings, loaded from the environment (12-factor)."""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    """Deployment environment the process is running in."""

    LOCAL = "local"
    CI = "ci"
    PRODUCTION = "production"


class EmbedderProvider(StrEnum):
    """Which ``Embedder`` implementation the ingest pipeline constructs (A2).

    ``MOCK`` stays the default deliberately (build plan A6 Phase 1.5c): a
    developer with no Ollama running must be able to run the whole pipeline
    exactly as before. The real embedder is opt in, via ``EMBEDDER=bge-m3``.
    """

    MOCK = "mock"
    BGE_M3 = "bge-m3"


class LLMProvider(StrEnum):
    """Which ``app.llm`` provider ``/ask`` (build plan A6 Phase 3.7) constructs.

    ``MOCK`` is the default, for the same reason ``EmbedderProvider.MOCK`` is: a
    developer with no Ollama running must be able to start the app and get a
    real HTTP response from ``/ask``, not a crash from an unreachable model.
    ``MockLLM`` is not a real answer -- see its own docstring -- so it is also
    what CI uses, the same way ``MockEmbedder`` is. The real model is opt in,
    via ``LLM_PROVIDER=ollama``.
    """

    MOCK = "mock"
    OLLAMA = "ollama"


class RerankerProvider(StrEnum):
    """Which ``app.rerankers`` cross-encoder ``/ask`` (issue #114) constructs, if any.

    ``NONE`` is the default, unlike ``EmbedderProvider.MOCK`` and ``LLMProvider.MOCK``:
    those stand in for a stage nothing works without (a question must be embedded, an answer
    must come from somewhere), while reranking is an *optional* stage. Its offline default is
    to skip it, which is exactly today's behaviour, rather than a ``MOCK`` that would quietly
    cut real passages by word overlap in a deployment that never asked for it. ``MOCK`` is the
    offline stand-in for wiring tests and demos; ``LLAMA_SERVER`` is the real cross-encoder,
    opt in via ``RERANKER=llama-server`` (``docs/adr/0002-cross-encoder-reranker-runtime.md``).
    """

    NONE = "none"
    MOCK = "mock"
    LLAMA_SERVER = "llama-server"


class Settings(BaseSettings):
    """Runtime configuration.

    Attributes:
        app_env: Deployment environment; gates developer-only surfaces.
        log_level: Root log level name accepted by the standard logging module.
        database_url: ``postgresql+asyncpg://`` connection string for the
            Postgres + pgvector database (build plan A6 Phase 1.5b). ``None``
            means no database is configured -- callers that persist (the
            ingest CLI, migrations) treat that as "run without a database"
            rather than failing to start. A local Postgres for development
            comes from the repository's ``docker-compose.yml``; in CI it is a
            service container. Supabase (Phase 5.2) is just a value for this
            setting in production, not a different code path.
        embedder: Which ``Embedder`` the ingest pipeline constructs (A6 Phase
            1.5c). Defaults to ``MOCK`` -- see ``EmbedderProvider``.
        ollama_base_url: Base URL of the local Ollama daemon, used only when
            ``embedder`` is ``BGE_M3`` or ``llm_provider`` is ``OLLAMA``.
            ``ollama serve``'s default port.
        llm_provider: Which ``app.llm`` provider ``/ask`` constructs (A6 Phase
            3.7). Defaults to ``MOCK`` -- see ``LLMProvider``.
        ollama_model: Chat model ``OllamaProvider`` asks Ollama for, only read
            when ``llm_provider`` is ``OLLAMA``. The build plan's own dev
            target (``ollama pull qwen3:8b``) -- switched from
            ``qwen2.5:7b-instruct-q4_K_M`` in issue #156, once #112's own
            comparison and #160's fix to the Phụ lục VI column-header gap
            showed it citing more accurately and refusing less than the
            model it replaces.
        ollama_num_ctx: Context window, in tokens, ``OllamaProvider`` requests on every call
            (issue #129). Ollama's own default is 4,096, which silently truncates the front of
            a prompt -- the system prompt's own citation rules, then the best-ranked passages
            -- once the generation prompt passes that size; measured against the gold set,
            over half did. 16384 is chosen to clear the largest observed generation prompt
            (9,549 tokens) with headroom for the reply, while staying inside the resident
            memory budget measured for ``qwen2.5:7b`` plus ``bge-m3`` both loaded at this size
            (see the issue's own README/PR note) -- raising it further needs a fresh memory
            measurement first, not just a config edit.
        reranker: Which cross-encoder reranks the retrieved passages before generation
            (issue #114). Defaults to ``NONE`` -- see ``RerankerProvider``.
        reranker_base_url: Base URL of the ``llama-server --rerank`` sidecar, only read when
            ``reranker`` is ``LLAMA_SERVER``. Not Ollama's: Ollama has no rerank endpoint.
        rerank_top_n: How many passages survive reranking, only read when ``reranker`` is not
            ``NONE``. Chosen by measurement, not taste -- see the ADR's table.
        grounding_check: Whether generation verifies that each cited clause supports the answer.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: AppEnv = AppEnv.LOCAL
    log_level: str = "INFO"
    database_url: str | None = None
    embedder: EmbedderProvider = EmbedderProvider.MOCK
    ollama_base_url: str = "http://localhost:11434"
    llm_provider: LLMProvider = LLMProvider.MOCK
    ollama_model: str = "qwen3:8b"
    ollama_num_ctx: int = Field(default=16384, ge=1)
    reranker: RerankerProvider = RerankerProvider.NONE
    reranker_base_url: str = "http://localhost:8081"
    rerank_top_n: int = Field(default=5, ge=1)
    grounding_check: bool = True

    @property
    def docs_enabled(self) -> bool:
        """Whether the interactive OpenAPI docs should be served."""
        return self.app_env is not AppEnv.PRODUCTION


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (FastAPI dependency)."""
    return Settings()
