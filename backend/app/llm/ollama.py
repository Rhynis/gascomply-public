"""Real LLM provider, backed by a local Ollama daemon (build plan A6 Phase 3.6, issue #90).

``ollama serve`` must be running, with the chat model pulled (``ollama pull
qwen2.5:7b-instruct-q4_K_M``) -- neither is true by default, which is exactly why
``mock`` stays the default ``LLMProvider`` (``app/core/config.py``): a developer with
no Ollama running must still get a real HTTP response from ``/ask``, not a crash
from an unreachable model. The real provider is opt in, via ``LLM_PROVIDER=ollama``.

Structured output, not free-text scraping. When ``generate`` is given a
``json_schema``, it is passed as Ollama's ``format`` field, which constrains the
model's decoding to that shape (checked against a real ``qwen2.5:7b-instruct-q4_K_M``
response while building this: the reply's ``message.content`` is a JSON string
matching the schema, not prose around it). This is what makes ``generator.py``'s
"parse through a schema, never scrape free text" rule enforceable against a real
model -- but the caller still parses and validates the result itself (issue #90):
constrained decoding narrows what the model can say, it does not verify that what
it said is *true*, which is the whole reason citations are checked against the
retrieved context afterwards, not trusted because they arrived in the right shape.
"""

import logging
from typing import Any

import httpx2

from app.llm.base import ContextOverflowError, LLMProvider, LLMUnavailableError, LLMUnreachableError

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 120.0
"""Generous: a quantized 7B model's first call after a cold load can take several
seconds even for a short reply (measured, building this module, at ~8.5s including a
~5.2s model load), and a hung request must never stall a request handler forever --
a real failure should surface as a timeout within a bound, not hang the process."""

DEFAULT_NUM_CTX = 16384
"""Mirrors ``Settings.ollama_num_ctx``'s own default (``app/core/config.py``) -- a
fallback for a provider built with no settings object at all (tests, ``eval.compare``'s
``main`` module-level use), never meant to drift from the configured default in practice."""

_REPLY_BUDGET_TOKENS = 256
"""Tokens of ``num_ctx`` reserved for the reply, below which a call is treated as overflowed
(issue #129). Every reply this provider's own callers ask for is a short, schema-constrained
JSON object -- the generator's own answer-plus-citations shape, or the grounding verifier's
single boolean -- never free-form prose, so 256 tokens is comfortably above any reply actually
observed while comfortably below the point a model that cannot even find room for a stop token
starts running on indefinitely (the mechanism behind issue #129's own HTTP-500 trap: a 6,972
token prompt filled the 4,095-token default window completely, and the reply never terminated
inside the client's 120s timeout). ``prompt_eval_count >= num_ctx - _REPLY_BUDGET_TOKENS`` is
treated as proof the prompt was truncated, or left too little room to trust the reply that
came back -- Ollama gives no separate truncation flag, so this count is the only signal there
is (the issue's own measurement: every truncated case logged a context limit of exactly
``num_ctx - 1``, i.e. ``prompt_eval_count == num_ctx - 1``, which this threshold always
catches for any budget >= 1)."""

_TEMPERATURE = 0.1
"""Low, not zero: this is extraction and citation, not creative writing, and a
near-deterministic model is easier to reason about -- but Ollama does not guarantee
exact reproducibility even at ``0.0``, so the caller (``generator.py``) still
validates every citation rather than trusting a low temperature to make that safe."""


class LLMError(Exception):
    """Raised when Ollama's response cannot be trusted as a real reply.

    Covers a response with no ``message.content`` at all -- the one shape ``.json()``
    would not itself catch, since a well-formed JSON body can still be missing the
    field this module reads.
    """


class OllamaProvider(LLMProvider):
    """Calls a local Ollama daemon's ``/api/chat`` endpoint with a configured chat model."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        num_ctx: int = DEFAULT_NUM_CTX,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            base_url: Ollama's base URL, e.g. ``http://localhost:11434``.
            model: Name of the pulled chat model to ask, e.g.
                ``qwen2.5:7b-instruct-q4_K_M``.
            timeout: Seconds to wait for one reply before raising.
            num_ctx: Context window, in tokens, sent as Ollama's own ``options.num_ctx`` on
                every call -- generation and the per-citation grounding verifier alike, since
                both go through this same method (issue #129). Left unset, Ollama falls back
                to its own 4,096-token default, which silently truncates the front of a long
                generation prompt (the system prompt's own rules, then the best-ranked
                passages) rather than raising anything.
            transport: Overrides the HTTP transport -- real network I/O by default
                (``None``), or a ``httpx2.MockTransport`` in tests, so request
                shape and error handling are verified with no model server
                anywhere (CI has no Ollama, and CI fails on any skipped test).
        """
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._num_ctx = num_ctx
        self._transport = transport
        self.last_prompt_eval_count: int | None = None
        """Set after every call that reaches Ollama, successful or overflowed -- read by
        ``eval.compare`` (never by ``/ask`` itself) right after the one call it cares about,
        before a later call through the same provider can overwrite it. ``None`` before the
        first call, or if a reply carried no ``prompt_eval_count`` at all."""

    @property
    def num_ctx(self) -> int:
        """The context window, in tokens, this provider sends on every call."""
        return self._num_ctx

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        """Ask the configured model for one chat reply.

        Awaits the HTTP round trip (``httpx2.AsyncClient``, issue #89's rule for any
        network call on the request path), so the event loop keeps serving other
        requests while Ollama works. A new client per call, same reasoning as
        ``BgeM3Embedder``: one short-lived question does not earn a shared
        connection's lifecycle.

        Args:
            system: System prompt.
            user: User prompt.
            json_schema: Passed through as Ollama's ``format``, when given.

        Returns:
            ``response["message"]["content"]``, exactly as Ollama returned it.

        Raises:
            LLMError: If the response has no ``message.content``.
            ContextOverflowError: If ``prompt_eval_count`` left less than
                ``_REPLY_BUDGET_TOKENS`` of ``num_ctx`` -- proof the prompt was truncated, or
                too close to it to trust the reply (issue #129). Never answered from; the
                caller (``app.services.generator``) turns this into a refusal.
            LLMUnavailableError: If the request timed out (``httpx2.TimeoutException``) --
                issue #129's second bug: this used to propagate raw and surface as an
                unhandled HTTP 500.
            LLMUnreachableError: If the request failed for any other transport reason
                (``httpx2.TransportError`` that is not a timeout -- connection refused, a
                dropped connection) -- issue #183: this used to propagate raw and surface as an
                unhandled HTTP 500 too, exactly like the timeout #129 already converted. A
                non-2xx status (e.g. the model is not pulled) is a real bug and still
                propagates raw, unconverted.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": _TEMPERATURE, "num_ctx": self._num_ctx},
            "stream": False,
            # issue #112: a thinking-capable model (e.g. qwen3) can emit a long reasoning
            # passage before its answer -- slower, and liable to break the strict JSON
            # ``generator.py`` depends on. Ollama accepts and ignores "think" for a model
            # that has no thinking mode at all (checked against a real qwen2.5 reply while
            # making this change), so setting it unconditionally is safe for every model
            # this provider is configured with, not only the ones it is meant to silence.
            "think": False,
        }
        if json_schema is not None:
            payload["format"] = json_schema
        async with httpx2.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            try:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
            except httpx2.TimeoutException as error:
                # Caught first: TimeoutException is itself a TransportError (issue #183's own
                # trap), and a timeout must stay LLMUnavailableError, never fall through to the
                # broader branch below.
                msg = f"Ollama request timed out after {self._timeout}s"
                raise LLMUnavailableError(msg) from error
            except httpx2.TransportError as error:
                msg = f"Ollama unreachable at {self._base_url}: {error}"
                raise LLMUnreachableError(msg) from error
            response.raise_for_status()
        body = response.json()
        content = body.get("message", {}).get("content")
        if not isinstance(content, str):
            msg = f"Ollama response carried no message.content: {body!r}"
            raise LLMError(msg)
        prompt_eval_count = body.get("prompt_eval_count")
        self.last_prompt_eval_count = (
            prompt_eval_count if isinstance(prompt_eval_count, int) else None
        )
        if (
            self.last_prompt_eval_count is not None
            and self.last_prompt_eval_count >= self._num_ctx - _REPLY_BUDGET_TOKENS
        ):
            logger.error(
                "Ollama context overflow: prompt_eval_count=%d left less than the "
                "%d-token reply budget inside num_ctx=%d -- refusing to trust this reply",
                self.last_prompt_eval_count,
                _REPLY_BUDGET_TOKENS,
                self._num_ctx,
            )
            msg = (
                f"prompt_eval_count={self.last_prompt_eval_count} is within the "
                f"{_REPLY_BUDGET_TOKENS}-token reply budget of num_ctx={self._num_ctx}"
            )
            raise ContextOverflowError(msg)
        logger.debug(
            "Generated a reply via %s (%d char(s), prompt_eval_count=%s)",
            self._model,
            len(content),
            self.last_prompt_eval_count,
        )
        return content
