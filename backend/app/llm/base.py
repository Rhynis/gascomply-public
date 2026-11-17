"""Shared interface for chat-style LLM providers (build plan A6 Phase 3.6, issue #90).

One turn in, one turn out: a system prompt, a user prompt, and the model's raw text
reply. ``app.services.generator`` owns the prompt (the cite-or-refuse rules, the
schema the reply must match, the retrieved context) and owns parsing the reply; a
provider knows nothing about citations, units, or the corpus -- it only knows how to
reach one particular model. That split is what lets ``MockLLM`` stand in for
``OllamaProvider`` (and, later, a hosted provider) with no change to the caller, the
same way ``MockEmbedder`` stands in for ``BgeM3Embedder`` (issue #89).
"""

from abc import ABC, abstractmethod
from typing import Any


class LLMUnavailableError(Exception):
    """Raised by a provider when it timed out, never produced a reply at all this time.

    Covers a request timeout (the case this exception was added for, issue #129's own
    second bug: a 6,972-token prompt filling a truncated window ran into the client's 120s
    timeout and surfaced as an HTTP 500) -- never a malformed or missing reply, which is
    ``LLMError`` (``app.llm.ollama``), and never a non-2xx status, which still propagates raw:
    that is a real bug a 500 should surface, never hidden behind a refusal.

    A non-timeout transport failure (connection refused, a dropped connection) is
    ``LLMUnreachableError`` below, a subclass of this one (issue #183): measured against this
    deployment's own Ollama daemon, restarted mid-reply or mid-outage by another project's
    LaunchAgent (issue #182's "Outage during the candidate run"), a connection failure is an
    operational fact about the deployment, exactly like a timeout, not a defect in this
    codebase's own logic -- so it must never become an unhandled 500 either. Only a non-2xx
    status still points at a bug in this codebase or its configuration.

    ``app.services.generator`` catches this (and ``LLMUnreachableError``) and turns it into
    ``AskResponse.refusal_reason`` -- ``"llm_unavailable"`` for a plain timeout,
    ``"llm_unreachable"`` for the subclass -- so a frontend (and this project's own evaluation)
    can tell a transient timeout apart from a connection failure.
    """


class LLMUnreachableError(LLMUnavailableError):
    """Raised by a provider when a transport failure, not a timeout, kept it unreached.

    Covers any ``httpx2.TransportError`` other than ``httpx2.TimeoutException`` --
    ``ConnectError`` (nothing listening on the configured port) and ``RemoteProtocolError`` (the
    daemon dropped the connection mid-reply), both measured for real against this project's own
    Ollama daemon restarting under another project's LaunchAgent (issue #183, built on #182's
    "Outage during the candidate run"). A non-2xx response (``HTTPStatusError``) is not covered
    here and still propagates raw -- that is a real bug, not an outage.

    Subclasses ``LLMUnavailableError`` so every caller that already catches that exception
    (``app.services.generator``'s existing ``except`` clauses) catches this too, with no change
    to their own code -- only ``_refuse_for_llm_error`` tells the two apart, to give this one its
    own ``refusal_reason`` (``"llm_unreachable"``, never folded into ``"llm_unavailable"``: issue
    #112 already relies on that code meaning "the model itself was slow", and folding a
    connection failure into it would silently lower a measured rate that is about the model, not
    the infrastructure under it).
    """


class ContextOverflowError(Exception):
    """Raised by a provider when a reply's prompt size left too little of its context window.

    Issue #129: Ollama gives no explicit truncation flag, but a reply's own
    ``prompt_eval_count`` is proof after the fact -- if it lands at or within the model's reply
    budget inside ``num_ctx``, the prompt either was truncated outright or left so little room
    for the reply that the call cannot be trusted (the mechanism behind the issue's own
    HTTP-500 trap: a model that cannot see enough of its own context to find a stop token can
    run on for thousands of tokens instead of answering). A provider that can measure this
    (``OllamaProvider``) raises this instead of returning a reply built on a truncated prompt.

    ``app.services.generator`` catches this and turns it into ``AskResponse.refusal_reason ==
    "context_overflow"`` -- never silently answered from a window that could not hold the
    whole prompt.
    """


class LLMProvider(ABC):
    """Answers one chat turn. Never called on the emergency path (``app.services.safety``)."""

    @abstractmethod
    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None = None
    ) -> str:
        """Return the model's raw text reply to one turn, with no memory of any other.

        Args:
            system: System prompt: the model's instructions, fixed per call.
            user: User prompt: the question and its retrieved context.
            json_schema: When given, a JSON Schema the reply should conform to. A
                provider that can enforce this at decode time (``OllamaProvider``,
                via Ollama's structured-output ``format``) does; one that cannot is
                free to ignore it and rely on the prompt's own instructions instead
                -- the caller (``app.services.generator``) parses and validates the
                reply itself either way, and never trusts the schema alone.

        Returns:
            The model's raw reply, exactly as returned. The caller decides how to
            parse it; a provider never returns a parsed object here -- one contract
            for every provider, real or not.
        """
