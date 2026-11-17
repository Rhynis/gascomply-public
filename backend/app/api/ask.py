"""``POST /ask`` -- safety, retrieval and generation composed (build plan A6 Phase 3.7, issue #90).

Safety runs first, and strictly first: ``check_emergency`` is the very first thing
the handler does, before any dependency that would touch a database or a network is
even resolved. That ordering is deliberate and load-bearing, not incidental --
``session``/``embedder``/``llm`` are looked up from ``app.state`` by hand, inside the
handler body, *after* the emergency check has already returned, rather than declared
as ``Depends`` parameters (FastAPI resolves every declared parameter dependency
before the handler body runs at all, which would open a database connection for an
emergency question even if the body never went on to use it). A test proves this
structurally: an app built with no database configured still answers an emergency
question with 200, never a 503.
"""

import logging

from fastapi import APIRouter, Depends, Request

from app.api.deps import require_session_factory
from app.embedders.base import EmbedderUnreachableError
from app.repositories.documents import DocumentRepository
from app.schemas.ask import AskOutcome, AskRequest, AskResponse
from app.services.generator import generate_answer
from app.services.retriever import Retriever
from app.services.safety import check_emergency

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])

RATE_LIMIT_PER_MINUTE = 30


def _rate_limit(request: Request) -> None:
    """Enforce this process's ``/ask`` limiter, stored on ``app.state`` (``app.main.create_app``).

    Not a module-level ``RateLimiter`` bound into the route at import time: the route object
    is shared by every app built from this router (every test's own ``create_app`` call
    included), so a singleton here would let one test's requests count against another's
    limit. Reading it off ``request.app.state`` each call gives every app -- and so every
    test -- its own counters, the same fix already applied to the session factory, embedder
    and LLM provider.
    """
    request.app.state.ask_rate_limiter(request)


@router.post("/ask", dependencies=[Depends(_rate_limit)])
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Answer one question: the emergency check first, then retrieval and generation.

    Args:
        body: The question.
        request: Used to reach the process's session factory, embedder, LLM provider,
            reranker and retrieval config (``app.state``, set in ``app.main.create_app``) --
            only once the question has already been found not to be an emergency.

    Returns:
        The emergency response, a refusal (including ``refusal_reason ==
        "embedder_unreachable"`` when the embedder's own Ollama call could not be reached,
        issue #183), or a grounded answer with citations.
    """
    safety = check_emergency(body.question)
    if safety.escalated:
        logger.info("ask: escalated (%s)", safety.matched)
        return AskResponse(outcome=AskOutcome.EMERGENCY, safety_response=safety.response)

    session_factory = require_session_factory(request)
    embedder = request.app.state.embedder
    llm = request.app.state.llm_provider
    reranker = request.app.state.reranker
    config = request.app.state.retrieval_config

    async with session_factory() as session, session.begin():
        retriever = Retriever(session, embedder, config=config, reranker=reranker)
        try:
            retrieval = await retriever.retrieve(body.question)
        except EmbedderUnreachableError as error:
            # Issue #183: this used to propagate raw and surface as an unhandled HTTP 500,
            # exactly the bug #129 already fixed for the LLM provider's own calls.
            logger.error("ask: refused -- the embedder was unreachable: %s", error)
            return AskResponse(outcome=AskOutcome.REFUSAL, refusal_reason="embedder_unreachable")
        return await generate_answer(body.question, retrieval, llm, DocumentRepository(session))
