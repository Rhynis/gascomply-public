"""Schemas for the pre-LLM gas-emergency check (build plan A6 Phase 3.5).

The question that reaches ``/ask`` (Phase 3.7) is checked for a gas emergency before any
retrieval or LLM call. These models are what that check hands back, so the caller can tell
the two paths apart without inspecting text: ``SafetyCheck.escalated`` is the flag, and
everything else is detail for logging and for tests.
"""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

AnswerT = TypeVar("AnswerT")
"""Whatever the caller's normal (non-emergency) pipeline returns."""


class SafetyCheck(BaseModel):
    """The verdict of the emergency check on one question.

    Attributes:
        escalated: ``True`` when the question reads as a gas emergency happening now (a
            smell of gas, a leak, a fire, an explosion), as opposed to a question *about*
            those subjects. This is the escalation flag: the caller must answer with
            ``response`` and must not run retrieval or call a model.
        matched: Name of the detection rule that fired, for tests, logs and review. ``None``
            when ``escalated`` is ``False``.
        response: The hard-coded emergency copy (Vietnamese, names hotline 114), returned
            byte-identical every time. ``None`` when ``escalated`` is ``False``.
    """

    model_config = ConfigDict(frozen=True)

    escalated: bool = Field(description="True when the caller must take the emergency path.")
    matched: str | None = Field(
        default=None, description="Name of the rule that fired; None when not escalated."
    )
    response: str | None = Field(
        default=None, description="Hard-coded emergency copy; None when not escalated."
    )


class GuardedAnswer(BaseModel, Generic[AnswerT]):
    """The outcome of running a pipeline behind the emergency check.

    Attributes:
        safety: The emergency verdict. Branch on ``safety.escalated``.
        answer: What the guarded pipeline returned. Always ``None`` on the emergency path,
            because the pipeline is never started there.
    """

    model_config = ConfigDict(frozen=True)

    safety: SafetyCheck
    answer: AnswerT | None = None
