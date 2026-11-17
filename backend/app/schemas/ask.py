"""Schemas for ``POST /ask`` (build plan A6 Phase 3.6-3.7, issue #90).

Three outcomes, never free prose alone (``AskOutcome``): a grounded answer with
structured citations, a refusal, or the hard-coded emergency response. A frontend
(Phase 5.1) branches on ``outcome`` and never has to parse ``answer`` to find a
citation -- ``citations`` is already unit id, Điều/Khoản, document number and
effective date, so each one can become a link on its own.
"""

import re
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AskRequest(BaseModel):
    """The body of ``POST /ask``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The question, in Vietnamese, exactly as the user typed it.",
    )


class AskOutcome(StrEnum):
    """Which shape an ``/ask`` response is -- what the frontend branches on."""

    ANSWER = "answer"
    """A grounded answer with at least one verified citation."""

    REFUSAL = "refusal"
    """Nothing survived cite-or-refuse: no supporting context, or every citation the
    model offered was fabricated. Never an apology in ``answer`` -- a first-class
    outcome with ``answer=None``."""

    EMERGENCY = "emergency"
    """``app.services.safety`` escalated before retrieval or generation ran at all."""


class Citation(BaseModel):
    """One citation backing a claim in the answer, as structured data (issue #90).

    Every ``Citation`` on a response has already been checked against the retrieved
    context: a citation the model gave that named a unit not actually retrieved is
    dropped before it ever reaches this schema, never included and flagged --
    dropped, silently to the caller, loudly to the log (``app.services.generator``).

    Attributes:
        unit_id: The exact unit id the answer relies on (A5.1 scheme).
        document_id: Owning document.
        document_number: Official document number, e.g. ``"87/2018/NĐ-CP"``.
        dieu_no: Article number.
        khoan_no: Clause number, set only when the citation is to a clause rather
            than a whole article.
        effective: Date the cited document took effect.
    """

    model_config = ConfigDict(frozen=True)

    unit_id: str
    document_id: str
    document_number: str
    dieu_no: int | None = None
    khoan_no: int | None = None
    appendix_no: int | None = None
    item_no: int | None = None
    effective: date | None = None

    @model_validator(mode="after")
    def _derive_appendix_coordinates(self) -> "Citation":
        """Derive appendix coordinates from the canonical unit id when applicable."""
        appendix = re.search(r"#phu-luc-(\d+)", self.unit_id)
        item = re.search(r"#muc-(\d+)", self.unit_id)
        if appendix is not None:
            object.__setattr__(self, "appendix_no", int(appendix.group(1)))
            object.__setattr__(self, "item_no", int(item.group(1)) if item else None)
            object.__setattr__(self, "dieu_no", None)
            object.__setattr__(self, "khoan_no", None)
        return self


class AskResponse(BaseModel):
    """What ``POST /ask`` returns.

    Attributes:
        outcome: Which of the three shapes this is; the frontend's first branch.
        answer: The generated answer text. ``None`` for a refusal, and for an
            emergency -- an emergency's own fixed text is ``safety_response``, never
            this field, since it is not model output and must never be confused
            with one.
        citations: Structured citations backing ``answer``, best first. Always
            empty for a refusal or an emergency.
        safety_response: The hard-coded emergency response (names hotline 114), set
            only when ``outcome`` is ``emergency``.
        disclaimer: The fixed disclaimer, set only when ``outcome`` is ``answer`` --
            a refusal gives no information to disclaim, and the emergency response
            carries its own instructions instead.
        possibly_superseded: ``True`` when a cited unit's document carries a
            ``REPLACED_BY`` edge that is still pending review (issue #75's flag,
            passed through by the retriever) -- shown so a reader can double check
            before relying on it. Always ``False`` off the ``answer`` outcome.
        refusal_reason: A short, machine-readable reason code (never localized
            prose -- a frontend owns how it is shown), set only when ``outcome`` is
            ``refusal``: ``"no_context"`` (nothing retrieved), ``"model_declined"``,
            ``"no_citations_given"``, ``"no_valid_citations"`` (every citation
            offered was fabricated), ``"unsupported_citations"`` (none of the cited
            clauses supported the answer), ``"status_unknown"`` (the answer claimed a
            document was in force although the corpus has no matching ``in_force``
            document), ``"invalid_model_output"`` (the reply
            could not be parsed against the expected schema), ``"context_overflow"``
            (issue #129: the provider's own ``prompt_eval_count`` showed the prompt was
            truncated, or left too little of the context window for the reply to be
            trusted), ``"llm_unavailable"`` (issue #129: the provider timed out or
            could not be reached for this call), or ``"verdict_contradicts_clause"`` (issue
            #195: a deterministic check found the answer's own opening "Có"/"Không" contradicts
            the threshold its own kept citation's text states).
    """

    model_config = ConfigDict(frozen=True)

    outcome: AskOutcome
    answer: str | None = None
    citations: tuple[Citation, ...] = ()
    safety_response: str | None = None
    disclaimer: str | None = None
    possibly_superseded: bool = False
    refusal_reason: str | None = None
