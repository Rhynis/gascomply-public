"""Deterministic and model-based checks that keep cited clauses on-topic."""

import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from app.llm.base import LLMProvider
from app.schemas.ask import Citation

logger = logging.getLogger(__name__)

# Keep these alternatives byte-for-byte aligned with ingest/references.py. The app deliberately
# owns its copy so it does not import ingest at runtime; the bare NNN/YYYY alternative is added
# only after the complete document-number alternatives, never by weakening them.
_NUMBERED_DOC = r"\d{1,4}/\d{4}/[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)*"
_STANDARD_DOC = r"(?:TCVN|QCVN)\s+\d+(?::\d{4})?(?:/[A-ZĐ]+)?"
EXTERNAL_PATTERN = re.compile(rf"{_NUMBERED_DOC}|{_STANDARD_DOC}", re.IGNORECASE)

_LEGAL_NOUN = r"(?:Nghị định|Thông tư|Luật|Quyết định|Nghị quyết)"
_BARE_NUMBER = re.compile(rf"{_LEGAL_NOUN}\s+(?:số\s+)?(\d{{1,4}}/\d{{4}})", re.IGNORECASE)
"""Only a bare ``NNN/YYYY`` immediately preceded by a legal-instrument noun (optionally with
``số`` in between) counts as a document number. Without this, a plain date written
``NN/M/YYYY`` (e.g. ``ngày 01/7/2025``) has a ``\\d{1,4}/\\d{4}`` substring (``7/2025``) that
the old unanchored pattern read as a document number no clause contains, refusing a correct
answer that correctly cites the correct effective-date clause (reported against this branch:
see the PR's review thread). A noun immediately before the number is never part of a date, so
this cannot fire on one; ``EXTERNAL_PATTERN`` above (the full ``157/2022/NĐ-CP`` /
``QCVN 08:2019/BCT`` forms) is unchanged and still matches with or without a noun in front."""


_VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}},
    "required": ["supported"],
}


class _VerifierReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    supported: bool


@dataclass
class GroundingTrace:
    """Per-request grounding decisions collected for evaluation without changing AskResponse.

    Attributes:
        decisions: One ``(unit_id, supported)`` pair per citation the layer-2 verifier judged,
            in the order it judged them.
        verifier_latencies_ms: One entry per verifier call actually made (``llm.generate`` for
            the narrow per-citation question), so the eval harness can report the check's own
            added latency separately from the whole ``/ask`` request's (the PR review: the
            previously-reported number was total request time, generation included, not what
            the check itself costs).
        layer1_refusals: Incremented when the deterministic document-number guard (layer 1)
            refuses, so a run can report how often it fired -- previously invisible, since a
            layer-1 refusal short-circuits before the verifier (and this trace) ever runs
            otherwise.
        prompt_eval_count: The main generation call's own ``prompt_eval_count`` (issue #129),
            set by ``generate_answer`` right after that one call, before any later verifier
            call through the same provider can overwrite the value it was read from
            (``OllamaProvider.last_prompt_eval_count``). ``None`` for ``MockLLM`` and any
            provider that does not expose this -- there is no generic ``LLMProvider`` field
            for it, this is read by attribute, best-effort, for evaluation only.
        pre_quote_answer: The model's own answer text (issue #167), set by ``generate_answer``
            right before it appends issue #146's verbatim appendix quote -- after fabrication
            validation and the grounding check have already settled ``citations``, but before
            any text from the corpus itself is appended. ``eval.compare``'s own deciding-fact
            check (``eval.metrics.answer_check``) matches a gold row's patterns against this
            text, never the final, quote-appended ``AskResponse.answer`` -- a quoted clause can
            contain words the model itself never wrote (e.g. a threshold number), and crediting
            a match found only inside that quote would check the corpus, not the model.
            ``None`` when the request refused before an answer existed at all.
        retried: Whether ``generate_answer`` ran its one allowed retry (issue #202): the
            per-citation verifier rejected every citation of the first attempt, and at least
            one rejected id was a passage ``generate_answer`` had actually offered. ``False``
            for the ordinary, single-call path -- the great majority of requests -- and also
            false for a refusal that never reaches the retry at all (a layer-1 refusal, a
            missing answer, or a rejected id that was only a focus unit never offered as its
            own passage). The retry's own verifier decisions are appended to ``decisions`` like
            any other call's, never kept separate.
    """

    decisions: list[tuple[str, bool]] = field(default_factory=list)
    verifier_latencies_ms: list[float] = field(default_factory=list)
    layer1_refusals: int = 0
    prompt_eval_count: int | None = None
    pre_quote_answer: str | None = None
    retried: bool = False


def _canonical_number(value: str) -> str:
    """Normalize a legal number to the comparison key used by the guard."""
    return re.sub(r"\s+", "", value).upper()


def _base_number(value: str) -> str | None:
    """Return the bare NNN/YYYY prefix of a numbered document, when present."""
    match = re.match(r"(?P<base>\d{1,4}/\d{4})(?:/.*)?$", value)
    return None if match is None else match.group("base")


def legal_numbers(text: str) -> frozenset[str]:
    """Extract complete external numbers and noun-prefixed bare numbered-document forms."""
    found: set[str] = set()
    spans: list[tuple[int, int]] = []
    for match in EXTERNAL_PATTERN.finditer(text):
        found.add(_canonical_number(match.group(0)))
        spans.append(match.span())
    for match in _BARE_NUMBER.finditer(text):
        number_start = match.start(1)
        if not any(start <= number_start < end for start, end in spans):
            found.add(_canonical_number(match.group(1)))
    return frozenset(found)


_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+|\n+|,\s*(?=nhưng\b)", re.IGNORECASE)
"""Splits on sentence punctuation, and also on a comma immediately before "nhưng" ("but").

The extra split matters for a real phrasing the issue's own acceptance criteria require:
"Nghị định 87/2018/NĐ-CP ... vẫn còn hiệu lực, nhưng đã bị sửa đổi bởi Nghị định
17/2020/NĐ-CP." names two document numbers in what would otherwise be a single sentence --
the one actually claimed in force (87/2018), and one merely named as having amended it
(17/2020, never claimed to be anything). Without this split, the sentence-local check below
would wrongly test 17/2020 against ``document_statuses`` too (it has no row at all, since
only 87/2018 is seeded in that test), refusing an answer the issue requires to be kept. The
comma stays consumed, "nhưng" stays in the following clause (lookahead), matching how the
sentence-final punctuation above is kept."""
_IN_FORCE_CLAIM = re.compile(
    r"(?:còn|vẫn)(?:được)?ápdụng|(?:còn|vẫn|đang)(?:có)?hiệulực|chưahếthiệulực"
)
"""Matches "còn/vẫn (được) áp dụng", "còn/vẫn/đang (có) hiệu lực" and "chưa hết hiệu lực".

The middle alternative's ``(?:có)?`` is deliberately optional: "còn hiệu lực" and "vẫn còn
hiệu lực" are both ordinary Vietnamese for "still in force" with no "có" in them at all (the
issue's own required phrasing lists "còn hiệu lực" bare, never "còn có hiệu lực"), and an
earlier version of this pattern required the literal "có" between còn/vẫn/đang and hiệulực --
silently never matching either, verified directly against this exact pair of phrasings before
the fix."""
_NEGATED_STATUS_CLAIM = re.compile(
    r"khôngcòn(?:có)?hiệulực|khôngcònápdụng|đãhết(?:có)?hiệulực|đãbịthaythế"
)


def _same_document_number(left: str, right: str) -> bool:
    """Return whether complete or bare legal numbers identify the same document."""
    return left == right or (
        (left_base := _base_number(left)) is not None
        and (right_base := _base_number(right)) is not None
        and left_base == right_base
    )


def in_force_claims_without_in_force_document(
    answer: str, document_statuses: Mapping[str, str]
) -> frozenset[str]:
    """Return in-force claims the corpus cannot establish from document status.

    The check is sentence-local, so an unrelated stub number cannot affect a
    supported claim elsewhere in the answer. Whitespace is collapsed before
    phrase matching because source and model text can split Vietnamese words
    such as ``"hiệ u lực"``. Negative claims remain for layer 2, where repeal
    clauses can support them.
    """
    unverified: set[str] = set()
    for sentence in _SENTENCE_BREAK.split(answer):
        numbers = legal_numbers(sentence)
        if not numbers:
            continue
        compact = re.sub(r"\s+", "", sentence).casefold()
        if _NEGATED_STATUS_CLAIM.search(compact) or not _IN_FORCE_CLAIM.search(compact):
            continue
        for number in numbers:
            statuses = [
                status
                for document_number, status in document_statuses.items()
                if _same_document_number(number, document_number)
            ]
            if "in_force" not in statuses:
                unverified.add(number)
    return frozenset(unverified)


def ungrounded_numbers(
    answer: str, citations: tuple[Citation, ...], source_texts: dict[str, str]
) -> frozenset[str]:
    """Return every legal document number in ``answer`` that no citation grounds.

    A number is grounded when it is a cited unit's own document number, or appears in that
    unit's own text (``source_texts``) -- compared on the number itself (``_base_number``), not
    the exact string, so ``157/2022`` in the answer matches a citation's ``157/2022/NĐ-CP``.
    """
    answer_numbers = legal_numbers(answer)
    if not answer_numbers:
        return frozenset()
    ungrounded: set[str] = set()
    for number in answer_numbers:
        citation_numbers = {_canonical_number(citation.document_number) for citation in citations}
        citation_bases = {
            base
            for citation_number in citation_numbers
            if (base := _base_number(citation_number)) is not None
        }
        source_numbers = {
            source_number
            for citation in citations
            for source_number in legal_numbers(source_texts.get(citation.unit_id, ""))
        }
        source_bases = {
            base
            for source_number in source_numbers
            if (base := _base_number(source_number)) is not None
        }
        bare_number = number if re.fullmatch(r"\d{1,4}/\d{4}", number) else None
        if not (
            number in citation_numbers
            or (bare_number is not None and bare_number in citation_bases)
            or number in source_numbers
            or (bare_number is not None and bare_number in source_bases)
        ):
            ungrounded.add(number)
    return frozenset(ungrounded)


def document_numbers_are_grounded(
    answer: str, citations: tuple[Citation, ...], source_texts: dict[str, str]
) -> bool:
    """Return whether every legal document number in answer is supported by a citation."""
    return not ungrounded_numbers(answer, citations, source_texts)


@dataclass(frozen=True)
class SupportCheck:
    """What every per-citation verifier call judges a citation against.

    Bundled into one object so ``filter_supported_citations`` takes citations, the provider
    and the trace as its own separate arguments rather than six positional parameters (the
    PR review: this is what used to carry a ``# noqa: PLR0913``).
    """

    question: str
    answer: str
    source_texts: dict[str, str]


def _verifier_prompt(check: SupportCheck, clause_text: str) -> str:
    """Build the one-citation-at-a-time support question for the verifier model.

    Carries no test-only marker: the real model receives exactly the question it needs to
    answer, nothing added to let a test double recognise it (the PR review). ``MockLLM`` and
    every scripted test double instead key on the verifier's own ``_VERIFIER_SCHEMA`` --
    specifically, that it is the only schema in this codebase with a ``supported`` property.
    """
    return (
        "Câu hỏi của người dùng:\n"
        f"{check.question}\n\n"
        "Câu trả lời dự kiến:\n"
        f"{check.answer}\n\n"
        "Nội dung của đúng một điều khoản được trích dẫn:\n"
        f"{clause_text}\n\n"
        "Trả lời supported=true chỉ khi nội dung điều khoản nêu một điều mà câu trả lời khẳng định."
    )


async def filter_supported_citations(
    check: SupportCheck,
    citations: tuple[Citation, ...],
    llm: LLMProvider,
    trace: GroundingTrace | None = None,
) -> tuple[Citation, ...]:
    """Keep only citations whose own text supports the answer, preserving order."""
    kept: list[Citation] = []
    for citation in citations:
        start = time.perf_counter()
        raw = await llm.generate(
            system=(
                "Bạn là bộ kiểm tra grounding. Đánh giá duy nhất citation hiện tại, không suy "
                "diễn từ các điều khoản khác."
            ),
            user=_verifier_prompt(check, check.source_texts.get(citation.unit_id, "")),
            json_schema=_VERIFIER_SCHEMA,
        )
        if trace is not None:
            trace.verifier_latencies_ms.append((time.perf_counter() - start) * 1000)
        try:
            supported = _VerifierReply.model_validate_json(raw).supported
        except (ValidationError, ValueError):
            # Fail closed. A reply the verifier schema cannot parse -- a real provider
            # misbehaving (truncated output, a hosted model ignoring the JSON-schema format,
            # Ollama returning something else under load) -- is treated exactly like an
            # explicit "not supported", not trusted by default. This project's standing rule
            # is "when in doubt, refuse, don't guess" (#90, #93, #106, #113); a parse failure
            # is the one situation where nothing has actually verified the citation, so
            # keeping it silently would be the most dangerous of the two wrong answers.
            # There is no known offline double that still needs the old fail-open behavior:
            # MockLLM answers the verifier's schema with a well-formed {"supported": true}
            # reply (see app/llm/mock.py), so this branch is reached only by a genuinely
            # malformed reply from a real provider.
            logger.warning(
                "generator: could not parse the verifier's reply for %r -- treating as "
                "unsupported: %r",
                citation.unit_id,
                raw,
            )
            supported = False
        if trace is not None:
            trace.decisions.append((citation.unit_id, supported))
        if supported:
            kept.append(citation)
        else:
            logger.warning(
                "generator: dropped an unsupported citation -- %r does not support the answer",
                citation.unit_id,
            )
    return tuple(kept)
