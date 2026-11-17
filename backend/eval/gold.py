"""Load and validate the gold evaluation set (build plan A6 Phase 4.1, issue #95).

``data/gold.yaml`` is the ground truth every later evaluation number is computed
against: recall@k, citation accuracy, and the GraphRAG-vs-naive comparison table
(Phase 4.2-4.3) all read it through this module. This module is deliberately a
loader/schema, not a metrics module -- it turns the YAML file into typed,
internally-consistent rows and nothing more. Computing recall@k or the
GraphRAG-vs-naive table is out of scope here (Phase 4.2/4.3, issue #96).

Row schema
----------
Each row of ``data/gold.yaml`` is one question, with the fields below. See
``GoldRow`` for the authoritative field-by-field description; this is the
narrative version.

- ``id``: a unique, stable slug for the row (e.g. ``compliance-min-store-area``),
  so a later phase can refer to one question without depending on its position
  in the file.
- ``category``: one of ``compliance``, ``multi_hop``, ``refusal``, ``emergency``
  (see ``GoldCategory``). Classifies the row's primary purpose; ``multi_hop`` and
  ``expected_refusal`` below are the machine-checkable flags a metrics module
  actually filters on, kept separate from -- but constrained to agree with --
  the category.
- ``question``: the question exactly as a shop owner would type it, in
  Vietnamese (C8 allows Vietnamese in eval content). Not a paraphrase of a
  clause heading.
- ``reference_answer``: the correct answer, in Vietnamese, verified against the
  real clause text in the database (never from memory or a web summary -- see
  the PR for the per-row verification evidence). For an ``emergency`` row this
  is byte-identical to ``app.services.safety.EMERGENCY_RESPONSE_VI``, since
  that path never calls generation; for a ``refusal`` row it is the refusal a
  correct system should give, not a fabricated citation.
- ``source_unit_ids``: the ``units.id`` values the answer is grounded in.
  Non-empty for ``compliance``/``multi_hop`` rows (at least one, at least two
  for ``multi_hop``); always empty for ``refusal``/``emergency`` rows, which by
  definition cite nothing.
- ``multi_hop``: ``True`` when answering requires combining more than one
  source unit (e.g. two clauses reached through the requirement cluster's
  graph, or a document-level edge plus a clause). Only ``category: multi_hop``
  rows set this ``True``.
- ``expected_refusal``: ``True`` when a correct system must refuse -- the
  question is out of corpus, or answerable only from a document whose text was
  never ingested (``acquired: false`` in the manifest). Only
  ``category: refusal`` rows set this ``True``.
- ``answer_must_match``/``answer_must_not_match`` (issue #167): optional lists of regular
  expressions checking the *deciding fact* of a generated answer, not just which unit it
  cited -- a wrong conclusion ("yes" instead of "no, below the threshold") can still name a
  gold unit and score full citation precision, which these patterns catch instead. Compiled
  with ``re.IGNORECASE`` at load (an invalid pattern fails ``load_gold_set`` immediately);
  never set on a ``refusal`` row, which by definition has no "deciding fact" an answer states.

``load_gold_set`` is what a caller (a test, or a later eval phase) actually
calls; it reads the YAML file, validates every row through ``GoldRow``, and
checks the whole-file invariant ``GoldRow`` cannot check on its own (id
uniqueness across rows).
"""

import re
import unicodedata
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "gold.yaml"
"""Default location of the gold set, mirroring ``ingest.run.REQUIREMENTS_PATH``'s
own ``Path(__file__)``-relative resolution -- stable regardless of the caller's
current working directory."""

_UNIT_ID_PATTERN = re.compile(
    r"^[a-z][a-z0-9-]*"  # document id
    r"(#dieu-[1-9]\d*(#khoan-[1-9]\d*(#diem-[a-zđ]{1,2})?)?"  # dieu/khoan/diem tail
    r"|#phu-luc-[1-9]\d*(#muc-[1-9]\d*(#diem-[a-zđ]{1,2})?"  # numeric item/diem (#140)
    r"|#muc-[ivxlcdm]+(#khoan-[1-9]\d*(#diem-[a-zđ]{1,2})?)?"  # Roman section (#147/#160)
    r")?"
    r")?$"
)
"""A loose structural check on a unit id's shape (A5.1), independent of the
document/article/khoan/point prefixes' construction logic already owned by
``app.core.ids`` -- this checks *shape* only ("looks like a unit id"), so it
never drifts out of sync with that module by duplicating it. Whether an id
actually resolves to a row in the database is what ``tests/test_gold.py``
checks against a migrated corpus; that check needs a live database and this
one deliberately does not.

The Roman-section branch (``#muc-[ivxlcdm]+``) covers Phụ lục VI's own
"Mục I"/"Mục II" prose (``appendix_section``/``appendix_section_clause``/
``appendix_section_point``, issue #147's fix round on PR #151) -- distinct
from a numeric table row's own ``#muc-<n>`` tail just above it, which stays
numeric-only."""

_ID_SLUG_PATTERN = r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$"
"""Shape of a gold row's own ``id`` -- a plain kebab-case slug, unrelated to
the corpus unit-id scheme above."""


class GoldCategory(StrEnum):
    """What a gold row is testing.

    Attributes:
        COMPLIANCE: A single-source compliance question -- the answer lives in
            one source unit.
        MULTI_HOP: The answer requires combining at least two source units,
            reached through a real graph relationship in this corpus (the
            requirement cluster's ``REFERENCES``/``APPLIES_TO`` edges, or a
            document-level edge like ``AMENDED_BY``).
        REFUSAL: A question a correct system must refuse to answer -- out of
            corpus, or answerable only from a document whose text was never
            ingested.
        EMERGENCY: One of the emergency phrasings ``app.services.safety``
            detects before any retrieval or LLM call (issue #91), included so
            a later phase can measure safety recall from this same file.
    """

    COMPLIANCE = "compliance"
    MULTI_HOP = "multi_hop"
    REFUSAL = "refusal"
    EMERGENCY = "emergency"


class GoldRow(BaseModel):
    """One verified question of the gold evaluation set.

    See this module's own docstring for the narrative description of every
    field; this is the schema ``data/gold.yaml`` is validated against.

    Attributes:
        id: Unique, stable slug identifying this row.
        category: What this row is testing; see ``GoldCategory``.
        question: The question in Vietnamese, phrased the way a shop owner
            would actually type it.
        reference_answer: The verified correct answer, in Vietnamese.
        source_unit_ids: ``units.id`` values the answer is grounded in. Empty
            for ``refusal``/``emergency`` rows.
        multi_hop: Whether answering needs more than one source unit.
        expected_refusal: Whether a correct system must refuse this question.
        answer_must_match: Optional patterns (issue #167) every one of which must match
            somewhere in a generated answer's own text for this row to pass the deciding-fact
            check (``eval.metrics.answer_check``) -- e.g. the numeric threshold a correct
            answer must actually state, not just gesture at. Compiled with ``re.IGNORECASE``
            at load; empty for a row with nothing to check. Never set on a ``refusal`` row.
        answer_must_not_match: The mirror of ``answer_must_match``: none of these may match --
            e.g. the wrong number a known failure mode states instead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=_ID_SLUG_PATTERN)
    category: GoldCategory
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    source_unit_ids: tuple[str, ...] = Field(default_factory=tuple)
    multi_hop: bool = False
    expected_refusal: bool = False
    answer_must_match: tuple[re.Pattern[str], ...] = Field(default_factory=tuple)
    answer_must_not_match: tuple[re.Pattern[str], ...] = Field(default_factory=tuple)

    @field_validator("answer_must_match", "answer_must_not_match", mode="before")
    @classmethod
    def _compile_answer_patterns(
        cls, value: object, info: ValidationInfo
    ) -> tuple[re.Pattern[str], ...]:
        """Compile the YAML's raw pattern strings into case-insensitive regexes (issue #167).

        Compiled at load, not lazily at check time, so a malformed pattern fails
        ``load_gold_set`` immediately -- never surfaces only when a later eval run happens to
        reach that row. Each pattern is NFC-normalised before compiling (``eval.metrics.
        answer_check`` normalises the answer text the same way before matching, so composed
        and decomposed Vietnamese diacritics never silently fail to match each other).
        """
        if value is None:
            return ()
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError(
                f"{info.data.get('id', '<unknown>')}: answer_must_match/answer_must_not_match "
                f"must be a list of regex strings, got {type(value).__name__}"
            )
        row_id = info.data.get("id", "<unknown>")
        compiled: list[re.Pattern[str]] = []
        for pattern in value:
            if not isinstance(pattern, str):
                raise ValueError(
                    f"{row_id}: pattern must be a string, got {type(pattern).__name__}"
                )
            normalized = unicodedata.normalize("NFC", pattern)
            try:
                compiled.append(re.compile(normalized, re.IGNORECASE))
            except re.error as error:
                raise ValueError(
                    f"{row_id}: invalid answer pattern {pattern!r}: {error}"
                ) from error
        return tuple(compiled)

    @model_validator(mode="after")
    def _check_answer_patterns_not_on_a_refusal(self) -> "GoldRow":
        """Reject an ``answer_must_match``/``answer_must_not_match`` pattern on a refusal row.

        A ``refusal`` row's ``reference_answer`` is the refusal text itself, never an answer a
        real ``/ask`` call would generate -- ``generate_answer`` never even reaches the
        deciding-fact text for this category, so a pattern here could never mean anything
        (issue #167's own acceptance criterion).
        """
        if self.expected_refusal and (self.answer_must_match or self.answer_must_not_match):
            raise ValueError(
                f"{self.id}: a refusal row cannot set answer_must_match/answer_must_not_match"
            )
        return self

    @model_validator(mode="after")
    def _check_source_unit_ids(self) -> "GoldRow":
        """Reject a malformed or duplicated unit id before any DB check runs.

        Raises:
            ValueError: If ``source_unit_ids`` contains a duplicate, or an id
                that does not look like a unit id.
        """
        if len(set(self.source_unit_ids)) != len(self.source_unit_ids):
            raise ValueError(f"{self.id}: source_unit_ids contains a duplicate")
        for unit_id in self.source_unit_ids:
            if not _UNIT_ID_PATTERN.match(unit_id):
                raise ValueError(f"{self.id}: {unit_id!r} does not look like a unit id (A5.1)")
        return self

    @model_validator(mode="after")
    def _check_category_agrees_with_flags(self) -> "GoldRow":
        """Enforce the fixed relationship between ``category`` and the other fields.

        Raises:
            ValueError: If ``multi_hop``, ``expected_refusal`` or
                ``source_unit_ids`` disagree with what ``category`` requires.
        """
        checks: dict[GoldCategory, list[tuple[bool, str]]] = {
            GoldCategory.MULTI_HOP: [
                (self.multi_hop, "category 'multi_hop' requires multi_hop=True"),
                (not self.expected_refusal, "category 'multi_hop' cannot be a refusal"),
                (len(self.source_unit_ids) >= 2, "category 'multi_hop' needs >= 2 source_unit_ids"),
            ],
            GoldCategory.COMPLIANCE: [
                (not self.multi_hop, "category 'compliance' cannot set multi_hop"),
                (not self.expected_refusal, "category 'compliance' cannot be a refusal"),
                (
                    len(self.source_unit_ids) >= 1,
                    "category 'compliance' needs >= 1 source_unit_ids",
                ),
            ],
            GoldCategory.REFUSAL: [
                (self.expected_refusal, "category 'refusal' requires expected_refusal=True"),
                (not self.multi_hop, "category 'refusal' cannot set multi_hop"),
                (not self.source_unit_ids, "category 'refusal' must cite no source units"),
            ],
            GoldCategory.EMERGENCY: [
                (not self.expected_refusal, "category 'emergency' is not a refusal"),
                (not self.multi_hop, "category 'emergency' cannot set multi_hop"),
                (
                    not self.source_unit_ids,
                    "category 'emergency' must cite no source units -- the response is hard-coded",
                ),
            ],
        }
        for condition, message in checks[self.category]:
            if not condition:
                raise ValueError(f"{self.id}: {message}")
        return self


def load_gold_set(path: Path = GOLD_PATH) -> tuple[GoldRow, ...]:
    """Load, validate and return the gold question set.

    Args:
        path: Location of the YAML file. Defaults to the committed
            ``data/gold.yaml``.

    Returns:
        Every row, in file order, each validated through ``GoldRow``.

    Raises:
        ValueError: If the file's top level is not a list, if any row fails
            ``GoldRow`` validation, or if two rows share an ``id``.
    """
    return parse_gold_set(path.read_text(encoding="utf-8"), str(path))


def parse_gold_set(text: str, source: str) -> tuple[GoldRow, ...]:
    """Parse and validate gold-set YAML held in memory (issue #178).

    The file-less core of ``load_gold_set``: ``eval.holdout`` decrypts the holdout straight
    into a string, and this lets it be parsed without the plaintext ever touching a disk.

    Args:
        text: The YAML document.
        source: What to call the document in an error message (a path, for ``load_gold_set``).

    Returns:
        Every row, in document order, each validated through ``GoldRow``.

    Raises:
        ValueError: If the top level is not a list, if any row fails ``GoldRow`` validation,
            or if two rows share an ``id``. A YAML syntax error, or a ``GoldRow`` error,
            quotes the offending line or row id; a caller handling a secret set must not
            print the message.
    """
    raw = yaml.safe_load(text)
    if not isinstance(raw, list):
        raise ValueError(
            f"{source}: expected a top-level YAML list of rows, got {type(raw).__name__}"
        )
    rows = tuple(_parse_row(entry) for entry in raw)
    _check_ids_are_unique(rows)
    return rows


def _parse_row(entry: object) -> GoldRow:
    """Parse one YAML mapping into a ``GoldRow``, raising a clear error on the wrong shape."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"expected a mapping for each gold row, got {type(entry).__name__}: {entry!r}"
        )
    return GoldRow.model_validate(entry)


def _check_ids_are_unique(rows: Sequence[GoldRow]) -> None:
    """Raise ``ValueError`` if any two rows share an ``id``."""
    seen: set[str] = set()
    for row in rows:
        if row.id in seen:
            raise ValueError(f"duplicate gold row id: {row.id!r}")
        seen.add(row.id)
