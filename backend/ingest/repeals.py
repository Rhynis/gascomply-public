"""Read a document's repeal statements: which whole documents does its text repeal (issue #68).

A repeal is the highest-value edge in the corpus -- it is what answers "is NĐ
136/2020 still in force?" -- and the most dangerous to get wrong. ``REPLACED_BY``
makes the in-force filter drop the *old* document from every answer, so an edge
that claims a live decree was superseded silently removes it. This module
therefore recognises a small number of shapes it can prove, and for everything
else emits nothing and *counts* it:

* ``partial``: a repeal act whose object is not a whole document -- an article,
  a clause, a point, a chapter, a phrase ("Bãi bỏ khoản 1 và 2 Điều 30", "Bãi bỏ
  Chương V của Nghị định ..."). Never an edge.
* ``unclear``: a statement about a document that fits no recognised shape, or
  names a document more than once. Never an edge.

Direction is always the same: the text that *states* the repeal is the new
document, so the repealed document is the old one. ``ingest.graph`` builds
``old -> this document``.

The recognised shapes, each measured against the real corpus:

* **List.** A clause "Các <loại văn bản> sau đây hết hiệu lực từ ngày <văn bản>
  này ...:" and its child points. Each point's *own* document is the one that
  opens it. A point that describes the document it names -- "Thông tư số
  149/2020/TT-BCA ... quy định chi tiết ... Nghị định số 136/2020/NĐ-CP" -- also
  mentions other documents, and reading every number in it would repeal the
  decree its circular merely implements. (NĐ 105/2025 Điều 45, TT 36/2025
  Điều 14.)
* **Passive.** "<Văn bản> [đã được sửa đổi, bổ sung ... theo <những văn bản
  khác>] hết hiệu lực từ ngày <văn bản> này ...". The subject is repealed; the
  laws that amended it are named only to describe it and stay in force. (Luật
  PCCC 2024 Điều 54.)
* **Active.** "<Văn bản> này thay thế | bãi bỏ <văn bản>" or the imperative "Bãi bỏ
  <văn bản>", with exactly one document named. (NĐ 87/2018 Điều 61 khoản 2, whose
  next sentence -- "Bãi bỏ Chương V của ..." -- is a partial repeal in the same
  unit, so recognition is per *sentence*, never per unit.)

Every shape ties the expiry to the document being read ("từ ngày <văn bản> này")
or to the document doing the repealing, which is what makes ``old -> this
document`` a claim the text actually makes rather than a guess.

PDF artifacts are handled where the real corpus shows them. Inside a *number*
they are repaired once, at extraction (``ingest.extract``, issue #80), so the text
read here already carries whole numbers; the three cue phrases tolerate whitespace
between their letters ("bã i  bỏ" is printed in NĐ 105/2025 Điều 44); and nothing
else depends on words a PDF may have split
("s au đây", "hiệ u lực" both occur in the real corpus's own repeal clauses). A
noun split the same way ("Nghị đị nh") is not recognised as a document, which can
only ever skip an edge, never make one.
"""

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from app.schemas.unit import Unit
from ingest.parser import ParseResult
from ingest.references import EXTERNAL_PATTERN

_NOUN = (
    r"(?:Nghị định|Thông tư(?: liên tịch)?|Nghị quyết(?: liên tịch)?|Quyết định|Chỉ thị"
    r"|Bộ luật|Luật|Pháp lệnh|Quy chuẩn|Tiêu chuẩn)"
)
"""The kinds of document this corpus repeals. Case-sensitive on purpose: an
official title capitalises them, ordinary prose does not ("pháp luật"), and a
title's own text is what ``_TITLE_CHAR`` must not run through."""


def _cue(phrase: str) -> str:
    """Return a pattern for ``phrase`` that survives a PDF splitting its words.

    Optional whitespace between every letter: the real corpus prints "bã i  bỏ"
    (NĐ 105/2025 Điều 44) and "hiệ u lực", and a whole-document repeal whose cue
    was split would otherwise be lost silently -- not built, and not counted.
    """
    letters = [re.escape(letter) for letter in phrase if not letter.isspace()]
    return r"(?i:" + r"\s*".join(letters) + r")"


_BAI_BO = _cue("bãi bỏ")
_THAY_THE = _cue("thay thế")
_HET_HIEU_LUC = _cue("hết hiệu lực")

_TITLE_CHAR = rf"(?:(?!{_NOUN}\b)[^\d;:()])"
_DOC_REF = re.compile(
    rf"{_NOUN}\b(?:\s+{_TITLE_CHAR}{{1,120}}?)?\s+(?:số\s+)?(?=\d{{1,4}}/\d{{4}}/)"
)
"""A document named by its type, optionally its title, and its number: "Nghị
định số 19/2016/NĐ-CP", "Luật Phòng cháy và chữa cháy số 27/2001/QH10". The title
may not itself contain a document type, so "Nghị định về kinh doanh khí và Nghị
định số 12/2019/NĐ-CP" is not read as one document with a long title."""

_STARTS_WITH_DOCUMENT = re.compile(rf"(?:các\s+)?{_NOUN}\b|văn bản\b")

_ACTIVE = re.compile(
    rf"^(?:(?P<self>{_NOUN})\s+này\s+)?(?P<verb>{_BAI_BO}|{_THAY_THE})\b"
    r"(?:\s+(?:hoàn toàn|toàn bộ))?\s*(?P<object>.*)$"
)
_VERB = re.compile(
    rf"\b(?P<verb>{_BAI_BO}|{_THAY_THE})\b(?:\s+(?:hoàn toàn|toàn bộ))?\s*(?P<object>.*)$"
)
_EXPIRES_WITH_THIS_DOCUMENT = rf"{_HET_HIEU_LUC}\s+(?:kể\s+)?từ\s+ngày\s+{_NOUN}\s+này\b"
_EXPIRY = re.compile(rf"^(?P<subject>{_NOUN}\b.*?)\s+{_EXPIRES_WITH_THIS_DOCUMENT}(?P<rest>.*)$")
_LIST_LEAD_IN = re.compile(
    rf"^Các\s+(?:{_NOUN}|văn bản(?: quy phạm pháp luật)?)\b[^:]*?"
    rf"{_EXPIRES_WITH_THIS_DOCUMENT}(?P<rest>[^:]*):$"
)
_EXPIRES = re.compile(_HET_HIEU_LUC)
_SELF_SUBJECT = re.compile(rf"^{_NOUN}\s+này\b")
_AMENDED_BY = re.compile(r"(?i:sửa đổi|bổ sung)\b[^:;]*?\b(?:theo|bởi)\b")
_EXCEPTION = re.compile(r"\btrừ\b(?!\s+trường hợp)")
"""An exception after the object ("..., trừ Điều 5") makes a repeal partial. A
"trừ trường hợp quy định tại Điều 46" is a transitional caveat, not one: the
corpus's own repeal lists carry it and the repeal still covers the whole
document."""

_BULLETS = "-\u2013\u2022+*"


class _Kind(Enum):
    """What one sentence is."""

    NONE = "none"
    WHOLE = "whole"
    PARTIAL = "partial"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class _Outcome:
    """The reading of one sentence; ``number`` is set only for ``WHOLE``."""

    kind: _Kind
    number: str | None = None


_PARTIAL = _Outcome(_Kind.PARTIAL)
_UNCLEAR = _Outcome(_Kind.UNCLEAR)


@dataclass(frozen=True)
class Repeal:
    """One whole document a text repeals.

    Attributes:
        origin_id: The unit whose text names the repealed document -- the
            child point of a list, else the unit holding the sentence.
        number: The repealed document's number, whitespace repaired but not yet
            checked for completeness: ``ingest.graph`` decides whether it earns
            an edge.
        evidence: The statement, verbatim: the sentence, or a list's lead-in
            together with the point it names.
    """

    origin_id: str
    number: str
    evidence: str


@dataclass(frozen=True)
class RepealScan:
    """What one document's text says about repealing other documents.

    Attributes:
        repeals: Every whole-document repeal recognised, in reading order.
        partial: Repeal acts whose object is not a whole document, counted and
            never built into an edge.
        unclear: Statements about a document that fit no recognised shape,
            counted and never built into an edge.
        statement_units: The units carrying a statement that produced at least
            one repeal -- a list's lead-in as well as its points -- so the
            caller can tell the cue reference sitting there is now resolved.
    """

    repeals: tuple[Repeal, ...] = ()
    partial: int = 0
    unclear: int = 0
    statement_units: frozenset[str] = frozenset()


@dataclass
class _Tally:
    """Mutable accumulator behind ``scan_repeals``."""

    repeals: list[Repeal] = field(default_factory=list)
    statement_units: set[str] = field(default_factory=set)
    partial: int = 0
    unclear: int = 0

    def add(self, repeal: Repeal, statement_unit: str) -> None:
        """Record one recognised repeal and the unit whose statement produced it."""
        self.repeals.append(repeal)
        self.statement_units.update((statement_unit, repeal.origin_id))

    def record(self, outcome: _Outcome, origin_id: str, evidence: str) -> None:
        """Record one sentence's reading."""
        if outcome.kind is _Kind.WHOLE and outcome.number is not None:
            self.add(Repeal(origin_id, outcome.number, evidence), origin_id)
        elif outcome.kind is _Kind.PARTIAL:
            self.partial += 1
        elif outcome.kind is _Kind.UNCLEAR:
            self.unclear += 1


def scan_repeals(result: ParseResult) -> RepealScan:
    """Read every unit of a parsed document for the whole documents it repeals.

    Args:
        result: Output of ``parse_document``. Units only: a preamble is web-page
            furniture in the corpus's HTML documents (a "Văn bản bị thay thế"
            listing), not text the document states.

    Returns:
        The recognised repeals and the counts of what was seen and skipped.
    """
    children: dict[str, list[Unit]] = defaultdict(list)
    for unit in result.units:
        if unit.parent_id is not None:
            children[unit.parent_id].append(unit)

    tally = _Tally()
    consumed: set[str] = set()
    for unit in result.units:
        if unit.id in consumed:
            continue
        text = _unit_text(unit)
        points = children.get(unit.id, ())
        if _scan_list(unit, text, points, tally):
            consumed.update(point.id for point in points)
        else:
            _scan_sentences(unit, text, tally)
    return RepealScan(
        repeals=tuple(tally.repeals),
        partial=tally.partial,
        unclear=tally.unclear,
        statement_units=frozenset(tally.statement_units),
    )


def _unit_text(unit: Unit) -> str:
    """Return a unit's heading and body as one whitespace-collapsed text.

    A heading and the first line of a body are often one wrapped sentence
    ("Bãi bỏ một số điều, khoản của Nghị định số 116/2017/NĐ-CP ngày 17" /
    "tháng 10 năm 2017 của Chính phủ ..."), so they are read together.
    """
    return " ".join(f"{unit.heading or ''} {unit.body}".split())


def _scan_list(unit: Unit, text: str, points: Sequence[Unit], tally: _Tally) -> bool:
    """Read ``unit`` as a list lead-in over ``points``; return whether it is one.

    A lead-in that fails its own checks makes every point unclear rather than
    falling through to be read sentence by sentence, and a lead-in with no points
    beneath it names nothing at all.
    """
    lead_in = _LIST_LEAD_IN.match(text)
    if lead_in is None:
        return False
    if not points or EXTERNAL_PATTERN.search(lead_in["rest"]) or _EXCEPTION.search(lead_in["rest"]):
        tally.unclear += max(len(points), 1)
        return True
    for point in points:
        point_text = _unit_text(point)
        number = _document_number(point_text)
        if number is None:
            tally.unclear += 1
        else:
            tally.add(Repeal(point.id, number, f"{text} {point_text}"), unit.id)
    return True


def _scan_sentences(unit: Unit, text: str, tally: _Tally) -> None:
    """Read each sentence of a unit's text on its own."""
    for sentence in _sentences(text):
        tally.record(_classify(sentence), unit.id, sentence)


def _sentences(text: str) -> list[str]:
    """Split ``text`` into sentences at ``.`` or ``;`` followed by a capital or a bullet.

    A ``;`` before a lower-case word stays inside one sentence ("...; thời hạn
    sử dụng ..."): a repeal statement is never cut in the middle of a clause.
    """
    sentences: list[str] = []
    for piece in re.split(r"(?<=[.;])\s+", text):
        if sentences and not (piece[:1].isupper() or piece[:1] in _BULLETS):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _classify(sentence: str) -> _Outcome:
    """Read one sentence; the shapes are tried from most to least specific."""
    return _active(sentence) or _expiry(sentence) or _elsewhere(sentence) or _Outcome(_Kind.NONE)


def _verb(matched: str) -> str:
    """Return a matched cue verb lower-cased and with any PDF-split whitespace put back."""
    return "bãi bỏ" if "".join(matched.split()).lower() == "bãibỏ" else "thay thế"


def _document_number(text: str) -> str | None:
    """Return the number of the document ``text`` *opens with*, or ``None``."""
    reference = _DOC_REF.match(text)
    if reference is None:
        return None
    number = EXTERNAL_PATTERN.match(text, reference.end())
    return number.group(0) if number else None


def _active(text: str) -> _Outcome | None:
    """Read "<văn bản> này thay thế | bãi bỏ X" and the imperative "Bãi bỏ X"."""
    match = _ACTIVE.match(text)
    if match is None:
        return None
    verb = _verb(match["verb"])
    if verb == "thay thế" and match["self"] is None:
        return None
    number = _document_number(match["object"])
    if number is None:
        if _STARTS_WITH_DOCUMENT.match(match["object"]):
            return _UNCLEAR
        return _PARTIAL if verb == "bãi bỏ" else None
    named = {found.group(0) for found in EXTERNAL_PATTERN.finditer(match["object"])}
    if named != {number} or _EXCEPTION.search(match["object"]):
        return _UNCLEAR
    return _Outcome(_Kind.WHOLE, number)


def _expiry(text: str) -> _Outcome | None:
    """Read "<văn bản> [đã được sửa đổi ... theo <văn bản>] hết hiệu lực từ ngày ... này"."""
    match = _EXPIRY.match(text)
    if match is None:
        return None
    subject = match["subject"]
    if _SELF_SUBJECT.match(subject):
        return None
    number = _document_number(subject)
    if number is None or not _describes_the_subject(subject, number):
        return _UNCLEAR
    if EXTERNAL_PATTERN.search(match["rest"]) or _EXCEPTION.search(match["rest"]):
        return _UNCLEAR
    return _Outcome(_Kind.WHOLE, number)


def _describes_the_subject(subject: str, number: str) -> bool:
    """Whether every other number in ``subject`` follows an "amended by" qualifier.

    "Luật ... số 27/2001/QH10 đã được sửa đổi, bổ sung một số điều theo Luật số
    40/2013/QH13 và Luật số 30/2023/QH15": the later numbers are the laws that
    amended the subject. A second number anywhere else could be a second
    subject ("A và B hết hiệu lực"), which this module cannot tell apart from a
    description, so it is not read.
    """
    first = subject.find(number)
    later = list(EXTERNAL_PATTERN.finditer(subject, first + len(number)))
    if not later:
        return True
    qualifier = _AMENDED_BY.search(subject, first + len(number))
    return qualifier is not None and all(found.start() >= qualifier.end() for found in later)


def _elsewhere(text: str) -> _Outcome | None:
    """Read a repeal cue that is not where the recognised shapes need it.

    ``bãi bỏ`` is a repeal act wherever it stands: a document after it is a
    statement in a shape this module does not read (counted as unclear), anything
    else is a repeal of a part (counted as partial). ``thay thế`` is also an
    ordinary verb ("thay thế thiết bị"), so it is only counted when a numbered
    document follows it. ``hết hiệu lực`` is counted only for a sentence that
    opens with a numbered document -- a statement that document has expired, that
    is not tied to this one taking effect -- and not for "Giấy chứng nhận hết hiệu
    lực", a certificate expiring, which the real corpus also contains.
    """
    match = _VERB.search(text)
    if match is None:
        opens_with_a_document = _document_number(text) is not None
        return _UNCLEAR if opens_with_a_document and _EXPIRES.search(text) else None
    if _verb(match["verb"]) == "bãi bỏ":
        return _UNCLEAR if _STARTS_WITH_DOCUMENT.match(match["object"]) else _PARTIAL
    return _UNCLEAR if _document_number(match["object"]) else None
