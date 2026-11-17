"""Detect in-text cross-references by regex (build plan A6 Phase 1.4).

Every "Căn cứ ...", "quy định tại Điều ...", amendment cue, and document-number
citation in a parsed document is written as one draft ``Reference`` row, for a
human and the graph to review in Phase 2.2. Extraction is pure regex -- no
LLM, no resolution to a real identifier -- so a run is deterministic and
offline-testable; ambiguity in ``target_hint`` is expected output here, not a
bug.

Recall matters more than precision: a wrong row costs a reviewer one click in
Phase 2.3, a missed row silently drops an edge from the graph and nobody ever
notices. A wrong ``kind`` is worse than a miss -- it still emits a row, so
Phase 2.2 would build the wrong edge type without anyone noticing.

Patterns encode Vietnamese legal drafting convention (smallest structural unit
cited first, e.g. "điểm a khoản 1 Điều 38"), not measured against a real
statute -- ``corpus/`` is still empty at the time of writing. They deserve a
second, evidence-based pass once real documents are available.

A "quy định tại ..." unit reference can also name another document by its
exact title instead of by number ("...Điều 17 Luật Phòng cháy, chữa cháy và
cứu nạn, cứu hộ..."), with no "của" at all -- a real, high-frequency
Vietnamese drafting convention a first pass over the corpus missed entirely
(issue #71: every one of #69's 16 "internal" misses was this). Recognizing it
is still pure regex, not a step away from "no resolution to a real
identifier" above: ``extract_references`` takes the manifest's own closed set
of document titles as input and only ever matches one of them verbatim, never
an invented "what a legal title looks like" pattern -- a title naming no
manifest document is not a citation this module can see.

Issue #121 closes the gap #71 left open: "của" was the only way a bare
document number or a legal-instrument noun with no exact title match could
ever mark a unit reference external, so "Điều 17 Nghị định số
136/2020/NĐ-CP" (no "của") and "Điều 16 Luật Phòng cháy, chữa cháy và cứ u
nạn, cứu hộ" (an exact title match, just split by a PDF extraction artifact)
both fell through to ``_classify_unit_reference``'s internal default --
turning a citation of someone else's article into a wrong, real-looking edge
into this document's own. "của" is now optional in front of every suffix
shape, a bare legal-instrument noun not followed by "này" is external even
when nothing else about it resolves, and title matching tolerates a stray
space PDF extraction can insert mid-word -- all three paths carry their unit
chain forward too, so ``ingest/graph.py`` can resolve into the target
document's own unit, not just the document itself.

Issue #130 adds a second unit-chain shape alongside "Điều/Khoản/Điểm/Chương":
an appendix citation ("Phụ lục VII", "Mục 1 Phụ lục V", "Mẫu số 02 Phụ lục
VI"). Measured on the real corpus, these never follow "quy định tại" alone
the way an article citation does -- "thuộc Phụ lục III kèm theo Nghị định
này", "thuộc nhóm 1 Phụ lục II kèm theo Nghị định số 105/2025/NĐ-CP", "theo
Mẫu số 02 Phụ lục VI kèm theo Thông tư này", and the repeal clause "của Phụ
lục I kèm theo Nghị định số 113/2017/NĐ-CP" (gas-decree-17-2020) are all real
sentences in the corpus, each introduced by a different cue word. The
trailing document mention is marked by "kèm theo" (an appendix is always
"issued together with" its document), not by "của" the way an article
citation is -- except the one real exception above, where "của" sits
*between* the appendix and its document directly, so both connectors are
accepted. The same four suffix outcomes apply (self-reference, a cited
document number, an exact manifest title, an unresolved legal-instrument
noun), reusing ``_unit_suffix_pattern`` with its connector made a parameter.
A form citation ("Mẫu số 02 Phụ lục VI") names the appendix, never a specific
item inside it -- the form number is matched and discarded, same as a group
number ("nhóm 1") is -- so it resolves exactly like a bare "Phụ lục VI"
citation. See ``ingest/graph.py``'s ``_APPENDIX_HINT`` for how the resulting
target_hint becomes an appendix or appendix-item id.
"""

import csv
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from app.schemas.reference import Reference, ReferenceKind
from app.schemas.unit import Unit
from ingest.parser import LEGAL_INSTRUMENT_NOUN, ParseResult

_NUMBERED_DOC = r"\d{1,4}/\d{4}/[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)*"
_STANDARD_DOC = r"(?:TCVN|QCVN)\s+\d+(?::\d{4})?(?:/[A-ZĐ]+)?"
_SELF_REFERENCE = r"(?:Nghị định|Luật|Thông tư|Quy chuẩn|Tiêu chuẩn)\s+này"

# A basis clause ends at ";", ".", or end of line -- whichever comes first.
# "[^;.\n]" also keeps the match from ever crossing a line, so a missing
# terminator can no longer swallow the following lines into one raw_text.
# The lookahead requires what follows "Căn cứ [vào]?" to actually name a
# legal instrument or a bare document/standard number; it consumes nothing,
# so the match (and therefore raw_text) still starts at "Căn cứ" itself.
# LEGAL_INSTRUMENT_NOUN (build plan A6 issue #9) is what gates a match here
# rather than a document-number requirement -- see its docstring in
# ingest/parser.py, its home since issue #56 also needs it there.
BASIS_PATTERN = re.compile(
    rf"Căn cứ\b(?=\s*(?:vào\s+)?(?:{LEGAL_INSTRUMENT_NOUN}|{_NUMBERED_DOC}|{_STANDARD_DOC}))"
    r"[^;.\n]*[;.]?"
)

AMENDMENT_PATTERN = re.compile(
    r"(?:được sửa đổi(?:,\s*bổ sung)?\s+bởi|được bổ sung bởi|được thay thế bởi"
    r"|thay thế bởi|bãi bỏ|hết hiệu lực)"
    rf"(?:\s+(?:Nghị định|Luật|Thông tư)?\s*(?:số\s+)?(?:{_NUMBERED_DOC}))?"
)

# Vietnamese drafting cites the smallest unit first ("điểm a khoản 1 Điều
# 38"); "Điều 38 Khoản 1 Điểm a" also occurs. Both orders are accepted and
# normalized to the same target_hint. Every component is optional except the
# article itself; a chapter-level citation ("Chương II") is also internal.
_UNIT_ORDER_SMALLEST_FIRST = (
    r"(?:điểm\s+(?P<diem_a>[a-zđ])\)?\s*,?\s*)?"
    r"(?:khoản\s+(?P<khoan_a>\d+)\s*,?\s*)?"
    r"Điều\s+(?P<dieu_a>\d+)"
)
_UNIT_ORDER_LARGEST_FIRST = (
    r"Điều\s+(?P<dieu_b>\d+)"
    r"(?:\s*,?\s*khoản\s+(?P<khoan_b>\d+))?"
    r"(?:\s*,?\s*điểm\s+(?P<diem_b>[a-zđ])\)?)?"
)
_CHAPTER_REFERENCE = r"Chương\s+(?P<chuong>[IVXLCDM]+|\d+)"

# Appendix citations (issue #130). Measured on the real corpus, the word
# immediately before "Phụ lục" varies far more than it does for an article
# citation -- "quy định tại", "thuộc", "theo" (before a form number), "của"
# (the one repeal-clause case where "của" sits between the appendix and its
# own trailing "kèm theo <document>", gas-decree-17-2020#dieu-9#khoan-3), or
# a bare "tại" -- so all four are accepted as the leading cue, rather than
# hard-coding "quy định tại" the way ``_UNIT_ORDER_SMALLEST_FIRST``/
# ``_UNIT_ORDER_LARGEST_FIRST`` do. ``_APPENDIX_FILLER`` absorbs the one
# optional token real drafting inserts between the cue and "Phụ lục": a
# group number ("thuộc nhóm 1 Phụ lục II") or a form number ("theo Mẫu số
# 02 Phụ lục VI") -- both discarded, never carried into target_hint, since
# neither names an item inside the appendix (a form citation names the
# appendix itself, issue #130's own scope note; a group number is metadata
# about facilities, not a sub-unit). ``_APPENDIX_LOCATION`` covers the one
# measured shape with *two* cue words: gas-decree-87-2018's real "... theo
# Mẫu số 01 tại Phụ lục kèm theo Nghị định này" names the form with "theo"
# and the appendix itself (bare, no number -- the document has only one)
# with a second, optional "tại" right before "Phụ lục".
_APPENDIX_CUE = r"(?:quy định tại|thuộc|theo|của|tại)"
_APPENDIX_FILLER = r"(?:(?:nhóm\s+\d+|Mẫu\s+số\s+\S+)\s+)?"
_APPENDIX_LOCATION = r"(?:tại\s+)?"
_APPENDIX_ITEM_PREFIX = r"(?:[Mm]ục\s+(?P<muc_phu_luc>\d+)\s+)?"
_PHU_LUC_REFERENCE = r"Phụ lục(?:\s+(?P<phu_luc>[IVXLCDM]+|\d+))?"
_UNIT_ORDER_APPENDIX = (
    rf"{_APPENDIX_CUE}\s+{_APPENDIX_FILLER}{_APPENDIX_LOCATION}"
    rf"{_APPENDIX_ITEM_PREFIX}{_PHU_LUC_REFERENCE}"
)

# A trailing "của <X>" decides whether the unit chain is internal or external:
# a self-reference ("của Nghị định này") keeps it internal; a document/
# standard number ("của Nghị định số 87/2018/NĐ-CP") makes it external and is
# consumed here so the plain EXTERNAL_PATTERN pass below does not also emit a
# bare row for the same number. No "của <X>" at all defaults to internal --
# an unqualified article reference inside a document is presumed to point at
# that same document; Phase 2.2's human review catches it if that is wrong.
#
# "của" is optional here (issue #121): real drafting cites another
# document's number directly after the unit chain just as often as it does
# by exact title (see _unit_suffix_pattern) -- "Điều 17 Nghị định số
# 136/2020/NĐ-CP" names decree 136/2020 with no "của" anywhere. The noun
# alternation also grows by the three instruments #121 measured on the dev
# database citing this way and missing from the original set: Quyết định,
# Nghị quyết, Bộ luật. The noun stays optional either way -- the number
# alone is still what makes a match here, same as before #121.
_DOC_MENTION = (
    rf"(?:(?:Nghị định|Luật|Thông tư|Quy chuẩn|Tiêu chuẩn|Quyết định|Nghị quyết|Bộ luật)\s*)?"
    rf"(?:số\s+)?(?:{_NUMBERED_DOC}|{_STANDARD_DOC})"
)

_NEVER_MATCHES = r"(?!)"
"""A pattern that never matches anything -- not the same as an empty
alternative, which would match a zero-width string everywhere. Used as the
title alternation's fallback when no manifest titles are given (issue #71),
so the optional ``title_ref`` group it sits in simply never fires rather than
matching empty text at every position."""


def _normalize_for_title_match(text: str) -> str:
    """Return ``text`` with every run of whitespace removed.

    Used on both sides of a title comparison (issue #121): PDF extraction
    can insert a stray space mid-word ("cứu" printed as "cứ u"), which an
    exact, space-for-space title match cannot see past even with
    ``re.IGNORECASE``. Comparing with all whitespace stripped is what
    ``_tolerant_title_pattern`` matches for; this is the same normalization
    applied to a captured match before it is looked up against a known
    title's own normalized form.
    """
    return re.sub(r"\s+", "", text)


def _tolerant_title_pattern(title: str) -> str:
    r"""Return ``title`` as a regex of its exact characters, whitespace tolerated between them.

    Still every character of the title, in the same order -- this is no
    looser a match at identifying which document is named, only looser
    about the stray mid-word space a PDF can leave between two of them
    (issue #121; see ``_normalize_for_title_match``). A real space the
    title itself calls for (between two words) matches too: the literal
    character is still required, ``\s*`` only adds tolerance around it.
    """
    return r"\s*".join(re.escape(char) for char in title)


def _title_alternation(document_titles: Iterable[str]) -> str:
    """Return a regex alternation matching any of ``document_titles``, whitespace-tolerant.

    Built fresh per call from the manifest's own closed set of titles --
    never a hand-written "what a legal title looks like" pattern (issue
    #71) -- so a title naming no manifest document can never match. Titles
    are tried longest first, so one that happens to be a textual prefix of
    another (none are, in the corpus today, but nothing guarantees that)
    can never shadow the more specific match.
    """
    titles = sorted({title for title in document_titles if title}, key=len, reverse=True)
    if not titles:
        return _NEVER_MATCHES
    return "|".join(_tolerant_title_pattern(title) for title in titles)


def _unit_suffix_pattern(document_titles: Iterable[str], *, connector: str = r"của") -> str:
    """Build the trailing-document suffix that follows a unit chain (issue #71, #121, #130).

    One optional suffix, with four alternatives tried in this order at the
    same position -- ``connector`` is optional before every one of them
    (issue #121: real drafting cites another document just as often with no
    "của" at all, by number ("Điều 17 Nghị định số 136/2020/NĐ-CP") or by
    title ("Điều 17 Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ")).
    ``connector`` defaults to the bare word an article/chapter citation uses
    ("của"); an appendix citation (issue #130) passes ``_APPENDIX_CONNECTOR``
    instead -- an appendix's trailing document mention is always marked by
    "kèm theo" ("issued together with") or "ban hành kèm theo" ("issued and
    promulgated together with"), never "của", even in the one real sentence
    where "của" also appears ("của Phụ lục I kèm theo Nghị định số
    113/2017/NĐ-CP", gas-decree-17-2020): there "của" sits *before* "Phụ
    lục", consumed by ``_APPENDIX_CUE`` below, not by this suffix at all:

    1. ``self_ref``: a self-reference ("Nghị định này") -- internal, same
       as before #121.
    2. ``ext_doc``: a document/standard number, with or without a leading
       legal-instrument noun -- see ``_DOC_MENTION``. External.
    3. ``title_ref``: the manifest's own closed set of document titles,
       whitespace-tolerant (issue #121's own measured case: an exact title
       the PDF split with a stray mid-word space). External.
    4. ``ext_noun`` (issue #121): none of the above matched, but the text
       still opens with a legal-instrument noun not followed by "này" --
       some *other* document is named, even though this pass cannot tell
       which. Still external, just unresolved: the alternative is falling
       back to "internal", which is the exact defect #121 exists to close
       (a reference to a different instrument must never become an edge
       into this document). ``LEGAL_INSTRUMENT_NOUN`` is the closed noun
       set, imported from ``ingest/parser.py``; the negative lookahead is
       deliberately broader than ``self_ref``'s own noun set, so a noun
       ``self_ref`` does not cover (e.g. "Nghị quyết này") still stays
       internal rather than falling through to here.

    Each alternative that names a real document keeps the unit chain
    appended after it (handled by the caller building ``target_hint``,
    build plan A6 issue #9 originally, extended by #121), so
    ``ingest/graph.py`` can resolve into that document's own unit, not
    only the document itself -- a bare, undated mention never satisfies
    ``ext_doc`` (it requires a document number), so these alternatives
    never compete for the same span.
    """
    title_alternation = _title_alternation(document_titles)
    return (
        rf"(?:\s+(?:{connector}\s+)?"
        rf"(?:(?P<self_ref>{_SELF_REFERENCE})"
        rf"|(?P<ext_doc>{_DOC_MENTION})"
        rf"|(?P<title_ref>{title_alternation})"
        rf"|(?P<ext_noun>{LEGAL_INSTRUMENT_NOUN})(?!\s+này\b)"
        rf"))?"
    )


def _internal_pattern(document_titles: Iterable[str] = ()) -> re.Pattern[str]:
    """Compile the "quy định tại ..." pattern for one ``extract_references`` call.

    Rebuilt per call, never a single fixed module constant, because
    ``document_titles`` (issue #71) is corpus data, not something knowable
    at import time the way every other pattern in this module is.
    """
    return re.compile(
        r"quy định tại\s+"
        # Largest-first goes first: it always starts with the literal "Điều", so
        # it never fires on điểm/khoản-first text -- but trying smallest-first
        # first would let its all-optional leading groups match a bare "Điều N"
        # and stop there, before a trailing "khoản M" that only largest-first
        # looks for.
        rf"(?:{_UNIT_ORDER_LARGEST_FIRST}|{_UNIT_ORDER_SMALLEST_FIRST}|{_CHAPTER_REFERENCE})"
        rf"{_unit_suffix_pattern(document_titles)}",
        re.IGNORECASE,
    )


INTERNAL_PATTERN = _internal_pattern()
"""The title-less pattern -- what ``_internal_pattern`` compiles to with no
manifest to match against. Kept as a module constant for readability and for
anything (a test, a REPL session) that wants the pre-#71 behaviour without
building a call to ``extract_references`` around it; extraction itself always
rebuilds its own copy from the titles it was actually given."""


_APPENDIX_CONNECTOR = r"(?:ban hành )?kèm theo"
"""The word connecting an appendix to its own document -- "kèm theo" ("issued
together with") on its own, or, just as often, "ban hành kèm theo" ("issued
and promulgated together with"). Missing from the PR #135 review: a citation
using the longer form -- "Phụ lục ban hành kèm theo Nghị định số
116/2017/NĐ-CP" (gas-decree-17-2020#dieu-1#khoan-9), "... Thông tư số
149/2020/TT-BCA" (pccc-circular-36-2025#dieu-14#khoan-3), "... Nghị định số
81/2018/NĐ-CP" (gas-decree-17-2020#dieu-21#khoan-4) -- matched no connector at
all with only "kèm theo" recognised, so the suffix past it (the actual cited
document, in every one of these cases outside the corpus) never matched
either, and the match fell through to the internal default: an edge straight
into the citing document's own appendix, exactly the #121 mistake #130 exists
to rule out. "ban hành " is optional here, not a second, separate
alternative, so a bare "kèm theo" still matches exactly as before."""


def _appendix_pattern(document_titles: Iterable[str] = ()) -> re.Pattern[str]:
    """Compile the appendix-citation pattern for one ``extract_references`` call (issue #130).

    Rebuilt per call for the same reason ``_internal_pattern`` is: the
    manifest's titles are corpus data, not import-time constants.
    """
    suffix = _unit_suffix_pattern(document_titles, connector=_APPENDIX_CONNECTOR)
    return re.compile(rf"{_UNIT_ORDER_APPENDIX}{suffix}", re.IGNORECASE)


APPENDIX_PATTERN = _appendix_pattern()
"""The title-less appendix pattern, mirroring ``INTERNAL_PATTERN`` above."""

EXTERNAL_PATTERN = re.compile(rf"{_NUMBERED_DOC}|{_STANDARD_DOC}")


def find_document_number(text: str) -> str | None:
    """Return the first document number named in ``text``, or ``None``.

    The number a ``target_hint`` names, so a citation and its hint agree on one
    spelling: a numbered document exactly as printed, a TCVN/QCVN standard with its
    whitespace collapsed to single spaces (``TCVN 6223:2017``).

    This only *finds* a number; it repairs nothing. A number pypdf split apart
    (``N Đ-CP``, ``68/ 2006/QH11``) is repaired once, on the extracted text, by
    ``ingest.extract`` (issue #80) -- by the time a hint exists it is already cut
    short (``EXTERNAL_PATTERN`` captures ``N Đ-CP`` as ``87/2018/N``), so nothing
    here could put it back, and a second copy of that rule at this level would
    only drift from the first.

    Args:
        text: A ``target_hint`` or any text that may contain a document number.

    Returns:
        The number, or ``None`` when ``text`` names none.
    """
    match = EXTERNAL_PATTERN.search(text)
    return " ".join(match.group(0).split()) if match else None


_CSV_HEADER = ("src_id", "kind", "raw_text", "target_hint")

_Classifier = Callable[[re.Match[str]], tuple[ReferenceKind, str]]


def extract_references(
    result: ParseResult, document_titles: Iterable[str] = ()
) -> tuple[Reference, ...]:
    """Detect every cross-reference in a parsed document, in reading order.

    Args:
        result: Output of ``parse_document``, whose preamble and every unit's
            heading and body are scanned for citation cues.
        document_titles: Titles of the *other* documents in the manifest --
            never this one's own, or a title citing itself could build a
            same-document edge instead of staying internal (issue #71). The
            closed set a "quy định tại ... <title>" citation is matched
            against; defaults to empty, which reproduces this function's
            behaviour from before #71 exactly -- a title-only citation
            stays unmatched and the reference falls through to
            ``ReferenceKind.INTERNAL``, as it always has for a caller (a
            unit test, mainly) that has no manifest to offer.

    Returns:
        References in reading order: the preamble first, then each unit's
        heading and body together, deduplicated within their own source text.
    """
    extractors = _build_extractors(document_titles)
    references: list[Reference] = list(_scan(result.document_id, result.preamble, extractors))
    for unit in result.units:
        references.extend(_scan(unit.id, _headed_text(unit), extractors))
    return tuple(references)


def _headed_text(unit: Unit) -> str:
    """Return a unit's heading and body joined into one text to scan.

    A heading is content, not a label: some single-sentence articles pack
    their entire substance -- citations included -- into the heading, with no
    separate body line at all (build plan issue #22). Scanning both as one
    text, rather than as two independent calls, lets the existing
    ``(src_id, kind, raw_text, target_hint)`` dedup collapse a citation that
    happens to appear in both into the single row it should be.
    """
    parts = [part for part in (unit.heading, unit.body) if part]
    return "\n\n".join(parts)


def write_references_csv(references: Iterable[Reference], path: Path) -> None:
    """Write references to a UTF-8 CSV with a header row.

    Args:
        references: Rows to write, in the order given.
        path: Destination file; created or overwritten.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_HEADER)
        for reference in references:
            writer.writerow(
                (reference.src_id, reference.kind.value, reference.raw_text, reference.target_hint)
            )


def _classify_basis(match: re.Match[str]) -> tuple[ReferenceKind, str]:
    """Classify a "Căn cứ ..." match."""
    return ReferenceKind.BASIS, _document_target(match.group(0), prefix="Căn cứ")


def _classify_amendment(match: re.Match[str]) -> tuple[ReferenceKind, str]:
    """Classify an amendment or repeal cue match."""
    return ReferenceKind.AMENDMENT, _document_target(match.group(0), prefix=None)


def _classify_external(match: re.Match[str]) -> tuple[ReferenceKind, str]:
    """Classify a bare document- or standard-number match."""
    return ReferenceKind.EXTERNAL, match.group(0)


def _classify_unit_reference(
    match: re.Match[str], title_index: dict[str, str]
) -> tuple[ReferenceKind, str]:
    """Classify a "quy định tại ..." match as internal or external.

    See ``_classify_with_suffix`` for the four outcomes this dispatches to;
    this function's only job is building the Điều/Khoản/Điểm/Chương unit
    chain that classification is based on.
    """
    return _classify_with_suffix(match, title_index, _unit_chain_target(match))


def _classify_appendix_reference(
    match: re.Match[str], title_index: dict[str, str]
) -> tuple[ReferenceKind, str]:
    """Classify an appendix-citation match as internal or external (issue #130).

    Same four outcomes as ``_classify_unit_reference``, over the appendix
    unit chain (``_appendix_chain_target``) instead of an article one --
    both match against the same ``_unit_suffix_pattern``-built named groups,
    just compiled into two different patterns (``_internal_pattern`` requires
    "quy định tại"; ``_appendix_pattern`` accepts the wider ``_APPENDIX_CUE``
    set), so the suffix-handling logic is shared rather than duplicated.
    """
    return _classify_with_suffix(match, title_index, _appendix_chain_target(match))


def _classify_with_suffix(
    match: re.Match[str], title_index: dict[str, str], unit: str
) -> tuple[ReferenceKind, str]:
    """Classify a unit-chain match as internal or external, given its already-built ``unit``.

    Four outcomes, checked in order -- see ``_unit_suffix_pattern`` for what
    each named group actually matched:

    1. ``ext_doc`` (a document/standard number): external, with the unit
       chain kept appended to ``target_hint`` -- ``ingest/graph.py``'s
       resolver finds the number anywhere in the hint, then resolves the
       remainder (the unit chain) into that document's own unit (issue
       #121; document-level only if that fails).
    2. ``title_ref`` (an exact manifest title, whitespace-tolerant, issues
       #71/#121): external too, ``target_hint`` built the same way as case
       1 -- the *canonical* title (from ``title_index``, not whatever
       whitespace variant was actually matched) followed by the unit
       chain, so ``ingest/graph.py``'s exact dict lookup still finds it
       even when a PDL artifact split the matched text.
    3. ``ext_noun`` (issue #121): a different legal instrument is named,
       but neither a number nor a known title identifies which one.
       External anyway, with the noun and the unit chain as
       ``target_hint`` -- it will not resolve to any document (there is
       nothing here to look up), but it must not fall back to internal
       either; see ``_unit_suffix_pattern``.
    4. None matched: internal, same as before #71 had a manifest to check
       against.
    """
    ext_doc = match.group("ext_doc")
    if ext_doc and (doc_number := EXTERNAL_PATTERN.search(ext_doc)):
        return ReferenceKind.EXTERNAL, f"{doc_number.group(0)} {unit}"
    if title_ref := match.group("title_ref"):
        canonical = title_index.get(_normalize_for_title_match(title_ref), title_ref)
        return ReferenceKind.EXTERNAL, f"{canonical} {unit}"
    if ext_noun := match.group("ext_noun"):
        return ReferenceKind.EXTERNAL, f"{ext_noun} {unit}"
    return ReferenceKind.INTERNAL, unit


def _appendix_chain_target(match: re.Match[str]) -> str:
    """Build the normalized ``Phụ lục N [Mục K]`` target (issue #130).

    A bare "Phụ lục" with no number at all (e.g. "Phụ lục ban hành kèm theo
    Thông tư số 149/2020/TT-BCA" -- one document, one appendix, so drafting
    sometimes omits the number) defaults to appendix 1, the same default
    ``ingest/parser.py``'s own ``_open_appendix`` uses when it opens an
    appendix heading with no number. A form or group number matched by
    ``_APPENDIX_FILLER`` is discarded here, never reaching ``target_hint``
    (see the module docstring): it names neither the appendix nor an item
    inside it.
    """
    phu_luc = match.group("phu_luc") or "1"
    parts = [f"Phụ lục {phu_luc}"]
    if muc := match.group("muc_phu_luc"):
        parts.append(f"Mục {muc}")
    return " ".join(parts)


def _unit_chain_target(match: re.Match[str]) -> str:
    """Build the normalized ``Điều X [Khoản Y] [Điểm z]`` or ``Chương N`` target."""
    if chuong := match.group("chuong"):
        return f"Chương {chuong}"
    dieu = match.group("dieu_a") or match.group("dieu_b")
    khoan = match.group("khoan_a") or match.group("khoan_b")
    diem = match.group("diem_a") or match.group("diem_b")
    parts = [f"Điều {dieu}"]
    if khoan:
        parts.append(f"Khoản {khoan}")
    if diem:
        parts.append(f"Điểm {diem.lower()}")
    return " ".join(parts)


def _document_target(raw_text: str, *, prefix: str | None) -> str:
    """Return the document number embedded in ``raw_text``, or its cleaned prose."""
    if found := EXTERNAL_PATTERN.search(raw_text):
        return found.group(0)
    cleaned = raw_text
    if prefix and cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix) :]
    return " ".join(cleaned.rstrip(";.").split())


# Precedence when spans overlap (build plan A6 Phase 1.4): a basis citation
# that embeds a document number, e.g. "Căn cứ Luật ... số 55/2024/QH15", is
# one basis row, never a basis row plus a separate external row. The unit
# reference pattern's own kind (internal/external) is decided per match, but
# it still ranks ahead of the plain external pass so a consumed document
# number inside a "quy định tại ... của Nghị định số X" span is not also
# reported as a second, bare external row.
def _build_extractors(
    document_titles: Iterable[str],
) -> tuple[tuple[re.Pattern[str], _Classifier], ...]:
    """Return the (pattern, classifier) pipeline for one ``extract_references`` call.

    Only the unit-reference entry varies call to call, rebuilt from
    ``document_titles`` (issue #71, see ``_internal_pattern``); the other
    three patterns are fixed at import time and reused unchanged. The
    unit-reference classifier also needs ``title_index`` (issue #121, a
    normalized-title -> canonical-title map) to recover a clean title for
    ``target_hint`` out of a possibly whitespace-mangled match -- built
    once here, per call, and closed over rather than threaded through
    ``_Classifier``'s single-argument signature. The appendix extractor
    (issue #130) shares the same ``title_index`` and is ordered right after
    the article/chapter one and before ``EXTERNAL_PATTERN`` for the same
    reason that one ranks ahead of ``EXTERNAL_PATTERN``: a document number
    consumed inside "... Phụ lục N kèm theo Nghị định số X" must not also be
    reported as a second, bare external row.
    """
    title_index = {_normalize_for_title_match(title): title for title in document_titles if title}

    def classify_unit_reference(match: re.Match[str]) -> tuple[ReferenceKind, str]:
        return _classify_unit_reference(match, title_index)

    def classify_appendix_reference(match: re.Match[str]) -> tuple[ReferenceKind, str]:
        return _classify_appendix_reference(match, title_index)

    return (
        (BASIS_PATTERN, _classify_basis),
        (AMENDMENT_PATTERN, _classify_amendment),
        (_internal_pattern(document_titles), classify_unit_reference),
        (_appendix_pattern(document_titles), classify_appendix_reference),
        (EXTERNAL_PATTERN, _classify_external),
    )


def _scan(
    src_id: str, text: str, extractors: tuple[tuple[re.Pattern[str], _Classifier], ...]
) -> list[Reference]:
    """Return the deduplicated references found in one unit's or the preamble's text."""
    consumed: list[tuple[int, int]] = []
    found: list[tuple[int, ReferenceKind, str, str]] = []

    for pattern, classify in extractors:
        for match in pattern.finditer(text):
            if _overlaps(match.span(), consumed):
                continue
            consumed.append(match.span())
            kind, target_hint = classify(match)
            found.append((match.start(), kind, match.group(0), target_hint))

    found.sort(key=lambda item: item[0])
    return _dedup(src_id, found)


def _dedup(src_id: str, found: list[tuple[int, ReferenceKind, str, str]]) -> list[Reference]:
    """Build one ``Reference`` per match, dropping exact repeats within ``src_id``."""
    seen: set[tuple[str, str, str]] = set()
    references: list[Reference] = []
    for _, kind, raw_text, target_hint in found:
        key = (kind.value, raw_text, target_hint)
        if key in seen:
            continue
        seen.add(key)
        references.append(
            Reference(src_id=src_id, kind=kind, raw_text=raw_text, target_hint=target_hint)
        )
    return references


def _overlaps(span: tuple[int, int], consumed: list[tuple[int, int]]) -> bool:
    """Return whether ``span`` intersects any already-consumed span."""
    start, end = span
    return any(start < c_end and end > c_start for c_start, c_end in consumed)
