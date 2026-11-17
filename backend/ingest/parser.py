"""Split raw Vietnamese legal text into the ``Chương -> Điều -> Khoản -> Điểm`` tree.

The parser is deterministic: pure regex over line starts, no LLM, no clock, no
randomness. Running it twice over the same text yields byte-identical units, which
is what lets Phases 2 and 3 join on the identifiers built here (A5.1).

Structure is opened only by an anchor at the start of a line, so a sentence that
wraps onto the next line always stays inside one unit -- the parser never splits
mid-sentence. A unit's ``body`` holds its own text only, without its number
marker, its heading, or the text of its children; an article's full text is
recomposed from its descendants when chunking (Phase 1.5). Text before the
first anchor -- title lines, "Căn cứ ..." basis clauses -- is kept verbatim as
``ParseResult.preamble`` instead of being dropped, since it is exactly where
Phase 1.4's reference extractor finds a document's ``BASED_ON`` citations.

Real corpora are messy -- tables, appendices, clauses amended in place, page
furniture left over from a PDF. Anything that cannot be placed in the tree is
logged at WARNING, skipped, and counted in ``ParseStats.skipped``, so corpus
damage is visible instead of silently changing an answer.

A "Công báo" extract of a QCVN standard often carries the promulgating
Thông tư's own closing articles ahead of the standard's own ``Chương I`` --
the circular's "Điều 1. Ban hành kèm theo Thông tư này..." is not this
document's own Điều 1. When a document has a chapter anywhere in it, an
``Điều`` line appearing before the first one is folded into the preamble
instead of opened as a real article, so the standard's genuine, later Điều 1
is not rejected as a duplicate of an impostor (issue #54). A document with
no chapter at all is unaffected -- there is nothing to compare its first
article's position against.

A mid-sentence citation to *another* document's article -- "...quy định tại
Điều 17 Luật Phòng cháy, chữa cháy..." -- can also fall at the start of a
physical line (an ordinary PDF line-wrap) and match ``ARTICLE_ANCHOR`` as if
it opened this document's own Điều 17, later evicting the real one the same
way. A genuine heading never opens with a legal-instrument noun (that word
only ever starts a reference to a different document) and always lacks the
period a genuine "Điều N. <this article's own subject>" carries when it
does, so an unpunctuated ``Điều N <noun>`` line is treated as prose, not an
anchor (issue #56).

An appendix made of sample forms ("CÁC BIỂU MẪU") is split into its forms,
not into its forms' own printed fields (issue #131): a form opens at its own
``Mẫu số ...`` line (``FORM_ANCHOR``) and becomes one ``appendix_form`` unit,
``<doc>#phu-luc-<n>#mau-<code>`` -- never a child ``appendix_item``/
``appendix_point`` per numbered field, which would cite a coordinate
("Phụ lục VIII, mục 9") that names nothing in the actual law, only a line on
a sample document. A genuine appendix of numbered items (a danger-category
list, say) is unaffected: it never contains a ``Mẫu số`` line, so it is
never reclassified and keeps splitting into ``appendix_item``/
``appendix_point`` exactly as before.

A row numbered hierarchically inside an appendix table -- Phụ lục VI's own
insurance-premium table prints ``35``, then its two sub-rows ``35.1``/``35.2``
(issue #147) -- opens a sub-item (``appendix_subitem``), a child of the item
it continues, rather than a duplicate of the item itself (``CLAUSE_ANCHOR``
alone cannot tell the two apart; see ``APPENDIX_SUBITEM_ANCHOR``). Scoped to
this one appendix (``_PREMIUM_TABLE_APPENDIX_NO``): every other appendix,
including this same document's own Phụ lục IV (whose real hierarchical
numbering always carries a *trailing* dot, e.g. "1.1.", which this anchor
never matches -- see its own docstring), keeps opening a dotted number as an
ordinary appendix item exactly as before.

Phụ lục VI's own prose is organized independently of that same table: its
"I. MỨC PHÍ BẢO HIỂM ..."/"II. MỨC KHẤU TRỪ BẢO HIỂM" sections print their
*own* "1./2./3." numbered paragraphs and lettered points, which share
``CLAUSE_ANCHOR``/``POINT_ANCHOR``'s plain shape with the table's own rows --
a genuine ambiguity already in the source document (issue #147's fix round,
reviewing PR #151). ``ingest/extract.py``'s own ``_restructure_appendix_6``
resolves it before this module ever sees the text: it rewrites each such
line, outside the table, into a sentinel (``APPENDIX_SECTION_MARKER``) this
module's ``APPENDIX_SECTION_ANCHOR``/``APPENDIX_SECTION_CLAUSE_ANCHOR``/
``APPENDIX_SECTION_POINT_ANCHOR`` recognize, so a section's own clause can
never be read as a duplicate of a same-numbered table row, or vice versa.
Khoản 1 Mục I is the one clause whose own body is split by the table itself
(its intro before the table, its formula/"Ghi chú" text after); both reach
the same sentinel, and ``_open_appendix_section_clause``'s own merge logic
(not ``_claim``'s ordinary duplicate-eviction) is what stitches the two
halves back into one unit instead of discarding the first.

A row of that same table, read on its own, never says what its own "N 0,3"
cell means -- "N" is a deductible-type code, "0,3" a percentage, defined only
by the column header printed once at the top of the table (issue #160). This
module never types that header itself: ``ingest/extract.py``'s own
``_restructure_appendix_6`` reads it out of the source text and carries it
through another sentinel (``APPENDIX_COLUMN_HEADER_MARKER``), consumed here
by ``APPENDIX_COLUMN_HEADER_ANCHOR``/``_open_appendix_column_header`` into
``self._column_header``, which ``_open_appendix_item``/``_open_appendix_subitem``
then attach to the row/sub-row unit itself (``Unit.column_header``) for
``ingest/chunk.py`` to compose into that row's chunk header -- never into the
unit's own ``body``.
"""

import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from app.core.ids import (
    appendix_form_id,
    appendix_id,
    appendix_item_id,
    appendix_point_id,
    appendix_section_clause_id,
    appendix_section_id,
    appendix_section_point_id,
    appendix_subitem_id,
    article_id,
    chapter_id,
    clause_id,
    document_id,
    point_id,
)
from app.schemas.unit import Unit, UnitType

logger = logging.getLogger(__name__)

LEGAL_INSTRUMENT_NOUN = (
    r"(?:Hiến pháp|Nghị quyết|Pháp lệnh|Nghị định|Luật|Thông tư|Quy chuẩn|Tiêu chuẩn|Công ước)"
)
"""The closed set of Vietnamese legal-instrument nouns that only ever start a
reference to a *different* document -- either by number ("Nghị định số
87/2018/NĐ-CP") or by title and date alone ("Luật Tổ chức Chính phủ ngày
..."). Originally ``ingest/references.py``'s own private constant (build plan
A6 issue #9); moved here, and made public, so ``ingest/parser.py`` can use it
too (issue #56) without a circular import -- ``references.py`` imports
``ParseResult`` from this module, not the other way around. ``references.py``
re-exports it from here. Measured on the real corpus: every "Căn cứ ..."
whose object is neither one of these nouns nor a bare number is ordinary
Vietnamese prose reusing the same word ("Căn cứ tình hình, yêu cầu bảo đảm an
toàn ...", "Căn cứ quyết định của Ủy ban nhân dân cấp tỉnh ..."), never a
citation to an existing, identifiable document. "Quyết định" (a decision) is
deliberately excluded even though it can itself be a numbered legal
instrument: every real instance found names a future or generic decision
("quyết định của Ủy ban nhân dân cấp tỉnh"), never one with its own number --
adding it back the moment a genuine "Quyết định số N" citation is found is a
small, safe change; guessing it in now, with no evidence, is not."""

CHAPTER_ANCHOR = re.compile(r"^Chương\s+(?P<number>[IVXLCDM]+|\d+)\b\s*\.?\s*(?P<heading>.*)$")
ARTICLE_ANCHOR = re.compile(r"^Điều\s+(?P<number>\d+)\s*(?P<dot>\.?)\s*(?P<heading>.*)$")
_THOUSANDS_GROUP = r"\d{1,3}(?:\.\d{3})+(?!\d)"
"""A thousands-separated number -- ``1.000``, ``30.000``, ``1.000.000`` -- never
a clause/item anchor (issue #141, narrowed per PR #142's review). Exactly one to
three leading digits, then one or more ``.DDD`` groups of exactly three digits
each, with the whole run not immediately followed by another digit (so a longer
number is never mistaken for a shorter prefix of itself). Deliberately narrower
than "any digit after the dot": hierarchical sub-numbering like ``1.1.`` and a
citation like ``4.2.7.2`` are *not* thousands groups (their groups are not all
exactly three digits) and must keep matching ``CLAUSE_ANCHOR`` exactly as before
this issue -- the first version of this fix rejected those too, which silently
merged Phụ lục IV's own hierarchical items into one and (via extract.py's reflow,
which reuses this same anchor) let a now-uninterrupted QCVN 10 clause reach
``_figure_legend_reason`` with its own citation digits, tripping a false
"figure legend" rejection and losing the clause outright."""

CLAUSE_ANCHOR = re.compile(rf"^(?!{_THOUSANDS_GROUP})(?P<number>\d+)\s*\.\s*(?P<rest>.*)$")
"""Matches ``N.`` opening a clause or appendix item, but never a thousands
separator (``_THOUSANDS_GROUP``, issue #141). Measured on the real corpus
(pccc-decree-105-2025's own Phụ lục II, a danger-category table of area/volume
thresholds): "1.000 m²", "30.000 m³" and friends were being read as ``N.``
anchors opening fake items "mục 1", "mục 30", etc., merging the table's real
rows into them out of STT order. Every other ``N.`` line -- including
hierarchical sub-numbering ("1.1.") and a citation ("4.2.7.2 TCVN...") -- matches
exactly as it always has; only a genuine thousands group is excluded."""
POINT_ANCHOR = re.compile(r"^(?P<letter>[a-zđ])\s*\)\s*(?P<rest>.*)$")

APPENDIX_6_NO = 6
"""Phụ lục VI's own appendix number, exposed publicly (unlike
``_PREMIUM_TABLE_APPENDIX_NO`` below) so ``ingest/extract.py`` can scope its
own section/clause/point rewriting (``_restructure_appendix_6``, issue #147)
by appendix number -- the same number the real ``APPENDIX_HEADING`` carries
-- rather than duplicating a bare literal ``6`` in a second module."""

_PREMIUM_TABLE_APPENDIX_NO = APPENDIX_6_NO
"""The one appendix ``APPENDIX_SUBITEM_ANCHOR``, ``APPENDIX_SECTION_ANCHOR``,
``APPENDIX_SECTION_CLAUSE_ANCHOR`` and ``APPENDIX_SECTION_POINT_ANCHOR`` are
allowed to open a unit in -- Phụ lục VI's own insurance-premium table and its
surrounding prose (issues #147). Checked as
``self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO`` before any of these
anchors is ever consulted, so no other appendix in the corpus -- including
this same document's own Phụ lục IV, numbered differently -- changes
behavior. The invariant this issue is built against (every unit outside
``#phu-luc-6`` byte-identical) depends on this guard, not only on each
anchor's own shape."""

APPENDIX_SUBITEM_ANCHOR = re.compile(
    r"^(?P<item>\d{1,2})\.(?P<subitem>\d{1,2})(?=\s|$)\s*(?P<rest>.*)$"
)
"""Matches a table row numbered one level under its parent, e.g. ``35.1`` or
``14.1`` (issue #147, Phụ lục VI's own premium table) -- never a thousands
group (the lookahead after ``subitem`` already rejects a third digit, the
same protection ``_THOUSANDS_GROUP`` gives ``CLAUSE_ANCHOR``, e.g. "5.000"
fails since "00" is followed by another "0") and never Phụ lục IV's own real
hierarchical numbering, which always carries a *trailing* dot ("1.1.",
"4.2.7.2") that this anchor's own lookahead excludes just as surely (a dot is
neither whitespace nor end-of-line). Checked only while
``self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO`` and only when its
``item`` equals the appendix item currently open (``self._item_no``) --
belt-and-suspenders on top of the shape itself, so this can only ever attach
a sub-row to the specific parent row it was printed under, never open a
fresh, wrongly-parented unit from an unrelated coincidence of digits."""

APPENDIX_SECTION_MARKER = "§"
"""Prefix ``ingest/extract.py``'s own ``_restructure_appendix_6`` puts on a
synthetic line it builds from Phụ lục VI's own prose (Mục I/Mục II, issue
#147's fix round on PR #151) -- the Unicode section sign, a character that
can never appear in the extracted Vietnamese text on its own. Marking these
lines this way is what lets ``APPENDIX_SECTION_ANCHOR``/
``APPENDIX_SECTION_CLAUSE_ANCHOR``/``APPENDIX_SECTION_POINT_ANCHOR`` below
never be confused with the table's own rows, which keep exactly the same
plain "N."/"N.M" shape ``CLAUSE_ANCHOR``/``APPENDIX_SUBITEM_ANCHOR`` already
recognize, unchanged by this issue's fix round."""

APPENDIX_SECTION_ANCHOR = re.compile(
    rf"^{APPENDIX_SECTION_MARKER}(?P<section>[IVXLCDM]+)\.\s*(?P<heading>.*)$"
)
"""Opens a named section inside an appendix -- Phụ lục VI's own "I. MỨC PHÍ
BẢO HIỂM ..."/"II. MỨC KHẤU TRỪ BẢO HIỂM" (issue #147's fix round). Only ever
matched after ``ingest/extract.py`` has already rewritten the line with
``APPENDIX_SECTION_MARKER``, and only consulted while ``self._appendix_no ==
_PREMIUM_TABLE_APPENDIX_NO`` (see the guard in ``feed``); never reached for
any other appendix."""

APPENDIX_SECTION_CLAUSE_ANCHOR = re.compile(
    rf"^{APPENDIX_SECTION_MARKER}(?P<section>[IVXLCDM]+)\.(?P<khoan>\d{{1,2}})\.\s*(?P<rest>.*)$"
)
"""Opens, or resumes, a clause inside the current appendix section (issue
#147's fix round) -- Mục I/Mục II's own "1./2./3." paragraphs, independent of
the table's numeric STT. Checked before ``CLAUSE_ANCHOR`` in ``feed``'s
appendix branch, but never actually ambiguous with it in practice:
``ingest/extract.py`` only ever emits this exact marked shape for a line
outside the table, never for one of the table's own rows."""

APPENDIX_SECTION_POINT_ANCHOR = re.compile(
    rf"^{APPENDIX_SECTION_MARKER}(?P<section>[IVXLCDM]+)\.(?P<khoan>\d{{1,2}})\."
    rf"(?P<letter>[a-zđ])\)\s*(?P<rest>.*)$"
)
"""Opens a lettered point inside the current appendix section clause (issue
#147's fix round) -- Mục II khoản 1's own "a)"/"b)"/"c)" deductible rules.
Checked before ``APPENDIX_SECTION_CLAUSE_ANCHOR`` in ``feed``'s appendix
branch: without that order, the clause anchor's own permissive ``rest``
group would just as happily swallow a point's "a) ..." tail as part of its
own clause text."""

APPENDIX_COLUMN_HEADER_MARKER = "‡"
"""Prefix ``ingest/extract.py``'s own ``_restructure_appendix_6`` puts on a
synthetic line carrying Phụ lục VI's own premium-table column header, read
out of the table's own header line in the source text (issue #160) -- the
Unicode double dagger, a character that can never appear in the extracted
Vietnamese text on its own, and distinct from ``APPENDIX_SECTION_MARKER``
(the section sign) so the two sentinels are never confused with each other
or with a real row's own plain "N."/"N.M" shape."""

APPENDIX_COLUMN_HEADER_ANCHOR = re.compile(rf"^{APPENDIX_COLUMN_HEADER_MARKER}(?P<header>.*)$")
"""Matches a column-header sentinel line (issue #160). Only ever matched
after ``ingest/extract.py`` has already rewritten the line with
``APPENDIX_COLUMN_HEADER_MARKER``, and only consulted while ``self._appendix_no
== _PREMIUM_TABLE_APPENDIX_NO`` (see the guard in ``feed``); never reached for
any other appendix. Carries no body text of its own -- ``_open_appendix_column_header``
only updates ``self._column_header``, consulted the next time
``_open_appendix_item``/``_open_appendix_subitem`` opens a row of this same
table."""

_EXTERNAL_ARTICLE_CITATION_HEADING = re.compile(rf"^{LEGAL_INSTRUMENT_NOUN}\b")
"""Matches a heading that opens with a legal-instrument noun -- only ever a
citation to another document, never this document's own article subject.
Combined with a missing period after the number, this is what tells a real
"Điều 17. Thủ tục huy động..." apart from a citation like "Điều 17 Luật
Phòng cháy..." that merely happens to start a physical line (issue #56)."""

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

_EMBEDDED_CAPTION_START = re.compile(r"\d+\s*\.\s*\S")
"""Matches a second ``<digits>.`` run followed by more text -- a fresh caption
starting mid-line, as opposed to a citation like "... Điều 40." that merely ends
the sentence with a number and the line's own closing period."""

_LEGEND_CAPTION = re.compile(r"^\s*(?:Hình|Bảng)\b")
"""Matches a "Hình"/"Bảng" (figure/table) caption line -- data, not prose (see module docstring)."""

APPENDIX_HEADING = re.compile(
    r"^\s*PHỤ LỤC(?:\s+(?P<number>[IVXLCDM]+|\d+))?\s*\.?\s*(?P<heading>.*)$",
    re.IGNORECASE,
)
APPENDIX_CAPTION = re.compile(r"^\s*DANH MỤC\b.*\bKèm theo\b", re.IGNORECASE)
FORM_ANCHOR = re.compile(
    r"^\s*(?:\d+\s*\.\s*)?Mẫu số\s+(?P<code>[A-Za-z0-9]+)\s*[:.]?\s*(?P<rest>.*)$"
)
"""Opens a sample form inside an appendix (issue #131): ``Mẫu số PC01``,
``Mẫu số 01a``, or a numbered reference to one in a form appendix's own table
of contents (``1. Mẫu số 01: Sổ theo dõi ...``, measured on circular
36/2025's Phụ lục VI). The optional leading ``N.`` is what tells a form
appendix's own table of contents apart from a numbered *item* in a genuine,
non-form appendix (``CLAUSE_ANCHOR``) -- checked first, in ``_Parser.feed``'s
``"appendix"`` branch, exactly so a line like that is read as naming a form,
never as opening an appendix item of its own.

Deliberately case-sensitive on "Mẫu số" itself, like ``SIGNATURE_START``: the
corpus's own citations to *another* document's form ("... theo Mẫu số 11 Mục
II tại Phụ lục ban hành kèm theo Nghị định này.", gas-decree-17-2020) always
continue a sentence rather than opening a line with nothing else on it, so
``extract.py``'s own ``_is_bare_form_heading`` -- not this pattern alone --
is what keeps a wrapped citation fragment like that from being misread as a
genuine form opener; see its docstring."""
RECIPIENT_START = re.compile(r"^\s*Nơi nhận\s*:", re.IGNORECASE)
SIGNATURE_START = re.compile(
    r"^\s*(?:(?:KT|TM|TL|TUQ)\.\s+)?"
    r"(?:CHÍNH PHỦ|THỦ TƯỚNG|PHÓ THỦ TƯỚNG|BỘ TRƯỞNG|CHỦ TỊCH)\b"
)
"""Deliberately case-*sensitive*, unlike ``APPENDIX_HEADING``/``APPENDIX_CAPTION``
above (issue #117): a real signer block is always rendered fully in capitals by
Vietnamese official-document convention ("TM. CHÍNH PHỦ", "BỘ TRƯỞNG", "KT. BỘ
TRƯỞNG"), while every one of these same words also occurs constantly as an
ordinary common noun in mixed-case running prose -- "Chính phủ ban hành...",
and, measured directly on decree 105/2025's own real Điều 47, "Bộ trưởng, Thủ
trưởng cơ quan ngang bộ, ... chịu trách nhiệm thi hành Nghị định này." itself:
the article's own genuine closing sentence, which ``re.IGNORECASE`` matched as
if it were the signature line that actually follows it, wiping the article's
real body. Matching case-sensitively is exactly the same all-caps signal
``ingest/extract.py``'s own ``_sentence_is_finished`` already uses to tell a
chapter/article title apart from ordinary prose."""
"""Public (not module-private) like ``CHAPTER_ANCHOR``/``ARTICLE_ANCHOR``/etc above:
``ingest/extract.py``'s PDF reflow needs these same four patterns to decide where a
genuine new block starts, for the same reason it already needs the chapter and
article anchors (issue #117). A PDF's own page furniture routinely sits between the
last article's real text and the appendix or signature that follows it -- a stray
running footer, a page break -- and none of the four are matched at all by
``CHAPTER_ANCHOR``/``ARTICLE_ANCHOR``/``CLAUSE_ANCHOR``/``POINT_ANCHOR``. Left
unrecognised by the reflow, a signature block's or an appendix's own opening line
gets silently glued onto whatever ordinary prose line precedes it instead of
starting a fresh line of its own, so it never reaches this module's ``feed`` as a
line ``match()`` can find at position 0 -- exactly the failure measured on the real
corpus: a page-footer artifact sitting in front of "Phụ lục I" on the same reflowed
line defeated this module's own anchor, which is correct and unchanged, but never
got the chance to see the line it was written to recognise."""


class ParseStats(BaseModel):
    """Counts describing one parse run.

    Attributes:
        chapters: Chapters placed in the tree.
        articles: Articles placed in the tree.
        clauses: Clauses placed in the tree.
        points: Points placed in the tree.
        skipped: Anchors that could not be placed and were dropped.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chapters: int = 0
    articles: int = 0
    clauses: int = 0
    points: int = 0
    skipped: int = 0


class ParseResult(BaseModel):
    """Units of one document, in reading order, with the run's statistics.

    Attributes:
        document_id: Owning document identifier.
        units: Structural units in reading order.
        stats: Counts describing this parse run.
        preamble: Text before the first anchor, e.g. title lines and
            "Căn cứ ..." basis clauses. Empty when the document opens directly
            with a chapter, article, clause, or point anchor.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    units: tuple[Unit, ...]
    stats: ParseStats
    preamble: str


def parse_document(text: str, doc_id: str) -> ParseResult:
    """Parse one document's plain text into its structural units.

    Args:
        text: Full plain text of the document, already extracted from its source file.
        doc_id: Document identifier from the corpus manifest, e.g. ``gas-decree-87-2018``.

    Returns:
        The units in reading order together with the parse statistics.

    Raises:
        ValueError: If ``doc_id`` is not a valid document identifier (A5.1).
    """
    lines = text.splitlines()
    has_chapters = any(CHAPTER_ANCHOR.match(line) for line in lines)
    parser = _Parser(doc_id, has_chapters=has_chapters)
    for line_no, line in enumerate(lines, start=1):
        next_line = lines[line_no] if line_no < len(lines) else None
        parser.feed(line_no, line, next_line)
    return parser.finish()


@dataclass
class _Pending:
    """A unit whose anchor has been read and whose body is still being collected."""

    unit_type: UnitType
    id: str
    parent_id: str | None = None
    chapter_no: int | None = None
    dieu_no: int | None = None
    khoan_no: int | None = None
    diem: str | None = None
    appendix_no: int | None = None
    appendix_section: str | None = None
    item_no: int | None = None
    subitem_no: int | None = None
    form_code: str | None = None
    column_header: str | None = None
    heading: str | None = None
    body_lines: list[str] = field(default_factory=list)


def _figure_legend_reason(
    rest: str, next_line: str | None, khoan_no: int | None, candidate_no: int
) -> str | None:
    r"""Return why a ``^\d+\.`` match looks like a figure/table legend, or ``None``.

    QCVN-style documents carry numbered legends under diagrams, e.g. "1. Các vị
    trí thay đổi ... 2. Mẫu thử ..." followed by "Hình 5. ...". Read naively,
    each numbered caption looks exactly like ``CLAUSE_ANCHOR``. Three signals
    tell a legend apart from a real clause without needing to understand the
    text: a legend crams a second numbered caption onto the same physical line
    where a real clause has exactly one (a trailing citation like "... Điều
    40." does not count -- nothing follows it); a legend is immediately
    followed by a "Hình"/"Bảng" caption line; and a legend's numbering does
    not continue the article's own strictly increasing clause sequence. Any
    one of the three is enough to distrust the match.

    Args:
        rest: The text captured after the anchor's own ``number.`` prefix
            (``CLAUSE_ANCHOR``'s ``rest`` group).
        next_line: The line immediately following the anchor's line, or
            ``None`` at end of text.
        khoan_no: The most recently opened clause number in the current
            article, or ``None`` when no clause has been opened yet.
        candidate_no: The clause number this line's anchor would open.

    Returns:
        A short reason naming the signal that triggered, or ``None`` when the
        line looks like a genuine clause anchor.
    """
    if _EMBEDDED_CAPTION_START.search(rest):
        return "a second numbered caption follows on the same line"
    if next_line is not None and _LEGEND_CAPTION.match(next_line):
        return "followed by a Hình/Bảng caption"
    if khoan_no is not None and candidate_no <= khoan_no:
        return "does not continue the article's clause sequence"
    return None


class _Parser:
    """Line-driven state machine building the unit tree of a single document."""

    def __init__(self, doc_id: str, *, has_chapters: bool = False) -> None:
        self._doc_id = document_id(doc_id)
        self._has_chapters = has_chapters
        self._units: list[Unit] = []
        self._seen_ids: set[str] = set()
        self._skipped = 0
        self._chapter_id: str | None = None
        self._dieu_no: int | None = None
        self._khoan_no: int | None = None
        self._pending: _Pending | None = None
        self._mode = "enacting"
        self._appendix_no: int | None = None
        self._item_no: int | None = None
        # Phụ lục VI's own Roman-numbered section ("I"/"II") and the clause
        # currently open inside it (issue #147's fix round), independent of
        # ``self._item_no``, which tracks the table's own numeric rows.
        self._appendix_section: str | None = None
        self._appendix_section_khoan: int | None = None
        # Phụ lục VI's own premium-table column header, read from the source
        # text by ``ingest/extract.py`` (issue #160) and consulted only by
        # ``_open_appendix_item``/``_open_appendix_subitem`` while
        # ``self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO``; reset whenever
        # a new appendix starts (``_open_appendix``) so it can never leak
        # into a later, unrelated appendix.
        self._column_header: str | None = None
        # Set once a "Mẫu số ..." form anchor has opened the current appendix's
        # first form (issue #131); ``None`` while still in a genuine, non-form
        # appendix (or before the first form of a form appendix has appeared).
        # Reset to ``None`` only by ``_open_appendix``, so it survives a
        # placeholder signature/recipient block *inside* the form it names --
        # see the guard on that check in ``feed`` below.
        self._form_code: str | None = None
        self._preamble_lines: list[str] = []
        self._preamble_done = False
        # Stricter than ``self._preamble_done``: set only when a genuine chapter or
        # article has actually opened, never by a clause or point (issue #117, see
        # the gate on the appendix/recipient/signature checks in ``feed`` below).
        self._structure_opened = False
        # True while "signature" mode was entered *from* "appendix" mode, not from
        # ordinary enacting text (issue #117, see the resume logic in ``feed``).
        self._signature_resumable = False

    def feed(  # noqa: C901, PLR0912, PLR0915
        self, line_no: int, line: str, next_line: str | None = None
    ) -> None:
        """Consume one source line.

        Args:
            line_no: 1-based line number, reported when an anchor is skipped.
            line: The raw line, without its trailing newline.
            next_line: The line immediately following ``line``, or ``None`` at
                end of text. Used only by the clause anchor, to check whether
                it is immediately followed by a "Hình"/"Bảng" figure caption.
        """
        appendix_match = APPENDIX_HEADING.match(line) if self._structure_opened else None
        if appendix_match is not None or (
            self._structure_opened and APPENDIX_CAPTION.match(line) and self._mode != "appendix"
        ):
            self._open_appendix(line_no, appendix_match, line)
        elif self._mode == "signature":
            # A signature block reached *from appendix mode* can still be
            # followed by more of the *same* appendix's own numbering (issue
            # #117): a multi-form appendix -- several "Mẫu số N" sample
            # documents bundled under one heading, measured on the real
            # corpus (gas-decree-87-2018's own Phụ lục) -- routinely repeats a
            # placeholder "Nơi nhận:"/signer block inside *each* sample form,
            # not only once at the very end. Treating every such placeholder
            # as the permanent end of all structure, the same way a genuine
            # final signature ends an enacting document, would silently drop
            # every sample form after the first one. A clause or point anchor
            # reappearing is exactly the signal that the same appendix's
            # enumeration has resumed, so appendix mode resumes with it;
            # anything else (an ordinary form label, a blank field) is still
            # dropped, same as before. Never resumed when signature mode was
            # entered from ordinary enacting text instead: a genuine
            # document's real signature is still always its true last
            # structural element.
            if self._signature_resumable and (match := CLAUSE_ANCHOR.match(line)):
                self._mode = "appendix"
                self._open_appendix_item(line_no, match)
            elif self._signature_resumable and (match := POINT_ANCHOR.match(line)):
                self._mode = "appendix"
                self._open_appendix_point(line_no, match)
            else:
                return
        elif (
            self._structure_opened
            and self._form_code is None
            and (RECIPIENT_START.match(line) or SIGNATURE_START.match(line))
        ):
            # Checked before the "appendix" mode branch below, not only the
            # enacting-text fall-through, so a signature or recipient block
            # trailing the *last appendix item* is recognised too (issue #117):
            # without this, appendix mode never re-checked for these patterns,
            # so a signature block trailing the last appendix item was silently
            # absorbed into that item's own body as if it were appendix content.
            #
            # Also gated on ``self._form_code is None`` (issue #131): once a
            # form has opened, a placeholder "Nơi nhận:"/signer block printed
            # *inside the sample form itself* ("ĐẠI DIỆN CƠ SỞ (Ký, ghi rõ họ
            # tên)", the form's own letterhead) is just more of that form's
            # body text, never a real document signature -- treating it as one
            # dropped every form after the first that happened to have no
            # clause or point left to resume on (measured on the real corpus:
            # gas-decree-87-2018's own forms 02-12 vanished entirely this way,
            # surviving only where a later form's genuine numbered field
            # happened to resume appendix mode by accident). A form never has
            # ``muc``/``khoan`` items to misinterpret in the first place, so
            # there is nothing left for "signature" mode to protect once
            # ``self._form_code`` is set -- unlike a genuine, non-form
            # appendix, where this gate is unchanged and issue #117's own
            # resumption logic below still applies exactly as before.
            #
            # All four patterns are gated on ``self._structure_opened`` -- True
            # only once a *real* chapter or article has actually been opened,
            # set in ``_open_chapter``/the accepted branch of ``ARTICLE_ANCHOR``
            # below, never by a clause or point anchor -- for the same reason
            # ``ARTICLE_ANCHOR`` already folds a pre-chapter ``Điều`` line into
            # the preamble (issue #54, see the module docstring): a QCVN
            # document's own Công báo extract carries the *promulgating*
            # Thông tư's front matter -- its own "Bộ trưởng ... ban hành...",
            # "Nơi nhận:", and signer block -- ahead of the standard's real
            # ``Chương I``. Measured on the real corpus (issue #117): without
            # this gate, that circular's own "Nơi nhận:" line, now correctly
            # isolated onto its own line by ``ingest/extract.py``'s reflow,
            # matched ``RECIPIENT_START`` while still in the front matter and put
            # the parser into signature mode before the standard's first chapter
            # had even opened -- silently dropping the entire document. Gating
            # on ``self._preamble_done`` instead is not enough: that flag is
            # already set True by a bare *clause*-shaped line in the same front
            # matter ("1. Đối với các trạm cấp...", rejected right after as
            # "clause outside any article"), well before the real ``Chương I``.
            self._signature_resumable = self._mode == "appendix"
            self._flush()
            self._mode = "signature"
            self._chapter_id = None
            self._dieu_no = None
            self._khoan_no = None
            # ``self._appendix_no`` survives when resumable, so a clause anchor
            # reappearing later (see the resume branch above) still builds the
            # right ``phu-luc-<n>#muc-<k>`` id, not a fresh, wrong appendix
            # number.
            if not self._signature_resumable:
                self._appendix_no = None
            self._item_no = None
            self._appendix_section = None
            self._appendix_section_khoan = None
        elif self._mode == "appendix":
            if match := FORM_ANCHOR.match(line):
                self._open_appendix_form(line_no, match)
            elif self._form_code is not None:
                # Already inside a form (issue #131): a line that *looks* like
                # a clause or point anchor is one of the form's own printed
                # fields ("1. Tên cơ sở:....", "a) ..."), not a fresh appendix
                # item -- the whole point of this fix is that a form's fields
                # are never split into ``muc``/``khoan``/``diem`` units, so
                # every such line just joins the form's own body text.
                self._append_body(line)
            elif self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO and (
                match := APPENDIX_COLUMN_HEADER_ANCHOR.match(line)
            ):
                self._open_appendix_column_header(match)
            elif self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO and (
                match := APPENDIX_SECTION_POINT_ANCHOR.match(line)
            ):
                self._open_appendix_section_point(line_no, match)
            elif self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO and (
                match := APPENDIX_SECTION_CLAUSE_ANCHOR.match(line)
            ):
                self._open_appendix_section_clause(line_no, match)
            elif self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO and (
                match := APPENDIX_SECTION_ANCHOR.match(line)
            ):
                self._open_appendix_section(line_no, match)
            elif (
                self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO
                and (match := APPENDIX_SUBITEM_ANCHOR.match(line))
                and self._item_no == int(match.group("item"))
            ):
                self._open_appendix_subitem(line_no, match)
            elif match := CLAUSE_ANCHOR.match(line):
                self._open_appendix_item(line_no, match)
            elif match := POINT_ANCHOR.match(line):
                self._open_appendix_point(line_no, match)
            else:
                self._append_body(line)
        elif match := CHAPTER_ANCHOR.match(line):
            self._preamble_done = True
            self._structure_opened = True
            self._open_chapter(line_no, match)
        elif match := ARTICLE_ANCHOR.match(line):
            if self._has_chapters and self._chapter_id is None:
                # A chapter exists somewhere in this document but none has
                # opened yet -- this Điều belongs to whatever promulgated the
                # document (its own closing articles, ahead of the real
                # Chương I in the extracted text), not to the document itself.
                self._reject_anchor_as_prose(line)
            elif not match.group("dot") and _EXTERNAL_ARTICLE_CITATION_HEADING.match(
                match.group("heading")
            ):
                # No period after the number, and the heading opens with a
                # legal-instrument noun -- a citation to another document's
                # article, not this one's own, that merely happens to start
                # a physical line.
                self._reject_anchor_as_prose(line)
            else:
                self._preamble_done = True
                self._structure_opened = True
                self._open_article(line_no, match)
        elif match := CLAUSE_ANCHOR.match(line):
            self._preamble_done = True
            self._open_clause(line_no, match, next_line)
        elif match := POINT_ANCHOR.match(line):
            self._preamble_done = True
            self._open_point(line_no, match)
        elif self._preamble_done:
            self._append_body(line)
        else:
            self._preamble_lines.append(line)

    def finish(self) -> ParseResult:
        """Flush the last pending unit and return the parse result.

        Returns:
            The units in reading order together with the parse statistics.
        """
        self._flush()
        units = tuple(self._units)
        stats = ParseStats(
            chapters=sum(1 for unit in units if unit.unit_type is UnitType.CHAPTER),
            articles=sum(1 for unit in units if unit.unit_type is UnitType.ARTICLE),
            clauses=sum(1 for unit in units if unit.unit_type is UnitType.CLAUSE),
            points=sum(1 for unit in units if unit.unit_type is UnitType.POINT),
            skipped=self._skipped,
        )
        preamble = _join_body(self._preamble_lines)
        logger.info("Parsed %s: %s", self._doc_id, stats.model_dump())
        return ParseResult(document_id=self._doc_id, units=units, stats=stats, preamble=preamble)

    def _open_chapter(self, line_no: int, match: re.Match[str]) -> None:
        """Start a chapter and reset the article and clause context."""
        self._flush()
        chapter_no = chapter_numeral_to_int(match.group("number"))
        unit_id = chapter_id(self._doc_id, chapter_no)
        self._chapter_id = None
        self._dieu_no = None
        self._khoan_no = None
        if self._claim(unit_id, line_no, "duplicate chapter"):
            return
        self._chapter_id = unit_id
        self._pending = _Pending(
            unit_type=UnitType.CHAPTER,
            id=unit_id,
            chapter_no=chapter_no,
            heading=_clean(match.group("heading")),
        )

    def _open_article(self, line_no: int, match: re.Match[str]) -> None:
        """Start an article and reset the clause context."""
        self._flush()
        dieu_no = int(match.group("number"))
        unit_id = article_id(self._doc_id, dieu_no)
        self._dieu_no = None
        self._khoan_no = None
        if self._claim(unit_id, line_no, "duplicate article"):
            return
        self._dieu_no = dieu_no
        self._pending = _Pending(
            unit_type=UnitType.ARTICLE,
            id=unit_id,
            parent_id=self._chapter_id,
            dieu_no=dieu_no,
            heading=_clean(match.group("heading")),
        )

    def _open_clause(self, line_no: int, match: re.Match[str], next_line: str | None) -> None:
        """Start a clause; skip it when misplaced, a duplicate, or a figure legend."""
        self._flush()
        if self._dieu_no is None:
            self._khoan_no = None
            self._skip(line_no, "clause outside any article")
            return
        khoan_no = int(match.group("number"))
        rest = match.group("rest")
        legend_reason = _figure_legend_reason(rest, next_line, self._khoan_no, khoan_no)
        if legend_reason is not None:
            self._skip(line_no, f"clause anchor rejected as a figure legend: {legend_reason}")
            return
        unit_id = clause_id(self._doc_id, self._dieu_no, khoan_no)
        if self._claim(unit_id, line_no, "duplicate clause", prefer_new=True):
            return
        self._khoan_no = khoan_no
        self._pending = _Pending(
            unit_type=UnitType.CLAUSE,
            id=unit_id,
            parent_id=article_id(self._doc_id, self._dieu_no),
            dieu_no=self._dieu_no,
            khoan_no=khoan_no,
            body_lines=[rest],
        )

    def _open_point(self, line_no: int, match: re.Match[str]) -> None:
        """Start a point, or skip it when it sits outside any clause."""
        self._flush()
        if self._dieu_no is None or self._khoan_no is None:
            self._skip(line_no, "point outside any clause")
            return
        diem = match.group("letter")
        unit_id = point_id(self._doc_id, self._dieu_no, self._khoan_no, diem)
        if self._claim(unit_id, line_no, "duplicate point"):
            return
        self._pending = _Pending(
            unit_type=UnitType.POINT,
            id=unit_id,
            parent_id=clause_id(self._doc_id, self._dieu_no, self._khoan_no),
            dieu_no=self._dieu_no,
            khoan_no=self._khoan_no,
            diem=diem,
            body_lines=[match.group("rest")],
        )

    def _open_appendix(self, line_no: int, match: re.Match[str] | None, line: str) -> None:
        """Start an appendix at a heading or a ``DANH MỤC ... Kèm theo`` caption."""
        self._flush()
        raw_number = match.group("number") if match is not None else None
        appendix_no = chapter_numeral_to_int(raw_number) if raw_number else 1
        unit_id = appendix_id(self._doc_id, appendix_no)
        self._mode = "appendix"
        self._chapter_id = None
        self._dieu_no = None
        self._khoan_no = None
        self._appendix_no = appendix_no
        self._item_no = None
        self._appendix_section = None
        self._appendix_section_khoan = None
        self._column_header = None
        self._form_code = None
        if self._claim(unit_id, line_no, "duplicate appendix"):
            return
        heading = _clean(match.group("heading")) if match is not None else line.strip()
        self._pending = _Pending(
            unit_type=UnitType.APPENDIX,
            id=unit_id,
            appendix_no=appendix_no,
            heading=heading,
        )

    def _open_appendix_item(self, line_no: int, match: re.Match[str]) -> None:
        """Start a numbered appendix item under the current appendix."""
        self._flush()
        if self._appendix_no is None:
            self._skip(line_no, "appendix item outside any appendix")
            return
        item_no = int(match.group("number"))
        unit_id = appendix_item_id(self._doc_id, self._appendix_no, item_no)
        if self._claim(unit_id, line_no, "duplicate appendix item"):
            return
        self._item_no = item_no
        self._pending = _Pending(
            unit_type=UnitType.APPENDIX_ITEM,
            id=unit_id,
            parent_id=appendix_id(self._doc_id, self._appendix_no),
            appendix_no=self._appendix_no,
            item_no=item_no,
            column_header=(
                self._column_header if self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO else None
            ),
            body_lines=[match.group("rest")],
        )

    def _open_appendix_subitem(self, line_no: int, match: re.Match[str]) -> None:
        """Start a numbered sub-row under the current appendix item (issue #147).

        Only ever reached while ``self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO``
        and the anchor's own ``item`` group already equals ``self._item_no`` --
        see the caller's guard in ``feed`` and ``APPENDIX_SUBITEM_ANCHOR``'s own
        docstring. The ``appendix_no is None`` branch below is therefore dead in
        practice; kept only so this method fails the same safe way
        ``_open_appendix_item``/``_open_appendix_point`` do if that guard is
        ever loosened, instead of raising on a ``None`` coordinate.
        """
        self._flush()
        if self._appendix_no is None or self._item_no is None:
            self._skip(line_no, "appendix sub-item outside any appendix item")
            return
        subitem_no = int(match.group("subitem"))
        unit_id = appendix_subitem_id(self._doc_id, self._appendix_no, self._item_no, subitem_no)
        if self._claim(unit_id, line_no, "duplicate appendix sub-item"):
            return
        self._pending = _Pending(
            unit_type=UnitType.APPENDIX_SUBITEM,
            id=unit_id,
            parent_id=appendix_item_id(self._doc_id, self._appendix_no, self._item_no),
            appendix_no=self._appendix_no,
            item_no=self._item_no,
            subitem_no=subitem_no,
            column_header=(
                self._column_header if self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO else None
            ),
            body_lines=[match.group("rest")],
        )

    def _open_appendix_point(self, line_no: int, match: re.Match[str]) -> None:
        """Start a point under the current appendix item."""
        self._flush()
        if self._appendix_no is None or self._item_no is None:
            self._skip(line_no, "appendix point outside any appendix item")
            return
        diem = match.group("letter")
        unit_id = appendix_point_id(self._doc_id, self._appendix_no, self._item_no, diem)
        if self._claim(unit_id, line_no, "duplicate appendix point"):
            return
        self._pending = _Pending(
            unit_type=UnitType.APPENDIX_POINT,
            id=unit_id,
            parent_id=appendix_item_id(self._doc_id, self._appendix_no, self._item_no),
            appendix_no=self._appendix_no,
            item_no=self._item_no,
            diem=diem,
            body_lines=[match.group("rest")],
        )

    def _open_appendix_column_header(self, match: re.Match[str]) -> None:
        """Record the premium table's own column header for later rows (issue #160).

        Reached only while ``self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO``
        (see the guard in ``feed``). Does not call ``self._flush()`` and does
        not touch ``self._pending`` -- this sentinel line, emitted by
        ``ingest/extract.py``'s own ``_restructure_appendix_6`` every time the
        table's header reprints at the top of a page, carries no body text of
        its own, only state ``_open_appendix_item``/``_open_appendix_subitem``
        read the next time either opens a row of this same table.
        """
        self._column_header = match.group("header")

    def _open_appendix_section(self, line_no: int, match: re.Match[str]) -> None:
        """Start a named section inside the current appendix (issue #147's fix round).

        Only ever reached while ``self._appendix_no == _PREMIUM_TABLE_APPENDIX_NO``
        (see the guard in ``feed``) -- Phụ lục VI's "Mục I"/"Mục II" prose is
        organized independently of the table's own numeric STT, so it gets its
        own id (``appendix_section_id``) under this same appendix, never
        colliding with a table row.
        """
        self._flush()
        if self._appendix_no is None:
            self._skip(line_no, "appendix section outside any appendix")
            return
        section = match.group("section")
        unit_id = appendix_section_id(self._doc_id, self._appendix_no, section)
        if self._claim(unit_id, line_no, "duplicate appendix section"):
            return
        self._appendix_section = section
        self._appendix_section_khoan = None
        self._pending = _Pending(
            unit_type=UnitType.APPENDIX_SECTION,
            id=unit_id,
            parent_id=appendix_id(self._doc_id, self._appendix_no),
            appendix_no=self._appendix_no,
            appendix_section=section,
            heading=_clean(match.group("heading")),
        )

    def _open_appendix_section_clause(self, line_no: int, match: re.Match[str]) -> None:
        """Start, or resume, a clause inside the current appendix section (issue #147).

        Khoản 1 Mục I's own body is split by the premium table itself: its
        intro sits before the table, its formula paragraph, "Ghi chú:", and
        nhóm-35 note sit right after the table's last row -- both reach this
        method through the exact same anchor (``APPENDIX_SECTION_CLAUSE_ANCHOR``),
        since ``ingest/extract.py``'s own ``_restructure_appendix_6`` re-opens
        the same sentinel the instant the table ends. When this clause's id
        was already flushed earlier in this appendix, that earlier unit is
        popped back out (``_pop_unit``) and its body becomes this
        occurrence's own starting body -- a merge, not a duplicate-eviction:
        ``_claim``'s own ``prefer_new`` discards the earlier text outright,
        which would be wrong here, since nothing about khoản 1's intro is
        stale, it is simply continuing after the table.
        """
        self._flush()
        if self._appendix_no is None or self._appendix_section is None:
            self._skip(line_no, "appendix section clause outside any section")
            return
        section = self._appendix_section
        khoan_no = int(match.group("khoan"))
        unit_id = appendix_section_clause_id(self._doc_id, self._appendix_no, section, khoan_no)
        body_lines = [match.group("rest")]
        if unit_id in self._seen_ids:
            existing = self._pop_unit(unit_id)
            if existing is not None and existing.body:
                body_lines = [existing.body, *body_lines]
        else:
            self._seen_ids.add(unit_id)
        self._appendix_section_khoan = khoan_no
        self._pending = _Pending(
            unit_type=UnitType.APPENDIX_SECTION_CLAUSE,
            id=unit_id,
            parent_id=appendix_section_id(self._doc_id, self._appendix_no, section),
            appendix_no=self._appendix_no,
            appendix_section=section,
            khoan_no=khoan_no,
            body_lines=body_lines,
        )

    def _open_appendix_section_point(self, line_no: int, match: re.Match[str]) -> None:
        """Start a lettered point inside the current appendix section clause (issue #147)."""
        self._flush()
        if (
            self._appendix_no is None
            or self._appendix_section is None
            or self._appendix_section_khoan is None
        ):
            self._skip(line_no, "appendix section point outside any section clause")
            return
        section = self._appendix_section
        khoan_no = self._appendix_section_khoan
        diem = match.group("letter")
        unit_id = appendix_section_point_id(
            self._doc_id, self._appendix_no, section, khoan_no, diem
        )
        if self._claim(unit_id, line_no, "duplicate appendix section point"):
            return
        self._pending = _Pending(
            unit_type=UnitType.APPENDIX_SECTION_POINT,
            id=unit_id,
            parent_id=appendix_section_clause_id(
                self._doc_id, self._appendix_no, section, khoan_no
            ),
            appendix_no=self._appendix_no,
            appendix_section=section,
            khoan_no=khoan_no,
            diem=diem,
            body_lines=[match.group("rest")],
        )

    def _open_appendix_form(self, line_no: int, match: re.Match[str]) -> None:
        """Start a sample form (``Mẫu số ...``) under the current appendix (issue #131).

        A form is a sibling of an appendix item, never nested under one and
        never split into its own ``muc``/``khoan``/``diem`` children -- see
        ``FORM_ANCHOR`` and the module docstring. ``prefer_new=True`` is what
        lets a form appendix's own table of contents name a form (``Mẫu số
        PC01`` with only its title as body) before the real form is reached
        later in the same appendix: the real occurrence evicts the earlier,
        thinner one, the same way a later clause is already preferred over an
        earlier one it continues (``_claim``).
        """
        self._flush()
        if self._appendix_no is None:
            self._skip(line_no, "form outside any appendix")
            return
        code = match.group("code").lower()
        unit_id = appendix_form_id(self._doc_id, self._appendix_no, code)
        self._item_no = None
        if self._claim(unit_id, line_no, "duplicate form", prefer_new=True):
            return
        self._form_code = code
        self._pending = _Pending(
            unit_type=UnitType.APPENDIX_FORM,
            id=unit_id,
            parent_id=appendix_id(self._doc_id, self._appendix_no),
            appendix_no=self._appendix_no,
            form_code=code,
            body_lines=[match.group("rest")],
        )

    def _reject_anchor_as_prose(self, line: str) -> None:
        """Treat an anchor-shaped line as ordinary text, not a real anchor.

        Used when ``ARTICLE_ANCHOR`` matches a line that is not actually this
        document's own heading (issues #54, #56) -- the line falls back to
        whatever a non-anchor line would do at this point: attach to the
        pending unit's body once real structure has started, or accumulate
        into the preamble before it has.
        """
        if self._preamble_done:
            self._append_body(line)
        else:
            self._preamble_lines.append(line)

    def _append_body(self, line: str) -> None:
        """Attach a non-anchor line to the pending unit, or drop it as preamble."""
        if self._pending is None:
            logger.debug("Ignoring text outside any unit: %r", line)
            return
        is_chapter_title = (
            self._pending.unit_type is UnitType.CHAPTER
            and self._pending.heading is None
            and not self._pending.body_lines
            and line.strip()
        )
        if is_chapter_title:
            self._pending.heading = line.strip()
            return
        self._pending.body_lines.append(line)

    def _claim(self, unit_id: str, line_no: int, reason: str, *, prefer_new: bool = False) -> bool:
        """Reserve an identifier, or resolve a second claim on one already taken.

        Args:
            unit_id: Identifier the anchor would produce.
            line_no: 1-based line number of the anchor.
            reason: Message logged when the identifier is already taken and
                the first occurrence is kept (``prefer_new`` is ``False``).
            prefer_new: When ``True``, a colliding identifier is reassigned to
                this later occurrence -- evicting the unit already built for
                the earlier one -- instead of keeping the first occurrence.
                Used by clauses, where an id that reaches this call has
                already passed ``_open_clause``'s figure-legend/monotonic
                check and is therefore more trustworthy than whatever claimed
                the id first; and by forms (issue #131), where a form
                appendix's own table of contents can name the same form
                before its real content is reached later in the same
                appendix -- the real occurrence is always the one worth
                keeping. Whichever occurrence is not kept is always logged at
                WARNING, so the loss stays visible in the anomaly log either
                way.

        Returns:
            ``True`` when the caller must stop building a unit for this
            anchor because the first occurrence is being kept. ``False``
            means the identifier is available to this occurrence -- either it
            was free, or (with ``prefer_new``) it was just reassigned here.
        """
        if unit_id in self._seen_ids:
            if prefer_new:
                self._evict(unit_id, line_no)
                return False
            self._skip(line_no, f"{reason} {unit_id}")
            return True
        self._seen_ids.add(unit_id)
        return False

    def _pop_unit(self, unit_id: str) -> Unit | None:
        """Remove and return the previously flushed unit with ``unit_id``, if any.

        Used by ``_open_appendix_section_clause`` (issue #147) to *merge* a
        clause whose body the premium table splits in two, not to discard it
        the way ``_evict`` does for a genuine duplicate -- the caller is
        responsible for folding the returned unit's own body back into the
        new occurrence.
        """
        for index, unit in enumerate(self._units):
            if unit.id == unit_id:
                return self._units.pop(index)
        return None

    def _evict(self, unit_id: str, line_no: int) -> None:
        """Drop the previously kept unit with ``unit_id`` in favor of a later occurrence.

        Args:
            unit_id: Identifier of the unit being replaced.
            line_no: 1-based line number of the occurrence that replaces it,
                reported in the anomaly log.
        """
        self._units = [unit for unit in self._units if unit.id != unit_id]
        self._skipped += 1
        logger.warning(
            "%s line %d: an earlier occurrence of %s was evicted in favor of "
            "this later one, which continues the article's clause sequence",
            self._doc_id,
            line_no,
            unit_id,
        )

    def _skip(self, line_no: int, reason: str) -> None:
        """Drop the current anchor, leaving no pending unit to collect its body."""
        self._skipped += 1
        self._pending = None
        logger.warning("%s line %d: skipped - %s", self._doc_id, line_no, reason)

    def _flush(self) -> None:
        """Turn the pending unit into a ``Unit`` and append it in reading order."""
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        self._units.append(
            Unit(
                id=pending.id,
                document_id=self._doc_id,
                unit_type=pending.unit_type,
                chapter_no=pending.chapter_no,
                dieu_no=pending.dieu_no,
                khoan_no=pending.khoan_no,
                diem=pending.diem,
                appendix_no=pending.appendix_no,
                appendix_section=pending.appendix_section,
                item_no=pending.item_no,
                subitem_no=pending.subitem_no,
                form_code=pending.form_code,
                column_header=pending.column_header,
                parent_id=pending.parent_id,
                heading=pending.heading,
                body=_join_body(pending.body_lines),
                ordinal=len(self._units),
            )
        )


def _join_body(lines: list[str]) -> str:
    """Join a unit's body lines, dropping blank lines at either end."""
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def _clean(value: str) -> str | None:
    """Return a stripped heading, or ``None`` when the source printed none."""
    stripped = value.strip()
    return stripped or None


def chapter_numeral_to_int(numeral: str) -> int:
    """Convert a chapter number written in Roman or Arabic digits to an integer.

    Public (not ``_``-prefixed) because ``ingest/graph.py`` needs the exact
    same conversion to resolve an internal chapter reference (e.g. "Chương
    II") back to a chapter id (build plan A6 Phase 2.2) -- reimplementing
    Roman numeral parsing a second time would risk it silently diverging
    from what this parser actually accepts as a chapter anchor.
    """
    if numeral.isdigit():
        return int(numeral)
    total = 0
    highest = 0
    for char in reversed(numeral):
        value = _ROMAN_VALUES[char]
        total = total - value if value < highest else total + value
        highest = max(highest, value)
    return total
