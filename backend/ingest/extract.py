r"""Extract parser-ready plain text from downloaded corpus files (build plan A6 Phase 1.6a).

``ingest/parser.py`` anchors structure on line starts -- ``^Chương``, ``^Điều\s+\d+``,
``^\d+\.``, ``^[a-zđ]\)``. HTML is the primary corpus format because its block
elements map cleanly onto those anchors: one line per ``<p>``/``<div>``/... survives
as one line of parser input. PDF is the fallback and is handled honestly, not
perfectly -- pages wrap text to column width rather than to sentences, and repeat
headers and footers mid-document, so the PDF path reflows wrapped lines using the
parser's own anchors to decide where a genuine new block starts, and drops lines
that repeat at the same page position across most pages.

A document that extracts to empty or near-empty text is worse than one that raises:
it parses to zero units with no error, and the cite-or-refuse rule then makes the
assistant silently refuse questions the corpus should answer. Every path here raises
``ExtractionError`` rather than returning too little text to be useful -- and, per the
same contract, rather than letting a parsing library's own exception through instead
(build plan A6 issue #33): a corrupt or truncated PDF raises ``pypdf.errors.PyPdfError``
from inside ``PdfReader``, which ``extract_text`` catches and re-raises as
``ExtractionError`` with the original preserved as ``__cause__``. selectolax has no
comparable failure mode to wrap -- it was measured, not assumed, against null bytes,
binary garbage, and unclosed tags, and it always returns a DOM rather than raising;
the near-empty-text floor below is what actually catches malformed HTML in practice.

A pypdf-extracted document number can arrive with its letters split apart
(``N Đ-CP``, ``NĐạCP``), so the citation is captured cut short (``87/2018/N``) and
the graph cannot represent it -- issue #80. ``extract_text`` repairs those numbers
once, on the finished text of either format; see ``_rejoin_split_document_numbers``
for why the repair lives here and not in ``ingest/references.py``.

Word 97-2003 (``.doc``) is a real corpus format -- some Công báo publications are
downloadable only as ``.doc`` -- but is deliberately not supported here; see
``SOURCES.md`` for the decision and the owner's manual conversion step.
"""

import logging
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError
from selectolax.lexbor import LexborHTMLParser as HTMLParser
from selectolax.lexbor import LexborNode as Node

from ingest.parser import (
    APPENDIX_6_NO,
    APPENDIX_CAPTION,
    APPENDIX_COLUMN_HEADER_ANCHOR,
    APPENDIX_COLUMN_HEADER_MARKER,
    APPENDIX_HEADING,
    APPENDIX_SECTION_ANCHOR,
    APPENDIX_SECTION_CLAUSE_ANCHOR,
    APPENDIX_SECTION_MARKER,
    APPENDIX_SECTION_POINT_ANCHOR,
    ARTICLE_ANCHOR,
    CHAPTER_ANCHOR,
    CLAUSE_ANCHOR,
    FORM_ANCHOR,
    POINT_ANCHOR,
    RECIPIENT_START,
    SIGNATURE_START,
    chapter_numeral_to_int,
)

logger = logging.getLogger(__name__)

_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "li",
        "ul",
        "ol",
        "br",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "tr",
        "td",
        "th",
        "blockquote",
        "section",
        "article",
    }
)
_DROP_TAGS = frozenset({"script", "style", "nav", "header", "footer", "noscript", "form"})
"""``form`` covers a site's search/filter widgets (measured on a real corpus
page: a document-type radio-button filter) -- legal article text is never
meaningfully held inside a form control, so dropping the whole tag needs no
narrower, page-specific signal the way the chrome in ``_DROP_CLASSES`` does."""

_MIN_TEXT_LENGTH = 200
"""A real corpus document runs to many pages; a scanned PDF or a stripped-down HTML
shell extracts to at most a few dozen characters. This floor is a coarse fail-safe,
not a quality bar -- ``parse_document`` is what actually validates structure."""


class ExtractionError(Exception):
    """Raised when a corpus file cannot be turned into usable plain text."""


def extract_text(path: Path) -> str:
    """Extract parser-ready plain text from one downloaded corpus file.

    Args:
        path: Path to an ``.html`` or ``.pdf`` corpus file. This function knows
            nothing about the manifest or the corpus layout -- it only reads the
            one file it is given.

    Returns:
        Plain text with one line per structural block, ready for ``parse_document``.

    Raises:
        ExtractionError: If the suffix is unsupported, the file cannot be
            parsed as its format (e.g. a truncated or corrupt PDF), or the
            extracted text is empty or too sparse to be usable -- most
            importantly, a scanned PDF with no text layer.
    """
    suffix = path.suffix.lower()
    if suffix == ".html":
        text = _extract_html(path)
    elif suffix == ".pdf":
        text = _extract_pdf_or_raise(path)
    else:
        raise ExtractionError(_unsupported_suffix_message(path, suffix))

    text = _rejoin_split_document_numbers(text, path)

    if len(text.strip()) < _MIN_TEXT_LENGTH:
        msg = (
            f"{path} extracted to only {len(text.strip())} character(s) of text -- "
            "treating this as a failed extraction (e.g. a scanned PDF with no text "
            "layer) rather than silently ingesting an empty document."
        )
        raise ExtractionError(msg)
    return text


def extract_text_parts(paths: Iterable[Path]) -> str:
    """Extract and concatenate a document's ordered source file(s) (issue #126).

    A document declared with a single ``path`` (``ingest.manifest.Manifest
    .parts_of`` returns a one-element tuple for it) is unaffected: this is
    exactly ``extract_text(paths[0])``, byte-identical to before this function
    existed. A document declared with ``parts`` -- decree 105/2025 was
    published across two issues of the Official Gazette -- has each part
    extracted on its own, in order, and every part after the first has its
    own continuation preamble stripped (``_strip_continuation_preamble``)
    before the parts are joined into one string, so ``parse_document`` only
    ever sees one document's worth of real content and never the Gazette's
    own running header or the decree's own title repeated at the top of the
    next issue.

    Args:
        paths: One or more corpus file paths, in reading order.

    Returns:
        The concatenated plain text, ready for ``parse_document``.

    Raises:
        ExtractionError: If any part fails to extract, propagated unchanged
            from ``extract_text``.
    """
    texts = [extract_text(path) for path in paths]
    merged = [texts[0]]
    for continuation_path, continuation_text in zip(list(paths)[1:], texts[1:], strict=True):
        merged.append(_strip_continuation_preamble(continuation_text, continuation_path))
    return "\n".join(merged)


_CONTINUATION_MARKER = re.compile(r"\(Tiếp theo Công báo số[^)]*\)")
"""Marks a continuation part's own repeated title line -- e.g. "Nghị định số
105/2025/NĐ-CP ngày 15 tháng 5 năm 2025 ... (Tiếp theo Công báo số 717 +
718)" -- the Official Gazette's own convention for "this issue continues a
document begun in an earlier one" (measured on decree 105/2025's real second
part, issue #126). Nothing else in the corpus ever uses this phrase, so it is
a safe, narrow signal: a continuation part's front matter (the Gazette's own
running header, then the issuing body's name repeated in capitals, then this
title line) is dropped up to and including the line carrying it, and nothing
is stripped at all when the marker is not found -- a part that does not
open this way is left completely untouched rather than guessed at (C12)."""


def _strip_continuation_preamble(text: str, path: Path) -> str:
    """Drop a continuation part's own front matter, up to its repeated title line.

    Without this, the Gazette's running header and the repeated title line
    that open a continuation part would reach ``parse_document`` as ordinary
    text and be silently appended onto whatever unit was still open at the
    end of the *previous* part -- here, the last appendix item of Phụ lục VI
    -- corrupting data that must stay byte-identical across a re-ingest
    (issue #126). Searched only in the first few lines, where the Gazette's
    own convention always places this block; a marker appearing deep inside
    a part's real content (not a real scenario, but not assumed impossible
    either) is deliberately left alone.

    Args:
        text: One continuation part's own extracted text.
        path: That part's source file, for the log message only.

    Returns:
        ``text`` with its continuation preamble removed, or ``text``
        unchanged if no continuation marker was found.
    """
    lines = text.split("\n")
    search_window = min(len(lines), 5)
    for index in range(search_window):
        if _CONTINUATION_MARKER.search(lines[index]):
            logger.info(
                "%s: stripped %d-line continuation preamble before %r",
                path.name,
                index + 1,
                lines[index + 1] if index + 1 < len(lines) else "<end of text>",
            )
            return "\n".join(lines[index + 1 :])
    return text


_WORD_SUFFIXES = frozenset({".doc", ".docx"})


def _unsupported_suffix_message(path: Path, suffix: str) -> str:
    """Build the rejection message for a suffix ``extract_text`` does not handle.

    ``.doc``/``.docx`` get a pointer to the documented conversion step (build
    plan A6 issue #33: Word 97-2003 is a real corpus format, deliberately not
    supported directly here) rather than a bare rejection, since that is
    genuinely the fix a caller needs, not just an explanation.
    """
    if suffix in _WORD_SUFFIXES:
        return (
            f"Unsupported corpus file type {suffix!r}: {path}. Word documents are not "
            "extracted directly -- convert with LibreOffice first and point the "
            'manifest at the resulting .html (see SOURCES.md, "Format" section).'
        )
    return f"Unsupported corpus file type {suffix!r}: {path}"


def _extract_pdf_or_raise(path: Path) -> str:
    """Run ``_extract_pdf``, converting any pypdf parsing failure to ``ExtractionError``.

    ``pypdf.PdfReader`` raises its own exception hierarchy (``PdfStreamError``,
    ``EmptyFileError``, ...) for a truncated, corrupt, or otherwise unreadable
    PDF -- all of them subclass ``pypdf.errors.PyPdfError``. Left unwrapped,
    that exception reaches every caller of ``extract_text`` even though its
    signature promises only ``ExtractionError``; ``ingest/run.py`` happens to
    survive today only because it catches bare ``Exception`` around the whole
    pipeline, which is not a substitute for the documented contract.
    """
    try:
        return _extract_pdf(path)
    except PyPdfError as error:
        msg = f"{path}: failed to read as a PDF ({error})"
        raise ExtractionError(msg) from error


# --- document numbers (both formats) ------------------------------------------------


_H = r"[^\S\n]*"
"""Any amount of horizontal whitespace, and never a line break. A line break is a
block boundary (a wrapped article heading closes its own line, a paragraph ends), so
a number is only ever repaired *within* a block: gluing across a boundary would change
the document's structure, not just a number."""

_SLASH = rf"{_H}/{_H}"
_HYPHEN = rf"(?:{_H}-{_H}|ạ)"
_DOCUMENT_CODE = (
    rf"(?:(?P<nd>N{_H}Đ{_HYPHEN}CP)"
    rf"|(?P<qd>Q{_H}Đ{_HYPHEN}TTg)"
    rf"|(?P<tt>TT{_HYPHEN}(?P<agency>[A-ZĐ]{{2,}}))"
    rf"|(?P<qh>QH(?P<khoa>\d{{1,2}})))"
    r"(?![0-9A-Za-zĐđ])"
)
_DOCUMENT_NUMBER = re.compile(
    r"(?:(?<=[Ss]ố )(?P<split>\d(?: \d){1,3})|(?<!\d)(?P<serial>\d{1,4}))"
    rf"{_SLASH}(?P<year>\d{{4}}){_SLASH}{_DOCUMENT_CODE}"
)
r"""A document number written in one of the closed forms a Vietnamese number can end
in -- ``NĐ-CP``, ``QĐ-TTg``, ``TT-<agency>``, ``QH<nn>`` -- with the damage pypdf and
the corpus's fonts really do to it (issue #80, measured on all seven acquired
documents), and *only* that damage:

* whitespace around either slash (``68/ 2006/QH11``, ``116/2017/ NĐ-CP``);
* a space between ``N`` (or ``Q``) and ``Đ``, the stroked letter pypdf splits off
  (``N Đ-CP``), and either side of the hyphen (``NĐ -CP``, ``NĐ- CP``);
* the hyphen decoded as ``ạ`` (``NĐạCP``, ``TTạBCT``, from one PDF's font map);
* a space inside the leading digits, but only right after the word ``số`` that
  introduces a number (``số 5 0/2024/NĐ-CP`` for ``số 50/2024/NĐ-CP``), where the
  split reading is the only one that is a number at all.

Everything is anchored on a full ``<serial>/<year>/<code>`` shape, so the pattern
cannot join "letters near a slash": a date (``15/10/2019``), an area (``12 m2``), a
citation coordinate followed by a number (``khoản 2 15/2020/NĐ-CP``) and a run of
distinct numbers (``số 12 15/2020/NĐ-CP``) all fail to match, and an issuer whose own
letters are split (``TT-B CA``) is left broken, never guessed at. ``ạ`` is an
ordinary Vietnamese letter, so it is only ever read as a hyphen *between* ``NĐ``/
``TT``/``QĐ`` and the rest of a code, with a number in front of it."""


def _canonical_document_number(match: re.Match[str]) -> str:
    """Return the number ``match`` was written as, without the damage."""
    serial = "".join((match["split"] or match["serial"]).split())
    if match["nd"]:
        code = "NĐ-CP"
    elif match["qd"]:
        code = "QĐ-TTg"
    elif match["tt"]:
        code = f"TT-{match['agency']}"
    else:
        code = f"QH{match['khoa']}"
    return f"{serial}/{match['year']}/{code}"


def _rejoin_split_document_numbers(text: str, path: Path) -> str:
    """Repair every document number in ``text`` that extraction split apart.

    **Why here, and not in ``ingest/references.py``** (issue #80). A hint is
    already cut short by the time ``references.py`` sees it: ``EXTERNAL_PATTERN``
    captures ``N Đ-CP`` as ``87/2018/N``, so nothing left in the hint can say what
    the number really was, and a repair there could only run on text a caller
    happens to still hold (issue #68's repeal reader did). Repaired once, on the
    extracted text, every consumer reads a whole number: the references, the
    repeal reader, the unit bodies a reviewer is shown as evidence, and the chunk
    text that is embedded. It is the same family as ``_rejoin_split_digits``
    (issue #72), which repaired an article number. This is the one implementation of
    the rule; ``references.py`` no longer carries a second, partial one.

    Runs on the finished text of either format, after reflow, and never across a
    line break (see ``_H``): a number that straddles a block boundary -- a wrapped
    article heading -- is left as it is rather than restructure the document.
    Every rejoin is logged, the same as ``_rejoin_split_words``, so the repair
    stays reviewable: what was merged, not just how many.
    """

    def _repair(match: re.Match[str]) -> str:
        repaired = _canonical_document_number(match)
        if repaired != match.group(0):
            logger.info(
                "%s: rejoined split document number %r into %r",
                path.name,
                match.group(0),
                repaired,
            )
        return repaired

    return _DOCUMENT_NUMBER.sub(_repair, text)


# --- HTML -------------------------------------------------------------------------


def _extract_html(path: Path) -> str:
    """Render an HTML corpus file as one line per block element."""
    html = path.read_text(encoding="utf-8", errors="replace")
    tree = HTMLParser(html)
    root = tree.body or tree.root
    if root is None:
        msg = f"{path} has no parseable HTML content"
        raise ExtractionError(msg)
    lines = _block_lines(root)
    return "\n".join(lines)


def _block_lines(root: Node) -> list[str]:
    """Walk the DOM in document order, emitting one line per block-level element.

    Text inside inline elements (``<b>``, ``<span>``, ...) is folded into the
    surrounding line; a block element flushes whatever text has accumulated so
    far, so nested blocks each land on their own line.
    """
    lines: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        text = _normalize_whitespace("".join(buffer))
        if text:
            lines.append(text)
        buffer.clear()

    def walk(node: Node) -> None:
        if node.tag == "-text":
            if node.text_content:
                buffer.append(node.text_content)
            return
        if node.tag in _DROP_TAGS or _has_drop_class(node):
            return
        is_block = node.tag in _BLOCK_TAGS
        if is_block:
            flush()
        for child in node.iter(include_text=True):
            walk(child)
        if is_block:
            flush()

    walk(root)
    flush()
    return lines


_DROP_CLASSES = frozenset({"navbar", "dropdown-menu", "badge"})
"""Site chrome that an aggregator renders inside ordinary tags rather than
inside ``<nav>``/``<header>``/``<footer>`` (build plan A6 issue #34): a
Bootstrap-style menu as ``<div class="navbar">`` (measured on a real corpus
page), and a small UI status badge such as ``<font class="badge ...
sdbs-badge">Sửa đổi, bổ sung</font>`` glued directly onto the very paragraph
holding a clause's real text, with no block boundary of its own to separate
it -- ``_BLOCK_TAGS`` alone cannot catch this, since ``<font>`` is inline and
its text folds straight into the surrounding line by design. Class name
only, not full context, so this also catches the same chrome wrapped in a
different tag on a different aggregator page."""


def _has_drop_class(node: Node) -> bool:
    """Return whether ``node``'s own ``class`` attribute names a dropped class."""
    classes = (node.attributes.get("class") or "").split()
    return not _DROP_CLASSES.isdisjoint(classes)


def _normalize_whitespace(text: str) -> str:
    """Collapse any run of whitespace -- including ``&nbsp;`` -- to single spaces."""
    return " ".join(text.split())


# --- PDF --------------------------------------------------------------------------


def _extract_pdf(path: Path) -> str:
    """Extract a PDF's text, dropping running headers/footers and reflowing wraps."""
    reader = PdfReader(path)
    pages = [_raw_page_lines(page.extract_text()) for page in reader.pages]
    pages = _drop_running_furniture(pages)
    pages = [
        [_rejoin_split_words(_rejoin_split_digits(line, path), path) for line in page]
        for page in pages
    ]
    raw_lines = [line for page in pages for line in page]
    raw_lines = _split_danger_table_rows(raw_lines)
    raw_lines = _restructure_appendix_6(raw_lines)
    raw_lines = _strip_gazette_colophon(raw_lines, path)
    lines = _reflow_lines(raw_lines, path)
    return "\n".join(lines)


def _raw_page_lines(page_text: str) -> list[str]:
    """Return one page's non-blank lines, as pypdf's layout heuristic broke them."""
    return [
        collapsed
        for line in page_text.splitlines()
        if (collapsed := _collapse_line_whitespace(line))
    ]


def _collapse_line_whitespace(line: str) -> str:
    """Collapse runs of horizontal whitespace within one line to a single space.

    Operates on exactly one already-``splitlines()``-separated line and never
    joins lines or touches a line break -- there is none left in ``line`` to
    touch, and nothing downstream of this function ever sees more than one raw line
    at a time until ``_reflow_lines`` makes that joining decision explicitly,
    anchored on the parser's own line-based anchors (issue #153).

    pypdf's own layout heuristic changed between 6.19.0 and 6.21.2: the newer
    release inserts an extra space between words in many places it previously
    emitted exactly one ("tháng 12  năm" for "tháng 12 năm", "Tên hóa chất
    theo" picking up a double gap before "theo") -- not new content, just a
    wider gap in the heuristic's own column-width estimate. Collapsing here,
    before any anchor or split-word regex in this module ever reads the line,
    removes that drift at the source rather than chasing it through every
    pattern that reads raw PDF text. A split word genuinely separated by one
    space (e.g. "đường ố ng") is unaffected -- collapsing a run of *more than
    one* whitespace character down to one never touches a line that already
    has only one.
    """
    return " ".join(line.split())


_SPLIT_ONSETS = frozenset(
    {
        # Every native Vietnamese syllable-initial consonant spelling -- single
        # letters and the standard digraphs/trigraph -- lowercased for lookup.
        "b",
        "c",
        "d",
        "đ",
        "g",
        "h",
        "k",
        "l",
        "m",
        "n",
        "p",
        "q",
        "r",
        "s",
        "t",
        "v",
        "x",
        "ch",
        "gh",
        "gi",
        "kh",
        "ng",
        "nh",
        "ph",
        "qu",
        "th",
        "tr",
        "ngh",
    }
)
"""Every valid Vietnamese syllable onset, and nothing else. A Vietnamese
syllable always has a vowel nucleus (build plan A6 issue #37's own safety
argument, for a single bare consonant) -- this is that same fact applied to
the full closed set of onset spellings: none of these strings contains a
vowel, so none of them can *ever* be a complete word on its own, no matter
how common the syllable that follows happens to be elsewhere.

That last clause matters and was checked, not assumed (build plan A6 issue
#39): a naive rule of "rejoin unless the continuation is a real syllable"
would refuse to fix ``"ch ứng"`` -> ``"chứng"``, because ``"ứng"`` is
independently a very common Vietnamese word (confirmed empirically -- 103
occurrences of onset-less ``"ứng"`` in this corpus's own clean, uncorrupted
HTML documents, in ``"đáp ứng"``, ``"phản ứng"``, ``"ứng dụng"``). But
``"ch"`` is never a word by itself, so ``"ch ứng"`` can never genuinely be
two separate words -- it is safe to merge regardless of what the
continuation means elsewhere. The rejoin trigger therefore checks the
*fragment*, not the continuation: is what precedes the space a complete
possible word, or is it structurally incapable of being one?

A fragment that already contains a vowel (``"ghi"``, ``"thu"``, ``"xu"``) is
excluded for the opposite reason -- each of those *is* an independently
real Vietnamese word (``"ghi"`` "to note", ``"thu"`` "autumn/to collect",
``"xu"`` "cent/trend"), so ``"ghi ệp"``, ``"thu ộc"``, ``"xu ất"`` are left
broken rather than risk silently merging across a genuine word boundary --
per the issue's own instruction, an ambiguous case stays broken."""

_DIACRITIC_VOWELS = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"

_LETTER = "a-zA-ZÀ-ỹ"

_SPLIT_WORD = re.compile(rf"(?<![{_LETTER}])([A-Za-zĐđ]{{1,3}}) ([{_DIACRITIC_VOWELS}][a-zà-ỹ]*)")
"""Matches an isolated 1-to-3-letter run, split by a stray space from a
vowel-initial continuation on the *same* raw line -- e.g. ``"B ổ"`` (for
``"Bổ"``), ``"Ngh ị"`` (for ``"Nghị"``), ``"ph ải"`` (for ``"phải"``). The
lookbehind requires the run to start right after a non-letter, and the
``{1,3}`` is immediately followed by a literal space, so this only ever
matches a *complete* isolated letter-run of length 1-3 -- a longer run
(a genuine short word like ``"khỏi"``) never has a length-1/2/3 prefix that
lands exactly on a space, so it is never partially captured.

Three letters is the ceiling because the longest Vietnamese onset spelling
(``"ngh"``) is three letters; a matched run longer than that could not be a
pure onset anyway. Whether a given match is actually safe to rejoin is
decided in ``_rejoin_split_words`` by checking the captured run against
``_SPLIT_ONSETS`` -- this pattern alone only narrows down *candidates*.

Operates per raw line, before ``_reflow_lines`` ever runs, so a split that
happens to straddle a page or reflow line break is never joined across it --
the same boundary the sentence/anchor logic in ``_reflow_lines`` respects.
"""


def _rejoin_split_words(line: str, path: Path) -> str:
    """Rejoin a pypdf-split word whose fragment cannot possibly stand alone.

    Every rejoin is logged so the fix stays reviewable: a sample of exactly
    what was merged, not just a count, per document. A candidate whose
    fragment is not a pure Vietnamese onset (``_SPLIT_ONSETS``) is left
    untouched -- it might be a genuine short word, and an ambiguous case
    stays broken rather than risk altering the statute's actual wording.
    """

    def _merge(match: re.Match[str]) -> str:
        before, after = match.group(1), match.group(2)
        if before.lower() not in _SPLIT_ONSETS:
            return match.group(0)
        logger.info("%s: rejoined split word %r into %r", path.name, match.group(0), before + after)
        return before + after

    return _SPLIT_WORD.sub(_merge, line)


_SPLIT_ARTICLE_NUMBER = re.compile(r"^(?P<prefix>Điều\s+)(?P<digits>\d(?:\s+\d)+)(?P<punct>\s*\.)")
r"""Matches a pypdf-split article number at the true anchor position -- the
very start of the line, right after ``Điều`` -- where the joined digits are
immediately followed by the period a genuine heading's own number actually
ends with (the same terminator ``ARTICLE_ANCHOR`` expects): e.g.
``"Điều  2 3 .  Cấp phép"`` for the real ``"Điều 23."``.

Real-corpus measurement (build plan A6 issue #72) found this exact shape --
each digit of a multi-digit article number pushed onto its own token by
pypdf -- in exactly one of the seven acquired documents,
``pccc-decree-105-2025``, for articles 23, 30 and 44, and nowhere else. The
same survey found no split ``Chương`` number and no split clause number
anywhere in the corpus, so neither gets a rule here: C12 forbids handling a
shape the corpus does not actually show.

Anchored to line start and gated on the trailing period, this can never
touch a legitimate digit sequence living anywhere else in the corpus's
prose -- a date (``"ngày 18 tháng 02 năm 2025"``), an area
(``"12 m2"``), or a citation list (``"khoản 1 và 2 Điều 30"``): none of
those sits at the exact position right after a line-opening ``Điều``, and
even a bare mid-sentence citation like ``"...Điều 30 và..."`` that did
reflow onto its own line would still fail the immediately-following-period
requirement. ``\d(?:\s+\d)+`` also requires every token between the digits
to be a single digit with nothing else between them, so ``"Điều 1 và 2"``
(two distinct numbers joined by a word, not a split single number) can never
match: after the first digit, ``(?:\s+\d)`` demands a digit right after the
whitespace, and "và" is not one."""


def _rejoin_split_digits(line: str, path: Path) -> str:
    """Rejoin a pypdf-split article number at the anchor position.

    Every rejoin is logged, the same as ``_rejoin_split_words``, so the fix
    stays reviewable: a sample of exactly what was merged, not just a count,
    per document.
    """

    def _merge(match: re.Match[str]) -> str:
        joined = re.sub(r"\s+", "", match.group("digits"))
        replacement = match.group("prefix") + joined + match.group("punct")
        logger.info(
            "%s: rejoined split article number %r into %r", path.name, match.group(0), replacement
        )
        return replacement

    return _SPLIT_ARTICLE_NUMBER.sub(_merge, line)


_MIN_PAGES_FOR_FURNITURE_DETECTION = 3
"""A running header or footer is defined by repeating across pages; below this many
pages there is nothing to compare against, so a 1-page document's only first/last
line would otherwise always look like "furniture repeated on every page"."""

_DIGIT_RUN = re.compile(r"\d+")


def _drop_running_furniture(pages: list[list[str]]) -> list[list[str]]:
    """Drop a line that repeats, possibly only in shape, as the first or last line of a page.

    A page header or footer is furniture repeated at the same position on most
    pages, not content -- e.g. a document title stamped at the top of every page,
    or a "Trang N" / "Page N" / "- N -" footer. Comparison is by *shape*, with
    digit runs normalized away, so a page-number footer that differs per page is
    still recognized as the same repeating shape; a constant header with no digits
    normalizes to itself, so exact-match furniture is still caught. A line that
    only sometimes repeats, or repeats but moves position, is left alone.
    """
    if len(pages) < _MIN_PAGES_FOR_FURNITURE_DETECTION:
        return pages
    header_shape = _majority_shape(page[0] for page in pages if page)
    footer_shape = _majority_shape(page[-1] for page in pages if page)
    cleaned: list[list[str]] = []
    for page in pages:
        trimmed = list(page)
        if trimmed and header_shape is not None and _furniture_shape(trimmed[0]) == header_shape:
            trimmed = trimmed[1:]
        if trimmed and footer_shape is not None and _furniture_shape(trimmed[-1]) == footer_shape:
            trimmed = trimmed[:-1]
        cleaned.append(trimmed)
    return cleaned


def _furniture_shape(line: str) -> str:
    """Normalize a line for furniture comparison, collapsing digit runs.

    "Trang 1" and "Trang 2" compare equal under this shape, so a page-number
    footer is recognized as one repeating shape even though its digits differ.
    """
    return _DIGIT_RUN.sub("#", line)


def _majority_shape(candidates: Iterable[str]) -> str | None:
    """Return the normalized shape that accounts for more than half of ``candidates``."""
    shapes = [_furniture_shape(candidate) for candidate in candidates]
    counts = Counter(shapes)
    if not counts:
        return None
    shape, count = counts.most_common(1)[0]
    return shape if count > sum(counts.values()) / 2 else None


_DANGER_TABLE_HEADER = re.compile(r"^TT\s+Loại hình cơ sở\s+Nhóm 1\s+Nhóm 2\s*$")
"""Phụ lục II's own column header (pccc-decree-105-2025's danger-category
table), reprinted at the top of every page the table spans -- measured on
the real corpus, it repeats on 6 of this document's ~90 pages, too few for
``_drop_running_furniture``'s majority-shape rule to ever catch it. Used by
``_split_danger_table_rows`` only to know when the table starts (and to drop
each repeat, which is not itself a row); nothing else in this module reads
it."""

_DANGER_TABLE_ROW_START = re.compile(r"^(?P<stt>\d{1,2})(?=\s|$)\s*(?P<rest>.*)$")
"""A bare STT cell opening one row of Phụ lục II's own table, at the start of
a raw pypdf line -- e.g. ``"1 Nhà chung cư, nhà ở tập thể"``, or a bare
``"39"`` alone when the next cell's text is too long to share the line. The
lookahead mirrors ``CLAUSE_ANCHOR``'s own fix for the same issue (#141): a
thousands separator like "1.000 m²" has a digit immediately after where a
dot would be, not whitespace or end-of-line, except there is no dot here at
all to begin with -- "1.000" simply never starts with a *bare* "1" followed
by whitespace, so this pattern alone could not tell a genuine STT "1" apart
from the "1" that opens "1.000" without the caller in
``_split_danger_table_rows`` also checking the matched number against the
STT sequence seen so far (strictly greater than the last one opened) -- an
accidental "1 " prefix appearing mid-cell would otherwise be read as a new
row.
"""

_GAZETTE_FOOTER_LINE = re.compile(
    r"^\s*\d{0,3}\s*CÔNG BÁO/Số[^/\n]*/Ngày\s+\d{1,2}-\d{1,2}-\d{4}\s*\d{0,3}\s*$"
)
"""Công Báo's running footer, as a raw line entirely on its own -- in either
of the two shapes pypdf hands back depending on which physical page side it
prints on: the page number first (``"78   CÔNG BÁO/Số .../Ngày ..."``) or
last (``"CÔNG BÁO/Số .../Ngày ... 77"``). ``_drop_running_furniture`` only
drops whichever one shape is the majority across the *whole* document, so
Phụ lục II's table -- which alternates between the two shapes exactly like
the rest of the document -- still has the minority shape's footer survive
as an ordinary raw line, landing mid-table between two of its rows. Used
only by ``_split_danger_table_rows``, which can drop it outright: the
generic ``_reflow_lines``/``_sentence_is_finished`` machinery that reads
this same footer shape *outside* the table is untouched by this pattern."""


_GAZETTE_FOOTER_MID_LINE = re.compile(
    r"\s*CÔNG BÁO/Số[^/\n]*/Ngày\s+\d{1,2}-\d{1,2}-\d{4}\s*\d{0,3}\s*"
)
"""The same footer phrase as ``_GAZETTE_FOOTER_LINE``, but with no line
anchor on either side -- matched (and removed) wherever it occurs inside a
raw PDF line, not only when it fills the whole line (issue #148).
``_drop_running_furniture`` and ``_GAZETTE_FOOTER_LINE`` both require the
footer to be a *whole* raw line; pypdf instead often fuses it onto real
content on both sides of the very same line, with the page's trailing text
before it and the next page's leading text right after -- e.g. "... Điều 12
Nghị định CÔNG BÁO/Số 233 + 234/Ngày 22-02-2020 19 số 15/2018/NĐ-CP ..."
(measured on the real corpus: gas-decree-17-2020's own Điều 10), where the
footer is not only noise but also splits the document number that follows
it in two.

Deliberately asymmetric, unlike ``_GAZETTE_FOOTER_LINE``: only a *trailing*
page number (after ``/Ngày ...``) is matched and removed here, never a
leading one (before ``CÔNG BÁO``). A trailing digit is unambiguous -- it is
the page-number-last shape ``_GAZETTE_FOOTER_LINE`` already documents, right
up against the footer's own text with nothing of the real document between
them. A digit *before* ``CÔNG BÁO`` on a line that also carries other real
content is not: real-corpus measurement (``test_pdf_a_table_row_ending_in_
a_digit_does_not_block_the_next_heading``'s own fixture, a QCVN
chemical-substance table) shows a bare trailing digit is routinely the
table's own last column, not a page number that happens to immediately
precede a footer fused onto the same line -- e.g. "... C2H3Br 18 CÔNG
BÁO/Số 233 + 234/Ngày 22-02-2020" keeps its "18". A line that is *nothing
but* the footer, page number included on either side, is unambiguous in a
different way -- covered separately below by ``_GAZETTE_FOOTER_LINE``, not
by this pattern."""


def _strip_gazette_footer(line: str, path: Path) -> str:
    """Remove a Công Báo running footer wherever it appears in ``line``.

    Called from ``_reflow_lines``, on the text that actually gets appended
    to a unit's body -- never on the text ``_reflow_lines`` reads to decide
    *where* a new block starts (issue #148). That distinction matters: a raw
    line that is nothing but this footer routinely sits directly in front of
    a genuine chapter/article heading, and the real corpus shows a real
    content line can end in running prose with no sentence terminator right
    before such a footer (a chemical-substance table row ending in a bare
    formula, gas-decree-17-2020's own real "744. Vinyl bromide ... C2H3Br").
    ``_sentence_is_finished`` has no rule for that shape; what actually
    opens the heading correctly today is the footer *line itself*, judged
    as mid-sentence-continuing prose ending in its own page number (a digit,
    not a citation coordinate) -- the same ``_is_pure_gazette_footer``/
    ``_sentence_is_finished`` machinery already in ``_reflow_lines``, built
    for the trailing-fused shape, happens to also carry the page-number-
    first shape through correctly, by accident, as long as the footer line
    itself survives into the gating decision. Stripping the footer earlier
    -- before ``_reflow_lines`` ever sees it -- silently dropped that
    accidental support and merged the next chapter into the table row
    instead of opening it (caught by this issue's own worktree rehearsal,
    not by any existing test). Calling this function only on what gets
    written to ``lines``/``buffer``, while every gating decision keeps
    reading the original, untouched ``raw`` line, keeps that behavior
    completely unchanged while still keeping the footer's own text out of
    the final unit body. Two cases, in order:

    1. The whole line, once any page number on either side is set aside, is
       nothing but the footer (``_GAZETTE_FOOTER_LINE``) -- dropped
       entirely, including that page number, since nothing else on the line
       could be real content for it to collide with.
    2. The footer sits alongside other real content on the same line
       (``_GAZETTE_FOOTER_MID_LINE``) -- only the footer's own text and an
       unambiguous *trailing* page number are removed, replaced by a single
       space so that whatever text sits on either side (if any) ends up
       separated by exactly one space, the same spacing ``_reflow_lines``
       already uses to join two raw lines that were never split by a
       footer at all. The trailing ``.strip()`` removes that single space
       again when the footer sat at the very start or end of the line.

    Every removal is logged, the same as ``_rejoin_split_words``, so the fix
    stays reviewable: a sample of exactly what was removed, not just a
    count, per document.
    """
    if _GAZETTE_FOOTER_LINE.match(line):
        logger.info("%s: dropped gazette footer line %r", path.name, line)
        return ""

    def _replace(match: re.Match[str]) -> str:
        logger.info("%s: stripped gazette footer %r", path.name, match.group(0).strip())
        return " "

    return _GAZETTE_FOOTER_MID_LINE.sub(_replace, line).strip()


def _split_danger_table_rows(lines: list[str]) -> list[str]:
    """Rewrite Phụ lục II's multi-column table into one line per real row (issue #141).

    pypdf hands back this table's text column by column within each row, not
    sentence by sentence: the STT cell, then the facility-type cell (itself
    wrapped over several raw lines when long), then the "Nhóm 1" threshold
    cell, then "Nhóm 2" -- with the table's own column header ("TT Loại hình
    cơ sở Nhóm 1 Nhóm 2") and Công Báo's running page footer both reprinted
    in the middle of this, on every page the table spans. None of that has
    anything a generic reflow could anchor on: a table row has no closing
    sentence punctuation, and before this fix the *only* thing that ever
    looked like an anchor inside it was a thousands separator like "1.000
    m²" -- read as a fake item "mục 1", which is the bug issue #141 as a
    whole exists to fix. ``CLAUSE_ANCHOR``'s own fix (see its docstring)
    keeps that from happening, but with no anchor left at all, every row
    would merge into a single formless block instead of splitting into 47
    units by STT -- which is what this function is for.

    Scoped to apply only between this table's own column header and the next
    ``PHỤ LỤC`` heading, so it can only ever change Phụ lục II; everywhere
    else in the corpus, including the appendix's own intro paragraph before
    the header and every other document, passes through untouched. A row is
    recognized by its STT cell matching ``_DANGER_TABLE_ROW_START`` *and*
    being strictly greater than the last STT opened -- the real table's own
    STT column runs 1 through 47 with no gaps, but the check only requires
    monotonic increase, not consecutive integers, so a future amendment that
    renumbers or drops a row does not desync and glue the rest of the table
    onto the wrong row. This is what tells a genuine row boundary apart from
    an unrelated line that happens to start with a small bare number, the
    same role ``_figure_legend_reason``'s own monotonic check plays for an
    ordinary clause. Everything collected for one row -- its
    facility-type cell and both threshold cells -- is joined with a single
    space into one synthetic ``"<stt>. <text>"`` line, which ``CLAUSE_ANCHOR``
    then recognizes as an ordinary appendix-item anchor once this function's
    output reaches ``_reflow_lines``/``parse_document``, exactly like a
    genuine ``"17. ..."`` line anywhere else in the corpus.
    """
    out: list[str] = []
    in_table = False
    current_stt: int | None = None
    row_parts: list[str] = []

    def flush_row() -> None:
        if current_stt is not None:
            text = " ".join(part for part in row_parts if part)
            out.append(f"{current_stt}. {text}")
        row_parts.clear()

    for line in lines:
        if not in_table:
            if _DANGER_TABLE_HEADER.match(line):
                in_table = True
            else:
                out.append(line)
            continue
        if _DANGER_TABLE_HEADER.match(line) or _GAZETTE_FOOTER_LINE.match(line):
            continue
        if APPENDIX_HEADING.match(line):
            flush_row()
            current_stt = None
            in_table = False
            out.append(line)
            continue
        match = _DANGER_TABLE_ROW_START.match(line)
        # Strictly greater than the last STT opened, not necessarily its
        # successor: real STT lists are consecutive on the real corpus, but a
        # future amendment that renumbers or drops a row would otherwise
        # desync "expected next" from what the table actually prints and
        # silently glue every remaining row onto the wrong one -- the same
        # monotonic-not-consecutive check ``_figure_legend_reason`` already
        # uses for an ordinary clause sequence.
        if match is not None and (current_stt is None or int(match.group("stt")) > current_stt):
            flush_row()
            current_stt = int(match.group("stt"))
            row_parts.append(match.group("rest"))
        else:
            row_parts.append(line)
    flush_row()
    return out


_PREMIUM_TABLE_HEADER_START = re.compile(
    r"^STT\s+(?P<category_column>Loại hình cơ sở có nguy hiểm về cháy, nổ)\s*$"
)
"""Phụ lục VI's own column header (pccc-decree-105-2025's insurance-premium
table, issue #147), reprinted at the top of every page the table spans --
the same role ``_DANGER_TABLE_HEADER`` plays for Phụ lục II, just a different
literal string. Only the header's own first line needs to be recognized to
find the table at all; see ``_PREMIUM_TABLE_HEADER_CONTINUATION`` for the
eight short column words that follow it on every page.

The ``category_column`` group (issue #160) captures this line's own second
column name ("Loại hình cơ sở có nguy hiểm về cháy, nổ") out of the matched
source text itself, so ``_premium_table_column_header`` can build the
displayed column header from what was actually read, never from a second
string typed separately into ``ingest/chunk.py``."""

_PREMIUM_TABLE_HEADER_CONTINUATION = (
    "Mức",
    "khấu",
    "trừ",
    "(loại)",
    "Tỷ lệ phí",
    "bảo hiểm",
    "tối thiểu/",
    "năm (%)",
)
"""The rest of the column header, each word its own raw pypdf line because
the header's own columns are narrow -- measured on the real corpus, this
exact eight-line sequence repeats immediately after
``_PREMIUM_TABLE_HEADER_START`` every time it appears. ``_skip_premium_table_header``
only skips a line here when it matches this *exact*, in-order continuation
of the header's own first line, never independently -- "Mức"/"trừ" are short,
ordinary Vietnamese words that could otherwise coincide with real row text
appearing on its own line."""

_PREMIUM_TABLE_END = re.compile(r"^Trường hợp thời hạn bảo hiểm khác\b")
"""The formula paragraph the law always prints immediately after this
table's last row, explaining how to pro-rate a premium for a term other than
one year -- the one unambiguous signal that the table itself is finished.
Needed because, unlike Phụ lục II's table (which runs straight into the next
appendix heading), Phụ lục VI's own section "I. MỨC PHÍ BẢO HIỂM" keeps going
past the table with further numbered paragraphs ("2. Đối với cơ sở ...",
"3. Đối với cơ sở hạt nhân...") before "II. MỨC KHẤU TRỪ BẢO HIỂM" even
starts; without this marker those paragraphs would read as unlabelled
continuation text of the table's last row and be silently absorbed into it."""

_PREMIUM_TABLE_ROW_START = re.compile(r"^(?P<stt>\d{1,2})(?=\s|$)\s*(?P<rest>.*)$")
"""A bare STT cell opening one top-level row of Phụ lục VI's own table --
the same shape and the same thousands-separator safety ``_DANGER_TABLE_ROW_START``
documents (a genuine STT is never immediately followed by another digit, so
"5.000 m3" can never match this at all, dot included)."""

_PREMIUM_TABLE_SUBROW_START = re.compile(
    r"^(?P<item>\d{1,2})\.(?P<sub>\d{1,2})(?=\s|$)\s*(?P<rest>.*)$"
)
"""A row numbered one level under its parent, e.g. "35.1" or "14.1" -- see
``ingest/parser.py``'s own ``APPENDIX_SUBITEM_ANCHOR``, which recognizes this
same shape once this function has joined a sub-row's own scattered raw lines
back into one. The lookahead after ``sub`` excludes a thousands group
exactly as ``_PREMIUM_TABLE_ROW_START``'s does."""


def _skip_premium_table_header(lines: list[str], index: int) -> tuple[int, list[str]]:
    """Return the index just past the header line at ``index`` and its own continuation.

    Only the header's first line is required to recognize the table
    (``_PREMIUM_TABLE_HEADER_START``, checked by the caller); the eight
    column words that follow are skipped here only when they appear exactly,
    in order, immediately after it -- a defensive match, not an independent
    one, so an unexpected reflow of the header can never silently eat real
    row content that merely starts with one of these same short words.

    Also returns the continuation words actually read off ``lines`` -- not
    ``_PREMIUM_TABLE_HEADER_CONTINUATION`` itself, even though they are
    required to be byte-identical to it for this function to skip them at
    all -- so a caller building the displayed column header (issue #160)
    reads it from the real extracted text, never from a second, separately
    typed Python string that could silently drift out of sync with the
    source. Fewer than eight words means the page's own header continuation
    did not fully match (a reflow anomaly); the caller treats that as "no
    header found" and logs it.
    """
    next_index = index + 1
    found: list[str] = []
    for expected in _PREMIUM_TABLE_HEADER_CONTINUATION:
        if next_index < len(lines) and lines[next_index] == expected:
            found.append(lines[next_index])
            next_index += 1
        else:
            break
    return next_index, found


_PREMIUM_TABLE_HEADER_DEDUCTIBLE_COLUMN_WORDS = 4
"""Number of ``_PREMIUM_TABLE_HEADER_CONTINUATION`` words, in order, that
make up the table's "Mức khấu trừ (loại)" column name -- the table's own
fixed two-column layout after the header's first line (``category_column``
plus this), not a guess: measured on the real corpus, the continuation is
always these exact eight words, the first four naming the deductible-type
column and the last four the premium-rate column (issue #160)."""


def _premium_table_column_header(category: str, continuation: list[str]) -> str | None:
    """Build the displayed column header from the source text's own header line (issue #160).

    ``category`` is ``_PREMIUM_TABLE_HEADER_START``'s own already-matched
    "Loại hình cơ sở ..." column name; ``continuation`` is whatever
    ``_skip_premium_table_header`` actually found in the source for the
    other two columns. Neither is typed again here -- this function only
    joins what was already read.

    Returns:
        ``"<category> | <deductible column> | <rate column>"``, or ``None``
        when fewer than all eight continuation words were found -- a header
        this function cannot safely build; the caller logs that and the
        row's chunk falls back to its unmodified header (issue #160's own
        fallback rule).
    """
    if len(continuation) < len(_PREMIUM_TABLE_HEADER_CONTINUATION):
        return None
    split = _PREMIUM_TABLE_HEADER_DEDUCTIBLE_COLUMN_WORDS
    deductible_column = _join_header_words(continuation[:split])
    rate_column = _join_header_words(continuation[split:])
    return f"{category} | {deductible_column} | {rate_column}"


def _join_header_words(words: list[str]) -> str:
    """Join a column header's own raw-line words the way the source prints them.

    Each raw line is its own word or short phrase (see
    ``_PREMIUM_TABLE_HEADER_CONTINUATION``); plain ``" ".join`` would insert
    a space after "tối thiểu/" that the source itself never prints before
    "năm (%)". No space is added when the previous word already ends in
    "/" -- the one place this table's header wraps without one.
    """
    result = ""
    for word in words:
        if result and not result.endswith("/"):
            result += " "
        result += word
    return result


def _enter_premium_table_header(
    lines: list[str], index: int, header_match: re.Match[str]
) -> tuple[int, str | None]:
    """Consume the table's header line and its continuation at ``index``.

    Returns the index just past them and the column-header text built from
    the source (``_premium_table_column_header``), or ``None`` when the
    continuation did not fully match -- logged here at WARNING, since a
    caller silently losing the header would ship an uninformative chunk with
    no visible trace (issue #160's own fallback rule: the chunk keeps its
    current header, and the fallback is logged).
    """
    next_index, continuation = _skip_premium_table_header(lines, index)
    column_header = _premium_table_column_header(
        header_match.group("category_column"), continuation
    )
    if column_header is None:
        logger.warning(
            "Phụ lục VI's premium table header continuation did not fully match at "
            "raw line %d (found %d/%d words) -- row chunks built before the next "
            "successful header match keep their unmodified header (issue #160)",
            index + 1,
            len(continuation),
            len(_PREMIUM_TABLE_HEADER_CONTINUATION),
        )
    return next_index, column_header


_APPENDIX_6_SECTION_HEADING = re.compile(r"^(?P<roman>[IVXLCDM]+)\.\s+(?P<heading>.+)$")
"""Phụ lục VI's own two Roman-numbered sections, "I. MỨC PHÍ BẢO HIỂM (CHƯA
BAO GỒM THUẾ GTGT)" and "II. MỨC KHẤU TRỪ BẢO HIỂM" (issue #147's fix round
on PR #151) -- organized independently of the table's own numeric STT, and
printed as their own complete raw pypdf line, never wrapped. Scoped to Phụ
lục VI alone by ``_restructure_appendix_6``'s own ``appendix6_active``
tracking, not by this shape alone: this function runs over the *entire*
document's text, where a genuine chapter heading never looks like this
(every chapter heading carries the word "Chương"), but a stray mid-sentence
citation wrapping onto its own line in exactly this shape is not
impossible -- the full-corpus, byte-identical-outside-``#phu-luc-6`` check
this issue's own acceptance criteria runs is what actually guards against
that in practice, the same way PR #151's own guards are measured rather than
proven."""


def _restructure_appendix_6(lines: list[str]) -> list[str]:  # noqa: C901, PLR0912, PLR0915
    """Rewrite Phụ lục VI's premium table *and* its own Mục I/Mục II prose (issue #147).

    Decree 105/2025's Phụ lục VI prices compulsory fire/explosion insurance by
    facility category, numbered the same way Phụ lục II is
    (``_split_danger_table_rows``) but one level deeper: most rows are a
    single STT ("17  Cơ sở kinh doanh khí đốt ... N 0,3"), but some are split
    into sub-rows printed as a compound number -- "35" names a facility
    group, then "35.1"/"35.2" each carry their own rate. Read with no
    splitting at all, pypdf hands the table back column by column exactly
    like Phụ lục II's, and the hierarchical numbering adds a second failure
    on top: ``CLAUSE_ANCHOR`` reads only the digits before a sub-row's own
    dot, so "35.1" and "35.2" both look like a second occurrence of item 35.
    This part of the function -- PR #151's original fix -- is unchanged by
    this issue's own fix round; the review on that PR confirmed it correct.

    Produces one synthetic line per row: ``"<n>. <text>"`` for a top-level
    row, the same shape ``_split_danger_table_rows`` already uses, recognized
    by the parser's existing ``CLAUSE_ANCHOR`` with no further change; and
    ``"<n>.<m> <text>"`` for a sub-row, recognized by the parser's
    ``APPENDIX_SUBITEM_ANCHOR``. A parent row that carries its own text
    before its first sub-row (e.g. row 35's own three-paragraph description
    of what "hạng nguy hiểm cháy nổ A, B"/"C"/"D, E" each mean) is flushed as
    that row's own line the moment its first sub-row opens -- never deferred
    to the next top-level row -- so the parent's synthetic line reaches the
    parser *before* its children's, the order ``self._item_no`` depends on
    to attach a sub-row to the right parent (``ingest/parser.py``'s own
    ``feed``). A lettered breakdown printed *inside* a row or sub-row's own
    cell ("a) Cơ sở sản xuất... N 0,2", "b) Cơ sở sản xuất dệt may... N
    0,25") is joined into that row's own text like everything else, not
    split into its own unit: those letters are the cell's own illustrative
    examples, not a further legal subdivision, and every rate inside them is
    still kept, just as running text (issue #147's own "keep every number in
    a cell").

    The table-row splitting above is scoped to between the table's own
    column header and ``_PREMIUM_TABLE_END``, the formula paragraph the law
    always prints right after the table's last row; see that constant's own
    docstring for why Phụ lục VI needs a dedicated end marker where Phụ lục
    II's splitter does not.

    New in this issue's fix round (reviewing PR #151): the prose *around*
    that table -- Mục I's own intro and post-table text, Mục I khoản 2/3,
    the "II. MỨC KHẤU TRỪ BẢO HIỂM" heading, and Mục II's own khoản 1
    (with its lettered points a/b/c) through khoản 3 -- shares
    ``CLAUSE_ANCHOR``/``POINT_ANCHOR``'s plain "N."/"letter)" shape with the
    table's own rows, a genuine ambiguity already in the source document.
    Read unmodified, that prose was either silently dropped as a duplicate of
    a same-numbered table row, or absorbed into whichever table row happened
    to be open at the time -- naming a wrong row in a citation (issue #131's
    own standard: an id must never name something that does not exist in the
    law). Tracked here by ``appendix6_active`` (entered the moment
    ``APPENDIX_HEADING`` matches Phụ lục VI itself, left the moment any
    *other* appendix heading appears) and, within that span, by ``section``/
    ``section_khoan`` (the Roman section and clause currently open in its
    prose, independent of the table's own ``current_top``/``current_sub``
    above): every such prose line, *outside* the table, is rewritten into a
    sentinel (``APPENDIX_SECTION_MARKER``) ``ingest/parser.py``'s own
    ``APPENDIX_SECTION_ANCHOR``/``APPENDIX_SECTION_CLAUSE_ANCHOR``/
    ``APPENDIX_SECTION_POINT_ANCHOR`` recognize -- never the table's own
    plain shape, so a table row can never collide with one of these again.
    Every line *inside* the table (``in_table`` True) reaches the unchanged
    row-splitting logic above exactly as before, whether or not
    ``appendix6_active`` -- the table's own header/end markers are already
    specific enough to this one table on their own.

    Khoản 1 Mục I is the one clause whose own body the table physically
    splits in two: its intro sits before the table ("Đối với cơ sở ... tỷ lệ
    phí bảo hiểm tối thiểu sau:"), and its formula paragraph, "Ghi chú:", and
    nhóm-35 note sit right after the table's last row. ``section_khoan`` is
    never reset while ``in_table`` is True, so the moment ``_PREMIUM_TABLE_END``
    matches, this function re-emits the *same* "§I.1." sentinel for that end
    line instead of passing it through unchanged (which is what PR #151 did,
    and which let that text be silently absorbed into row 37's own body
    instead) -- ``ingest/parser.py``'s own ``_open_appendix_section_clause``
    is what merges the two halves back into one unit.

    Scoped to Phụ lục VI alone for every rewrite above: every other appendix
    in the corpus, including this same document's own Phụ lục IV or V, is
    passed through this function completely unchanged, exactly as
    ``_split_premium_table_rows`` (this function's own previous name) left
    them.
    """
    out: list[str] = []
    appendix6_active = False
    in_table = False
    section: str | None = None
    section_khoan: int | None = None
    current_top: int | None = None
    current_sub: int | None = None
    top_emitted = False
    top_parts: list[str] = []
    sub_parts: list[str] = []

    def flush_top() -> None:
        nonlocal top_emitted
        if current_top is not None and not top_emitted:
            text = " ".join(part for part in top_parts if part)
            out.append(f"{current_top}. {text}")
            top_emitted = True
        top_parts.clear()

    def flush_sub() -> None:
        if current_sub is not None:
            text = " ".join(part for part in sub_parts if part)
            out.append(f"{current_top}.{current_sub} {text}")
        sub_parts.clear()

    index = 0
    while index < len(lines):
        line = lines[index]

        heading_match = APPENDIX_HEADING.match(line)
        if heading_match is not None:
            number = heading_match.group("number")
            appendix6_active = (
                number is not None and chapter_numeral_to_int(number) == APPENDIX_6_NO
            )
            section = None
            section_khoan = None
            if in_table:
                flush_sub()
                flush_top()
                current_top = None
                current_sub = None
                in_table = False
            out.append(line)
            index += 1
            continue

        if not in_table:
            header_match = _PREMIUM_TABLE_HEADER_START.match(line)
            if header_match is not None:
                in_table = True
                index, column_header = _enter_premium_table_header(lines, index, header_match)
                if column_header is not None:
                    out.append(f"{APPENDIX_COLUMN_HEADER_MARKER}{column_header}")
                continue
            if appendix6_active:
                section_match = _APPENDIX_6_SECTION_HEADING.match(line)
                if section_match is not None:
                    section = section_match.group("roman")
                    section_khoan = None
                    out.append(
                        f"{APPENDIX_SECTION_MARKER}{section}. {section_match.group('heading')}"
                    )
                    index += 1
                    continue
                if section is not None:
                    clause_match = CLAUSE_ANCHOR.match(line)
                    if clause_match is not None:
                        section_khoan = int(clause_match.group("number"))
                        out.append(
                            f"{APPENDIX_SECTION_MARKER}{section}.{section_khoan}. "
                            f"{clause_match.group('rest')}"
                        )
                        index += 1
                        continue
                    if section_khoan is not None:
                        point_match = POINT_ANCHOR.match(line)
                        if point_match is not None:
                            out.append(
                                f"{APPENDIX_SECTION_MARKER}{section}.{section_khoan}."
                                f"{point_match.group('letter')}) {point_match.group('rest')}"
                            )
                            index += 1
                            continue
            out.append(line)
            index += 1
            continue
        header_match = _PREMIUM_TABLE_HEADER_START.match(line)
        if header_match is not None:
            index, column_header = _enter_premium_table_header(lines, index, header_match)
            if column_header is not None:
                out.append(f"{APPENDIX_COLUMN_HEADER_MARKER}{column_header}")
            continue
        if _GAZETTE_FOOTER_LINE.match(line):
            index += 1
            continue
        if _PREMIUM_TABLE_END.match(line):
            flush_sub()
            flush_top()
            current_top = None
            current_sub = None
            in_table = False
            if section is not None and section_khoan is not None:
                out.append(f"{APPENDIX_SECTION_MARKER}{section}.{section_khoan}. {line}")
            else:
                out.append(line)
            index += 1
            continue
        sub_match = _PREMIUM_TABLE_SUBROW_START.match(line)
        if (
            sub_match is not None
            and current_top is not None
            and int(sub_match.group("item")) == current_top
            and (current_sub is None or int(sub_match.group("sub")) > current_sub)
        ):
            if current_sub is None:
                # This row's own intro (if any) is now complete -- nothing
                # more will ever be added to it once its first child opens.
                flush_top()
            else:
                flush_sub()
            current_sub = int(sub_match.group("sub"))
            sub_parts = [sub_match.group("rest")]
            index += 1
            continue
        top_match = _PREMIUM_TABLE_ROW_START.match(line)
        if top_match is not None and (
            current_top is None or int(top_match.group("stt")) > current_top
        ):
            flush_sub()
            flush_top()
            current_top = int(top_match.group("stt"))
            current_sub = None
            top_emitted = False
            top_parts = [top_match.group("rest")]
            index += 1
            continue
        if current_sub is not None:
            sub_parts.append(line)
        else:
            top_parts.append(line)
        index += 1
    flush_sub()
    flush_top()
    return out


_GAZETTE_COLOPHON_LINES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^VĂN PHÒNG CHÍNH PHỦ XUẤT BẢN$"),
    re.compile(r"^Địa chỉ:\s*Số 1,\s*Hoàng Hoa Thám,\s*Ba Đình,\s*Hà Nội$"),
    re.compile(r"^Điện thoại liên hệ:$"),
    re.compile(r"^-\s*Nội dung:\s*[\d.]+$"),
    re.compile(r"^-\s*Phát hành:\s*[\d.]+$"),
    re.compile(r"^Email:\s*congbao@chinhphu\.vn$"),
    re.compile(r"^Website\s+http://congbao\.chinhphu\.vn$"),
)
"""Each raw line of the Công Báo end-of-issue colophon (issue #159), in the
fixed order the publisher always prints them, matched against a raw line
*after* ``_raw_page_lines`` has already stripped its own leading/trailing
whitespace -- only the internal spacing around a punctuation mark or digits
is left tolerant (measured on the real corpus: ``pccc-decree-105-2025``'s
own extraction has a stray double space before both phone-style numbers,
"- Phát hành:  080.44243", that a literal match would miss).

This is deliberately the same kind of guard ``_GAZETTE_FOOTER_LINE`` and
``_GAZETTE_FOOTER_MID_LINE`` (#149) already rely on: matching this fixed,
publisher-specific wording verbatim is itself what keeps this scoped to
gazette PDFs only -- no other corpus document, and no other paragraph of a
gazette PDF, could ever contain "VĂN PHÒNG CHÍNH PHỦ XUẤT BẢN" followed by
this exact address/phone/email/website block, so there is no separate
filename or document-type check to add. This is not "strip the last N
characters of a PDF" -- a document that happened to end differently (a
scanned issue missing its last page, or a non-gazette PDF) is left alone
by ``_strip_gazette_colophon`` below, because the match simply fails."""

_GAZETTE_COLOPHON_CONTINUATION = re.compile(r"^\(Xem tiếp Công báo số[^()]*\)$")
"""The optional line immediately before the colophon proper, printed only
when a Công Báo issue is split across several PDF parts and the one at
hand is not the final part -- e.g. "(Xem tiếp Công báo số 719 + 720)" at
the end of real-corpus ``pccc-decree-105-2025.pdf`` (the 717+718 part).
Absent from the final part, whose own real extraction
(``pccc-decree-105-2025-cong-bao-719-720.pdf``) ends in the colophon alone
with nothing in front of it."""


def _strip_gazette_colophon(raw_lines: list[str], path: Path) -> list[str]:
    """Drop the Công Báo publisher's end-of-issue colophon from the document's tail.

    Called from ``_extract_pdf`` on the whole document's flattened raw
    lines, after every per-page cleanup has already run and before
    ``_reflow_lines`` joins wrapped lines back together -- the colophon
    itself spans a page break (its last page carries only the fixed block;
    "(Xem tiếp Công báo số ...)", when present, is the real content page's
    own very last line), so it can only ever be recognized once pages are
    flattened into one sequence.

    Matched, and removed, only as a contiguous *suffix* of the whole
    document -- never mid-document -- because this is the publisher's own
    closing block, printed exactly once, on the issue's last page (issue
    #159). 239 characters of this publisher text previously survived
    inside whichever unit happened to close the PDF (real corpus:
    ``pccc-decree-105-2025#phu-luc-8#mau-pc23`` on ``main``, or
    ``pccc-decree-105-2025#phu-luc-6#muc-ii#khoan-3`` once #151's Phụ lục
    VI splitter gives that unit its own id), where it got embedded and
    could be quoted as if it were law.
    """
    if len(raw_lines) < len(_GAZETTE_COLOPHON_LINES):
        return raw_lines
    cut = len(raw_lines) - len(_GAZETTE_COLOPHON_LINES)
    tail = raw_lines[cut:]
    if not all(
        pattern.match(line) for pattern, line in zip(_GAZETTE_COLOPHON_LINES, tail, strict=True)
    ):
        return raw_lines
    if cut > 0 and _GAZETTE_COLOPHON_CONTINUATION.match(raw_lines[cut - 1]):
        cut -= 1
    removed = raw_lines[cut:]
    logger.info("%s: dropped gazette colophon (%d lines): %r", path.name, len(removed), removed)
    return raw_lines[:cut]


def _reflow_lines(raw_lines: Iterable[str], path: Path) -> list[str]:
    """Join wrapped lines back together, keeping a break only at a real anchor.

    Reuses the parser's own anchor patterns so "what counts as a new block" is
    defined in exactly one place. A chapter or article heading is always its own
    line in the source and never legitimately wraps onto the next one -- only a
    clause or point anchor introduces text that can wrap -- so a line opening a
    chapter or article always closes itself off from whatever line follows it,
    regardless of whether that next line happens to look like an anchor too.

    A chapter- or article-shaped line is trusted as opening a genuine new
    heading only when the single raw line immediately before it actually
    finished a sentence (build plan A6 issue #34). Amending decrees ("sửa
    đổi, bổ sung ...") are saturated with citations to other documents' own
    Điều numbers, e.g. "... Luật sửa đổi, bổ sung một số điều" wrapping,
    purely because of column width, right before "Điều 6 và Phụ lục 4 về
    danh mục ...": anchor-shaped, but still the middle of the citing
    sentence, not a new article. Measured against the real corpus, this
    exact shape was the single largest source of noise: it does not just add
    a spurious unit, it silently steals the real article's identifier out
    from under it (the genuine "Điều 6" later in the same document was being
    discarded as a "duplicate" of this citation fragment) and leaves every
    following line without a known article context until the next real one
    recovers it.

    Checked against the one immediately preceding *raw* line, not the
    reflowed buffer accumulated so far: a long-running document preamble or
    foreword can go many raw lines without ever hitting a recognized
    terminator, and judging against the whole buffer let that unrelated
    unfinished-looking history block a real heading long after the actual
    text immediately before it had already finished its own sentence.

    An appendix heading, a ``DANH MỤC ... Kèm theo`` caption, ``Nơi nhận:``, and a
    signer-block opener are gated the same way as chapter/article headings (issue
    #117): a running footer or a bare page break routinely sits between a
    document's last real sentence and its appendix or signature, so without this
    gate that gap would defeat the parser's own anchors by gluing them onto
    whatever ordinary prose line precedes them instead of ever letting them start
    a fresh line the parser can recognise at line-start.

    This check is deliberately narrower than "every anchor type": a clause or
    point anchor still interrupts unconditionally, as before. QCVN-style
    technical standards intersperse clauses and points between formulas and
    tables whose garbled, non-sentence text (extracted equation fragments,
    bare numeric table cells) never reaches a recognizable sentence
    terminator -- gating clause/point anchors the same way silently merged
    the next real clause or point into that unfinished-looking text instead
    of opening it, which is a content loss the build plan explicitly forbids
    trading for less noise. Chapter and article headings carry no such risk:
    real corpus measurement found no case of a formula or table sitting
    directly before a genuine chapter/article heading.

    A *bare* sample form opener (``_is_bare_form_heading``, issue #131) joins
    the unconditional set for the same reason a bare appendix heading
    already does: the page break immediately before one routinely ends a
    form appendix's own table-of-contents line in ordinary, unpunctuated
    prose, which a gated check would read as unfinished and merge the real
    form into. A form opener that is *not* bare (a table-of-contents entry
    carrying its own title, or a mid-sentence citation to another document's
    form) is left alone here -- the parser's own ``FORM_ANCHOR`` recognises
    the table-of-contents shape without any reflow change, and a citation
    like "... theo Mẫu số 11 Mục II tại Phụ lục..." must keep wrapping onto
    the same line as the sentence it continues, exactly as before.

    ``path`` is used only to log a gazette footer stripped from the text
    that actually gets appended to a line (``_strip_gazette_footer``, issue
    #148) -- every gating decision below still reads ``raw`` completely
    untouched, so that stripping never changes where a break falls.
    """
    lines: list[str] = []
    buffer = ""
    buffer_is_closed = False
    previous_raw = ""
    previous_content_raw = ""
    for raw in raw_lines:
        is_heading = _is_heading_anchor(raw)
        # When the raw line immediately before this one is nothing but the
        # running footer on its own -- a routine page break, not a paragraph
        # boundary -- judge against the last real content line before *that*
        # instead (issue #117). A footer *fused onto* the end of a real
        # content line (what ``_TRAILING_GAZETTE_FOOTER`` was built for) still
        # judges against that same line with the footer suffix stripped off;
        # only a footer occupying an entire raw line by itself falls back
        # further. Measured on the real corpus: a mid-sentence line wrap
        # ("... của" / <page footer, its own raw line> / "Chính phủ quy định
        # ...") was wrongly read as "the previous sentence finished" purely
        # because the footer line, once stripped down to nothing, fell into
        # this function's *unrelated* empty-string case -- not because the
        # real prose actually ended there.
        #
        # This judgment, and ``anchor_applies`` below, are computed from
        # ``raw`` -- never from the footer-stripped ``clean`` further down
        # (issue #148). The real corpus shows a footer occupying its own raw
        # line, with its own page number first, sitting directly in front
        # of a genuine chapter heading after a table row that ends in a
        # bare chemical formula with no sentence terminator at all
        # (gas-decree-17-2020's own real "744. Vinyl bromide ... C2H3Br");
        # ``_sentence_is_finished`` has no rule for that shape, and what
        # actually opens the heading correctly is the footer line itself,
        # read by the fallback above as mid-sentence-continuing prose that
        # happens to end in a digit (its own page number) rather than a
        # citation coordinate. Stripping the footer before this judgment
        # ever ran would silently remove that accidental support and merge
        # the next chapter into the table row instead of opening it.
        judged_line = (
            previous_content_raw if _is_pure_gazette_footer(previous_raw) else previous_raw
        )
        anchor_applies = (_is_block_anchor(raw) and not is_heading) or (
            is_heading and _sentence_is_finished(judged_line)
        )
        clean = _strip_gazette_footer(raw, path)
        if not buffer or buffer_is_closed or anchor_applies:
            if buffer:
                lines.append(buffer)
            buffer = clean
            buffer_is_closed = is_heading
        elif clean:
            buffer = f"{buffer} {clean}"
        previous_raw = raw
        if not _is_pure_gazette_footer(raw):
            previous_content_raw = raw
    if buffer:
        lines.append(buffer)
    return lines


_SENTENCE_END_CHARS = '.;:”")'

_TRAILING_GAZETTE_FOOTER = re.compile(
    r"\s*CÔNG BÁO/Số[^/\n]*/Ngày\s+\d{1,2}-\d{1,2}-\d{4}\s*\d*\s*$"
)
"""Công Báo's own running footer ("CÔNG BÁO/Số 233 + 234/Ngày 22-02-2020",
sometimes with a trailing bare page number). ``_drop_running_furniture``
already drops this when pypdf gives it its own page-first/page-last line, but
pypdf sometimes fuses it onto the end of the real content's own last line
with no line break at all -- stripped here so a page boundary landing mid-
sentence, from the reader's perspective, does not also look mid-sentence to
``_sentence_is_finished``."""


def _is_pure_gazette_footer(raw: str) -> bool:
    """Return whether ``raw`` is nothing but the Công Báo running footer.

    ``_drop_running_furniture`` only drops a footer that repeats in the exact
    same *shape* on more than half of a document's pages; one that varies (a
    different page-number position, an extra trailing digit) survives as an
    ordinary raw line instead, and is judged by ``_reflow_lines`` like any
    other line except for this check (issue #117).
    """
    return not _TRAILING_GAZETTE_FOOTER.sub("", raw.rstrip()).rstrip()


_CITATION_COORDINATE_BEFORE_DIGIT = re.compile(
    r"(?:[Kk]hoản|[Đđ]iều|[Đđ]iểm|[Cc]hương|[Ss]ố)\s+\d+\s*$"
)
"""A trailing digit preceded by one of Vietnamese law's own citation
coordinate words is a citation fragment ("... khoản 2"), not a finished
table cell -- excluded from ``_sentence_is_finished``'s trailing-digit case
so it is not mistaken for the table-row shape that case exists for."""


def _sentence_is_finished(previous_raw: str) -> bool:
    """Return whether the single raw line before a candidate heading finished a sentence.

    Three shapes count as finished, besides an empty line: text ending in one
    of Vietnamese legal prose's own sentence terminators, an all-caps title
    line, and text ending in a digit that is not itself a citation
    coordinate. Titles need their own case, since a chapter or article's
    title conventionally has no terminal punctuation of its own -- "Chương I"
    is followed by a title line like "QUY ĐỊNH CHUNG", itself followed by
    "Điều 1." with nothing to mark the title as "finished" except that it is
    set entirely in capitals. A trailing digit covers a table row (measured
    in the real corpus: a chemical-substance table's last column, immediately
    before the next chapter) -- tabular data extracted from a PDF has no
    sentence of its own to finish, and withholding a heading interrupt after
    it would merge the heading into the table's last cell instead of opening
    it. That same shape -- ends in a digit -- is also how a citation fragment
    like "... khoản 2" ends, so a trailing citation coordinate is excluded
    from it rather than trusted. Ordinary mixed-case prose with no sentence
    break, no trailing digit, and no citation coordinate is not finished -- a
    chapter/article-shaped line right after it is far more likely a citation
    fragment than a genuine new heading.

    Checking only the single line immediately before the candidate, rather
    than everything accumulated in the current reflow buffer, matters: a long
    preamble or foreword can run for many raw lines without ever hitting a
    recognized terminator, and judging against that whole history would keep
    blocking a real heading long after the specific text right before it had
    already finished its own sentence.
    """
    stripped = _TRAILING_GAZETTE_FOOTER.sub("", previous_raw.rstrip()).rstrip()
    if not stripped:
        return True
    if stripped[-1] in _SENTENCE_END_CHARS:
        return True
    if stripped[-1].isdigit() and not _CITATION_COORDINATE_BEFORE_DIGIT.search(stripped):
        return True
    return stripped != stripped.lower() and stripped == stripped.upper()


def _is_bare_appendix_heading(line: str) -> bool:
    """Return whether ``line`` is a numeral-only appendix heading (issue #117).

    Nothing else shares its line, e.g. "Phụ lục III" alone -- its title always
    follows on later lines. Unlike a citation such as "Phụ lục III kèm theo
    Nghị định này phải được..." (real heading *text* on the very same line), a bare heading is
    never a wrapped mid-sentence fragment on the real corpus: every mid-sentence
    citation to an appendix continues with "kèm theo" or similar prose on that
    same raw line, never nothing at all. Treated as an *unconditional* block
    anchor below, the same as a clause or point, rather than gated on
    ``_sentence_is_finished`` -- which a long, punctuation-less table row
    directly in front of it would otherwise defeat: measured on the real
    corpus, Phụ lục II's own multi-page area table ends its last row in an
    ordinary word ("... Dưới 75 ha"), not a sentence terminator or a digit, so
    the gated check wrongly read the table row as unfinished prose and merged
    Phụ lục III's own heading into Phụ lục II's table instead of opening it.
    """
    match = APPENDIX_HEADING.match(line)
    return match is not None and not match.group("heading")


def _is_bare_form_heading(line: str) -> bool:
    """Return whether ``line`` is a sample form opener with nothing else on it (issue #131).

    Measured directly on the real corpus (pccc-decree-105-2025,
    gas-decree-17-2020): a form's own real content always opens on a line by
    itself, e.g. ``"Mẫu số PC01"`` alone -- the form's title, if any, is the
    *next* raw pypdf line, never glued onto the same one. A form appendix's
    own table of contents, by contrast, always carries the form's title on
    the *same* raw line (``"Mẫu số PC01 Phiếu thông tin của cơ sở"``), and a
    mid-sentence citation to another document's form continues with more of
    that sentence on the same line too (``"... theo Mẫu số 11 Mục II tại Phụ
    lục ban hành kèm theo Nghị định này."``, gas-decree-17-2020's own Điều
    31c). Bareness is what tells these apart without reading ahead: treated
    as an *unconditional* block anchor below, the same as ``_is_bare_appendix_
    heading`` already is and for the same reason -- a long, punctuation-less
    table or caption line directly in front of a real form opener must not
    defeat it the way a gated, ``_sentence_is_finished``-style check would
    (measured: the Công báo page break ahead of pccc-decree-105-2025's own
    "Mẫu số PC01" ends its preceding table-of-contents line in an ordinary
    word, "...phí", never a sentence terminator). The table-of-contents and
    citation lines are deliberately left alone here -- not block anchors at
    all -- so the citation inside Điều 31c keeps wrapping onto the same
    reflowed line as before, and ``ingest/parser.py``'s own ``FORM_ANCHOR``
    (which does recognise the numbered table-of-contents shape, since that
    text never needs a reflow fix) is what turns a table-of-contents entry
    into a (later-evicted) form instead.
    """
    match = FORM_ANCHOR.match(line)
    return match is not None and not match.group("rest")


def _is_heading_anchor(line: str) -> bool:
    """Return whether ``line`` opens a chapter, article, appendix, or a signer block (#117).

    These join ``CHAPTER_ANCHOR``/``ARTICLE_ANCHOR`` here, not ``CLAUSE_ANCHOR``/
    ``POINT_ANCHOR``'s unconditional set: like a chapter or article, each one
    opens a document-wide, one-time transition that a genuine occurrence never
    has to share a physical line with unrelated prose, but each can also *look*
    like the tail of an ordinary sentence that merely happens to start a new
    line ("Phụ lục 4 về danh mục ..." continuing a citation, "Chủ tịch Ủy ban
    nhân dân cấp xã có trách nhiệm ..." starting a genuine new sentence) --
    exactly the ambiguity ``_sentence_is_finished`` already exists to resolve
    for chapter/article headings. Gating these the same way is the same trade
    already made there, not a new one. A *bare* appendix heading is excluded
    here (``_is_bare_appendix_heading``) and handled unconditionally instead,
    in ``_is_block_anchor`` below.
    """
    return bool(
        CHAPTER_ANCHOR.match(line)
        or ARTICLE_ANCHOR.match(line)
        or (APPENDIX_HEADING.match(line) and not _is_bare_appendix_heading(line))
        or APPENDIX_CAPTION.match(line)
        or RECIPIENT_START.match(line)
        or SIGNATURE_START.match(line)
    )


def _is_block_anchor(line: str) -> bool:
    """Return whether ``line`` starts a new structural unit, per the parser's anchors."""
    return bool(
        CHAPTER_ANCHOR.match(line)
        or ARTICLE_ANCHOR.match(line)
        or CLAUSE_ANCHOR.match(line)
        or POINT_ANCHOR.match(line)
        or APPENDIX_HEADING.match(line)
        or APPENDIX_CAPTION.match(line)
        or RECIPIENT_START.match(line)
        or SIGNATURE_START.match(line)
        or _is_bare_form_heading(line)
        # Phụ lục VI's own sentinel-marked section/clause/point lines (issue
        # #147's fix round, ``_restructure_appendix_6``) -- unconditional,
        # the same as CLAUSE_ANCHOR/POINT_ANCHOR above: ``extract.py`` only
        # ever emits one of these on a line it has already decided opens a
        # fresh unit, never mid-sentence, so there is nothing to gate.
        or APPENDIX_SECTION_ANCHOR.match(line)
        or APPENDIX_SECTION_CLAUSE_ANCHOR.match(line)
        or APPENDIX_SECTION_POINT_ANCHOR.match(line)
        # The premium table's own column-header sentinel (issue #160,
        # ``_restructure_appendix_6``) -- unconditional for the same reason:
        # a complete, synthetic line this module only ever emits on its own.
        or APPENDIX_COLUMN_HEADER_ANCHOR.match(line)
    )
