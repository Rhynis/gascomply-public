"""Turn a parsed document's units into embedding-ready chunks (build plan A6 Phase 1.5).

``Unit.body`` holds a unit's own text only, excluding its children's text (see
``app/schemas/unit.py``), so an article's full text has to be recomposed from
its descendants before it can be embedded. This module does that recomposition
and decides the chunk boundary: one chunk per Article by default, or one chunk
per Clause when an article's composed text is too long for a single embedding
chunk to represent well.

Persisting the resulting chunks to pgvector is Phase 1.5b, once a database is
provisioned; this module only produces the ``Chunk`` rows.
"""

import logging

from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.schemas.unit import Unit, UnitType
from ingest.parser import APPENDIX_6_NO, ParseResult

logger = logging.getLogger(__name__)

ARTICLE_SPLIT_THRESHOLD = 2000
"""Character count -- not tokens, so this stays a stdlib-only measure (C12) --
above which an article's composed text is split into one chunk per Clause
instead of kept as one chunk for the whole article.

2000 is a round number chosen from general embedding-chunking convention, not
tuned against any fixture: there is no real corpus in this repo yet (Phase 0.1
has not landed real downloaded documents). Most embedding models retrieve best
on chunks in the low hundreds to low thousands of tokens; at a conservative
~4 characters per token, 2000 characters keeps a chunk comfortably under
~500 tokens, well inside the context window of every embedder under
consideration (A2), while still letting a typical single-topic decree article
(usually a few hundred to low thousands of characters) stay as one coherent
chunk instead of being fragmented across its clauses for no reason. THIS
THRESHOLD NEEDS RE-MEASURING once real corpus documents land and the actual
article-length distribution is known.

Measured on the composed text WITH its context header (``_context_header``)
already prepended, not on the article's own body alone: the header is part of
what actually gets embedded, so measuring without it would let the true chunk
size overshoot the budget the threshold is meant to enforce by however long
the header happens to be.
"""


def build_chunks(result: ParseResult, document: Document) -> tuple[Chunk, ...]:
    """Build embedding-ready chunks from a parsed document's units.

    One chunk is built per Article by default, carrying a context header (see
    ``_context_header``) plus the article's own text and its clauses' and
    points' text composed in reading order. When an article's composed text
    -- header included -- exceeds ``ARTICLE_SPLIT_THRESHOLD``, one chunk is
    built per Clause instead, each still carrying its own points' text and a
    header naming both the article and the clause. A unit with no heading and
    no composed text of its own produces no chunk; a unit with a heading but
    no other text still does, since the heading -- not yet part of ``body``
    -- is itself content some single-sentence articles carry entirely
    (build plan issue #22).

    Each numbered appendix item also gets its own chunk; a lettered point
    inside it is folded into that same chunk (never split out on its own),
    while a numbered sub-row inside it (e.g. ``35.1`` under item ``35``,
    issue #147) gets its own chunk instead, a sibling of the item's -- see
    ``_chunks_for_appendix``/``_chunks_for_appendix_subitems``.

    An appendix's own named sections (Phụ lục VI's "Mục I"/"Mục II" prose,
    issue #147's fix round) are chunked the same way an article is: one
    chunk per section when its composed text is short enough, or one per
    clause -- each still folding in its own lettered points -- when it is
    not. See ``_chunks_for_appendix_sections``.

    A Phụ lục VI premium-table row or sub-row's chunk header also carries
    that table's own column header, read from the source text (issue #160)
    -- see ``_appendix_header``/``_with_column_header``. Chunk text is the
    only thing this changes: the unit's own ``body`` is untouched, and every
    other appendix is unaffected.

    Args:
        result: Parse result whose units belong to ``document``.
        document: Owning document; ``effective`` and ``status`` are copied
            onto every chunk verbatim, never re-derived.

    Returns:
        Chunks in the same reading order as their source articles/clauses.
    """
    chunks: list[Chunk] = []
    for article in _units_of_type(result.units, UnitType.ARTICLE):
        chunks.extend(_chunks_for_article(article, result.units, document))
    for appendix in _units_of_type(result.units, UnitType.APPENDIX):
        chunks.extend(_chunks_for_appendix(appendix, result.units, document))
        chunks.extend(_chunks_for_appendix_sections(appendix, result.units, document))
    return tuple(chunks)


def _chunks_for_appendix(
    appendix: Unit, units: tuple[Unit, ...], document: Document
) -> list[Chunk]:
    """Build one chunk per numbered appendix item, or one for an unnumbered appendix."""
    appendix_no = _require_appendix_no(appendix)
    items = [
        unit
        for unit in units
        if unit.unit_type is UnitType.APPENDIX_ITEM and unit.appendix_no == appendix_no
    ]
    if not items:
        body = _compose_with_header(
            _appendix_header(appendix),
            _compose(appendix, []),
            heading=appendix.heading,
        )
        return _as_chunk_list(appendix, body, document)
    chunks: list[Chunk] = []
    for item in items:
        item_no = _require_item_no(item)
        points = [
            unit
            for unit in units
            if unit.unit_type is UnitType.APPENDIX_POINT
            and unit.appendix_no == appendix_no
            and unit.item_no == item_no
        ]
        body = _compose_with_header(
            _appendix_header(appendix, item),
            _compose(item, points),
            heading=item.heading or appendix.heading,
        )
        chunks.extend(_as_chunk_list(item, body, document))
        chunks.extend(_chunks_for_appendix_subitems(appendix, item, units, document))
    return chunks


def _chunks_for_appendix_subitems(
    appendix: Unit, item: Unit, units: tuple[Unit, ...], document: Document
) -> list[Chunk]:
    """Build one chunk per sub-row of ``item`` (issue #147), each its own, separate chunk.

    A sub-row (e.g. ``35.1``/``35.2`` under item ``35``) carries its own
    substantial, topically distinct text -- different facility sub-types at
    different rates -- so it is chunked the same way a sibling item is
    (its own chunk, never folded into the parent item's), not the way a
    lettered point is (folded into its parent item's single chunk). Not
    composing the parent item's own chunk with its sub-rows' text (unlike
    ``points`` above) is what keeps that text from being embedded twice.
    """
    appendix_no = _require_appendix_no(appendix)
    item_no = _require_item_no(item)
    subitems = [
        unit
        for unit in units
        if unit.unit_type is UnitType.APPENDIX_SUBITEM
        and unit.appendix_no == appendix_no
        and unit.item_no == item_no
    ]
    chunks: list[Chunk] = []
    for subitem in subitems:
        body = _compose_with_header(
            _appendix_header(appendix, item, subitem),
            _compose(subitem, []),
            heading=subitem.heading or item.heading or appendix.heading,
        )
        chunks.extend(_as_chunk_list(subitem, body, document))
    return chunks


def _chunks_for_appendix_sections(
    appendix: Unit, units: tuple[Unit, ...], document: Document
) -> list[Chunk]:
    """Build chunks for every named section inside an appendix (issue #147).

    Phụ lục VI's own "Mục I"/"Mục II" prose is organized independently of
    its table's numeric STT (``_chunks_for_appendix`` above), and needs the
    same chunking treatment an article gets: a short section is one chunk,
    composing itself with every one of its clauses' and points' text in
    reading order; a section whose composed text is too long (Mục I's own
    post-table formula/"Ghi chú"/nhóm-35 note make it so) is split into one
    chunk per clause instead, each still folding in its own lettered points
    -- see ``_chunks_for_appendix_section``/``_chunks_for_appendix_section_clauses``,
    which mirror ``_chunks_for_article``/``_chunks_for_clauses`` exactly.
    """
    appendix_no = _require_appendix_no(appendix)
    sections = [
        unit
        for unit in units
        if unit.unit_type is UnitType.APPENDIX_SECTION and unit.appendix_no == appendix_no
    ]
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(_chunks_for_appendix_section(appendix, section, units, document))
    return chunks


def _chunks_for_appendix_section(
    appendix: Unit, section: Unit, units: tuple[Unit, ...], document: Document
) -> list[Chunk]:
    """Build one chunk for ``section``, or one per clause when it is too long."""
    appendix_no = _require_appendix_no(appendix)
    section_label = _require_appendix_section(section)
    clauses = [
        unit
        for unit in units
        if unit.unit_type is UnitType.APPENDIX_SECTION_CLAUSE
        and unit.appendix_no == appendix_no
        and unit.appendix_section == section_label
    ]
    text = _compose(section, _appendix_section_descendants(units, appendix_no, section_label))
    composed = _compose_with_header(
        _appendix_section_header(appendix, section), text, heading=section.heading
    )
    if not clauses or len(composed) <= ARTICLE_SPLIT_THRESHOLD:
        return _as_chunk_list(section, composed, document)
    return _chunks_for_appendix_section_clauses(appendix, section, clauses, units, document)


def _chunks_for_appendix_section_clauses(
    appendix: Unit,
    section: Unit,
    clauses: list[Unit],
    units: tuple[Unit, ...],
    document: Document,
) -> list[Chunk]:
    """Build one chunk per clause, each carrying its own points' text and section context."""
    appendix_no = _require_appendix_no(appendix)
    section_label = _require_appendix_section(section)
    chunks: list[Chunk] = []
    for clause in clauses:
        khoan_no = _require_khoan_no(clause)
        points = [
            unit
            for unit in units
            if unit.unit_type is UnitType.APPENDIX_SECTION_POINT
            and unit.appendix_no == appendix_no
            and unit.appendix_section == section_label
            and unit.khoan_no == khoan_no
        ]
        text = _compose(clause, points)
        composed = _compose_with_header(
            _appendix_section_header(appendix, section, clause), text, heading=clause.heading
        )
        chunks.extend(_as_chunk_list(clause, composed, document))
    return chunks


def _appendix_section_descendants(
    units: tuple[Unit, ...], appendix_no: int, section_label: str
) -> list[Unit]:
    """Return a section's clause and point units, in reading order."""
    return [
        unit
        for unit in units
        if unit.appendix_no == appendix_no
        and unit.appendix_section == section_label
        and unit.unit_type in (UnitType.APPENDIX_SECTION_CLAUSE, UnitType.APPENDIX_SECTION_POINT)
    ]


def _appendix_section_header(appendix: Unit, section: Unit, clause: Unit | None = None) -> str:
    """Build the context header for an appendix section, or one of its clauses.

    The section's own Roman label is rendered upper-case ("Mục I"/"Mục II")
    regardless of how it was stored on the unit, so a shop owner or the model
    reading this header sees exactly the label the decree itself prints
    (issue #147).
    """
    header = f"Phụ lục {_require_appendix_no(appendix)}"
    if appendix.heading:
        header = f"{header}. {appendix.heading}"
    header = f"{header} — Mục {_require_appendix_section(section).upper()}"
    if section.heading:
        header = f"{header}. {section.heading}"
    if clause is not None:
        header = f"{header} — Khoản {_require_khoan_no(clause)}"
    return header


def _require_appendix_section(unit: Unit) -> str:
    """Return the Roman section label carried by an appendix section unit (issue #147)."""
    if unit.appendix_section is None:
        msg = f"appendix section unit missing appendix_section: {unit.id}"
        raise ValueError(msg)
    return unit.appendix_section


def _chunks_for_article(article: Unit, units: tuple[Unit, ...], document: Document) -> list[Chunk]:
    """Build one chunk for ``article``, or one per clause when it is too long."""
    dieu_no = _require_dieu_no(article)
    clauses = [
        unit for unit in units if unit.dieu_no == dieu_no and unit.unit_type is UnitType.CLAUSE
    ]
    text = _compose(article, _article_descendants(units, dieu_no))
    composed = _compose_with_header(_context_header(article), text, heading=article.heading)

    if not clauses or len(composed) <= ARTICLE_SPLIT_THRESHOLD:
        return _as_chunk_list(article, composed, document)
    return _chunks_for_clauses(clauses, units, article, document)


def _chunks_for_clauses(
    clauses: list[Unit], units: tuple[Unit, ...], article: Unit, document: Document
) -> list[Chunk]:
    """Build one chunk per clause, each carrying its own points' text and article context."""
    dieu_no = _require_dieu_no(article)
    chunks: list[Chunk] = []
    for clause in clauses:
        khoan_no = _require_khoan_no(clause)
        text = _compose(clause, _clause_descendants(units, dieu_no, khoan_no))
        composed = _compose_with_header(
            _context_header(article, clause), text, heading=clause.heading
        )
        chunks.extend(_as_chunk_list(clause, composed, document))
    return chunks


def _context_header(article: Unit, clause: Unit | None = None) -> str:
    """Build the one-line context header composed onto the front of a chunk's body.

    ``Unit.body`` never includes ``Unit.heading``, so without this header an
    article's title -- the single most query-shaped sentence in the article --
    never reaches the text that gets embedded, and a split clause chunk would
    carry no indication of which article it belongs to at all. The header
    re-states the article's own coordinate and heading, plus the clause's
    coordinate when building a clause-level chunk:

        Điều 38. Điều kiện đối với cửa hàng bán lẻ LPG chai
        Điều 38. Điều kiện đối với cửa hàng bán lẻ LPG chai — Khoản 2

    An article with no heading in the source (``heading is None``) omits the
    ". <heading>" part rather than printing the literal word "None".

    Args:
        article: The owning article unit, for its ``dieu_no`` and ``heading``.
        clause: The clause unit, when building a clause-level chunk; omitted
            for an article-level chunk.

    Returns:
        The header line, in Vietnamese -- user-facing corpus text (C8), not
        code prose.
    """
    header = f"Điều {_require_dieu_no(article)}"
    if article.heading:
        header = f"{header}. {article.heading}"
    if clause is not None:
        header = f"{header} — Khoản {_require_khoan_no(clause)}"
    return header


def _appendix_header(appendix: Unit, item: Unit | None = None, subitem: Unit | None = None) -> str:
    """Build the context header for an appendix, one of its items, or one item's sub-row.

    A sub-row's own coordinate is rendered the same compound-number shape
    the source itself prints ("Mục 35.1"), not a separate "sub-item N" label
    -- that compound number is the only way a shop owner or the model
    reading this header could recognize it as the same row the law cites
    (issue #147).

    A Phụ lục VI premium-table row or sub-row also gets its own table's
    column header appended, read from the source text (issue #160) -- see
    ``_with_column_header``; every other appendix is unaffected.
    """
    header = f"Phụ lục {_require_appendix_no(appendix)}"
    if appendix.heading:
        header = f"{header}. {appendix.heading}"
    if item is not None:
        header = f"{header} — Mục {_require_item_no(item)}"
    if subitem is not None:
        header = f"{header}.{_require_subitem_no(subitem)}"
    return _with_column_header(header, appendix, item, subitem)


def _with_column_header(
    header: str, appendix: Unit, item: Unit | None, subitem: Unit | None
) -> str:
    """Append the premium table's own column header to a row's chunk header (issue #160).

    Scoped to Phụ lục VI alone (``appendix.appendix_no == APPENDIX_6_NO``);
    every other appendix's chunk header is returned unchanged. The text
    itself always comes from ``Unit.column_header``, which
    ``ingest/parser.py`` set from what ``ingest/extract.py`` actually read
    out of the table's own header line in the source -- never a string
    typed into this module.

    When the row carries no captured header -- the extraction-time fallback
    for a header that did not fully match (see
    ``ingest/extract.py``'s own ``_premium_table_column_header``) -- the
    header is returned unchanged, and that fallback is logged here, so the
    loss stays visible instead of silently shipping an uninformative chunk
    (issue #160's own fallback rule).
    """
    if appendix.appendix_no != APPENDIX_6_NO:
        return header
    row = subitem if subitem is not None else item
    if row is None:
        return header
    if row.column_header is None:
        logger.warning(
            "%s: no premium table column header was captured for this row; "
            "its chunk keeps the unmodified header (issue #160)",
            row.id,
        )
        return header
    return f"{header} (cột: {row.column_header})"


def _compose_with_header(header: str, text: str, *, heading: str | None) -> str:
    """Prepend ``header`` to ``text``, or fall back to the header alone.

    A heading is content, not a label: Vietnamese drafting sometimes packs an
    entire single-sentence article's substance into its heading, with no
    separate body line and no clauses at all (build plan issue #22). When
    ``text`` is empty but the unit has a real ``heading``, the header --
    which already carries that heading text (see ``_context_header``) -- is
    itself worth a chunk. A unit with neither a heading nor any composed text
    is not content, only a bare coordinate like "Điều 16"; that case still
    returns an empty string so ``_as_chunk_list`` skips it as before.
    """
    if text:
        return f"{header}\n{text}"
    return header if heading else ""


def _as_chunk_list(unit: Unit, body: str, document: Document) -> list[Chunk]:
    """Wrap the chunk built from ``unit`` in a list, or return ``[]`` when ``body`` is empty."""
    if not body:
        logger.debug("Skipping %s: composed text is empty", unit.id)
        return []
    return [
        Chunk(
            unit_id=unit.id,
            document_id=document.id,
            dieu_no=unit.dieu_no,
            khoan_no=unit.khoan_no,
            effective=document.effective,
            status=document.status,
            body=body,
        )
    ]


def _units_of_type(units: tuple[Unit, ...], unit_type: UnitType) -> list[Unit]:
    """Return the units of one level, in reading order."""
    return [unit for unit in units if unit.unit_type is unit_type]


def _article_descendants(units: tuple[Unit, ...], dieu_no: int) -> list[Unit]:
    """Return an article's clause and point units, in reading order."""
    return [
        unit
        for unit in units
        if unit.dieu_no == dieu_no and unit.unit_type in (UnitType.CLAUSE, UnitType.POINT)
    ]


def _clause_descendants(units: tuple[Unit, ...], dieu_no: int, khoan_no: int) -> list[Unit]:
    """Return a clause's point units, in reading order."""
    return [
        unit
        for unit in units
        if unit.dieu_no == dieu_no
        and unit.khoan_no == khoan_no
        and unit.unit_type is UnitType.POINT
    ]


def _compose(unit: Unit, descendants: list[Unit]) -> str:
    """Join a unit's own text with its descendants' text, in reading order.

    ``Unit.body`` holds a unit's own text only, so this is how an article's or
    clause's full text is recomposed for embedding.

    Deliberately does not re-emit the "1."/"a)" markers the parser stripped
    into ``khoan_no``/``diem`` -- a composed body reads as a flat run of
    sentences instead. Each point already reads as one complete, semicolon-
    terminated requirement, and the blank line between parts (below) marks the
    boundary as clearly as a marker would; re-emitting markers would also make
    this function level-aware (a clause needs "N." but a point needs a
    letter + ")"), which is complexity this module does not otherwise need.
    The alternative (re-emitting markers) is equally defensible -- this is a
    deliberate choice, not an oversight.
    """
    parts = [candidate.body for candidate in (unit, *descendants) if candidate.body]
    return "\n\n".join(parts)


def _require_dieu_no(unit: Unit) -> int:
    """Return ``unit.dieu_no``, or raise -- every article unit carries one (app/schemas/unit.py)."""
    if unit.dieu_no is None:
        msg = f"article unit missing dieu_no: {unit.id}"
        raise ValueError(msg)
    return unit.dieu_no


def _require_khoan_no(unit: Unit) -> int:
    """Return ``unit.khoan_no``, or raise -- every clause unit carries one (app/schemas/unit.py)."""
    if unit.khoan_no is None:
        msg = f"clause unit missing khoan_no: {unit.id}"
        raise ValueError(msg)
    return unit.khoan_no


def _require_appendix_no(unit: Unit) -> int:
    """Return the appendix number carried by an appendix unit."""
    if unit.appendix_no is None:
        msg = f"appendix unit missing appendix_no: {unit.id}"
        raise ValueError(msg)
    return unit.appendix_no


def _require_item_no(unit: Unit) -> int:
    """Return the item number carried by an appendix item unit."""
    if unit.item_no is None:
        msg = f"appendix item missing item_no: {unit.id}"
        raise ValueError(msg)
    return unit.item_no


def _require_subitem_no(unit: Unit) -> int:
    """Return the sub-item number carried by an appendix sub-item unit (issue #147)."""
    if unit.subitem_no is None:
        msg = f"appendix sub-item missing subitem_no: {unit.id}"
        raise ValueError(msg)
    return unit.subitem_no
