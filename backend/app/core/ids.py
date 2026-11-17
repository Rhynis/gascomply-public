"""Stable identifier scheme for corpus nodes (build plan A5.1 - locked).

Every downstream table joins on these strings, so the format is frozen here and
pinned by tests. Vietnamese unit words are kept in the identifiers on purpose:
they mirror the source documents and keep an identifier readable next to the
statute it came from.

    Document     gas-decree-87-2018
    Cited doc    cited-136-2020-nd-cp   (a stub: known only from a citation, issue #67)
    Chapter      gas-decree-87-2018#chuong-3
    Article      gas-decree-87-2018#dieu-38
    Clause       gas-decree-87-2018#dieu-38#khoan-1
    Point        gas-decree-87-2018#dieu-38#khoan-1#diem-a
    Appendix     gas-decree-87-2018#phu-luc-1
    Appendix sub-row pccc-decree-105-2025#phu-luc-6#muc-35#muc-1   (a table row's own
                 sub-row, issue #147)
    Form         pccc-decree-105-2025#phu-luc-8#mau-pc01   (a sample form, issue #131)
    Entity       req:min-store-area, form:..., authority:..., concept:..., fee:...

The ``#`` and ``<type>:`` prefixes keep the identifier namespaces disjoint, so a
single ``edges(src_id, dst_id)`` table can link any node type to any other
unambiguously (A5.2).

A stub's id is a plain document id -- never containing ``:`` -- because
``ids.is_entity_id`` decides entity-ness from an id's prefix, and
an id with an unknown colon prefix would be dropped as a dangling edge (#61).
"""

import re
import unicodedata
from enum import StrEnum

UNIT_SEPARATOR = "#"
"""Separator between a document identifier and each nested unit segment."""

ENTITY_SEPARATOR = ":"
"""Separator between an entity type prefix and its slug."""

STUB_ID_PREFIX = "cited-"
"""Reserved prefix of every stub document id (issue #67). ``document_id`` refuses
it, so a manifest document can never take an id a stub could also derive: the
two namespaces are disjoint by construction, not by hoping nobody names a
document ``cited-...``."""

CHAPTER_PREFIX = "chuong"
ARTICLE_PREFIX = "dieu"
CLAUSE_PREFIX = "khoan"
POINT_PREFIX = "diem"
APPENDIX_PREFIX = "phu-luc"
APPENDIX_ITEM_PREFIX = "muc"
APPENDIX_FORM_PREFIX = "mau"
"""Prefix for a sample form (``Mẫu số ...``) inside an appendix (issue #131). A
form is its own node, sibling to ``APPENDIX_ITEM_PREFIX``, never nested under
it: a form's own numbered fields are part of the form's text, not separate
``muc``/``khoan``/``diem`` units -- see ``appendix_form_id``."""

_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_POINT_PATTERN = re.compile(r"^[a-zđ]{1,2}$")
_FORM_CODE_PATTERN = re.compile(r"^[a-z0-9]+$")
"""A form code is whatever follows "Mẫu số" in the source, lower-cased --
``PC01`` -> ``pc01``, ``01a`` -> ``01a``. No internal hyphen is accepted (unlike
``_SLUG_PATTERN``): every form code measured on the real corpus is a bare
alphanumeric run, and disallowing a hyphen here keeps ``_join``'s own
``prefix-value`` separator unambiguous when splitting an id back apart."""


class EntityType(StrEnum):
    """Non-document node types; the value doubles as the identifier prefix (A5.3)."""

    REQUIREMENT = "req"
    FORM = "form"
    AUTHORITY = "authority"
    CONCEPT = "concept"
    FEE = "fee"


def document_id(slug: str) -> str:
    """Return the document identifier for a manifest slug.

    Args:
        slug: Lower-case, hyphen-separated document slug, e.g. ``gas-decree-87-2018``.

    Returns:
        The validated document identifier.

    Raises:
        ValueError: If the slug is not a lower-case hyphen-separated slug, or
            starts with ``STUB_ID_PREFIX``, which is reserved for stubs.
    """
    if not _SLUG_PATTERN.match(slug):
        msg = f"Invalid document slug: {slug!r}"
        raise ValueError(msg)
    if slug.startswith(STUB_ID_PREFIX):
        msg = f"Document slug {slug!r} uses the prefix {STUB_ID_PREFIX!r}, reserved for stubs"
        raise ValueError(msg)
    return slug


def stub_document_id(number: str) -> str:
    """Return the id of the stub for a cited document number (issue #67).

    Deterministic, so a re-ingest derives exactly the same id, and injective
    over the numbers a stub can carry: ``136/2020/NĐ-CP`` becomes
    ``cited-136-2020-nd-cp``. Case, the ``Đ``/``D`` accent and the ``/`` and
    ``-`` separators fold away, which is what makes an accent-stripped
    spelling of the same number the same stub. Leading zeros are *not*
    folded: numbers are identified as printed.

    Args:
        number: A canonical document number, as ``ingest.references.find_document_number``
            returns it -- whitespace already removed.

    Returns:
        A plain document id, never containing ``:`` or ``#``.

    Raises:
        ValueError: If ``number`` leaves nothing to build a slug from.
    """
    folded = unicodedata.normalize("NFKD", number.lower().replace("đ", "d"))
    slug = re.sub(r"[^a-z0-9]+", "-", folded.encode("ascii", "ignore").decode("ascii")).strip("-")
    stub_id = f"{STUB_ID_PREFIX}{slug}"
    if not _SLUG_PATTERN.match(stub_id):
        msg = f"Cannot derive a stub id from document number {number!r}"
        raise ValueError(msg)
    return stub_id


def chapter_id(doc_id: str, chuong_no: int) -> str:
    """Return the stable identifier of a chapter (Chương).

    Args:
        doc_id: Owning document identifier.
        chuong_no: 1-based chapter number.

    Returns:
        An identifier such as ``gas-decree-87-2018#chuong-3``.

    Raises:
        ValueError: If the document identifier or the number is invalid.
    """
    return _join(doc_id, (CHAPTER_PREFIX, _positive(chuong_no, CHAPTER_PREFIX)))


def article_id(doc_id: str, dieu_no: int) -> str:
    """Return the stable identifier of an article (Điều).

    Args:
        doc_id: Owning document identifier.
        dieu_no: 1-based article number.

    Returns:
        An identifier such as ``gas-decree-87-2018#dieu-38``.

    Raises:
        ValueError: If the document identifier or the number is invalid.
    """
    return _join(doc_id, (ARTICLE_PREFIX, _positive(dieu_no, ARTICLE_PREFIX)))


def clause_id(doc_id: str, dieu_no: int, khoan_no: int) -> str:
    """Return the stable identifier of a clause (Khoản).

    Args:
        doc_id: Owning document identifier.
        dieu_no: 1-based article number.
        khoan_no: 1-based clause number within the article.

    Returns:
        An identifier such as ``gas-decree-87-2018#dieu-38#khoan-1``.

    Raises:
        ValueError: If the document identifier or a number is invalid.
    """
    return _join(
        doc_id,
        (ARTICLE_PREFIX, _positive(dieu_no, ARTICLE_PREFIX)),
        (CLAUSE_PREFIX, _positive(khoan_no, CLAUSE_PREFIX)),
    )


def point_id(doc_id: str, dieu_no: int, khoan_no: int, diem: str) -> str:
    """Return the stable identifier of a point (Điểm).

    Args:
        doc_id: Owning document identifier.
        dieu_no: 1-based article number.
        khoan_no: 1-based clause number within the article.
        diem: Point letter as printed in the source, e.g. ``a`` or ``đ``.

    Returns:
        An identifier such as ``gas-decree-87-2018#dieu-38#khoan-1#diem-a``.

    Raises:
        ValueError: If the document identifier, a number or the letter is invalid.
    """
    normalized = diem.strip().lower()
    if not _POINT_PATTERN.match(normalized):
        msg = f"Invalid point letter: {diem!r}"
        raise ValueError(msg)
    return _join(
        doc_id,
        (ARTICLE_PREFIX, _positive(dieu_no, ARTICLE_PREFIX)),
        (CLAUSE_PREFIX, _positive(khoan_no, CLAUSE_PREFIX)),
        (POINT_PREFIX, normalized),
    )


def appendix_id(doc_id: str, appendix_no: int) -> str:
    """Return the stable identifier of an appendix."""
    return _join(doc_id, (APPENDIX_PREFIX, _positive(appendix_no, APPENDIX_PREFIX)))


def appendix_item_id(doc_id: str, appendix_no: int, item_no: int) -> str:
    """Return the stable identifier of a numbered item inside an appendix."""
    return _join(
        doc_id,
        (APPENDIX_PREFIX, _positive(appendix_no, APPENDIX_PREFIX)),
        (APPENDIX_ITEM_PREFIX, _positive(item_no, APPENDIX_ITEM_PREFIX)),
    )


def _appendix_section_segment(section: str) -> tuple[str, str]:
    """Return the validated ``(prefix, value)`` segment for an appendix section.

    Shared by ``appendix_section_id``/``appendix_section_clause_id``/
    ``appendix_section_point_id`` so the lower-casing and shape check happen
    in exactly one place. ``_join`` only ever accepts a raw document id as
    its first argument (see its own docstring) -- each of those three
    functions builds its own full segment chain from ``doc_id`` down,
    rather than chaining through another function's already-joined string.
    """
    normalized = section.strip().lower()
    if not re.fullmatch(r"[ivxlcdm]+", normalized):
        msg = f"Invalid appendix section: {section!r}"
        raise ValueError(msg)
    return (APPENDIX_ITEM_PREFIX, normalized)


def appendix_section_id(doc_id: str, appendix_no: int, section: str) -> str:
    """Return a named section id inside an appendix.

    Phụ lục VI has two independent numbering schemes: numeric ``mục`` values
    name premium-table rows, while its prose is organized as Roman sections I
    and II. Keeping the prose's own labels prevents a clause such as "1." in
    Mục I from colliding with table row 1.
    """
    return _join(
        doc_id,
        (APPENDIX_PREFIX, _positive(appendix_no, APPENDIX_PREFIX)),
        _appendix_section_segment(section),
    )


def appendix_section_clause_id(doc_id: str, appendix_no: int, section: str, clause_no: int) -> str:
    """Return a clause id inside a Roman-labelled appendix section."""
    return _join(
        doc_id,
        (APPENDIX_PREFIX, _positive(appendix_no, APPENDIX_PREFIX)),
        _appendix_section_segment(section),
        (CLAUSE_PREFIX, _positive(clause_no, CLAUSE_PREFIX)),
    )


def appendix_section_point_id(
    doc_id: str, appendix_no: int, section: str, clause_no: int, diem: str
) -> str:
    """Return a point id inside a clause of a Roman-labelled appendix section."""
    normalized_diem = diem.strip().lower()
    if not _POINT_PATTERN.match(normalized_diem):
        msg = f"Invalid point letter: {diem!r}"
        raise ValueError(msg)
    return _join(
        doc_id,
        (APPENDIX_PREFIX, _positive(appendix_no, APPENDIX_PREFIX)),
        _appendix_section_segment(section),
        (CLAUSE_PREFIX, _positive(clause_no, CLAUSE_PREFIX)),
        (POINT_PREFIX, normalized_diem),
    )


def appendix_subitem_id(doc_id: str, appendix_no: int, item_no: int, subitem_no: int) -> str:
    """Return the stable identifier of a numbered sub-row inside an appendix item.

    Some appendix tables number a row hierarchically in the source -- Phụ lục
    VI's own insurance-premium table (issue #147) prints ``35``, then ``35.1``
    and ``35.2`` as its two sub-rows, each carrying its own rate. A sub-row is
    a genuine child of the item it continues, never a sibling that happens to
    reuse the parent's own ``item_no`` -- which is exactly the bug this issue
    exists to fix: read naively, ``CLAUSE_ANCHOR`` sees only the leading
    ``35`` before the dot and treats every sub-row as a duplicate of item 35
    itself.

    Nested with the same ``APPENDIX_ITEM_PREFIX`` segment the parent item
    already uses, not a new prefix -- a sub-row is one more ``mục``, numbered
    one level under its own parent ``mục``, the same relationship
    ``clause_id``/``point_id`` already encode for Điều/Khoản. ``35.1``
    becomes ``...#phu-luc-6#muc-35#muc-1``: unambiguous (the parent segment
    always comes first) and readable next to the source's own compound
    number.

    Args:
        doc_id: Owning document identifier.
        appendix_no: 1-based appendix number.
        item_no: 1-based number of the parent appendix item (the part before
            the dot in the source, e.g. ``35``).
        subitem_no: 1-based number of the sub-row itself (the part after the
            dot, e.g. ``1`` for ``35.1``) -- local to its parent item, not a
            running count across the whole appendix.

    Returns:
        An identifier such as ``pccc-decree-105-2025#phu-luc-6#muc-35#muc-1``.

    Raises:
        ValueError: If the document identifier or a number is invalid.
    """
    return _join(
        doc_id,
        (APPENDIX_PREFIX, _positive(appendix_no, APPENDIX_PREFIX)),
        (APPENDIX_ITEM_PREFIX, _positive(item_no, APPENDIX_ITEM_PREFIX)),
        (APPENDIX_ITEM_PREFIX, _positive(subitem_no, APPENDIX_ITEM_PREFIX)),
    )


def appendix_form_id(doc_id: str, appendix_no: int, form_code: str) -> str:
    """Return the stable identifier of a sample form (``Mẫu số ...``) inside an appendix.

    A form is split out as its own unit, sibling to an appendix item, never a
    child of one: ``<doc>#phu-luc-<n>#mau-<code>`` (issue #131). A form's own
    numbered fields stay inside its ``body`` text rather than becoming
    ``appendix_item_id``/``appendix_point_id`` children -- those fields are not
    legal subdivisions of the appendix, they are the form's own printed
    layout, and citing one by a ``muc``/``khoan`` coordinate names something
    that does not exist in the source law.

    Args:
        doc_id: Owning document identifier.
        appendix_no: 1-based appendix number.
        form_code: The code printed after "Mẫu số", e.g. ``PC01`` or ``01a``.
            Case-folded to lower case; the source's own case (``PC01`` vs
            ``pc01``) is not legally significant and is not worth a second
            identifier for the same form.

    Returns:
        An identifier such as ``pccc-decree-105-2025#phu-luc-8#mau-pc01``.

    Raises:
        ValueError: If the document identifier, the appendix number, or the
            form code is invalid.
    """
    normalized = form_code.strip().lower()
    if not _FORM_CODE_PATTERN.match(normalized):
        msg = f"Invalid form code: {form_code!r}"
        raise ValueError(msg)
    return _join(
        doc_id,
        (APPENDIX_PREFIX, _positive(appendix_no, APPENDIX_PREFIX)),
        (APPENDIX_FORM_PREFIX, normalized),
    )


def appendix_point_id(doc_id: str, appendix_no: int, item_no: int, diem: str) -> str:
    """Return the stable identifier of a point inside an appendix item."""
    normalized = diem.strip().lower()
    if not _POINT_PATTERN.match(normalized):
        msg = f"Invalid point letter: {diem!r}"
        raise ValueError(msg)
    return _join(
        doc_id,
        (APPENDIX_PREFIX, _positive(appendix_no, APPENDIX_PREFIX)),
        (APPENDIX_ITEM_PREFIX, _positive(item_no, APPENDIX_ITEM_PREFIX)),
        (POINT_PREFIX, normalized),
    )


def entity_id(entity_type: EntityType, slug: str) -> str:
    """Return the stable identifier of a non-document node.

    Args:
        entity_type: Node type; its value is used as the identifier prefix.
        slug: Lower-case, hyphen-separated slug, e.g. ``min-store-area``.

    Returns:
        An identifier such as ``req:min-store-area``.

    Raises:
        ValueError: If the slug is not a lower-case hyphen-separated slug.
    """
    if not _SLUG_PATTERN.match(slug):
        msg = f"Invalid entity slug: {slug!r}"
        raise ValueError(msg)
    return f"{entity_type.value}{ENTITY_SEPARATOR}{slug}"


ENTITY_PREFIXES = tuple(f"{entity_type.value}{ENTITY_SEPARATOR}" for entity_type in EntityType)
"""Every entity id prefix (``req:``, ``concept:``, ...), built from ``EntityType`` rather than
hard-coded, so a new ``EntityType`` member needs no second edit here."""


def is_entity_id(node_id: str) -> bool:
    """True when ``node_id`` is an entity id, never a unit or document id.

    Decided from the id's own shape (the ``<type>:`` prefix), never from whether a lookup
    happens to miss -- a document that simply does not exist must not be mistaken for an
    entity (issue #58's review). The one definition ``graph_traversal`` and ``retriever``
    both use.
    """
    return node_id.startswith(ENTITY_PREFIXES)


def unit_and_ancestors(unit_id: str) -> tuple[str, ...]:
    """Return ``unit_id`` followed by each unit that encloses it, deepest first.

    The walk stops short of the document itself: a document has no text of its own to
    return, so the bare document id is never a candidate. This is how a unit that has no
    chunk of its own is mapped back to the chunk that holds its text (issue #89): a
    chunk is one article, or one clause of a long article, and the deepest of these ids
    that names a chunk is the one that contains the unit.

    Args:
        unit_id: A unit id, e.g. ``gas-decree-87-2018#dieu-53#khoan-1``. An entity id or
            a bare document id has no ``#`` and yields ``()``.

    Returns:
        ``("gas-decree-87-2018#dieu-53#khoan-1", "gas-decree-87-2018#dieu-53")`` for the
        example above; ``()`` for an id that names no unit.
    """
    parts = unit_id.split(UNIT_SEPARATOR)
    return tuple(UNIT_SEPARATOR.join(parts[:size]) for size in range(len(parts), 1, -1))


def _join(doc_id: str, *segments: tuple[str, str | int]) -> str:
    """Join a validated document identifier with ``prefix-value`` unit segments."""
    parts = [document_id(doc_id)]
    parts.extend(f"{prefix}-{value}" for prefix, value in segments)
    return UNIT_SEPARATOR.join(parts)


def _positive(value: int, label: str) -> int:
    """Return ``value`` when it is a positive integer, otherwise raise."""
    if value < 1:
        msg = f"{label} number must be >= 1, got {value}"
        raise ValueError(msg)
    return value
