"""Typed corpus units, mirroring the ``units`` table (build plan A5.3).

A unit is one node of the ``Chương -> Điều -> Khoản -> Điểm`` tree of a legal
document. Two invariants are enforced here rather than in the parser, so no
caller can build a unit that would break a downstream join (A5.1 is locked and
Phases 2 and 3 join on these identifiers):

* ``id`` must equal the canonical identifier derived from the unit's own
  coordinates, so there is exactly one way to spell a unit.
* ``parent_id`` must name the unit's real parent, or be ``None`` when the parent
  is the document itself. ``units.parent_id`` references ``units(id)`` in the
  DDL, so a chapter -- whose parent is the document -- carries ``None`` and
  relies on ``document_id`` instead.

Each unit carries only its **own** coordinates: an article records ``dieu_no``
and reaches its chapter through ``parent_id``, never through ``chapter_no``.
"""

from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.ids import (
    CHAPTER_PREFIX,
    UNIT_SEPARATOR,
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


class UnitType(StrEnum):
    """Level of a unit within the document tree."""

    CHAPTER = "chapter"
    ARTICLE = "article"
    CLAUSE = "clause"
    POINT = "point"
    APPENDIX = "appendix"
    APPENDIX_ITEM = "appendix_item"
    APPENDIX_SECTION = "appendix_section"
    APPENDIX_SECTION_CLAUSE = "appendix_section_clause"
    APPENDIX_SECTION_POINT = "appendix_section_point"
    APPENDIX_SUBITEM = "appendix_subitem"
    APPENDIX_POINT = "appendix_point"
    APPENDIX_FORM = "appendix_form"


_COORDINATE_FIELDS = (
    "chapter_no",
    "dieu_no",
    "khoan_no",
    "diem",
    "appendix_no",
    "appendix_section",
    "item_no",
    "subitem_no",
    "form_code",
)

_OWN_COORDINATES: dict[UnitType, frozenset[str]] = {
    UnitType.CHAPTER: frozenset({"chapter_no"}),
    UnitType.ARTICLE: frozenset({"dieu_no"}),
    UnitType.CLAUSE: frozenset({"dieu_no", "khoan_no"}),
    UnitType.POINT: frozenset({"dieu_no", "khoan_no", "diem"}),
    UnitType.APPENDIX: frozenset({"appendix_no"}),
    UnitType.APPENDIX_ITEM: frozenset({"appendix_no", "item_no"}),
    UnitType.APPENDIX_SECTION: frozenset({"appendix_no", "appendix_section"}),
    UnitType.APPENDIX_SECTION_CLAUSE: frozenset({"appendix_no", "appendix_section", "khoan_no"}),
    UnitType.APPENDIX_SECTION_POINT: frozenset(
        {"appendix_no", "appendix_section", "khoan_no", "diem"}
    ),
    UnitType.APPENDIX_SUBITEM: frozenset({"appendix_no", "item_no", "subitem_no"}),
    UnitType.APPENDIX_POINT: frozenset({"appendix_no", "item_no", "diem"}),
    UnitType.APPENDIX_FORM: frozenset({"appendix_no", "form_code"}),
}
"""Coordinates a unit of each type must carry; every other coordinate stays unset."""

_T = TypeVar("_T")


class Unit(BaseModel):
    """One node of a document's structural tree.

    Attributes:
        id: Stable identifier per A5.1, e.g. ``gas-decree-87-2018#dieu-38#khoan-1``.
        document_id: Owning document identifier.
        unit_type: Level of the unit.
        chapter_no: Chapter number, on chapters only.
        dieu_no: Article number, on articles and below.
        khoan_no: Clause number, on clauses and below.
        diem: Point letter, on points only.
        subitem_no: Sub-row number inside an appendix item (e.g. ``1`` for
            ``35.1``), on appendix sub-items only (issue #147).
        form_code: Sample form code (e.g. ``pc01``), on forms only (issue #131).
        column_header: Phụ lục VI's own premium-table column header, read
            from the source text by ``ingest/extract.py`` (issue #160), on an
            ``appendix_item``/``appendix_subitem`` row of that one table
            only. Never a string typed into this module, and never
            persisted -- ``app/repositories/units.py`` does not write it --
            since it exists only to carry the header from parsing through to
            chunk-building (``ingest/chunk.py``) within a single ingest run.
        parent_id: Parent unit identifier, or ``None`` when the parent is the document.
        heading: Title printed next to the unit number, when the source has one.
        body: The unit's own text, excluding the text of its children.
        ordinal: 0-based position in document reading order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    document_id: str
    unit_type: UnitType
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
    parent_id: str | None = None
    heading: str | None = None
    body: str = ""
    ordinal: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_identifiers(self) -> "Unit":
        """Pin ``id`` and ``parent_id`` to the locked identifier scheme.

        Returns:
            The validated unit.

        Raises:
            ValueError: If a coordinate is missing or set on the wrong unit type,
                or if ``id`` or ``parent_id`` does not match the coordinates.
        """
        doc_id = document_id(self.document_id)
        self._reject_foreign_coordinates()
        self._reject_misplaced_column_header()
        expected_id, expected_parent = self._canonical(doc_id)

        if self.id != expected_id:
            msg = f"id must be {expected_id!r} for these coordinates, got {self.id!r}"
            raise ValueError(msg)
        if self.parent_id != expected_parent:
            msg = f"parent_id must be {expected_parent!r} for {self.id!r}, got {self.parent_id!r}"
            raise ValueError(msg)
        return self

    def _reject_foreign_coordinates(self) -> None:
        """Raise when a coordinate belonging to another unit type is set."""
        own = _OWN_COORDINATES[self.unit_type]
        extra = [
            name
            for name in _COORDINATE_FIELDS
            if name not in own and getattr(self, name) is not None
        ]
        if extra:
            msg = f"{self.unit_type.value} unit must not set {', '.join(extra)}"
            raise ValueError(msg)

    def _reject_misplaced_column_header(self) -> None:
        """Raise when ``column_header`` is set on a unit type that cannot carry one.

        Only a Phụ lục VI premium-table row (``appendix_item``) or sub-row
        (``appendix_subitem``) ever carries a captured column header (issue
        #160); every other unit type -- including this same appendix's own
        ``appendix_section``/``appendix_section_clause`` prose units -- keeps
        it unset.
        """
        if self.column_header is not None and self.unit_type not in (
            UnitType.APPENDIX_ITEM,
            UnitType.APPENDIX_SUBITEM,
        ):
            msg = f"{self.unit_type.value} unit must not set column_header"
            raise ValueError(msg)

    def _canonical(self, doc_id: str) -> tuple[str, str | None]:  # noqa: C901
        """Return the identifier and parent identifier implied by the coordinates."""
        match self.unit_type:
            case UnitType.CHAPTER:
                return chapter_id(doc_id, _require(self.chapter_no, "chapter_no")), None
            case UnitType.ARTICLE:
                return article_id(doc_id, _require(self.dieu_no, "dieu_no")), self._article_parent(
                    doc_id
                )
            case UnitType.CLAUSE:
                dieu_no = _require(self.dieu_no, "dieu_no")
                khoan_no = _require(self.khoan_no, "khoan_no")
                return clause_id(doc_id, dieu_no, khoan_no), article_id(doc_id, dieu_no)
            case UnitType.POINT:
                dieu_no = _require(self.dieu_no, "dieu_no")
                khoan_no = _require(self.khoan_no, "khoan_no")
                diem = _require(self.diem, "diem")
                return (
                    point_id(doc_id, dieu_no, khoan_no, diem),
                    clause_id(doc_id, dieu_no, khoan_no),
                )
            case UnitType.APPENDIX:
                return appendix_id(doc_id, _require(self.appendix_no, "appendix_no")), None
            case UnitType.APPENDIX_ITEM:
                appendix_no = _require(self.appendix_no, "appendix_no")
                item_no = _require(self.item_no, "item_no")
                return appendix_item_id(doc_id, appendix_no, item_no), appendix_id(
                    doc_id, appendix_no
                )
            case UnitType.APPENDIX_SECTION:
                appendix_no = _require(self.appendix_no, "appendix_no")
                section = _require(self.appendix_section, "appendix_section")
                return appendix_section_id(doc_id, appendix_no, section), appendix_id(
                    doc_id, appendix_no
                )
            case UnitType.APPENDIX_SECTION_CLAUSE:
                appendix_no = _require(self.appendix_no, "appendix_no")
                section = _require(self.appendix_section, "appendix_section")
                clause_no = _require(self.khoan_no, "khoan_no")
                return (
                    appendix_section_clause_id(doc_id, appendix_no, section, clause_no),
                    appendix_section_id(doc_id, appendix_no, section),
                )
            case UnitType.APPENDIX_SECTION_POINT:
                appendix_no = _require(self.appendix_no, "appendix_no")
                section = _require(self.appendix_section, "appendix_section")
                clause_no = _require(self.khoan_no, "khoan_no")
                diem = _require(self.diem, "diem")
                return (
                    appendix_section_point_id(doc_id, appendix_no, section, clause_no, diem),
                    appendix_section_clause_id(doc_id, appendix_no, section, clause_no),
                )
            case UnitType.APPENDIX_SUBITEM:
                appendix_no = _require(self.appendix_no, "appendix_no")
                item_no = _require(self.item_no, "item_no")
                subitem_no = _require(self.subitem_no, "subitem_no")
                return (
                    appendix_subitem_id(doc_id, appendix_no, item_no, subitem_no),
                    appendix_item_id(doc_id, appendix_no, item_no),
                )
            case UnitType.APPENDIX_POINT:
                appendix_no = _require(self.appendix_no, "appendix_no")
                item_no = _require(self.item_no, "item_no")
                diem = _require(self.diem, "diem")
                return (
                    appendix_point_id(doc_id, appendix_no, item_no, diem),
                    appendix_item_id(doc_id, appendix_no, item_no),
                )
            case UnitType.APPENDIX_FORM:
                appendix_no = _require(self.appendix_no, "appendix_no")
                form_code = _require(self.form_code, "form_code")
                return appendix_form_id(doc_id, appendix_no, form_code), appendix_id(
                    doc_id, appendix_no
                )

    def _article_parent(self, doc_id: str) -> str | None:
        """Return the article's chapter identifier, or ``None`` when it has no chapter."""
        if self.parent_id is None:
            return None
        prefix = f"{doc_id}{UNIT_SEPARATOR}{CHAPTER_PREFIX}-"
        if not self.parent_id.startswith(prefix):
            msg = f"an article's parent must be a chapter of {doc_id}, got {self.parent_id!r}"
            raise ValueError(msg)
        return self.parent_id


def _require(value: _T | None, field_name: str) -> _T:
    """Return ``value`` when it is set, otherwise raise.

    Args:
        value: Coordinate to check.
        field_name: Name reported in the error message.

    Returns:
        The coordinate, narrowed to a non-optional type.

    Raises:
        ValueError: If the coordinate is unset.
    """
    if value is None:
        msg = f"{field_name} is required for this unit type"
        raise ValueError(msg)
    return value
