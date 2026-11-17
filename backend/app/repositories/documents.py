"""Async repository for the ``documents`` table (build plan A5.3, A6 Phase 1.5b).

Persists the existing ``Document`` model verbatim -- no second, ORM-shaped
domain object. Every query is parameterized (C3); nothing here builds SQL by
string interpolation.
"""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ids import UNIT_SEPARATOR, unit_and_ancestors
from app.schemas.document import Document, DocumentStub
from app.schemas.unit import UnitType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentRow:
    """One ``documents`` row, read back exactly as stored.

    Not the write-side ``Document`` model: that one requires ``path`` (a real,
    non-empty file location), which a stub row genuinely does not have --
    ``path`` is ``NULL`` for every stub (``DocumentRepository.insert_stubs``).
    Building a ``Document`` from a stub's row would therefore fail its own
    validation. This is the read-side shape instead, one field per column,
    that a caller (``GET /doc/{id}``, issue #90) turns into whatever response
    it needs.

    Attributes:
        id: Document id.
        number: Official document number, or the number itself for a stub
            (``DocumentStub.title_vi``'s own rule).
        title_vi: Vietnamese title.
        doc_type: Raw stored type string, ``"unknown"`` for a stub.
        issued: Signature date, when known.
        effective: Date the document took effect, when known.
        status: Raw stored status string, ``"unknown"`` for a stub.
        source_url: Page the document was downloaded from.
        fetched: Date the file was downloaded.
        path: Location of the downloaded file, or ``None`` for a stub.
        meta: Free-form notes.
    """

    id: str
    number: str
    title_vi: str
    doc_type: str
    issued: date | None
    effective: date | None
    status: str
    source_url: str | None
    fetched: date | None
    path: str | None
    meta: dict[str, Any]


@dataclass(frozen=True)
class AppendixItemRow:
    """One appendix item or sub-item unit, with what a verbatim quote needs (issue #146).

    ``DocumentRepository.get_appendix_items`` is the only source of these: its own SQL
    filters to ``unit_type IN ('appendix_item', 'appendix_subitem')`` server-side, so a
    caller can never receive a form, an article, a clause, an appendix's own Roman-numbered
    prose section, or a lettered point under either name -- only the two unit types issue
    #146 allows a legal quote to be built from.

    Attributes:
        id: Unit id. An ``appendix_subitem`` row's own sub-item number (e.g. ``1`` for
            ``35.1``) is not a column on ``units`` at all -- ``item_no`` holds the
            *parent* item's number for this unit type (e.g. ``35``) -- so a caller that
            needs it reads the id's own trailing ``#muc-<n>`` segment instead, the same
            regex-on-the-locked-id-scheme approach ``app.schemas.ask.Citation`` already
            uses for ``appendix_no``/``item_no``.
        unit_type: Raw stored type -- ``"appendix_item"`` or ``"appendix_subitem"``.
        appendix_no: Owning appendix's 1-based position (``VII`` is ``7``).
        item_no: Item number inside the appendix (the *parent* item, for a sub-item row).
        body: The unit's own text, exactly as stored -- never re-normalized, so a corpus
            quirk (a stray space inside a word) survives into the quote unchanged, as a
            real legal quote must (issue #146's own trap 2).
    """

    id: str
    unit_type: str
    appendix_no: int | None
    item_no: int | None
    body: str


_GET_APPENDIX_ITEMS_SQL = text(
    "SELECT id, unit_type, appendix_no, item_no, body FROM units "
    "WHERE id = ANY(:ids) AND unit_type = ANY(:unit_types)"
)


@dataclass(frozen=True)
class FocusUnitRow:
    """One focus unit's own body, parent and sibling order (issue #187).

    An article- or clause-level passage's focus units -- the real units nested inside
    it (``Passage.focus_units``'s own docstring) -- are rendered in the generation
    prompt as their own lines, each showing that unit's own body exactly once, in the
    order the corpus itself nests them: parent before child, siblings by ``ordinal``,
    never by sorting the id strings (``khoan-10`` sorts before ``khoan-2`` as text).
    ``DocumentRepository.get_focus_unit_contexts`` is the only source of these; unlike
    ``get_unit_contexts`` (a unit id's full recomposed text, descendants folded in, for
    the grounding verifier), this is each unit's *own* body alone, plus just enough
    tree shape to order several of them relative to each other.

    Attributes:
        id: The unit's own id, exactly as stored.
        body: The unit's own text, verbatim; empty for a unit type whose content lives
            entirely in its children (e.g. an article the corpus subdivided into
            clauses has an empty ``body`` of its own).
        parent_id: The immediate containing unit's id, or ``None`` for a document root.
        ordinal: Sibling order under ``parent_id``, as stored at ingest time.
    """

    id: str
    body: str
    parent_id: str | None
    ordinal: int


_GET_FOCUS_UNIT_CONTEXTS_SQL = text(
    "SELECT id, coalesce(body, '') AS body, parent_id, coalesce(ordinal, 0) AS ordinal "
    "FROM units WHERE id = ANY(:ids)"
)


@dataclass(frozen=True)
class RuleUnitRow:
    """One Phụ lục VI ``appendix_section_point`` unit found by its own text (issue #165).

    A premium-table row's own "M"/"N" cell names a deductible type, but the row's own
    text never states that type's actual rule (the real cap -- e.g. 1% for M, 10% for
    N -- lives in Phụ lục VI's own Mục II khoản 1, on a lettered điểm).
    ``DocumentRepository.get_deductible_rule_units`` is the only source of these: it
    finds a letter's rule unit by searching every candidate điểm's own stored text for
    "thuộc loại M"/"thuộc loại N", never by computing an id from the scheme -- so a
    parse that moved the rule to a different điểm, or renumbered it, is still found
    correctly (or, if truly missing, simply absent from the result).

    Attributes:
        id: The điểm's unit id, exactly as stored.
        body: The điểm's own text, verbatim.
    """

    id: str
    body: str


_DEDUCTIBLE_RULE_TEXT_PATTERN = re.compile(r"thuộc loại ([MN])\b")
"""Matches the exact phrase Phụ lục VI's own Mục II khoản 1 uses to name which
deductible type a điểm states the rule for, e.g. "... thuộc loại M quy định tại
khoản 1 Mục I ..." -- read off each candidate row's own stored text, never assumed
from the id scheme (issue #165; see ``RuleUnitRow``)."""

_GET_DEDUCTIBLE_RULE_UNITS_SQL = text(
    "SELECT id, body FROM units "
    "WHERE document_id = :document_id AND appendix_no = :appendix_no "
    "AND unit_type = :unit_type AND body ~ :pattern"
)

_GET_SQL = text(
    "SELECT id, number, title_vi, doc_type, issued, effective, status, "
    "source_url, fetched, path, meta FROM documents WHERE id = :id"
)
_GET_MANY_SQL = text(
    "SELECT id, number, title_vi, doc_type, issued, effective, status, "
    "source_url, fetched, path, meta FROM documents WHERE id = ANY(:ids)"
)
_GET_BY_NUMBERS_SQL = text(
    "SELECT id, number, title_vi, doc_type, issued, effective, status, "
    "source_url, fetched, path, meta FROM documents "
    "WHERE upper(regexp_replace(number, '\\s+', '', 'g')) = ANY(:numbers) "
    "OR (split_part(upper(regexp_replace(number, '\\s+', '', 'g')), '/', 1) || '/' "
    "|| split_part(upper(regexp_replace(number, '\\s+', '', 'g')), '/', 2)) = ANY(:numbers)"
)


def _to_row(row: Any) -> DocumentRow:  # noqa: ANN401 - a driver row, not a typed value
    """Build one ``DocumentRow`` from a ``_GET_SQL``/``_GET_MANY_SQL`` result row."""
    meta = row.meta
    return DocumentRow(
        id=row.id,
        number=row.number,
        title_vi=row.title_vi,
        doc_type=row.doc_type,
        issued=row.issued,
        effective=row.effective,
        status=row.status,
        source_url=row.source_url,
        fetched=row.fetched,
        path=row.path,
        meta=json.loads(meta) if isinstance(meta, str) else (meta or {}),
    )


_UPSERT_SQL = text(
    """
    INSERT INTO documents (id, number, title_vi, doc_type, issued, effective,
                            status, source_url, fetched, path, meta)
    VALUES (:id, :number, :title_vi, :doc_type, :issued, :effective,
            :status, :source_url, :fetched, :path, :meta ::jsonb)
    ON CONFLICT (id) DO UPDATE SET
        number = EXCLUDED.number,
        title_vi = EXCLUDED.title_vi,
        doc_type = EXCLUDED.doc_type,
        issued = EXCLUDED.issued,
        effective = EXCLUDED.effective,
        status = EXCLUDED.status,
        source_url = EXCLUDED.source_url,
        fetched = EXCLUDED.fetched,
        path = EXCLUDED.path,
        meta = EXCLUDED.meta
    """
)

_INSERT_STUB_SQL = text(
    """
    INSERT INTO documents (id, number, title_vi, doc_type, status, path, meta)
    VALUES (:id, :number, :title_vi, :doc_type, :status, NULL, '{}'::jsonb)
    ON CONFLICT (id) DO NOTHING
    """
)
"""``DO NOTHING``, not ``DO UPDATE`` (issue #67): a stub carries no information
beyond its number, which is exactly what its id is derived from, so a repeat
insert has nothing to refresh -- and an existing row is never overwritten with
stub values. ``path`` is ``NULL`` because there is no file."""


class DocumentRepository:
    """Persists ``Document`` rows, upserted by their stable A5.1 id."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository.

        Args:
            session: Session to run queries on. The caller owns its
                transaction (see ``app.core.db.session_scope``).
        """
        self._session = session

    async def upsert(self, document: Document) -> None:
        """Insert or update one document, keyed on its stable id.

        A re-run over the same document replaces its row in place rather
        than duplicating it, which is what makes re-running the ingest CLI
        idempotent.

        Args:
            document: The document to persist.
        """
        await self._session.execute(
            _UPSERT_SQL,
            {
                "id": document.id,
                "number": document.number,
                "title_vi": document.title_vi,
                "doc_type": document.doc_type.value,
                "issued": document.issued,
                "effective": document.effective,
                "status": document.status.value,
                "source_url": document.source_url,
                "fetched": document.fetched,
                "path": document.storage_path,
                "meta": json.dumps(document.meta),
            },
        )

    async def insert_stubs(self, stubs: Sequence[DocumentStub]) -> None:
        """Insert stub documents, leaving any row that already exists untouched.

        Idempotent by construction: a stub's id is derived from its number, so
        re-ingesting the same corpus inserts nothing new. One batched
        statement, since a stub has no foreign key to order around.

        Args:
            stubs: Stubs to persist; may be empty.
        """
        if not stubs:
            return
        await self._session.execute(
            _INSERT_STUB_SQL,
            [
                {
                    "id": stub.id,
                    "number": stub.number,
                    "title_vi": stub.title_vi,
                    "doc_type": stub.doc_type.value,
                    "status": stub.status.value,
                }
                for stub in stubs
            ],
        )

    async def get(self, document_id: str) -> DocumentRow | None:
        """Return one document by id, or ``None`` if it has no row.

        Args:
            document_id: Document id -- a manifest document or a stub alike.

        Returns:
            The row, or ``None``.
        """
        result = await self._session.execute(_GET_SQL, {"id": document_id})
        row = result.first()
        return None if row is None else _to_row(row)

    async def get_many(self, document_ids: Sequence[str]) -> dict[str, DocumentRow]:
        """Return the documents named by ``document_ids``, keyed by id.

        Args:
            document_ids: Document ids to look up; duplicates are harmless.

        Returns:
            ``{id: DocumentRow}`` for each id that names a row; an id that
            does not is simply absent, the same rule ``EntityRepository.get_many``
            follows for ``edges``' own lack of a foreign key.
        """
        if not document_ids:
            return {}
        result = await self._session.execute(_GET_MANY_SQL, {"ids": list(set(document_ids))})
        return {row.id: row for row in (_to_row(raw) for raw in result)}

    async def get_by_numbers(self, numbers: Sequence[str]) -> tuple[DocumentRow, ...]:
        """Return corpus documents matching complete or bare legal document numbers."""
        if not numbers:
            return ()
        result = await self._session.execute(_GET_BY_NUMBERS_SQL, {"numbers": list(set(numbers))})
        return tuple(_to_row(raw) for raw in result)

    async def get_unit_contexts(self, unit_ids: Sequence[str]) -> dict[str, str]:
        """Return the full text each cited id denotes, for grounding.

        "The text a unit id denotes" is its own heading and body, plus every descendant
        unit's body in reading (``ordinal``) order, under its ancestors' headings for
        context -- the same recomposition ``ingest.chunk.build_chunks`` does at embedding
        time. This matters because the corpus is chunked at article level: an article
        whose own body is empty (its real content lives entirely in its clauses) must not
        be handed to the grounding check as a bare heading when the generator itself was
        shown that article's full composed text as its retrieved passage -- the PR review
        measured 146 articles with an empty own body, 108 of them offered to the generator
        as article-level chunks, every one of which the previous version of this method
        silently reduced to a heading with nothing to verify.

        Args:
            unit_ids: Cited unit ids to fetch the full text of.

        Returns:
            ``{unit_id: text}`` for every id that resolved to some text (its own, an
            ancestor's heading, or a descendant's body); an id with none of those is
            simply absent.
        """
        if not unit_ids:
            return {}
        ancestor_ids = {
            ancestor for unit_id in unit_ids for ancestor in unit_and_ancestors(unit_id)
        }
        descendant_patterns = [f"{unit_id}{UNIT_SEPARATOR}%" for unit_id in unit_ids]
        result = await self._session.execute(
            text(
                "SELECT id, heading, body FROM units "
                "WHERE id = ANY(:ids) OR id LIKE ANY(:patterns) "
                "ORDER BY ordinal"
            ),
            {"ids": list(ancestor_ids), "patterns": descendant_patterns},
        )
        rows = list(result)
        by_id = {row.id: row for row in rows}
        contexts: dict[str, str] = {}
        for unit_id in unit_ids:
            parts: list[str] = []
            for ancestor in reversed(unit_and_ancestors(unit_id)):
                row = by_id.get(ancestor)
                if row is None:
                    continue
                if row.heading:
                    parts.append(row.heading)
                if ancestor == unit_id and row.body:
                    parts.append(row.body)
            prefix = f"{unit_id}{UNIT_SEPARATOR}"
            parts.extend(row.body for row in rows if row.id.startswith(prefix) and row.body)
            if parts:
                contexts[unit_id] = "\n".join(parts)
        return contexts

    async def get_focus_unit_contexts(self, unit_ids: Sequence[str]) -> dict[str, FocusUnitRow]:
        """Return each focus unit's own body, parent and ordinal, keyed by id (issue #187).

        Used by the generator to render a passage's focus units as their own lines,
        each showing that unit's own text once rather than the passage's flattened
        ``text`` -- see ``FocusUnitRow``'s own docstring for why this is a different
        shape from ``get_unit_contexts``.

        Args:
            unit_ids: Focus unit ids to look up; duplicates are harmless.

        Returns:
            ``{id: FocusUnitRow}`` for every id that names a row; an id with no row
            (a dangling reference) is simply absent.
        """
        if not unit_ids:
            return {}
        result = await self._session.execute(
            _GET_FOCUS_UNIT_CONTEXTS_SQL, {"ids": list(set(unit_ids))}
        )
        return {
            row.id: FocusUnitRow(
                id=row.id, body=row.body, parent_id=row.parent_id, ordinal=row.ordinal
            )
            for row in result
        }

    async def get_appendix_items(self, unit_ids: Sequence[str]) -> dict[str, AppendixItemRow]:
        """Return the appendix item/sub-item rows among ``unit_ids``, keyed by id (issue #146).

        Used only to build the generator's deterministic, post-grounding verbatim quote: an
        id that is not an appendix item or sub-item -- a form, an article, a clause, a Roman
        prose section, or a lettered point -- is filtered out by the SQL itself, never by a
        caller remembering to check ``unit_type`` afterwards.

        Args:
            unit_ids: Cited unit ids to look up; duplicates are harmless, and an id with no
                matching row (or the wrong ``unit_type``) is simply absent from the result.

        Returns:
            ``{id: AppendixItemRow}`` for every id that is an appendix item or sub-item.
        """
        if not unit_ids:
            return {}
        result = await self._session.execute(
            _GET_APPENDIX_ITEMS_SQL,
            {
                "ids": list(set(unit_ids)),
                "unit_types": [UnitType.APPENDIX_ITEM.value, UnitType.APPENDIX_SUBITEM.value],
            },
        )
        return {
            row.id: AppendixItemRow(
                id=row.id,
                unit_type=row.unit_type,
                appendix_no=row.appendix_no,
                item_no=row.item_no,
                body=row.body,
            )
            for row in result
        }

    async def get_deductible_rule_units(
        self, document_id: str, appendix_no: int
    ) -> dict[str, RuleUnitRow]:
        """Return one appendix's own deductible-type rule units, keyed by letter (issue #165).

        Used by the generator to offer a Phụ lục VI premium-table row's own deductible
        type (its "M"/"N" cell) its rule as a separate citable passage, since the row's
        own text never states it. The candidate ``appendix_section_point`` units of
        this document's named appendix are searched by their own stored text --
        ``RuleUnitRow``'s own docstring explains why this is never computed from the id
        scheme.

        Args:
            document_id: Owning document (the decree whose appendix this is).
            appendix_no: The appendix's own number (``6`` for Phụ lục VI).

        Returns:
            ``{"M": RuleUnitRow, "N": RuleUnitRow}`` for each letter whose rule was
            found in this document's appendix; a letter with no matching điểm is
            simply absent.
        """
        result = await self._session.execute(
            _GET_DEDUCTIBLE_RULE_UNITS_SQL,
            {
                "document_id": document_id,
                "appendix_no": appendix_no,
                "unit_type": UnitType.APPENDIX_SECTION_POINT.value,
                "pattern": "thuộc loại [MN]",
            },
        )
        rule_units: dict[str, RuleUnitRow] = {}
        for row in result:
            match = _DEDUCTIBLE_RULE_TEXT_PATTERN.search(row.body)
            if match is None:
                continue
            rule_units.setdefault(match.group(1), RuleUnitRow(id=row.id, body=row.body))
        return rule_units
