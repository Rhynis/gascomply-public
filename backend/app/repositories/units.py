"""Async repository for the ``units`` table (build plan A5.3, A6 Phase 1.5b).

Persists the existing ``Unit`` model verbatim -- no second, ORM-shaped domain
object. Every query is parameterized (C3).
"""

import logging
from collections import defaultdict
from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.unit import Unit

logger = logging.getLogger(__name__)

_UPSERT_SQL = text(
    """
    INSERT INTO units (id, document_id, unit_type, chapter_no, dieu_no, khoan_no,
                        diem, appendix_no, appendix_section, item_no, parent_id,
                        heading, body, ordinal)
    VALUES (:id, :document_id, :unit_type, :chapter_no, :dieu_no, :khoan_no,
            :diem, :appendix_no, :appendix_section, :item_no, :parent_id,
            :heading, :body, :ordinal)
    ON CONFLICT (id) DO UPDATE SET
        document_id = EXCLUDED.document_id,
        unit_type = EXCLUDED.unit_type,
        chapter_no = EXCLUDED.chapter_no,
        dieu_no = EXCLUDED.dieu_no,
        khoan_no = EXCLUDED.khoan_no,
        diem = EXCLUDED.diem,
        appendix_no = EXCLUDED.appendix_no,
        appendix_section = EXCLUDED.appendix_section,
        item_no = EXCLUDED.item_no,
        parent_id = EXCLUDED.parent_id,
        heading = EXCLUDED.heading,
        body = EXCLUDED.body,
        ordinal = EXCLUDED.ordinal
    """
)

_LIST_BY_DOCUMENT_SQL = text(
    "SELECT id, document_id, unit_type, chapter_no, dieu_no, khoan_no, diem, appendix_no, "
    "appendix_section, item_no, parent_id, heading, body, ordinal FROM units "
    "WHERE document_id = :document_id ORDER BY ordinal"
)

_LIST_IDS_BY_DOCUMENTS_SQL = text(
    "SELECT id, document_id FROM units WHERE document_id = ANY(:document_ids)"
)


class UnitRepository:
    """Persists ``Unit`` rows, upserted by their stable A5.1 id."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository.

        Args:
            session: Session to run queries on. The caller owns its
                transaction (see ``app.core.db.session_scope``).
        """
        self._session = session

    async def upsert_many(self, units: Sequence[Unit]) -> None:
        """Insert or update units, keyed on their stable ids.

        Units are upserted one at a time, in the given order, rather than as
        one batched statement: ``units.parent_id`` references ``units(id)``,
        so a child unit's row must not be written before its parent's. The
        parser's own reading-order guarantee (a chapter or article always
        precedes its descendants) already gives that order for free.

        Args:
            units: Units to persist, in reading order.
        """
        for unit in units:
            await self._session.execute(
                _UPSERT_SQL,
                {
                    "id": unit.id,
                    "document_id": unit.document_id,
                    "unit_type": unit.unit_type.value,
                    "chapter_no": unit.chapter_no,
                    "dieu_no": unit.dieu_no,
                    "khoan_no": unit.khoan_no,
                    "diem": unit.diem,
                    "appendix_no": unit.appendix_no,
                    "appendix_section": unit.appendix_section,
                    "item_no": unit.item_no,
                    "parent_id": unit.parent_id,
                    "heading": unit.heading,
                    "body": unit.body,
                    "ordinal": unit.ordinal,
                },
            )

    async def list_by_document(self, document_id: str) -> list[Unit]:
        """Return every unit of a document, in reading order (issue #90).

        Args:
            document_id: Owning document id.

        Returns:
            Units ordered as the parser produced them; empty for a document
            with no units (a metadata-only document, or a stub).
        """
        result = await self._session.execute(_LIST_BY_DOCUMENT_SQL, {"document_id": document_id})
        return [Unit(**row._mapping) for row in result]

    async def existing_ids(self, document_id: str) -> tuple[str, ...]:
        """Return every unit id currently stored for this document.

        Read-only, and run *before* ``prune_document`` mutates anything --
        issue #117's review (point 2) needs this document's prior unit count
        to decide whether a parse that would delete most of them should be
        refused, which only holds if nothing has been deleted yet when this
        is called.
        """
        result = await self._session.execute(
            text("SELECT id FROM units WHERE document_id = :document_id"),
            {"document_id": document_id},
        )
        return tuple(row.id for row in result)

    async def prune_document(self, document_id: str, keep_ids: Sequence[str]) -> tuple[str, ...]:
        """Delete units no longer produced by a document parse, deepest first."""
        result = await self._session.execute(
            text("SELECT id FROM units WHERE document_id = :document_id"),
            {"document_id": document_id},
        )
        stale = [row.id for row in result if row.id not in set(keep_ids)]
        for unit_id in sorted(stale, key=len, reverse=True):
            await self._session.execute(text("DELETE FROM units WHERE id = :id"), {"id": unit_id})
        return tuple(stale)

    async def list_ids_by_documents(
        self, document_ids: Sequence[str]
    ) -> dict[str, tuple[str, ...]]:
        """Return every real unit id of each document in ``document_ids``, grouped by document.

        Bodies are not fetched -- callers of this method only need to know which ids the
        corpus actually has (issue #106: a citation to an id that merely *looks* nested under
        a retrieved chunk, e.g. a clause number past the end of the article, must still be
        rejected, which only holds if "real" means "has a row here", never "matches a shape").

        Args:
            document_ids: Owning document ids to fetch units for.

        Returns:
            Each document id mapped to the ids of its units, in no particular order. A document
            with no units, or not present in ``document_ids``, is simply absent from the result
            rather than mapped to an empty tuple.
        """
        if not document_ids:
            return {}
        result = await self._session.execute(
            _LIST_IDS_BY_DOCUMENTS_SQL, {"document_ids": list(document_ids)}
        )
        grouped: dict[str, list[str]] = defaultdict(list)
        for row in result:
            grouped[row.document_id].append(row.id)
        return {document_id: tuple(ids) for document_id, ids in grouped.items()}
