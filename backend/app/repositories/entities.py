"""Async repository for the ``entities`` table (build plan A5.3, A6 Phase 2.4).

Persists and reads back the existing ``Entity`` model verbatim -- no second, ORM-shaped
domain object. Every query is parameterized (C3).
"""

import json
import logging
from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.entity import Entity

logger = logging.getLogger(__name__)

_UPSERT_SQL = text(
    """
    INSERT INTO entities (id, entity_type, label_vi, meta)
    VALUES (:id, :entity_type, :label_vi, :meta ::jsonb)
    ON CONFLICT (id) DO UPDATE SET
        entity_type = EXCLUDED.entity_type,
        label_vi = EXCLUDED.label_vi,
        meta = EXCLUDED.meta
    """
)

_GET_MANY_SQL = text("SELECT id, entity_type, label_vi, meta FROM entities WHERE id = ANY(:ids)")


class EntityRepository:
    """Persists ``Entity`` rows, upserted by their stable A5.1 id."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository.

        Args:
            session: Session to run queries on. The caller owns its
                transaction (see ``app.core.db.session_scope``).
        """
        self._session = session

    async def upsert_many(self, entities: Sequence[Entity]) -> None:
        """Insert or update entities, keyed on their stable id.

        A re-run over the same seed data replaces each row in place rather
        than duplicating it -- what makes re-running the ingest CLI
        idempotent (build plan A6 issue #51).

        Args:
            entities: Entities to persist.
        """
        if not entities:
            return
        await self._session.execute(
            _UPSERT_SQL,
            [
                {
                    "id": entity.id,
                    "entity_type": entity.entity_type.value,
                    "label_vi": entity.label_vi,
                    "meta": json.dumps(entity.meta),
                }
                for entity in entities
            ],
        )

    async def get_many(self, entity_ids: Sequence[str]) -> list[Entity]:
        """Return the entities with these ids, in no particular order.

        An id that names no row is simply absent: ``edges`` has no foreign key to ``entities``,
        so a graph walk can reach an id that was never inserted, and that is for the caller to
        notice by comparing what it asked for with what came back.

        Args:
            entity_ids: Entity ids (``req:...``, ``concept:...``).

        Returns:
            The entities found.
        """
        if not entity_ids:
            return []
        result = await self._session.execute(_GET_MANY_SQL, {"ids": list(entity_ids)})
        return [
            Entity(
                id=row.id,
                entity_type=row.entity_type,
                label_vi=row.label_vi,
                meta=json.loads(row.meta) if isinstance(row.meta, str) else (row.meta or {}),
            )
            for row in result
        ]
