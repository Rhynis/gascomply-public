"""Typed graph entities, mirroring the ``entities`` table (build plan A5.3, A6 Phase 2.4).

An entity is a non-document, non-unit node of the graph -- a requirement, a
form, an authority, a concept, or a fee (``EntityType``). This phase (2.4)
fills only ``Requirement`` and one ``Concept`` node; ``Form``/``Authority``/
``Fee`` are reserved for whichever later phase has seed data for them (A5.6
covers requirements only).

Mirrors ``app/schemas/unit.py``'s own invariant: ``id`` must equal the
canonical identifier derived from the entity's own type and slug
(``app.core.ids.entity_id``), so there is exactly one way to spell an entity
and no caller can build one that would break a downstream join.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.ids import ENTITY_SEPARATOR, EntityType, entity_id


class Entity(BaseModel):
    """One node of the requirement/form/authority/concept/fee graph.

    Attributes:
        id: Stable identifier per A5.1, e.g. ``req:min-store-area``.
        entity_type: Node type; must match ``id``'s own prefix.
        label_vi: Vietnamese-language label shown to a user.
        meta: Structured data about the entity, e.g. ``{"value": "≥ 12 m²"}``
            for a requirement (A5.3's own example). Empty for a node with
            nothing further to record, such as the one ``Concept``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    entity_type: EntityType
    label_vi: str = Field(min_length=1)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_id(self) -> "Entity":
        """Pin ``id`` to the locked identifier scheme.

        Returns:
            The validated entity.

        Raises:
            ValueError: If ``id`` does not round-trip through
                ``entity_id(entity_type, slug)`` for its own ``entity_type``.
        """
        prefix = f"{self.entity_type.value}{ENTITY_SEPARATOR}"
        if not self.id.startswith(prefix):
            msg = f"id must start with {prefix!r} for entity_type {self.entity_type.value}"
            raise ValueError(msg)
        slug = self.id[len(prefix) :]
        if entity_id(self.entity_type, slug) != self.id:
            msg = f"id {self.id!r} is not a valid {self.entity_type.value} identifier"
            raise ValueError(msg)
        return self
