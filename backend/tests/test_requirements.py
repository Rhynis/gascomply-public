"""Tests for the requirement cluster (build plan A5.3, A5.6, A6 Phase 2.4).

Covers entities, seed data, and the graph pass.

DB-backed tests need a live Postgres: set ``TEST_DATABASE_URL``. Every test that
needs one depends, directly or transitively, on the ``migrated_db`` fixture,
so a missing ``TEST_DATABASE_URL`` skips them with one clear reason rather than a
silent pass. No ``pytest-asyncio`` (C12): each DB-backed test drives its own
async body with a single ``asyncio.run()`` call, matching
``tests/test_repositories.py``.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.core.ids import EntityType, entity_id
from app.repositories.entities import EntityRepository
from app.schemas.edge import EdgeSource, EdgeType, ReviewStatus
from app.schemas.entity import Entity
from ingest.graph import (
    ConceptSeed,
    RequirementCluster,
    RequirementSeed,
    build_requirement_graph,
    load_requirement_cluster,
)
from ingest.run import REQUIREMENTS_PATH

# --- Entity schema -----------------------------------------------------------------


def test_entity_id_must_match_its_own_entity_type():
    with pytest.raises(ValueError, match="must start with"):
        Entity(id="form:x", entity_type=EntityType.REQUIREMENT, label_vi="X")


def test_entity_id_must_be_a_real_slug():
    # Raised by app.core.ids.entity_id itself, reused rather than
    # duplicated -- Entity's own validator only adds the prefix check above.
    with pytest.raises(ValueError, match="Invalid entity slug"):
        Entity(id="req:Not_A_Slug", entity_type=EntityType.REQUIREMENT, label_vi="X")


def test_a_well_formed_entity_round_trips():
    entity = Entity(
        id=entity_id(EntityType.REQUIREMENT, "min-store-area"),
        entity_type=EntityType.REQUIREMENT,
        label_vi="Diện tích tối thiểu",
        meta={"value": "≥ 12 m²"},
    )
    assert entity.id == "req:min-store-area"
    assert entity.meta == {"value": "≥ 12 m²"}


def test_meta_defaults_to_empty():
    entity = Entity(id="concept:x", entity_type=EntityType.CONCEPT, label_vi="X")
    assert entity.meta == {}


# --- load_requirement_cluster --------------------------------------------------------


def test_the_real_seed_file_loads_and_has_eight_requirements():
    # A schema-drift guard on the actual committed data/requirements.json,
    # not just the loader logic -- a domain expert editing this file by
    # hand should have something catch a malformed edit immediately.
    cluster = load_requirement_cluster(REQUIREMENTS_PATH)
    assert cluster.concept.slug == "lpg-retail-store"
    assert len(cluster.requirements) == 8
    slugs = {requirement.slug for requirement in cluster.requirements}
    assert slugs == {
        "min-store-area",
        "ignition-distance",
        "fire-resistance",
        "electrical",
        "display-racks",
        "supply-contract",
        "licensing-time",
        "cylinder-inspection",
    }


def test_load_requirement_cluster_parses_a_minimal_file(tmp_path: Path):
    path = tmp_path / "requirements.json"
    path.write_text(
        json.dumps(
            {
                "concept": {"slug": "x", "label_vi": "X"},
                "requirements": [
                    {
                        "slug": "y",
                        "label_vi": "Y",
                        "value": "Z",
                        "source_unit_id": "doc#dieu-1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cluster = load_requirement_cluster(path)
    assert cluster.concept == ConceptSeed(slug="x", label_vi="X")
    assert cluster.requirements == (
        RequirementSeed(slug="y", label_vi="Y", value="Z", source_unit_id="doc#dieu-1"),
    )


# --- build_requirement_graph ---------------------------------------------------------


def _cluster(*requirements: RequirementSeed) -> RequirementCluster:
    return RequirementCluster(
        concept=ConceptSeed(slug="lpg-retail-store", label_vi="Cửa hàng bán lẻ LPG chai"),
        requirements=requirements,
    )


def test_build_requirement_graph_fails_the_ingest_when_a_unit_does_not_exist():
    # "A citation to nothing must fail the ingest rather than writing it."
    cluster = _cluster(
        RequirementSeed(slug="x", label_vi="X", value="V", source_unit_id="ghost#dieu-1")
    )
    with pytest.raises(ValueError, match="not a real unit"):
        build_requirement_graph(cluster, unit_ids=set())


def test_build_requirement_graph_builds_one_concept_and_one_requirement_entity():
    cluster = _cluster(
        RequirementSeed(slug="x", label_vi="X label", value="≥ 1", source_unit_id="doc#dieu-1")
    )
    entities, _edges = build_requirement_graph(cluster, unit_ids={"doc#dieu-1"})
    by_id = {entity.id: entity for entity in entities}
    assert by_id["concept:lpg-retail-store"].entity_type is EntityType.CONCEPT
    assert by_id["req:x"].entity_type is EntityType.REQUIREMENT
    assert by_id["req:x"].label_vi == "X label"
    assert by_id["req:x"].meta == {"value": "≥ 1"}


def test_applies_to_direction_is_requirement_to_concept():
    # Pinned the way #30 pinned BELONGS_TO's direction and #32 pinned both
    # REPLACED_BY voices -- getting this backwards would silently invert
    # "what a store must satisfy" into "what a store's own concept requires
    # of a requirement", with nothing else to catch it.
    cluster = _cluster(
        RequirementSeed(slug="x", label_vi="X", value="V", source_unit_id="doc#dieu-1")
    )
    _entities, edges = build_requirement_graph(cluster, unit_ids={"doc#dieu-1"})
    applies_to = next(edge for edge in edges if edge.edge_type is EdgeType.APPLIES_TO)
    assert applies_to.src_id == "req:x"
    assert applies_to.dst_id == "concept:lpg-retail-store"


def test_requirement_references_its_source_unit():
    cluster = _cluster(
        RequirementSeed(slug="x", label_vi="X", value="V", source_unit_id="doc#dieu-1")
    )
    _entities, edges = build_requirement_graph(cluster, unit_ids={"doc#dieu-1"})
    reference = next(edge for edge in edges if edge.edge_type is EdgeType.REFERENCES)
    assert reference.src_id == "req:x"
    assert reference.dst_id == "doc#dieu-1"


def test_every_curated_edge_is_pending_review_and_sourced_as_curated():
    # The issue's own decision: curated data still lands PENDING, like
    # REGEX/LLM -- one uniform gate for every non-structural edge, not a
    # silent exception because a human wrote the file.
    cluster = _cluster(
        RequirementSeed(slug="x", label_vi="X", value="V", source_unit_id="doc#dieu-1")
    )
    _entities, edges = build_requirement_graph(cluster, unit_ids={"doc#dieu-1"})
    for edge in edges:
        assert edge.review_status is ReviewStatus.PENDING
        assert edge.source is EdgeSource.CURATED


def test_three_requirements_sharing_one_clause_is_many_to_one():
    # The exact shape the real data hits: #dieu-53#khoan-2 states the
    # ignition distance, the fire rating and the electrical rule in three
    # consecutive sentences -- any design assuming one requirement per
    # clause is wrong, and this is the case that exposes it.
    shared_unit = "gas-decree-87-2018#dieu-53#khoan-2"
    cluster = _cluster(
        RequirementSeed(slug="a", label_vi="A", value="1", source_unit_id=shared_unit),
        RequirementSeed(slug="b", label_vi="B", value="2", source_unit_id=shared_unit),
        RequirementSeed(slug="c", label_vi="C", value="3", source_unit_id=shared_unit),
    )
    entities, edges = build_requirement_graph(cluster, unit_ids={shared_unit})
    assert len({entity.id for entity in entities}) == 4  # 1 concept + 3 requirements
    reference_targets = [edge.dst_id for edge in edges if edge.edge_type is EdgeType.REFERENCES]
    assert reference_targets == [shared_unit, shared_unit, shared_unit]


def test_target_document_ids_extracts_the_owning_document_of_each_source_unit():
    cluster = _cluster(
        RequirementSeed(slug="a", label_vi="A", value="1", source_unit_id="doc-1#dieu-1"),
        RequirementSeed(slug="b", label_vi="B", value="2", source_unit_id="doc-2#dieu-1#khoan-1"),
    )
    assert cluster.target_document_ids() == {"doc-1", "doc-2"}


# --- EntityRepository (DB-backed) ---------------------------------------------------


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    """Create an engine, run ``body(engine)`` to completion, then dispose it."""

    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


def test_entity_upsert_many_replaces_rather_than_duplicates(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        entity = Entity(id="concept:x", entity_type=EntityType.CONCEPT, label_vi="First label")
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await EntityRepository(session).upsert_many([entity])
        async with session_scope(session_factory) as session:
            await EntityRepository(session).upsert_many(
                [Entity(id="concept:x", entity_type=EntityType.CONCEPT, label_vi="Updated label")]
            )
        async with engine.begin() as conn:
            rows = (
                await conn.execute(text("SELECT label_vi FROM entities WHERE id = 'concept:x'"))
            ).all()
        assert len(rows) == 1
        assert rows[0].label_vi == "Updated label"

    _run(migrated_db, body)


def test_entity_meta_round_trips_as_jsonb(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        entity = Entity(
            id="req:x",
            entity_type=EntityType.REQUIREMENT,
            label_vi="X",
            meta={"value": "≥ 12 m²"},
        )
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await EntityRepository(session).upsert_many([entity])
        async with engine.begin() as conn:
            row = (await conn.execute(text("SELECT meta FROM entities WHERE id = 'req:x'"))).one()
        assert row.meta == {"value": "≥ 12 m²"}

    _run(migrated_db, body)


def test_entity_upsert_many_with_no_entities_is_a_no_op(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            await EntityRepository(session).upsert_many([])
        async with engine.begin() as conn:
            count = (await conn.execute(text("SELECT count(*) FROM entities"))).scalar_one()
        assert count == 0

    _run(migrated_db, body)
