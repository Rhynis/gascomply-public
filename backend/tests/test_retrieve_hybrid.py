"""Tests for hybrid retrieval: graph expansion, entities and supersession (issue #89).

``Retriever.retrieve`` is tested end to end against a live Postgres, with a fixture graph
shaped like the real requirement cluster (``tests/test_traversal_reverse.py``'s own shape) plus
a repealed-document pair (like ``cited-136-2020-nd-cp`` / ``pccc-decree-105-2025`` on the real
corpus). Every vector is a hand-built one-hot vector (as in ``tests/test_retrieve_vector.py``),
so which chunk the vector step finds is controlled exactly and the graph's own contribution can
be isolated: what ``retrieve`` returns beyond the vector hit is provably the graph's doing, not
an artifact of the mock embedder's hashes.

Needs a live Postgres with pgvector (``TEST_DATABASE_URL``). No Ollama: a tiny fixed-vector fake
embedder stands in for real similarity, since the point here is graph wiring, not recall -- recall
is measured on the real corpus with real embeddings in the PR body.

No ``pytest-asyncio`` (C12): each test drives its own ``asyncio.run()``.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.embedders.base import Embedder
from app.repositories.chunks import CHUNKS_EMBEDDING_DIMENSION, ChunkRepository
from app.repositories.documents import DocumentRepository
from app.repositories.edges import EdgeRepository
from app.repositories.entities import EntityRepository
from app.repositories.units import UnitRepository
from app.schemas.chunk import Chunk
from app.schemas.document import DocStatus, DocType, Document, DocumentStub
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from app.schemas.entity import Entity
from app.schemas.retrieval import RetrievalConfig, RetrievalResult, Why
from app.schemas.unit import Unit, UnitType
from app.services.retriever import Retriever

REFERENCES = EdgeType.REFERENCES
APPLIES_TO = EdgeType.APPLIES_TO
BELONGS_TO = EdgeType.BELONGS_TO
REPLACED_BY = EdgeType.REPLACED_BY

QUESTION = "diện tích tối thiểu cửa hàng LPG"


# --- fixture builders (mirroring test_traversal_reverse.py / test_retrieve_vector.py) -----------


def _document(doc_id: str, **overrides: Any) -> Document:
    fields: dict[str, Any] = {
        "id": doc_id,
        "number": "87/2018/NĐ-CP",
        "title_vi": doc_id,
        "doc_type": DocType.DECREE,
        "issued": date(2018, 6, 15),
        "effective": date(2018, 8, 1),
        "status": DocStatus.IN_FORCE,
        "path": f"corpus/{doc_id}.html",
    }
    fields.update(overrides)
    return Document(**fields)


def _article(unit_id: str, document_id: str, dieu_no: int, **overrides: Any) -> Unit:
    fields: dict[str, Any] = {
        "id": unit_id,
        "document_id": document_id,
        "unit_type": UnitType.ARTICLE,
        "dieu_no": dieu_no,
        "body": f"Điều {dieu_no}.",
        "ordinal": dieu_no,
    }
    fields.update(overrides)
    return Unit(**fields)


def _clause(unit_id: str, document_id: str, dieu_no: int, khoan_no: int, **overrides: Any) -> Unit:
    fields: dict[str, Any] = {
        "id": unit_id,
        "document_id": document_id,
        "unit_type": UnitType.CLAUSE,
        "dieu_no": dieu_no,
        "khoan_no": khoan_no,
        "parent_id": f"{document_id}#dieu-{dieu_no}",
        "body": f"Khoản {khoan_no}.",
        "ordinal": khoan_no,
    }
    fields.update(overrides)
    return Unit(**fields)


def _chunk(unit_id: str, document_id: str, **overrides: Any) -> Chunk:
    fields: dict[str, Any] = {
        "unit_id": unit_id,
        "document_id": document_id,
        "effective": date(2018, 8, 1),
        "status": DocStatus.IN_FORCE,
        "body": f"text of {unit_id}",
    }
    fields.update(overrides)
    return Chunk(**fields)


def _edge(src_id: str, dst_id: str, edge_type: EdgeType = REFERENCES, **overrides: Any) -> Edge:
    fields: dict[str, Any] = {
        "src_id": src_id,
        "dst_id": dst_id,
        "edge_type": edge_type,
        "review_status": ReviewStatus.APPROVED,
        "source": EdgeSource.CURATED,
    }
    fields.update(overrides)
    return Edge(**fields)


def _entity(entity_id: str, label_vi: str = "label", **overrides: Any) -> Entity:
    from app.core.ids import EntityType

    fields: dict[str, Any] = {
        "id": entity_id,
        "entity_type": EntityType.REQUIREMENT
        if entity_id.startswith("req:")
        else EntityType.CONCEPT,
        "label_vi": label_vi,
    }
    fields.update(overrides)
    return Entity(**fields)


def _unit_vector(hot_index: int, dimension: int = CHUNKS_EMBEDDING_DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    vector[hot_index] = 1.0
    return vector


class _FixedEmbedder(Embedder):
    """Returns a preset vector for one known question; fails loudly on any other input.

    Stands in for a real embedder so which chunk the vector step finds is chosen by the test,
    not by ``MockEmbedder``'s hash (which has no geometry to control).
    """

    dimension = CHUNKS_EMBEDDING_DIMENSION

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        assert list(texts) == [QUESTION], texts
        return [self._vector]


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


@dataclass
class _Fixture:
    """Everything one ``_seed`` call persists, bundled so ``_seed`` takes one argument, not six."""

    documents: Sequence[Document] = ()
    stubs: Sequence[DocumentStub] = ()
    units: Sequence[Unit] = ()
    chunks: Sequence[tuple[Chunk, list[float]]] = ()
    entities: Sequence[Entity] = ()
    edges: Sequence[Edge] = ()


def _seed(database_url: str, fixture: _Fixture) -> None:
    async def body(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            for document in fixture.documents:
                await DocumentRepository(session).upsert(document)
            await DocumentRepository(session).insert_stubs(fixture.stubs)
            await UnitRepository(session).upsert_many(list(fixture.units))
            await ChunkRepository(session).upsert_many(
                [c for c, _ in fixture.chunks], [v for _, v in fixture.chunks]
            )
            await EntityRepository(session).upsert_many(list(fixture.entities))
            await EdgeRepository(session).upsert_many(list(fixture.edges))

    _run(database_url, body)


def _retrieve(
    database_url: str, vector: list[float], config: RetrievalConfig | None = None, **kwargs: Any
) -> RetrievalResult:
    result_holder: list[RetrievalResult] = []

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            retriever = Retriever(session, _FixedEmbedder(vector), config)
            result_holder.append(await retriever.retrieve(QUESTION, **kwargs))

    _run(database_url, body)
    return result_holder[0]


# --- the requirement-cluster fixture: descent, citation, applies_to expansion --------------------
#
# doc-a#dieu-53 (article, the vector hit, chunk)
#   khoan-1 (clause, no chunk of its own -- inside the article chunk)
#     req:min-area  --REFERENCES-->  khoan-1        --APPLIES_TO-->  concept:store
#   khoan-2 (clause, no chunk of its own)
#     req:distance  --REFERENCES-->  khoan-2
# req:other-biz     --REFERENCES--> doc-b#dieu-1 (a *different* document's own chunk)
#                   --APPLIES_TO-->  concept:store
# doc-a#dieu-53     --REFERENCES--> doc-c#dieu-1 (forward citation, a third document's chunk)
# doc-a#dieu-54     (a sibling article under the same document -- must NOT be reached)


def _cluster_documents() -> list[Document]:
    return [_document("doc-a"), _document("doc-b"), _document("doc-c")]


def _cluster_units() -> list[Unit]:
    return [
        _article("doc-a#dieu-53", "doc-a", 53),
        _clause("doc-a#dieu-53#khoan-1", "doc-a", 53, 1),
        _clause("doc-a#dieu-53#khoan-2", "doc-a", 53, 2),
        # khoan-3: a real unit inside the hit's own chunk that no edge ever points at (issue
        # #106) -- unlike khoan-1/khoan-2, nothing cites it and it carries no BELONGS_TO edge
        # either, so the graph walk alone would never reach it.
        _clause("doc-a#dieu-53#khoan-3", "doc-a", 53, 3),
        _article("doc-a#dieu-54", "doc-a", 54),
        _article("doc-b#dieu-1", "doc-b", 1),
        _article("doc-c#dieu-1", "doc-c", 1),
    ]


def _cluster_chunks() -> list[tuple[Chunk, list[float]]]:
    return [
        (_chunk("doc-a#dieu-53", "doc-a"), _unit_vector(0)),  # the vector hit
        (_chunk("doc-a#dieu-54", "doc-a"), _unit_vector(9)),  # sibling: must not be reached
        (_chunk("doc-b#dieu-1", "doc-b"), _unit_vector(8)),  # low similarity to the question
        (_chunk("doc-c#dieu-1", "doc-c"), _unit_vector(7)),  # low similarity to the question
    ]


def _cluster_entities() -> list[Entity]:
    return [
        _entity("req:min-area", meta={"value": "≥ 12 m²"}),
        _entity("req:distance", meta={"value": "≥ 3 m"}),
        _entity("req:other-biz", meta={"value": "other"}),
        _entity("concept:store", label_vi="Cửa hàng"),
    ]


def _cluster_edges() -> list[Edge]:
    return [
        _edge("doc-a#dieu-53#khoan-1", "doc-a#dieu-53", BELONGS_TO),
        _edge("doc-a#dieu-53#khoan-2", "doc-a#dieu-53", BELONGS_TO),
        # The article -> document BELONGS_TO edges themselves: what a wrong implementation that
        # put BELONGS_TO in both follow_types and reverse_types would use to reach the document
        # from the hit and climb back down to a sibling article (the 83-hit blow-up the issue
        # measured). The retriever must never take the first (forward) half of that path.
        _edge("doc-a#dieu-53", "doc-a", BELONGS_TO),
        _edge("doc-a#dieu-54", "doc-a", BELONGS_TO),
        _edge("req:min-area", "doc-a#dieu-53#khoan-1"),
        _edge("req:distance", "doc-a#dieu-53#khoan-2"),
        _edge("req:min-area", "concept:store", APPLIES_TO),
        _edge("req:distance", "concept:store", APPLIES_TO),
        _edge("req:other-biz", "doc-b#dieu-1"),
        _edge("req:other-biz", "concept:store", APPLIES_TO),
        _edge("doc-a#dieu-53", "doc-c#dieu-1"),
    ]


def _seed_cluster(database_url: str, extra_edges: Sequence[Edge] = ()) -> None:
    _seed(
        database_url,
        _Fixture(
            documents=_cluster_documents(),
            units=_cluster_units(),
            chunks=_cluster_chunks(),
            entities=_cluster_entities(),
            edges=[*_cluster_edges(), *extra_edges],
        ),
    )


# --- the vector hit itself -----------------------------------------------------------------------


def test_the_vector_hit_is_first_and_carries_why_similar(migrated_db: str):
    _seed_cluster(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0))

    assert result.passages[0].unit_id == "doc-a#dieu-53"
    assert result.passages[0].hop == 0
    assert Why.SIMILAR in result.passages[0].why


# --- BELONGS_TO: descend only, never climb to siblings --------------------------------------------


def test_a_sibling_article_under_the_same_document_is_never_reached(migrated_db: str):
    # BELONGS_TO is only ever put in reverse_types by the retriever, never in follow_types too --
    # the 83-hit chapter blow-up the issue measured is what forward BELONGS_TO would cause. k=1
    # so dieu-54 (a distinct one-hot vector, similarity 0) is not itself a vector hit: its only
    # possible route into the result is the (wrong) forward-BELONGS_TO path this test forbids.
    _seed_cluster(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0), config=RetrievalConfig(k=1))

    assert "doc-a#dieu-54" not in {passage.unit_id for passage in result.passages}


def test_a_clause_with_no_chunk_of_its_own_is_returned_as_its_containing_article(
    migrated_db: str,
):
    # khoan-1/khoan-2 have no chunk row; a requirement that cites one must still surface the
    # article chunk that holds it, with the clause named in focus_units.
    _seed_cluster(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0))

    hit = next(p for p in result.passages if p.unit_id == "doc-a#dieu-53")
    assert "doc-a#dieu-53#khoan-1" in hit.focus_units
    assert "doc-a#dieu-53#khoan-2" in hit.focus_units


# --- a clause no edge reaches is still citable (issue #106) --------------------------------------


def test_a_real_clause_no_edge_points_at_is_still_a_focus_unit(migrated_db: str):
    # khoan-3 has a real row in `units`, nested under the hit's own chunk, but is cited by
    # nothing and carries no BELONGS_TO edge -- before issue #106's fix, the graph walk was the
    # only way into `focus_units`, so this clause's text was shown (it is part of the hit's own
    # chunk body) but a correct citation to it would still have been rejected as fabricated.
    _seed_cluster(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0))

    hit = next(p for p in result.passages if p.unit_id == "doc-a#dieu-53")
    assert "doc-a#dieu-53#khoan-3" in hit.focus_units


def test_an_id_shaped_like_a_real_clause_with_no_row_in_units_is_never_a_focus_unit(
    migrated_db: str,
):
    # doc-a#dieu-53#khoan-99 has the exact shape a real clause of this article would have, and
    # would satisfy a prefix/containment check the same way khoan-3 does -- but no such row was
    # ever inserted into `units`. It must never enter focus_units: the anti-fabrication guarantee
    # depends on "real" meaning "the corpus actually has this row", never "matches the pattern".
    _seed_cluster(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0))

    hit = next(p for p in result.passages if p.unit_id == "doc-a#dieu-53")
    assert "doc-a#dieu-53#khoan-99" not in hit.focus_units


# --- REFERENCES: forward from a hit, backward from a hit -----------------------------------------


def test_a_document_the_hit_cites_forward_is_found_and_labelled(migrated_db: str):
    # k=1 so doc-c#dieu-1 (similarity 0 to the question) is not itself a vector hit: it can only
    # appear through the forward REFERENCES walk from the hit.
    _seed_cluster(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0), config=RetrievalConfig(k=1))

    cited = next(p for p in result.passages if p.unit_id == "doc-c#dieu-1")
    assert Why.CITED_BY_HIT in cited.why
    assert cited.hop >= 1


def test_a_requirement_that_cites_a_clause_inside_the_hit_is_returned_as_an_entity(
    migrated_db: str,
):
    _seed_cluster(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0))

    entity_ids = {entity.entity_id for entity in result.entities}
    assert {"req:min-area", "req:distance"} <= entity_ids
    min_area = next(e for e in result.entities if e.entity_id == "req:min-area")
    assert min_area.sources == ("doc-a#dieu-53#khoan-1",)
    assert min_area.meta == {"value": "≥ 12 m²"}


# --- descending into a container the graph reaches with no chunk of its own (issue #136) ---------
#
# The real case: an article cites Phụ lục VII, which has no chunk of its own (ingest never
# chunks the appendix container, only its items), and traversal reaches the appendix but stops
# there -- its items never reach the prompt. These fixtures mirror that shape in miniature.


def _weighted_vector(weight: float, other_index: int = 20) -> list[float]:
    """A vector whose cosine similarity to ``_unit_vector(0)`` increases strictly with ``weight``.

    Two nonzero components only: ``weight`` at index 0, a fixed small amount at ``other_index``
    -- so several descendants can be given a deliberately ordered similarity to the question
    (``_unit_vector(0)``) without any of them reaching the hit's own 1.0.
    """
    vector = [0.0] * CHUNKS_EMBEDDING_DIMENSION
    vector[0] = weight
    vector[other_index] = 0.1
    return vector


def _appendix_unit(unit_id: str, document_id: str, appendix_no: int, **overrides: Any) -> Unit:
    fields: dict[str, Any] = {
        "id": unit_id,
        "document_id": document_id,
        "unit_type": UnitType.APPENDIX,
        "appendix_no": appendix_no,
        "body": f"Phụ lục {appendix_no}.",
        "ordinal": 100 + appendix_no,
    }
    fields.update(overrides)
    return Unit(**fields)


def _appendix_item(
    unit_id: str, document_id: str, appendix_no: int, item_no: int, **overrides: Any
) -> Unit:
    fields: dict[str, Any] = {
        "id": unit_id,
        "document_id": document_id,
        "unit_type": UnitType.APPENDIX_ITEM,
        "appendix_no": appendix_no,
        "item_no": item_no,
        "parent_id": f"{document_id}#phu-luc-{appendix_no}",
        "body": f"Mục {item_no}.",
        "ordinal": 100 + appendix_no + item_no,
    }
    fields.update(overrides)
    return Unit(**fields)


def _seed_appendix_descent(
    database_url: str,
    bodies: tuple[str, str, str, str] = (
        "text of muc-1",
        "text of muc-2",
        "text of muc-3",
        "text of muc-4",
    ),
) -> None:
    """Seed a hit that cites a chunkless appendix with four chunked items.

    ``doc-x#dieu-1#khoan-1`` (inside the hit) --REFERENCES--> ``doc-x#phu-luc-1`` (no chunk of
    its own), which has four items ranked ``muc-1 > muc-2 > muc-3 > muc-4`` by similarity to
    the question. ``bodies`` sets each item's chunk text, in that same order, so a budget test
    can control exactly how many characters each one spends.
    """
    _seed(
        database_url,
        _Fixture(
            documents=[_document("doc-x")],
            units=[
                _article("doc-x#dieu-1", "doc-x", 1),
                _clause("doc-x#dieu-1#khoan-1", "doc-x", 1, 1),
                _appendix_unit("doc-x#phu-luc-1", "doc-x", 1),
                _appendix_item("doc-x#phu-luc-1#muc-1", "doc-x", 1, 1),
                _appendix_item("doc-x#phu-luc-1#muc-2", "doc-x", 1, 2),
                _appendix_item("doc-x#phu-luc-1#muc-3", "doc-x", 1, 3),
                _appendix_item("doc-x#phu-luc-1#muc-4", "doc-x", 1, 4),
            ],
            chunks=[
                (_chunk("doc-x#dieu-1", "doc-x"), _unit_vector(0)),
                (_chunk("doc-x#phu-luc-1#muc-1", "doc-x", body=bodies[0]), _weighted_vector(0.9)),
                (_chunk("doc-x#phu-luc-1#muc-2", "doc-x", body=bodies[1]), _weighted_vector(0.6)),
                (_chunk("doc-x#phu-luc-1#muc-3", "doc-x", body=bodies[2]), _weighted_vector(0.3)),
                (_chunk("doc-x#phu-luc-1#muc-4", "doc-x", body=bodies[3]), _weighted_vector(0.05)),
            ],
            edges=[
                _edge("doc-x#dieu-1#khoan-1", "doc-x#dieu-1", BELONGS_TO),
                _edge("doc-x#dieu-1#khoan-1", "doc-x#phu-luc-1"),
            ],
        ),
    )


def test_a_container_reached_only_through_references_offers_its_chunked_items(migrated_db: str):
    # k=1 so none of the four items is itself a vector hit: the only way any of them can appear
    # is through the REFERENCES walk reaching the appendix and descending into its items. Each
    # item's body is tiny here, as a real appendix item's is (Phụ lục VII averages 116 characters
    # each), so the default 6,000-character budget holds all four -- unlike a fixed top-k, which
    # would have dropped the 4th regardless of how little room it actually needed.
    _seed_appendix_descent(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0), config=RetrievalConfig(k=1))

    offered = {p.unit_id for p in result.passages}
    assert {
        "doc-x#phu-luc-1#muc-1",
        "doc-x#phu-luc-1#muc-2",
        "doc-x#phu-luc-1#muc-3",
        "doc-x#phu-luc-1#muc-4",
    } <= offered
    assert "doc-x#phu-luc-1" not in offered  # the container itself has no chunk to show


def test_a_descended_item_carries_the_containers_hop_and_why(migrated_db: str):
    _seed_appendix_descent(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0), config=RetrievalConfig(k=1))

    item = next(p for p in result.passages if p.unit_id == "doc-x#phu-luc-1#muc-1")
    assert Why.CITED_BY_HIT in item.why
    assert item.hop >= 1


def test_descend_char_budget_cuts_off_the_lower_similarity_items(migrated_db: str):
    # Four 100-character bodies: a budget of 250 fits muc-1 (sim 0.9, spent 100) and muc-2
    # (sim 0.6, spent 200) but not muc-3 (would make 300) -- the cut lands by similarity rank,
    # not by a fixed count, and it lands between ranks exactly where the budget runs out.
    _seed_appendix_descent(migrated_db, bodies=("x" * 100,) * 4)

    result = _retrieve(
        migrated_db, _unit_vector(0), config=RetrievalConfig(k=1, descend_char_budget=250)
    )

    offered = {p.unit_id for p in result.passages}
    assert {"doc-x#phu-luc-1#muc-1", "doc-x#phu-luc-1#muc-2"} <= offered
    assert offered.isdisjoint({"doc-x#phu-luc-1#muc-3", "doc-x#phu-luc-1#muc-4"})


def test_the_best_descendant_is_kept_even_alone_over_the_budget(migrated_db: str):
    # A budget smaller than even the single best item's body -- it is still offered: the budget
    # bounds what else joins it, not whether the container is emptied.
    _seed_appendix_descent(migrated_db, bodies=("x" * 100,) * 4)

    result = _retrieve(
        migrated_db, _unit_vector(0), config=RetrievalConfig(k=1, descend_char_budget=10)
    )

    offered = {p.unit_id for p in result.passages}
    assert "doc-x#phu-luc-1#muc-1" in offered
    assert offered.isdisjoint(
        {"doc-x#phu-luc-1#muc-2", "doc-x#phu-luc-1#muc-3", "doc-x#phu-luc-1#muc-4"}
    )


def test_a_container_with_no_chunked_descendant_contributes_nothing(migrated_db: str):
    # The appendix has no items at all -- `_descend` must find nothing and add nothing, not
    # raise, and the container (still chunkless) must still not appear as a passage.
    _seed(
        migrated_db,
        _Fixture(
            documents=[_document("doc-w")],
            units=[
                _article("doc-w#dieu-1", "doc-w", 1),
                _clause("doc-w#dieu-1#khoan-1", "doc-w", 1, 1),
                _appendix_unit("doc-w#phu-luc-1", "doc-w", 1),
            ],
            chunks=[(_chunk("doc-w#dieu-1", "doc-w"), _unit_vector(0))],
            edges=[
                _edge("doc-w#dieu-1#khoan-1", "doc-w#dieu-1", BELONGS_TO),
                _edge("doc-w#dieu-1#khoan-1", "doc-w#phu-luc-1"),
            ],
        ),
    )

    result = _retrieve(migrated_db, _unit_vector(0), config=RetrievalConfig(k=1))

    assert "doc-w#phu-luc-1" not in {p.unit_id for p in result.passages}


def test_an_appendix_that_is_itself_the_vector_hit_behaves_as_today(migrated_db: str):
    # The appendix has its own chunk and is the vector hit directly -- descent must never run,
    # and its item is offered only as a focus unit (issue #106), exactly as before this issue.
    _seed(
        migrated_db,
        _Fixture(
            documents=[_document("doc-z")],
            units=[
                _appendix_unit("doc-z#phu-luc-1", "doc-z", 1),
                _appendix_item("doc-z#phu-luc-1#muc-1", "doc-z", 1, 1),
            ],
            chunks=[(_chunk("doc-z#phu-luc-1", "doc-z"), _unit_vector(0))],
        ),
    )

    result = _retrieve(migrated_db, _unit_vector(0), config=RetrievalConfig(k=1))

    assert result.passages[0].unit_id == "doc-z#phu-luc-1"
    assert "doc-z#phu-luc-1#muc-1" in result.passages[0].focus_units
    assert "doc-z#phu-luc-1#muc-1" not in {p.unit_id for p in result.passages}


def test_an_appendix_reached_through_references_with_its_own_chunk_behaves_as_today(
    migrated_db: str,
):
    # The appendix has its own chunk (the #134 form-appendix case) but is reached through the
    # graph, not as a vector hit -- `_containing_chunk` finds it directly, so descent never runs.
    _seed(
        migrated_db,
        _Fixture(
            documents=[_document("doc-y")],
            units=[
                _article("doc-y#dieu-1", "doc-y", 1),
                _clause("doc-y#dieu-1#khoan-1", "doc-y", 1, 1),
                _appendix_unit("doc-y#phu-luc-1", "doc-y", 1),
                _appendix_item("doc-y#phu-luc-1#muc-1", "doc-y", 1, 1),
            ],
            chunks=[
                (_chunk("doc-y#dieu-1", "doc-y"), _unit_vector(0)),
                (_chunk("doc-y#phu-luc-1", "doc-y"), _weighted_vector(0.5)),
            ],
            edges=[
                _edge("doc-y#dieu-1#khoan-1", "doc-y#dieu-1", BELONGS_TO),
                _edge("doc-y#dieu-1#khoan-1", "doc-y#phu-luc-1"),
            ],
        ),
    )

    result = _retrieve(migrated_db, _unit_vector(0), config=RetrievalConfig(k=1))

    appendix = next(p for p in result.passages if p.unit_id == "doc-y#phu-luc-1")
    assert Why.CITED_BY_HIT in appendix.why
    assert "doc-y#phu-luc-1#muc-1" in appendix.focus_units
    assert "doc-y#phu-luc-1#muc-1" not in {p.unit_id for p in result.passages}


# --- APPLIES_TO: both ways, reaching the whole obligation set of the concept ----------------------


def test_the_concept_and_a_sibling_requirement_are_reached_through_applies_to(migrated_db: str):
    # req:other-biz's own clause (doc-b#dieu-1) has low similarity to the question and is not the
    # vector hit -- reaching it at all is only possible via concept:store, forward then backward.
    _seed_cluster(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0))

    entity_ids = {entity.entity_id for entity in result.entities}
    assert "concept:store" in entity_ids
    assert "req:other-biz" in entity_ids


def test_hybrid_finds_a_unit_that_pure_vector_search_at_the_same_k_misses(migrated_db: str):
    # The claim the whole GraphRAG framing rests on, isolated in a controlled fixture: doc-b#dieu-1
    # is req:other-biz's legal basis, reachable only through the graph (concept:store, two hops),
    # and its similarity to the question (hot index 8) is low enough that plain top-k (k=1) never
    # returns it on its own.
    _seed_cluster(migrated_db)
    config = RetrievalConfig(k=1)

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            retriever = Retriever(session, _FixedEmbedder(_unit_vector(0)), config)
            vector_only = {m.unit_id for m in await retriever.vector_search(QUESTION)}
            hybrid = await retriever.retrieve(QUESTION)
        hybrid_ids = {p.unit_id for p in hybrid.passages}
        assert "doc-b#dieu-1" not in vector_only
        assert "doc-b#dieu-1" in hybrid_ids

    _run(migrated_db, body)


# --- approved edges only, at every step -----------------------------------------------------


def test_a_pending_references_edge_is_not_followed(migrated_db: str):
    _seed_cluster(
        migrated_db,
        extra_edges=[
            _edge(
                "req:pending",
                "doc-a#dieu-53#khoan-1",
                review_status=ReviewStatus.PENDING,
            )
        ],
    )
    _seed(migrated_db, _Fixture(entities=[_entity("req:pending")]))

    result = _retrieve(migrated_db, _unit_vector(0))

    assert "req:pending" not in {e.entity_id for e in result.entities}


def test_a_pending_applies_to_edge_is_not_followed(migrated_db: str):
    _seed_cluster(
        migrated_db,
        extra_edges=[
            _edge("req:min-area", "concept:pending", APPLIES_TO, review_status=ReviewStatus.PENDING)
        ],
    )
    # Seeded as a real row: if the pending edge were wrongly walked, this would actually show up
    # in the result, rather than the assertion passing vacuously because the id names nothing.
    _seed(migrated_db, _Fixture(entities=[_entity("concept:pending")]))

    result = _retrieve(migrated_db, _unit_vector(0))

    assert "concept:pending" not in {e.entity_id for e in result.entities}


def test_a_rejected_references_edge_is_not_followed(migrated_db: str):
    _seed_cluster(
        migrated_db,
        extra_edges=[
            _edge(
                "req:rejected",
                "doc-a#dieu-53#khoan-2",
                review_status=ReviewStatus.REJECTED,
            )
        ],
    )
    _seed(migrated_db, _Fixture(entities=[_entity("req:rejected")]))

    result = _retrieve(migrated_db, _unit_vector(0))

    assert "req:rejected" not in {e.entity_id for e in result.entities}


# --- supersession: the real cited-136-2020-nd-cp / pccc-decree-105-2025 pair, in miniature --------


def _supersession_stub() -> DocumentStub:
    return DocumentStub(number="999/2020/NĐ-CP")


def _seed_supersession(database_url: str) -> DocumentStub:
    """Seed a citing clause (inside a vector hit) -> stub --REPLACED_BY--> an in-force document."""
    stub = _supersession_stub()
    _seed(
        database_url,
        _Fixture(
            documents=[_document("doc-a"), _document("doc-new")],
            stubs=[stub],
            units=[
                _article("doc-a#dieu-1", "doc-a", 1),
                _clause("doc-a#dieu-1#khoan-1", "doc-a", 1, 1),
                _article("doc-new#dieu-1", "doc-new", 1),
            ],
            chunks=[
                (_chunk("doc-a#dieu-1", "doc-a"), _unit_vector(0)),
                (_chunk("doc-new#dieu-1", "doc-new"), _unit_vector(5)),
            ],
            edges=[
                _edge("doc-a#dieu-1#khoan-1", "doc-a#dieu-1", BELONGS_TO),
                _edge("doc-a#dieu-1#khoan-1", stub.id, origin_id="doc-a#dieu-1#khoan-1"),
                _edge(
                    stub.id,
                    "doc-new",
                    REPLACED_BY,
                    origin_id="doc-a#dieu-1#khoan-1",
                    evidence="Nghị định này thay thế Nghị định số 999/2020/NĐ-CP.",
                ),
            ],
        ),
    )
    return stub


def test_a_citation_of_a_retired_stub_is_reported_with_its_in_force_replacement(
    migrated_db: str,
):
    stub = _seed_supersession(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0))

    assert len(result.supersessions) == 1
    notice = result.supersessions[0]
    assert notice.retired_document == stub.id
    assert notice.replaced_by == ("doc-new",)
    assert notice.declared_in == "doc-a#dieu-1#khoan-1"
    assert notice.cited_by == ("doc-a#dieu-1#khoan-1",)
    assert "999/2020" in (notice.evidence or "")


def test_the_in_force_replacement_is_fetched_as_a_successor_passage(migrated_db: str):
    _seed_supersession(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0))

    successor = next(p for p in result.passages if p.unit_id == "doc-new#dieu-1")
    assert Why.SUCCESSOR in successor.why
    assert successor.replaces == (_supersession_stub().id,)


def test_the_stub_itself_never_appears_as_a_passage(migrated_db: str):
    stub = _seed_supersession(migrated_db)

    result = _retrieve(migrated_db, _unit_vector(0))

    assert stub.id not in {p.unit_id for p in result.passages}


def test_a_stub_merely_cited_with_no_replacement_is_not_reported_as_retired(migrated_db: str):
    # Most of the real corpus's stubs (63 of them) are cited but never named by a REPLACED_BY
    # edge -- a stub's own status ('unknown') must not, by itself, trigger a supersession notice.
    stub = DocumentStub(number="1/2000/NĐ-CP")
    _seed(
        migrated_db,
        _Fixture(
            documents=[_document("doc-a")],
            stubs=[stub],
            units=[_article("doc-a#dieu-1", "doc-a", 1)],
            chunks=[(_chunk("doc-a#dieu-1", "doc-a"), _unit_vector(0))],
            edges=[_edge("doc-a#dieu-1", stub.id)],
        ),
    )

    result = _retrieve(migrated_db, _unit_vector(0))

    assert result.supersessions == ()


def test_an_expired_document_with_no_replaced_by_edge_is_still_reported_as_retired(
    migrated_db: str,
):
    # The other branch of the retired-citation query: `status = 'expired'`, no REPLACED_BY at all.
    _seed(
        migrated_db,
        _Fixture(
            documents=[_document("doc-a"), _document("doc-old", status=DocStatus.EXPIRED)],
            units=[
                _article("doc-a#dieu-1", "doc-a", 1),
                _article("doc-old#dieu-1", "doc-old", 1),
            ],
            chunks=[(_chunk("doc-a#dieu-1", "doc-a"), _unit_vector(0))],
            edges=[_edge("doc-a#dieu-1", "doc-old")],
        ),
    )

    result = _retrieve(migrated_db, _unit_vector(0))

    assert len(result.supersessions) == 1
    assert result.supersessions[0].retired_document == "doc-old"
    assert result.supersessions[0].replaced_by == ()


def test_include_history_keeps_a_superseded_passage_and_flags_it(migrated_db: str):
    _seed(
        migrated_db,
        _Fixture(
            documents=[_document("doc-old", status=DocStatus.EXPIRED)],
            units=[_article("doc-old#dieu-1", "doc-old", 1)],
            chunks=[
                (_chunk("doc-old#dieu-1", "doc-old", status=DocStatus.EXPIRED), _unit_vector(0))
            ],
        ),
    )

    without_history = _retrieve(migrated_db, _unit_vector(0))
    with_history = _retrieve(migrated_db, _unit_vector(0), include_history=True)

    assert without_history.passages == ()
    [passage] = with_history.passages
    assert passage.unit_id == "doc-old#dieu-1"
    assert passage.superseded is True


# --- empty corpus / no hits -----------------------------------------------------------------------


def test_retrieve_on_an_empty_corpus_returns_an_empty_result(migrated_db: str):
    result = _retrieve(migrated_db, _unit_vector(0))

    assert result == RetrievalResult(question=QUESTION, passages=())


def test_retrieve_rejects_a_blank_question(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            retriever = Retriever(session, _FixedEmbedder(_unit_vector(0)))
            try:
                await retriever.retrieve("")
            except ValueError as error:
                assert "blank" in str(error)
                return
        raise AssertionError("expected ValueError")

    _run(migrated_db, body)
