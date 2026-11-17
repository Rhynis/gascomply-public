"""Tests for ``GET /doc/{id}`` (build plan A6 Phase 3.7, issue #90)."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import create_engine, create_session_factory, session_scope
from app.repositories.documents import DocumentRepository
from app.repositories.units import UnitRepository
from app.schemas.document import DocStatus, DocType, Document, DocumentStub
from app.schemas.unit import Unit, UnitType

DOC_ID = "gas-decree-87-2018"


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


def _seed_document_with_units(database_url: str) -> None:
    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            await DocumentRepository(session).upsert(
                Document(
                    id=DOC_ID,
                    number="87/2018/NĐ-CP",
                    title_vi="Nghị định về kinh doanh khí",
                    doc_type=DocType.DECREE,
                    status=DocStatus.IN_FORCE,
                    effective=date(2018, 8, 1),
                    path=f"corpus/{DOC_ID}.html",
                )
            )
            await UnitRepository(session).upsert_many(
                [
                    Unit(
                        id=f"{DOC_ID}#dieu-1",
                        document_id=DOC_ID,
                        unit_type=UnitType.ARTICLE,
                        dieu_no=1,
                        heading="Phạm vi điều chỉnh",
                        body="Nghị định này quy định về kinh doanh khí.",
                        ordinal=0,
                    ),
                    Unit(
                        id=f"{DOC_ID}#dieu-2",
                        document_id=DOC_ID,
                        unit_type=UnitType.ARTICLE,
                        dieu_no=2,
                        body="Điều 2.",
                        ordinal=1,
                    ),
                ]
            )

    _run(database_url, body)


def _seed_stub(database_url: str) -> str:
    stub = DocumentStub(number="136/2020/NĐ-CP")

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            await DocumentRepository(session).insert_stubs([stub])

    _run(database_url, body)
    return stub.id


def test_get_document_returns_the_document_with_its_units_in_reading_order(
    db_client: TestClient, migrated_db: str
):
    _seed_document_with_units(migrated_db)

    response = db_client.get(f"/doc/{DOC_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == DOC_ID
    assert body["number"] == "87/2018/NĐ-CP"
    assert body["status"] == "in_force"
    assert body["effective"] == "2018-08-01"
    assert [unit["id"] for unit in body["units"]] == [f"{DOC_ID}#dieu-1", f"{DOC_ID}#dieu-2"]
    assert body["units"][0]["heading"] == "Phạm vi điều chỉnh"


def test_get_document_404s_for_an_unknown_id(db_client: TestClient, migrated_db: str):
    response = db_client.get("/doc/no-such-document")

    assert response.status_code == 404


def test_get_document_serves_a_stub_with_no_units(db_client: TestClient, migrated_db: str):
    stub_id = _seed_stub(migrated_db)

    response = db_client.get(f"/doc/{stub_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unknown"
    assert body["doc_type"] == "unknown"
    assert body["units"] == []


def test_get_document_503s_with_no_database_configured(client: TestClient):
    response = client.get(f"/doc/{DOC_ID}")

    assert response.status_code == 503


def test_the_61st_request_in_a_minute_is_rejected(db_client: TestClient, migrated_db: str):
    for _ in range(60):
        response = db_client.get("/doc/no-such-document")
        assert response.status_code == 404

    response = db_client.get("/doc/no-such-document")
    assert response.status_code == 429


def test_ask_and_doc_rate_limits_are_independent(db_client: TestClient, migrated_db: str):
    for _ in range(30):
        assert db_client.post("/ask", json={"question": "cháy nhà"}).status_code == 200
    assert db_client.post("/ask", json={"question": "cháy nhà"}).status_code == 429

    # /doc's own limit (60) is untouched by /ask's 30 -- the two routers each read their
    # own RateLimiter off app.state, never sharing state or a global counter.
    assert db_client.get("/doc/no-such-document").status_code == 404
