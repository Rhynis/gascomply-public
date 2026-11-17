"""Tests for the edge review CLI and its repository queries (build plan A6 Phase 2.3a).

Needs a live Postgres with pgvector: set ``TEST_DATABASE_URL`` (``docker-compose.yml``
starts one locally on port 5433; CI runs a service container). Every test here
depends, directly or transitively, on the ``migrated_db`` fixture, so a missing
``TEST_DATABASE_URL`` skips them all with one clear reason rather than a silent pass.

No ``pytest-asyncio`` (C12): each test drives its own async body with a single
``asyncio.run()`` call, matching ``tests/test_repositories.py``. ``ingest.review.main``
already wraps its own body in ``asyncio.run``, so a CLI-level test just calls it
directly like any other synchronous function.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

import ingest.review as review
from app.core.db import create_engine, create_session_factory, session_scope
from app.repositories.documents import DocumentRepository
from app.repositories.edges import EdgeRepository, ReviewRow
from app.repositories.units import UnitRepository
from app.schemas.document import DocStatus, DocType, Document, DocumentStub
from app.schemas.edge import Edge, EdgeSource, EdgeType, ReviewStatus
from app.schemas.unit import Unit, UnitType

DOC_ID = "gas-decree-87-2018"
OTHER_DOC_ID = "gas-decree-17-2020"


def _document(doc_id: str = DOC_ID, **overrides: Any) -> Document:
    fields: dict[str, Any] = {
        "id": doc_id,
        "number": "87/2018/NĐ-CP",
        "title_vi": "Nghị định về kinh doanh khí",
        "doc_type": DocType.DECREE,
        "issued": date(2018, 6, 15),
        "effective": date(2018, 8, 1),
        "status": DocStatus.IN_FORCE,
        "path": f"corpus/{doc_id}.html",
    }
    fields.update(overrides)
    return Document(**fields)


def _article(**overrides: Any) -> Unit:
    fields: dict[str, Any] = {
        "id": f"{DOC_ID}#dieu-1",
        "document_id": DOC_ID,
        "unit_type": UnitType.ARTICLE,
        "dieu_no": 1,
        "heading": "Phạm vi điều chỉnh",
        "body": "Việc bảo quản chai LPG thực hiện theo quy định tại Điều 38.",
        "ordinal": 0,
    }
    fields.update(overrides)
    return Unit(**fields)


def _edge(**overrides: Any) -> Edge:
    fields: dict[str, Any] = {
        "src_id": f"{DOC_ID}#dieu-1",
        "dst_id": OTHER_DOC_ID,
        "edge_type": EdgeType.REFERENCES,
        "evidence": "quy định tại Điều 38 của Nghị định số 17/2020/NĐ-CP",
        "review_status": ReviewStatus.PENDING,
        "source": EdgeSource.REGEX,
    }
    fields.update(overrides)
    return Edge(**fields)


def _run(database_url: str, body: Callable[[AsyncEngine], Awaitable[None]]) -> None:
    """Create an engine, run ``body(engine)`` to completion, then dispose it."""

    async def _with_engine() -> None:
        engine = create_engine(database_url)
        try:
            await body(engine)
        finally:
            await engine.dispose()

    asyncio.run(_with_engine())


async def _seed(engine: AsyncEngine, edges: list[Edge], *, with_units: bool = True) -> None:
    """Persist one document (and, unless disabled, its one article) plus ``edges``."""
    session_factory = create_session_factory(engine)
    async with session_scope(session_factory) as session:
        await DocumentRepository(session).upsert(_document())
        await DocumentRepository(session).upsert(_document(OTHER_DOC_ID, number="17/2020/NĐ-CP"))
        if with_units:
            await UnitRepository(session).upsert_many([_article()])
        await EdgeRepository(session).upsert_many(edges)


# --- the trap: a rejected (and an approved) edge must survive a re-ingest --------


def test_upsert_never_resets_review_status_on_conflict(migrated_db: str):
    # This is the automated version of the acceptance proof: ingest, reject,
    # re-ingest the *same* edge (regex extraction always re-produces it as
    # PENDING), confirm the human decision is untouched.
    async def body(engine: AsyncEngine) -> None:
        await _seed(engine, [_edge()])
        session_factory = create_session_factory(engine)

        async with session_scope(session_factory) as session:
            edge_id = (await session.execute(text("SELECT id FROM edges"))).scalar_one()

        async with session_scope(session_factory) as session:
            updated = await EdgeRepository(session).set_review_status(
                [edge_id], ReviewStatus.REJECTED
            )
        assert updated == 1

        # Simulate a re-ingest: the exact same (src_id, dst_id, edge_type),
        # freshly re-extracted, always arrives as PENDING again.
        async with session_scope(session_factory) as session:
            await EdgeRepository(session).upsert_many([_edge(evidence="re-extracted text")])

        async with session_scope(session_factory) as session:
            row = (
                await session.execute(
                    text("SELECT review_status, evidence FROM edges WHERE id = :id"),
                    {"id": edge_id},
                )
            ).one()
        assert row.review_status == ReviewStatus.REJECTED.value
        assert row.evidence == "re-extracted text"  # evidence still refreshes

    _run(migrated_db, body)


def test_upsert_never_resets_an_approved_edge_either(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(engine, [_edge()])
        session_factory = create_session_factory(engine)

        async with session_scope(session_factory) as session:
            edge_id = (await session.execute(text("SELECT id FROM edges"))).scalar_one()
        async with session_scope(session_factory) as session:
            await EdgeRepository(session).set_review_status([edge_id], ReviewStatus.APPROVED)
        async with session_scope(session_factory) as session:
            await EdgeRepository(session).upsert_many([_edge()])
        async with session_scope(session_factory) as session:
            status = (
                await session.execute(
                    text("SELECT review_status FROM edges WHERE id = :id"), {"id": edge_id}
                )
            ).scalar_one()
        assert status == ReviewStatus.APPROVED.value

    _run(migrated_db, body)


# --- EdgeRepository.list_pending ---------------------------------------------------


def test_list_pending_joins_the_source_units_body(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(engine, [_edge()])
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            rows = await EdgeRepository(session).list_pending()
        assert len(rows) == 1
        assert rows[0].source_body == _article().body

    _run(migrated_db, body)


def test_list_pending_source_body_is_none_for_a_document_level_citation(migrated_db: str):
    # BASIS/AMENDMENT preamble citations have src_id = a bare document id,
    # which never appears in units.id -- the LEFT JOIN must not drop the row.
    async def body(engine: AsyncEngine) -> None:
        await _seed(
            engine,
            [_edge(src_id=DOC_ID, edge_type=EdgeType.GUIDED_BY)],
            with_units=False,
        )
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            rows = await EdgeRepository(session).list_pending()
        assert len(rows) == 1
        assert rows[0].source_body is None

    _run(migrated_db, body)


def test_list_pending_never_returns_an_approved_or_rejected_edge(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(
            engine,
            [
                _edge(dst_id=f"{OTHER_DOC_ID}#dieu-1", review_status=ReviewStatus.APPROVED),
                _edge(dst_id=f"{OTHER_DOC_ID}#dieu-2", review_status=ReviewStatus.REJECTED),
            ],
        )
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            rows = await EdgeRepository(session).list_pending()
        assert rows == []

    _run(migrated_db, body)


def test_list_pending_filters_by_edge_type(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(
            engine,
            [
                _edge(dst_id=f"{OTHER_DOC_ID}#dieu-1", edge_type=EdgeType.REFERENCES),
                _edge(dst_id=OTHER_DOC_ID, edge_type=EdgeType.GUIDED_BY, src_id=DOC_ID),
            ],
        )
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            rows = await EdgeRepository(session).list_pending(edge_type=EdgeType.GUIDED_BY)
        assert len(rows) == 1
        assert rows[0].edge_type == EdgeType.GUIDED_BY.value

    _run(migrated_db, body)


def test_list_pending_filters_by_src_document_including_units_under_it(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(
            engine,
            [
                _edge(src_id=f"{DOC_ID}#dieu-1", dst_id=f"{OTHER_DOC_ID}#dieu-1"),
                _edge(src_id=DOC_ID, dst_id=OTHER_DOC_ID, edge_type=EdgeType.BASED_ON),
                _edge(src_id=OTHER_DOC_ID, dst_id=DOC_ID, edge_type=EdgeType.BASED_ON),
            ],
        )
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            rows = await EdgeRepository(session).list_pending(src_document=DOC_ID)
        assert {row.src_id for row in rows} == {f"{DOC_ID}#dieu-1", DOC_ID}

    _run(migrated_db, body)


def test_list_pending_respects_limit(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(
            engine,
            [_edge(dst_id=f"{OTHER_DOC_ID}#dieu-{n}") for n in range(1, 4)],
        )
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            rows = await EdgeRepository(session).list_pending(limit=2)
        assert len(rows) == 2

    _run(migrated_db, body)


# --- EdgeRepository.set_review_status / stats --------------------------------------


def test_set_review_status_reports_how_many_ids_actually_matched(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(engine, [_edge()])
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            edge_id = (await session.execute(text("SELECT id FROM edges"))).scalar_one()
        async with session_scope(session_factory) as session:
            updated = await EdgeRepository(session).set_review_status(
                [edge_id, 999_999], ReviewStatus.APPROVED
            )
        assert updated == 1  # one real id, one that matches nothing

    _run(migrated_db, body)


def test_stats_groups_by_edge_type_review_status_and_source(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(
            engine,
            [
                _edge(dst_id=f"{OTHER_DOC_ID}#dieu-1"),
                _edge(dst_id=f"{OTHER_DOC_ID}#dieu-2", review_status=ReviewStatus.APPROVED),
            ],
        )
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            rows = await EdgeRepository(session).stats()
        by_status = {row.review_status: row.count for row in rows}
        assert by_status[ReviewStatus.PENDING.value] == 1
        assert by_status[ReviewStatus.APPROVED.value] == 1

    _run(migrated_db, body)


# --- CLI argument validation --------------------------------------------------------


def test_approve_all_without_edge_type_is_rejected():
    assert review._parse_args(["--approve-all", "--database-url", "postgresql://x"]) is None


def test_approve_all_with_edge_type_parses(migrated_db: str):
    args = review._parse_args(
        ["--approve-all", "--edge-type", "guided_by", "--database-url", migrated_db]
    )
    assert args is not None
    assert args.edge_type is EdgeType.GUIDED_BY


def test_an_invalid_edge_type_is_a_usage_error():
    with pytest.raises(SystemExit):
        review._parse_args(["--list-pending", "--edge-type", "not_a_real_type"])


# --- the CLI end to end --------------------------------------------------------------


def test_approve_all_only_touches_the_filtered_set(migrated_db: str):
    async def body(engine: AsyncEngine) -> None:
        await _seed(
            engine,
            [
                _edge(dst_id=f"{OTHER_DOC_ID}#dieu-1", edge_type=EdgeType.REFERENCES),
                _edge(dst_id=OTHER_DOC_ID, edge_type=EdgeType.GUIDED_BY, src_id=DOC_ID),
            ],
        )

    _run(migrated_db, body)

    exit_code = review.main(
        ["--approve-all", "--edge-type", "guided_by", "--database-url", migrated_db]
    )
    assert exit_code == 0

    async def check(engine: AsyncEngine) -> None:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            statuses = (
                await session.execute(text("SELECT edge_type, review_status FROM edges"))
            ).all()
        by_type = {row.edge_type: row.review_status for row in statuses}
        assert by_type[EdgeType.GUIDED_BY.value] == ReviewStatus.APPROVED.value
        assert by_type[EdgeType.REFERENCES.value] == ReviewStatus.PENDING.value  # untouched

    _run(migrated_db, check)


def test_approve_and_reject_via_the_cli(migrated_db: str, capsys: pytest.CaptureFixture[str]):
    async def body(engine: AsyncEngine) -> None:
        await _seed(engine, [_edge()])

    _run(migrated_db, body)

    async def get_id(engine: AsyncEngine) -> int:
        session_factory = create_session_factory(engine)
        async with session_scope(session_factory) as session:
            return (await session.execute(text("SELECT id FROM edges"))).scalar_one()

    edge_id: list[int] = []

    async def capture_id(engine: AsyncEngine) -> None:
        edge_id.append(await get_id(engine))

    _run(migrated_db, capture_id)
    the_id = edge_id[0]

    exit_code = review.main(["--reject", str(the_id), "--database-url", migrated_db])
    assert exit_code == 0
    assert "rejected: 1 of 1" in capsys.readouterr().out

    exit_code = review.main(["--list-pending", "--database-url", migrated_db])
    assert exit_code == 0
    assert "No pending edges match." in capsys.readouterr().out


def test_stats_cli_prints_the_count_table(migrated_db: str, capsys: pytest.CaptureFixture[str]):
    async def body(engine: AsyncEngine) -> None:
        await _seed(engine, [_edge()])

    _run(migrated_db, body)

    exit_code = review.main(["--stats", "--database-url", migrated_db])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "references" in output
    assert "pending" in output
    assert "regex" in output


def test_list_pending_cli_prints_evidence_and_source_body(
    migrated_db: str, capsys: pytest.CaptureFixture[str]
):
    async def body(engine: AsyncEngine) -> None:
        await _seed(engine, [_edge()])

    _run(migrated_db, body)

    exit_code = review.main(["--list-pending", "--database-url", migrated_db])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert _edge().evidence in output
    assert _article().body in output


def test_no_database_url_fails_loudly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    exit_code = review.main(["--stats", "--database-url", ""])
    assert exit_code == 1


# --- repeal edges: the reviewer sees the clause, and cannot bulk-approve them (issue #83) ----
#
# A REPLACED_BY edge runs OLD -> NEW but the repeal is written in the NEW document, so it
# starts at the repealed document -- here a stub with no text -- while the sentence that
# justifies it lives in a unit of the repealing one (``origin_id``, issue #68). The text below
# is NĐ 105/2025 Điều 45 khoản 2 điểm b as the real corpus prints it.

REPEALING_ID = "pccc-decree-105-2025"
REPEALED_STUB = DocumentStub(number="136/2020/NĐ-CP")
_POINT_ID = f"{REPEALING_ID}#dieu-45#khoan-2#diem-b"
_LEAD_IN = (
    "Các Nghị định sau đây hết hiệu lực kể từ ngày Nghị định này có hiệu lực thi hành, "
    "trừ trường hợp quy định tại Điều 46 của Nghị định này:"
)
_POINT_TEXT = (
    "Nghị định số 136/2020/NĐ-CP ngày 24 tháng 11 năm 2020 quy định chi tiết một số điều và "
    "biện pháp thi hành Luật Phòng cháy và chữa cháy;"
)
_PLACEHOLDER = "(document-level citation; no unit body)"


def _repeal_units() -> list[Unit]:
    common: dict[str, Any] = {"document_id": REPEALING_ID, "dieu_no": 45}
    article = Unit(
        id=f"{REPEALING_ID}#dieu-45",
        unit_type=UnitType.ARTICLE,
        heading="Điều khoản thi hành",
        ordinal=0,
        **common,
    )
    clause = Unit(
        id=f"{REPEALING_ID}#dieu-45#khoan-2",
        unit_type=UnitType.CLAUSE,
        khoan_no=2,
        parent_id=article.id,
        body=_LEAD_IN,
        ordinal=1,
        **common,
    )
    point = Unit(
        id=_POINT_ID,
        unit_type=UnitType.POINT,
        khoan_no=2,
        diem="b",
        parent_id=clause.id,
        body=_POINT_TEXT,
        ordinal=2,
        **common,
    )
    return [article, clause, point]


def _repeal_edge(**overrides: Any) -> Edge:
    fields: dict[str, Any] = {
        "src_id": REPEALED_STUB.id,
        "dst_id": REPEALING_ID,
        "edge_type": EdgeType.REPLACED_BY,
        "evidence": f"{_LEAD_IN} {_POINT_TEXT}",
        "review_status": ReviewStatus.PENDING,
        "source": EdgeSource.REGEX,
        "origin_id": _POINT_ID,
    }
    fields.update(overrides)
    return Edge(**fields)


async def _seed_repeal(engine: AsyncEngine, edges: list[Edge] | None = None) -> None:
    """Persist the repealing document, its units, the repealed stub, and ``edges``."""
    session_factory = create_session_factory(engine)
    async with session_scope(session_factory) as session:
        documents = DocumentRepository(session)
        await documents.upsert(_document(REPEALING_ID, number="105/2025/NĐ-CP"))
        await documents.insert_stubs([REPEALED_STUB])
        await UnitRepository(session).upsert_many(_repeal_units())
        await EdgeRepository(session).upsert_many([_repeal_edge()] if edges is None else edges)


def _rows(database_url: str, sql: str) -> list[Any]:
    """Run one read-only query and return its rows."""
    rows: list[Any] = []

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            rows.extend((await session.execute(text(sql))).all())

    _run(database_url, body)
    return rows


def _list_pending(database_url: str, **filters: Any) -> list[ReviewRow]:
    rows: list[ReviewRow] = []

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            rows.extend(await EdgeRepository(session).list_pending(**filters))

    _run(database_url, body)
    return rows


def _status_by_type(database_url: str) -> dict[str, str]:
    return {
        row.edge_type: row.review_status
        for row in _rows(database_url, "SELECT edge_type, review_status FROM edges")
    }


# 1. the reviewer sees the clause that states the repeal


def test_list_pending_returns_the_clause_that_states_a_repeal(migrated_db: str):
    _run(migrated_db, _seed_repeal)

    (row,) = _list_pending(migrated_db, edge_type=EdgeType.REPLACED_BY)

    # The edge starts at a stub with no text; the body is the repealing unit's, not None.
    assert row.src_id == REPEALED_STUB.id
    assert row.origin_id == _POINT_ID
    assert row.source_body == _POINT_TEXT


def test_the_listing_shows_the_repeal_clause_and_says_which_unit_it_is(
    migrated_db: str, capsys: pytest.CaptureFixture[str]
):
    _run(migrated_db, _seed_repeal)

    assert (
        review.main(["--list-pending", "--edge-type", "replaced_by", "--database-url", migrated_db])
        == 0
    )
    output = capsys.readouterr().out

    assert _PLACEHOLDER not in output
    assert repr(_POINT_TEXT) in output
    assert f"stated in: {_POINT_ID}" in output
    assert _LEAD_IN in output  # the edge's own evidence carries the lead-in as well


def test_src_document_lists_the_repeal_edges_the_repealing_document_produced(
    migrated_db: str, capsys: pytest.CaptureFixture[str]
):
    _run(migrated_db, _seed_repeal)

    listed = _list_pending(migrated_db, src_document=REPEALING_ID)
    assert [(row.src_id, row.dst_id) for row in listed] == [(REPEALED_STUB.id, REPEALING_ID)]

    assert (
        review.main(
            ["--list-pending", "--src-document", REPEALING_ID, "--database-url", migrated_db]
        )
        == 0
    )
    assert f"{REPEALED_STUB.id} --REPLACED_BY--> {REPEALING_ID}" in capsys.readouterr().out


def test_src_document_does_not_list_a_repeal_under_a_document_that_did_not_state_it(
    migrated_db: str,
):
    _run(migrated_db, _seed_repeal)

    # Attribution is by the text that produced the edge (as reconcile_document_edges
    # scopes it), so neither the repealed stub nor an unrelated document owns it.
    assert _list_pending(migrated_db, src_document=REPEALED_STUB.id) == []
    assert _list_pending(migrated_db, src_document=DOC_ID) == []


def test_a_row_with_no_origin_still_lists_and_filters_by_its_source(migrated_db: str):
    # A curated seed edge has no origin_id: nothing's text produced it. It falls back to
    # src_id, so it lists and filters exactly as it did before origin_id existed.
    _run(
        migrated_db,
        lambda engine: _seed(
            engine, [_edge(origin_id=None, source=EdgeSource.CURATED, src_id=DOC_ID)]
        ),
    )

    (row,) = _list_pending(migrated_db, src_document=DOC_ID)

    assert row.origin_id is None
    assert row.source_body is None  # src_id is a bare document id: no unit body


def test_an_edge_that_starts_where_it_is_stated_lists_exactly_as_before(
    migrated_db: str, capsys: pytest.CaptureFixture[str]
):
    edge = _edge(origin_id=f"{DOC_ID}#dieu-1")
    _run(migrated_db, lambda engine: _seed(engine, [edge]))

    assert review.main(["--list-pending", "--database-url", migrated_db]) == 0
    output = capsys.readouterr().out

    (edge_id,) = [row.id for row in _list_pending(migrated_db)]
    assert f"[{edge_id}] {edge.src_id} --REFERENCES--> {edge.dst_id}  (source: regex)" in output
    assert f"  evidence: {edge.evidence!r}" in output
    assert f"  source body: {_article().body!r}" in output
    assert "stated in:" not in output  # origin == src: nothing to add
    assert "1 pending edge(s) listed." in output


# 2. --approve-all refuses the edge type that retires a document


def test_approve_all_refuses_replaced_by_and_approves_nothing(
    migrated_db: str, capsys: pytest.CaptureFixture[str]
):
    _run(migrated_db, _seed_repeal)

    exit_code = review.main(
        ["--approve-all", "--edge-type", "replaced_by", "--database-url", migrated_db]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "--approve-all refuses --edge-type replaced_by" in output
    assert "--approve ID [ID ...]" in output  # says what to do instead
    assert "approved:" not in output
    assert _status_by_type(migrated_db) == {"replaced_by": "pending"}


def test_approve_all_refuses_replaced_by_even_when_narrowed_to_one_document(
    migrated_db: str,
):
    _run(migrated_db, _seed_repeal)

    exit_code = review.main(
        [
            "--approve-all",
            "--edge-type",
            "replaced_by",
            "--src-document",
            REPEALING_ID,
            "--database-url",
            migrated_db,
        ]
    )

    assert exit_code == 1
    assert _status_by_type(migrated_db) == {"replaced_by": "pending"}


def test_the_refusal_happens_before_any_database_is_touched():
    # An unreachable database: if the refusal came after connecting, this would raise
    # or log a connection error instead of returning the refusal's exit code.
    exit_code = review.main(
        [
            "--approve-all",
            "--edge-type",
            "replaced_by",
            "--database-url",
            "postgresql+asyncpg://nobody:nopass@127.0.0.1:1/nowhere",
        ]
    )

    assert exit_code == 1


def test_approve_all_itself_refuses_replaced_by_when_called_without_the_argument_check(
    migrated_db: str,
):
    _run(migrated_db, _seed_repeal)

    async def body(engine: AsyncEngine) -> None:
        async with session_scope(create_session_factory(engine)) as session:
            with pytest.raises(ValueError, match="refuses --edge-type replaced_by"):
                await review._approve_all(EdgeRepository(session), EdgeType.REPLACED_BY, None)

    _run(migrated_db, body)

    assert _status_by_type(migrated_db) == {"replaced_by": "pending"}


# 3. everything else behaves exactly as before


@pytest.mark.parametrize(
    "edge_type", [edge_type for edge_type in EdgeType if edge_type is not EdgeType.REPLACED_BY]
)
def test_approve_all_still_approves_every_other_edge_type(
    migrated_db: str, edge_type: EdgeType, capsys: pytest.CaptureFixture[str]
):
    _run(migrated_db, lambda engine: _seed(engine, [_edge(edge_type=edge_type)]))

    exit_code = review.main(
        ["--approve-all", "--edge-type", edge_type.value, "--database-url", migrated_db]
    )

    assert exit_code == 0
    assert f"approved: 1 pending {edge_type.value} edge(s)." in capsys.readouterr().out
    assert _status_by_type(migrated_db) == {edge_type.value: "approved"}


def test_approve_all_for_another_type_leaves_a_pending_repeal_alone(migrated_db: str):
    _run(migrated_db, lambda engine: _seed_repeal(engine, [_repeal_edge(), _edge()]))

    assert (
        review.main(["--approve-all", "--edge-type", "references", "--database-url", migrated_db])
        == 0
    )

    assert _status_by_type(migrated_db) == {"replaced_by": "pending", "references": "approved"}


def test_approve_by_id_still_approves_a_replaced_by_edge(
    migrated_db: str, capsys: pytest.CaptureFixture[str]
):
    # The individual path is the one the refusal points to, and it is unchanged.
    _run(migrated_db, _seed_repeal)
    (row,) = _list_pending(migrated_db, edge_type=EdgeType.REPLACED_BY)

    assert review.main(["--approve", str(row.id), "--database-url", migrated_db]) == 0

    assert "approved: 1 of 1 edge id(s) updated." in capsys.readouterr().out
    assert _status_by_type(migrated_db) == {"replaced_by": "approved"}


def test_reject_by_id_still_rejects_a_replaced_by_edge(
    migrated_db: str, capsys: pytest.CaptureFixture[str]
):
    _run(migrated_db, _seed_repeal)
    (row,) = _list_pending(migrated_db, edge_type=EdgeType.REPLACED_BY)

    assert review.main(["--reject", str(row.id), "--database-url", migrated_db]) == 0

    assert "rejected: 1 of 1 edge id(s) updated." in capsys.readouterr().out
    assert _status_by_type(migrated_db) == {"replaced_by": "rejected"}


def test_stats_still_counts_a_repeal_edge_like_any_other(
    migrated_db: str, capsys: pytest.CaptureFixture[str]
):
    _run(migrated_db, _seed_repeal)

    assert review.main(["--stats", "--database-url", migrated_db]) == 0
    stats_line = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("replaced_by")
    )

    assert stats_line.split() == ["replaced_by", "pending", "regex", "1"]
