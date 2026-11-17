"""Tests for stub documents -- cited documents the manifest does not list (issue #67).

Two layers. The first half is pure: identity, normalisation and the strict
"is this number complete" rule, driven with hand-built ``Reference`` rows so the
hints are exactly the ones the issue names. The second half needs a live Postgres
(``TEST_DATABASE_URL``, see ``tests/test_repositories.py``) and runs the real
pipeline -- extraction, resolution, persistence, the real traversal filter -- over a
small synthetic corpus, because a stub that only exists in memory proves nothing
about what the database holds.
"""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from app.core.db import create_engine, create_session_factory, session_scope
from app.core.ids import document_id, stub_document_id
from app.repositories.documents import DocumentRepository
from app.repositories.edges import EdgeRepository
from app.schemas.document import DocStatus, DocType, Document, DocumentStub
from app.schemas.edge import EdgeType, ReviewStatus
from app.schemas.reference import Reference, ReferenceKind
from app.services.graph_traversal import GraphTraversalService
from ingest.extract import _rejoin_split_document_numbers
from ingest.graph import ReferenceEdges, build_reference_edges
from ingest.manifest import MANIFEST_FILENAME, Manifest
from ingest.parser import ParseResult, parse_document
from ingest.references import find_document_number
from ingest.run import DocumentStats, _stub_ids, _totals, main

# --- pure: identity ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        ("136/2020/NĐ-CP", "cited-136-2020-nd-cp"),
        ("68/2006/QH11", "cited-68-2006-qh11"),
        ("06/2022/TT-BCA", "cited-06-2022-tt-bca"),
    ],
)
def test_a_stub_id_is_deterministic_and_is_a_plain_document_id(number: str, expected: str):
    first = stub_document_id(number)
    assert first == expected
    assert stub_document_id(number) == first
    # No colon: ids.is_entity_id decides entity-ness by prefix, and an
    # unknown "xxx:" prefix would be dropped as a dangling edge (#61). No "#" either --
    # that separates a document from its units, and a stub has none.
    assert ":" not in first
    assert "#" not in first


def test_the_accent_stripped_spelling_of_a_number_is_the_same_stub():
    assert stub_document_id("136/2020/ND-CP") == stub_document_id("136/2020/NĐ-CP")


def test_different_numbers_never_share_a_stub_id():
    numbers = ["136/2020/NĐ-CP", "136/2021/NĐ-CP", "137/2020/NĐ-CP", "136/2020/TT-CP"]
    assert len({stub_document_id(number) for number in numbers}) == len(numbers)


def test_a_number_that_leaves_no_slug_is_refused():
    with pytest.raises(ValueError, match="Cannot derive a stub id"):
        stub_document_id("///")


def test_a_manifest_document_cannot_take_the_id_a_stub_could_derive():
    stub_id = stub_document_id("136/2020/NĐ-CP")
    with pytest.raises(ValueError, match="reserved for stubs"):
        document_id(stub_id)
    with pytest.raises(ValueError, match="reserved for stubs"):
        Document(
            id=stub_id,
            number="136/2020/NĐ-CP",
            title_vi="Nghị định",
            doc_type=DocType.DECREE,
            status=DocStatus.IN_FORCE,
            path="corpus/x.html",
        )


def test_a_stub_is_never_reported_as_in_force():
    stub = DocumentStub(number="136/2020/NĐ-CP")
    assert stub.status is DocStatus.UNKNOWN
    assert stub.status is not DocStatus.IN_FORCE
    assert stub.doc_type is DocType.UNKNOWN
    assert stub.title_vi == "136/2020/NĐ-CP"
    assert stub.id == stub_document_id("136/2020/NĐ-CP")


# --- pure: normalising a number out of a hint (trap 1) -----------------------------

_SPACED_HINT = "Luật Tiêu chuẩn và Quy chuẩn kỹ thuật số 68/ 2006/QH11 ngày 21 tháng 7 năm 2006"
_PLAIN_HINT = "68/2006/QH11"


def _extracted(text: str) -> str:
    """Return ``text`` as ``extract_text`` hands it on: its document numbers already whole."""
    return _rejoin_split_document_numbers(text, Path("fixture.pdf"))


def test_the_two_hints_from_the_issue_are_one_number_once_extraction_has_repaired_the_text():
    # The spaced hint is a whole basis phrase. The repair is ingest.extract's (issue #80): by
    # the time a hint exists, the text it came from is already whole.
    assert find_document_number(_extracted(_SPACED_HINT)) == "68/2006/QH11"
    assert find_document_number(_PLAIN_HINT) == "68/2006/QH11"


def test_finding_a_number_repairs_nothing_that_extraction_left_split():
    # One implementation of the repair, in ingest.extract: the finder only finds.
    assert find_document_number(_SPACED_HINT) is None


def test_a_standard_number_is_returned_with_its_whitespace_collapsed():
    assert find_document_number("TCVN   6223:2017") == "TCVN 6223:2017"


def test_a_number_followed_by_a_unit_chain_still_yields_the_number():
    assert find_document_number("87/2018/NĐ-CP Điều 5 Khoản 1") == "87/2018/NĐ-CP"


@pytest.mark.parametrize(
    "hint",
    [
        "Luật Thương mại ngày 14 tháng 6 năm 2005",
        "Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam",
        "hết hiệu lực",
        "Nghị định số",
    ],
)
def test_a_hint_with_no_number_yields_none(hint: str):
    assert find_document_number(hint) is None


# --- pure: resolving hints against a manifest ----------------------------------------

DOC_ID = "citing-decree-200-2021"


def _document(doc_id: str, number: str, doc_type: DocType) -> Document:
    return Document(
        id=doc_id,
        number=number,
        title_vi=f"Văn bản {number}",
        doc_type=doc_type,
        status=DocStatus.IN_FORCE,
        path=f"corpus/{doc_id}.html",
    )


_SELF = _document(DOC_ID, "200/2021/NĐ-CP", DocType.DECREE)
_LAW = _document("law-fire-prevention-2001", "27/2001/QH10", DocType.LAW)
_MANIFEST = Manifest(root=Path("."), documents=(_SELF, _LAW))
_EMPTY_RESULT = ParseResult(
    document_id=DOC_ID, units=(), stats=parse_document("", DOC_ID).stats, preamble=""
)


def _resolve(*references: Reference) -> ReferenceEdges:
    return build_reference_edges(references, _MANIFEST, _EMPTY_RESULT)


def _reference(kind: ReferenceKind, target_hint: str, raw_text: str | None = None) -> Reference:
    return Reference(
        src_id=DOC_ID, kind=kind, raw_text=raw_text or target_hint, target_hint=target_hint
    )


def test_the_two_hints_from_the_issue_resolve_to_one_stub_not_two():
    resolution = _resolve(
        _reference(
            ReferenceKind.BASIS, _extracted(_SPACED_HINT), _extracted(f"Căn cứ {_SPACED_HINT};")
        ),
        _reference(ReferenceKind.EXTERNAL, _PLAIN_HINT),
    )

    assert [stub.number for stub in resolution.stubs] == ["68/2006/QH11"]
    assert {edge.dst_id for edge in resolution.edges} == {stub_document_id("68/2006/QH11")}
    assert resolution.unresolved == 0
    assert resolution.no_stub == 0


def test_a_title_only_citation_creates_no_stub_and_is_counted():
    resolution = _resolve(
        _reference(ReferenceKind.BASIS, "Luật Thương mại ngày 14 tháng 6 năm 2005"),
        _reference(ReferenceKind.BASIS, "Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam"),
    )

    assert resolution.stubs == ()
    assert resolution.edges == ()
    assert resolution.unresolved == 2
    assert resolution.no_stub == 2
    assert resolution.rejected_numbers == ()


def test_a_standard_creates_no_stub_because_its_identity_includes_an_edition():
    resolution = _resolve(
        _reference(ReferenceKind.EXTERNAL, "TCVN 7762"),
        _reference(ReferenceKind.EXTERNAL, "TCVN 7762:2007"),
    )

    assert resolution.stubs == ()
    assert resolution.unresolved == 2
    assert resolution.no_stub == 2
    assert resolution.rejected_numbers == ()


@pytest.mark.parametrize(
    "number",
    [
        "136/2020/NĐ-CP",
        "08/2018/NĐ-CP",
        "05/2007/QH12",
        "33/2013/QH13",
        "06/2022/TT-BCA",
        "20/2019/TT-BKHCN",
    ],
)
def test_a_complete_number_earns_a_stub(number: str):
    resolution = _resolve(_reference(ReferenceKind.EXTERNAL, number))

    assert [stub.number for stub in resolution.stubs] == [number]
    assert resolution.no_stub == 0


# Shapes a number used to reach the resolver in when extraction cut it short at an artifact
# (issue #80 now repairs the ones the corpus shows). The resolver must still refuse any that
# arrives this way, so none of them is ever made into a stub.
@pytest.mark.parametrize(
    "number",
    [
        "87/2018/N",  # "N" for "NĐ-CP" -- and the manifest's own decree, so a ghost node
        "78/2021/NĐ",  # "NĐ" for "NĐ-CP"
        "41/2011/TT",  # "TT" for "TT-<agency>"
        "0/2024/NĐ-CP",  # a serial cut short: "0" for "50"
        "12/2015/QĐ-TT",  # extraction stops at the lower-case g of "QĐ-TTg"
        "12/2020/TT-B",  # an issuer cut short to one letter
        "00/2020/NĐ-CP",  # no document is numbered zero
        "12/1899/NĐ-CP",  # not a plausible year
    ],
)
def test_a_malformed_number_gets_no_stub_and_is_named(number: str):
    resolution = _resolve(_reference(ReferenceKind.EXTERNAL, number))

    assert resolution.stubs == ()
    assert resolution.edges == ()
    assert resolution.unresolved == 1
    assert resolution.no_stub == 1
    assert resolution.rejected_numbers == (number,)


def test_rejected_numbers_are_distinct_and_sorted_but_no_stub_counts_every_citation():
    resolution = _resolve(
        _reference(ReferenceKind.EXTERNAL, "78/2021/NĐ"),
        _reference(ReferenceKind.EXTERNAL, "41/2011/TT"),
        _reference(ReferenceKind.EXTERNAL, "78/2021/NĐ"),
    )

    assert resolution.rejected_numbers == ("41/2011/TT", "78/2021/NĐ")
    assert resolution.no_stub == 3


def test_a_number_in_the_manifest_resolves_to_the_manifest_document_never_a_stub():
    resolution = _resolve(_reference(ReferenceKind.EXTERNAL, "27/2001/QH10"))

    assert resolution.stubs == ()
    assert [edge.dst_id for edge in resolution.edges] == [_LAW.id]


def test_a_number_the_pdf_split_that_is_in_the_manifest_resolves_to_it_not_to_a_stub():
    resolution = _resolve(_reference(ReferenceKind.EXTERNAL, _extracted("27/ 2001/QH10")))

    assert resolution.stubs == ()
    assert [edge.dst_id for edge in resolution.edges] == [_LAW.id]


def test_a_basis_edge_to_a_stub_is_based_on_never_guided_by():
    # A decree citing a *law* by number is GUIDED_BY when the law is in the manifest
    # (its type is known). A stub's type is unknown -- QH is a law or a National
    # Assembly resolution -- so the edge must not claim the target is a law.
    resolution = _resolve(
        _reference(ReferenceKind.BASIS, "39/2024/QH15", "Căn cứ Luật số 39/2024/QH15;")
    )

    (edge,) = resolution.edges
    assert edge.edge_type is EdgeType.BASED_ON
    assert edge.src_id == DOC_ID
    assert edge.dst_id == stub_document_id("39/2024/QH15")
    assert edge.review_status is ReviewStatus.PENDING


def test_an_external_edge_to_a_stub_is_a_pending_references_edge():
    resolution = _resolve(_reference(ReferenceKind.EXTERNAL, "136/2020/NĐ-CP"))

    (edge,) = resolution.edges
    assert edge.edge_type is EdgeType.REFERENCES
    assert edge.review_status is ReviewStatus.PENDING
    assert edge.evidence == "136/2020/NĐ-CP"


def test_an_amendment_reference_never_gets_a_stub_but_is_still_counted():
    # An amendment reference captures a cue, not a target, so it has no number to derive
    # a stub from. The REPLACED_BY edge that starts at a stub is built by the repeal
    # pass from the whole clause (issue #68), and reconcile scopes it by origin_id.
    resolution = _resolve(
        _reference(ReferenceKind.AMENDMENT, "136/2020/NĐ-CP", "bãi bỏ Nghị định 136/2020/NĐ-CP")
    )

    assert resolution.stubs == ()
    assert resolution.edges == ()
    assert resolution.unresolved == 1
    assert resolution.no_stub == 1


def test_an_unresolved_internal_reference_is_not_a_no_stub_citation():
    resolution = _resolve(_reference(ReferenceKind.INTERNAL, "Điều 99"))

    assert resolution.unresolved == 1
    assert resolution.no_stub == 0


def test_stubs_come_out_distinct_and_ordered_by_id_whatever_the_citation_order():
    forward = _resolve(
        _reference(ReferenceKind.EXTERNAL, "136/2020/NĐ-CP"),
        _reference(ReferenceKind.EXTERNAL, "08/2018/NĐ-CP"),
        _reference(ReferenceKind.EXTERNAL, "136/2020/NĐ-CP"),
    )
    backward = _resolve(
        _reference(ReferenceKind.EXTERNAL, "08/2018/NĐ-CP"),
        _reference(ReferenceKind.EXTERNAL, "136/2020/NĐ-CP"),
    )

    assert forward.stubs == backward.stubs
    assert [stub.id for stub in forward.stubs] == sorted(stub.id for stub in forward.stubs)
    assert len(forward.stubs) == 2


# --- pure: how the stats table counts stubs ------------------------------------------


def test_the_total_counts_a_stub_two_documents_share_once():
    shared = stub_document_id("136/2020/NĐ-CP")
    only_b = stub_document_id("08/2018/NĐ-CP")
    rows = [
        DocumentStats(document_id="a", stub_ids=frozenset({shared}), no_stub=2),
        DocumentStats(document_id="b", stub_ids=frozenset({shared, only_b}), no_stub=1),
    ]

    total = _totals(rows)

    assert len(total.stub_ids) == 2  # not 3: the shared decree is one document
    assert total.no_stub == 3  # citations, so these do add up


def test_a_document_reported_as_failed_reports_no_stubs():
    stubs = [DocumentStub(number="136/2020/NĐ-CP")]

    assert _stub_ids(stubs, "ingested to zero units") == frozenset()
    assert _stub_ids(stubs, None) == frozenset({stub_document_id("136/2020/NĐ-CP")})


# --- database: the stub row itself ----------------------------------------------------


def _query(database_url: str, sql: str, params: dict[str, Any] | None = None) -> list[Any]:
    """Run one read-only query in its own engine and return every row."""

    async def _run() -> list[Any]:
        engine = create_engine(database_url)
        try:
            async with engine.begin() as conn:
                return list((await conn.execute(text(sql), params or {})).all())
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def _insert_stubs(database_url: str, stubs: Sequence[DocumentStub]) -> None:
    async def _run() -> None:
        engine = create_engine(database_url)
        try:
            async with session_scope(create_session_factory(engine)) as session:
                await DocumentRepository(session).insert_stubs(stubs)
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_a_stub_row_is_honest_about_what_is_unknown(migrated_db: str):
    _insert_stubs(migrated_db, [DocumentStub(number="136/2020/NĐ-CP")])

    (row,) = _query(migrated_db, "SELECT * FROM documents")
    assert row.id == "cited-136-2020-nd-cp"
    assert row.number == "136/2020/NĐ-CP"
    assert row.title_vi == "136/2020/NĐ-CP"
    assert row.status == "unknown"
    assert row.doc_type == "unknown"
    assert row.path is None
    assert row.issued is None
    assert row.effective is None


def test_inserting_the_same_stub_twice_leaves_one_unchanged_row(migrated_db: str):
    stub = DocumentStub(number="136/2020/NĐ-CP")
    _insert_stubs(migrated_db, [stub, stub])
    _insert_stubs(migrated_db, [stub])

    assert _query(migrated_db, "SELECT count(*) AS n FROM documents")[0].n == 1


def test_inserting_a_stub_never_overwrites_an_existing_row(migrated_db: str):
    stub = DocumentStub(number="136/2020/NĐ-CP")

    async def _seed_a_real_row_with_the_stub_id() -> None:
        engine = create_engine(migrated_db)
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO documents (id, number, title_vi, doc_type, status, path) "
                        "VALUES (:id, 'real', 'Real title', 'decree', 'in_force', 'corpus/x')"
                    ),
                    {"id": stub.id},
                )
        finally:
            await engine.dispose()

    asyncio.run(_seed_a_real_row_with_the_stub_id())
    _insert_stubs(migrated_db, [stub])

    (row,) = _query(migrated_db, "SELECT * FROM documents")
    assert (row.title_vi, row.status, row.doc_type) == ("Real title", "in_force", "decree")


def test_inserting_no_stubs_is_a_no_op(migrated_db: str):
    _insert_stubs(migrated_db, [])

    assert _query(migrated_db, "SELECT count(*) AS n FROM documents")[0].n == 0


# --- database: the whole pipeline over a synthetic corpus ------------------------------

_LAW_ID = "law-fire-prevention-2001"
_LAW_NUMBER = "27/2001/QH10"
_LAW_HTML = (
    "<html><body><p>Điều 1. Nội dung luật.</p>"
    "<p>Luật này quy định về phòng cháy và chữa cháy trên toàn quốc, "
    "bao gồm trách nhiệm của tổ chức, cá nhân trong hoạt động sản "
    "xuất, kinh doanh, sinh hoạt hằng ngày, cũng như trách nhiệm "
    "của các cơ quan quản lý nhà nước có liên quan trong việc "
    "thanh tra, kiểm tra và xử lý vi phạm.</p></body></html>"
)

_CITING_A = "citing-decree-a-2021"
_CITING_A_HTML = """<html><body>
<p>Căn cứ Luật Tiêu chuẩn và Quy chuẩn kỹ thuật số 68/ 2006/QH11 ngày 21 tháng 7 năm 2006;</p>
<p>Căn cứ Luật Thương mại ngày 14 tháng 6 năm 2005;</p>
<p>Căn cứ Luật Phòng cháy và chữa cháy số 27/2001/QH10;</p>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Phạm vi điều chỉnh</p>
<p>Nghị định này quy định chi tiết việc thi hành Nghị định số 136/2020/NĐ-CP và Luật
số 68/2006/QH11 về điều kiện kinh doanh khí dầu mỏ hóa lỏng tại thị trường Việt Nam.</p>
<p>Điều 2. Điều khoản thi hành</p>
<p>Việc bảo quản chai phải tuân thủ TCVN 7441 và Nghị định số 88/2018/NĐ-C P về an toàn.</p>
</body></html>"""
"""One citing decree, chosen so every branch of the stub rule appears in real text:

  - basis "... số 68/ 2006/QH11 ngày ..."  the spaced number (trap 1)  -> stub 68/2006/QH11
  - basis by title and date only                                       -> no stub, counted
  - basis "... số 27/2001/QH10"            in the manifest             -> GUIDED_BY the law
  - "Nghị định số 136/2020/NĐ-CP"          plain external              -> stub 136/2020/NĐ-CP
  - "Luật số 68/2006/QH11"                 the same law again, plain   -> the *same* stub
  - "TCVN 7441"                            a standard                  -> no stub, counted
  - "88/2018/NĐ-C P"                       a split no repair covers    -> rejected "88/2018/NĐ-C"
"""

_CITING_B = "citing-decree-b-2021"
_CITING_B_HTML = """<html><body>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Phạm vi điều chỉnh</p>
<p>Nghị định này bổ sung việc thi hành Nghị định số 136/2020/NĐ-CP về điều kiện kinh
doanh khí dầu mỏ hóa lỏng tại thị trường Việt Nam, bao gồm các cửa hàng bán lẻ.</p>
</body></html>"""
"""Cites 136/2020/NĐ-CP too, so the same stub is reached from two documents."""

_STUB_136 = "cited-136-2020-nd-cp"
_STUB_68 = "cited-68-2006-qh11"


def _write_corpus(root: Path) -> Path:
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)

    def entry(doc_id: str, filename: str, number: str, doc_type: str) -> dict[str, Any]:
        return {
            "id": doc_id,
            "number": number,
            "title_vi": f"Văn bản {number}",
            "doc_type": doc_type,
            "status": "in_force",
            "effective": "2021-01-01",
            "path": f"corpus/{filename}",
        }

    manifest = {
        "documents": [
            entry(_CITING_A, "a.html", "200/2021/NĐ-CP", "decree"),
            entry(_CITING_B, "b.html", "201/2021/NĐ-CP", "decree"),
            entry(_LAW_ID, "law.html", _LAW_NUMBER, "law"),
        ]
    }
    (corpus / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    (corpus / "a.html").write_text(_CITING_A_HTML, encoding="utf-8")
    (corpus / "b.html").write_text(_CITING_B_HTML, encoding="utf-8")
    (corpus / "law.html").write_text(_LAW_HTML, encoding="utf-8")
    return corpus


def _args(corpus: Path, database_url: str, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--corpus",
        str(corpus),
        "--database-url",
        database_url,
        "--references",
        str(tmp_path / "references.csv"),
        *extra,
    ]


def _table_row(out: str, document_id_: str) -> dict[str, str]:
    lines = out.splitlines()
    header = next(line for line in lines if line.startswith("document"))
    row = next(line for line in lines if line.startswith(document_id_))
    return dict(zip(header.split(), row.split(), strict=True))


def _stub_rows(database_url: str) -> list[Any]:
    return _query(
        database_url,
        "SELECT id, number, status FROM documents WHERE status = 'unknown' ORDER BY id",
    )


def test_the_pipeline_makes_one_stub_per_distinct_normalised_number(
    migrated_db: str, tmp_path: Path, capsys: Any
):
    corpus = _write_corpus(tmp_path)

    assert main(_args(corpus, migrated_db, tmp_path)) == 0
    out = capsys.readouterr().out

    # 68/ 2006/QH11 (spaced, basis) and 68/2006/QH11 (plain, external) are ONE stub.
    assert [(row.id, row.number) for row in _stub_rows(migrated_db)] == [
        (_STUB_136, "136/2020/NĐ-CP"),  # ordered by id: "cited-1..." sorts before "cited-6..."
        (_STUB_68, "68/2006/QH11"),
    ]
    # ...and the in-manifest law (27/2001/QH10) and the cut-short 88/2018/NĐ-C are not stubs.
    assert _query(migrated_db, "SELECT count(*) AS n FROM documents")[0].n == 3 + 2

    a = _table_row(out, _CITING_A)
    assert int(a["stubs"]) == 2
    # title-only basis + TCVN 7441 + the cut-short 88/2018/NĐ-C; nothing internal missed.
    assert int(a["no_stub"]) == 3
    assert int(a["unresolved"]) == 3
    assert int(a["self_cite_miss"]) == 0
    assert (
        f"WARNING: {_CITING_A} cites 1 document number(s) that failed the complete-number "
        "check, so no stub: 88/2018/NĐ-C"
    ) in out


_DECREE_87_ID = "test-gas-decree-87-2018"
"""Deliberately not the real ``gas-decree-87-2018`` id: issue #51's requirement cluster cites
real units of that exact document, and this fixture's synthetic text never reaches them, so
reusing the id would fail the run's seed pass. Resolution is by *number*, which is the real one."""
_DECREE_87_HTML = (
    "<html><body><p>Điều 1. Phạm vi điều chỉnh</p>"
    "<p>Nghị định này quy định về điều kiện kinh doanh khí dầu mỏ hóa lỏng tại thị trường "
    "Việt Nam, bao gồm các cửa hàng bán lẻ và trạm chiết nạp khí, áp dụng thống nhất trên "
    "toàn quốc cho mọi tổ chức, cá nhân tham gia hoạt động kinh doanh khí.</p></body></html>"
)
_CITES_87_HTML = """<html><body>
<p>Căn cứ Nghị định số 87/2018/N Đ-CP ngày 16 tháng 5 năm 2018 của Chính phủ;</p>
<p>Chương I</p>
<p>QUY ĐỊNH CHUNG</p>
<p>Điều 1. Phạm vi điều chỉnh</p>
<p>Việc chiết nạp khí thực hiện theo quy định của Nghị định số 87/2018/N Đ-CP ngày 16 tháng 5
năm 2018 của Chính phủ về kinh doanh khí, áp dụng đối với mọi trạm chiết nạp trong nước.</p>
</body></html>"""
"""Real text: gas-decree-17-2020 prints "87/2018/N Đ-CP" -- the LPG retail decree's own number
with the stroked Đ split off -- and extraction used to capture it as ``87/2018/N``."""


def test_a_split_87_2018_citation_resolves_to_the_manifest_decree_and_makes_no_ghost_stub(
    migrated_db: str, tmp_path: Path, capsys: Any
):
    # The case that motivated issue #80: "87/2018/N" is NĐ 87/2018, the most important document
    # in the corpus. Keyed on a loose pattern it would become a permanent node shadowing the real
    # one; with the number repaired at extraction it is simply the manifest document.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    entries = [
        ("cites-87-decree-2021", "cites.html", "200/2021/NĐ-CP", _CITES_87_HTML),
        (_DECREE_87_ID, "d87.html", "87/2018/NĐ-CP", _DECREE_87_HTML),
    ]
    documents = [
        {
            "id": doc_id,
            "number": number,
            "title_vi": f"Văn bản {number}",
            "doc_type": "decree",
            "status": "in_force",
            "effective": "2018-08-01",
            "path": f"corpus/{filename}",
        }
        for doc_id, filename, number, _ in entries
    ]
    (corpus / MANIFEST_FILENAME).write_text(json.dumps({"documents": documents}), encoding="utf-8")
    for _, filename, _, html in entries:
        (corpus / filename).write_text(html, encoding="utf-8")

    assert main(_args(corpus, migrated_db, tmp_path)) == 0
    out = capsys.readouterr().out

    # The citation resolves to the manifest document, from the clause and from the basis line.
    edges = _query(
        migrated_db,
        "SELECT src_id, edge_type FROM edges "
        "WHERE dst_id = :dst AND edge_type <> 'belongs_to' ORDER BY src_id",
        {"dst": _DECREE_87_ID},
    )
    assert [(row.src_id, row.edge_type) for row in edges] == [
        ("cites-87-decree-2021", "based_on"),
        ("cites-87-decree-2021#dieu-1", "references"),
    ]
    # ...and no node is made for the truncated number, however it is spelled.
    assert _query(migrated_db, "SELECT id FROM documents WHERE id LIKE 'cited-87-2018%'") == []
    assert _stub_rows(migrated_db) == []
    assert "failed the complete-number check" not in out
    row = _table_row(out, "cites-87-decree-2021")
    assert (int(row["unresolved"]), int(row["no_stub"]), int(row["stubs"])) == (0, 0, 0)


def test_the_table_counts_a_shared_stub_once_and_never_as_an_ingested_document(
    migrated_db: str, tmp_path: Path, capsys: Any
):
    corpus = _write_corpus(tmp_path)

    main(_args(corpus, migrated_db, tmp_path))
    out = capsys.readouterr().out

    b = _table_row(out, _CITING_B)
    total = _table_row(out, "TOTAL")
    assert int(b["stubs"]) == 1  # 136/2020 only
    assert int(total["stubs"]) == 2  # a: {68, 136}, b: {136} -> the union, not 3
    # A run that found two stubs must never look like it ingested two more documents.
    assert "3 ingested, 0 metadata-only" in out


def test_no_stub_holds_any_unit_or_chunk(migrated_db: str, tmp_path: Path):
    main(_args(_write_corpus(tmp_path), migrated_db, tmp_path))

    assert _stub_rows(migrated_db)  # the assertions below are not vacuous
    units = _query(
        migrated_db,
        "SELECT count(*) AS n FROM units u JOIN documents d ON d.id = u.document_id "
        "WHERE d.status = 'unknown'",
    )
    chunks = _query(
        migrated_db,
        "SELECT count(*) AS n FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE d.status = 'unknown'",
    )
    assert units[0].n == 0
    assert chunks[0].n == 0


def test_every_edge_into_a_stub_is_pending_and_regex_sourced(migrated_db: str, tmp_path: Path):
    main(_args(_write_corpus(tmp_path), migrated_db, tmp_path))

    rows = _query(
        migrated_db,
        "SELECT src_id, dst_id, edge_type, review_status, source FROM edges "
        "WHERE dst_id LIKE 'cited-%' ORDER BY src_id, dst_id",
    )
    assert {(row.review_status, row.source) for row in rows} == {("pending", "regex")}
    assert {(row.src_id, row.dst_id, row.edge_type) for row in rows} == {
        # the spaced basis and the plain unit citation, both onto the one 68/2006 stub
        (_CITING_A, _STUB_68, "based_on"),
        (f"{_CITING_A}#dieu-1", _STUB_68, "references"),
        (f"{_CITING_A}#dieu-1", _STUB_136, "references"),
        (f"{_CITING_B}#dieu-1", _STUB_136, "references"),
    }
    # The in-manifest law is still resolved by number to the real document.
    law_edge = _query(
        migrated_db,
        "SELECT edge_type FROM edges WHERE src_id = :src AND dst_id = :dst",
        {"src": _CITING_A, "dst": _LAW_ID},
    )
    assert [row.edge_type for row in law_edge] == ["guided_by"]


def test_reingest_is_idempotent_and_keeps_a_human_decision_on_an_edge_into_a_stub(
    migrated_db: str, tmp_path: Path
):
    corpus = _write_corpus(tmp_path)
    args = _args(corpus, migrated_db, tmp_path)
    main(args)

    edge_sql = "SELECT id FROM edges WHERE src_id = :src AND dst_id = :dst"
    (edge,) = _query(migrated_db, edge_sql, {"src": f"{_CITING_A}#dieu-1", "dst": _STUB_136})

    async def _reject() -> None:
        engine = create_engine(migrated_db)
        try:
            async with session_scope(create_session_factory(engine)) as session:
                await EdgeRepository(session).set_review_status([edge.id], ReviewStatus.REJECTED)
        finally:
            await engine.dispose()

    asyncio.run(_reject())
    stubs_before = _stub_rows(migrated_db)
    counts_before = _query(
        migrated_db,
        "SELECT (SELECT count(*) FROM documents) AS docs, (SELECT count(*) FROM edges) AS edges",
    )[0]

    assert main(args) == 0

    counts_after = _query(
        migrated_db,
        "SELECT (SELECT count(*) FROM documents) AS docs, (SELECT count(*) FROM edges) AS edges",
    )[0]
    assert _stub_rows(migrated_db) == stubs_before  # same ids, same count
    assert (counts_after.docs, counts_after.edges) == (counts_before.docs, counts_before.edges)
    status = _query(migrated_db, "SELECT review_status FROM edges WHERE id = :id", {"id": edge.id})
    assert status[0].review_status == "rejected"  # a person's decision survives the re-ingest


def test_a_dry_run_reports_the_stubs_but_writes_none(migrated_db: str, tmp_path: Path, capsys: Any):
    corpus = _write_corpus(tmp_path)

    assert main(_args(corpus, migrated_db, tmp_path, "--dry-run")) == 0
    out = capsys.readouterr().out

    assert int(_table_row(out, "TOTAL")["stubs"]) == 2
    assert _query(migrated_db, "SELECT count(*) AS n FROM documents")[0].n == 0


# --- database: the real in-force filter ---------------------------------------------------


def _traverse(database_url: str, seed: str, *, include_history: bool) -> dict[str, bool]:
    """Walk one hop of REFERENCES/BASED_ON/GUIDED_BY edges; return ``{id: possibly_superseded}``."""

    async def _run() -> dict[str, bool]:
        engine = create_engine(database_url)
        try:
            async with session_scope(create_session_factory(engine)) as session:
                hits = await GraphTraversalService(session).traverse(
                    [seed],
                    max_hops=1,
                    follow_types=[EdgeType.REFERENCES, EdgeType.BASED_ON, EdgeType.GUIDED_BY],
                    include_history=include_history,
                )
            return {hit.unit_id: hit.possibly_superseded for hit in hits}
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def _approve_every_edge(database_url: str) -> None:
    async def _run() -> None:
        engine = create_engine(database_url)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("UPDATE edges SET review_status = 'approved'"))
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_a_traversal_reaching_a_stub_never_reports_it_as_current_law(
    migrated_db: str, tmp_path: Path
):
    # Real pipeline, real database, real GraphTraversalService -- no mock of the filter.
    main(_args(_write_corpus(tmp_path), migrated_db, tmp_path))
    _approve_every_edge(migrated_db)  # only approved edges are ever walked

    current = _traverse(migrated_db, _CITING_A, include_history=False)
    history = _traverse(migrated_db, _CITING_A, include_history=True)

    # Control: the in-force law the same document cites IS reachable, so the walk
    # works and the stub's absence below is the filter, not an empty graph.
    assert _LAW_ID in current
    assert _STUB_68 not in current
    # With history requested the stub is a real, reachable node -- not dropped as a
    # dangling edge, which is what an id with no documents row would have been.
    assert _STUB_68 in history
    assert _LAW_ID in history

    # And from a unit: the plain "Nghị định số 136/2020/NĐ-CP" citation.
    unit = f"{_CITING_A}#dieu-1"
    assert _STUB_136 not in _traverse(migrated_db, unit, include_history=False)
    assert _STUB_136 in _traverse(migrated_db, unit, include_history=True)
