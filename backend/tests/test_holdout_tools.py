"""Tests for ``eval.holdout_tools`` (issue #189, plus the PR #190 review's fix round).

Five groups:

1. The blind-check flag logic (``_flags_for_row``/``check_rows``), on hand-built rows, no
   database at all -- one test per flag, plus a clean row that gets none. ``NO_SOURCE``,
   ``REFUSAL_FLAG``, ``MULTIHOP_ONE_SOURCE`` and ``UNEXPECTED_SOURCE`` are unreachable through a
   row ``GoldRow`` itself validates, so those rows are built with ``GoldRow.model_construct``,
   bypassing the validator that would otherwise forbid them -- the ported flag code is still
   exercised, even though it is now dead through any row that ever reaches ``parse_gold_set``.
2. ``drop``/``append`` on temporary files, including every refusal case, leaving the file
   byte-identical when refused.
3. Export rendering order, on hand-built ``_UnitRow`` tuples, no database.
4. Privacy: a fixture whose question and reference answer carry a marker string; the marker
   must appear in neither stdout nor stderr, captured at the file-descriptor level
   (``capfd``), the way ``tests/test_holdout.py`` does.
5. Usage errors: a missing database URL, or ``check``'s own bad arguments -- both a fixed
   sentence with no row data, so ``main`` prints the message in full (not a privacy test).

The database-backed tests (``check``'s DB read, ``export-corpus``) seed ``documents``/``units``
directly with SQL rather than running the full ingest pipeline -- ``holdout_tools`` only ever
reads those two tables, so a fixture corpus and the real parser would test nothing more here.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from app.core.config import get_settings
from app.core.db import create_engine
from eval import holdout_tools
from eval.gold import GoldCategory, GoldRow, load_gold_set

LEAK = "zqx"
"""Every question and reference answer in the privacy fixtures below contains this."""


def _row(**overrides: Any) -> GoldRow:
    """Build a valid ``compliance`` ``GoldRow``, overriding whatever a test needs to vary."""
    fields: dict[str, Any] = {
        "id": "compliance-row",
        "category": GoldCategory.COMPLIANCE,
        "question": "Diện tích tối thiểu của cửa hàng là bao nhiêu?",
        "reference_answer": "Diện tích tối thiểu là 12 mét vuông.",
        "source_unit_ids": ("gas-decree-87-2018#dieu-7",),
    }
    fields.update(overrides)
    return GoldRow(**fields)


def _unvalidated_row(**overrides: Any) -> GoldRow:
    """Build a ``GoldRow`` bypassing its own category/flags validator (see module docstring)."""
    fields: dict[str, Any] = {
        "id": "unvalidated-row",
        "category": GoldCategory.COMPLIANCE,
        "question": "q",
        "reference_answer": "a",
        "source_unit_ids": (),
        "multi_hop": False,
        "expected_refusal": False,
        "answer_must_match": (),
        "answer_must_not_match": (),
    }
    fields.update(overrides)
    return GoldRow.model_construct(**fields)


# --- 1. flag logic (no database) --------------------------------------------------------------


def test_no_source_flag_on_an_answerable_row_with_no_source():
    row = _unvalidated_row(category=GoldCategory.COMPLIANCE, source_unit_ids=())

    flags = holdout_tools._flags_for_row(row, frozenset(), {}, {})

    assert flags == ["NO_SOURCE"]


def test_missing_unit_flag_when_a_source_id_is_not_in_the_database():
    row = _row(source_unit_ids=("gas-decree-87-2018#dieu-7",))

    flags = holdout_tools._flags_for_row(row, frozenset(), {}, {})

    assert flags == ["MISSING_UNIT:1"]


def test_ref_doc_mismatch_flag_when_the_answer_names_a_different_document():
    row = _row(
        source_unit_ids=("gas-decree-87-2018#dieu-7",),
        reference_answer="Theo Nghị định 99/2099/NĐ-CP, diện tích tối thiểu là mười hai mét.",
    )
    unit = row.source_unit_ids[0]

    flags = holdout_tools._flags_for_row(
        row,
        frozenset([unit]),
        {"gas-decree-87-2018": "87/2018/NĐ-CP"},
        {unit: row.reference_answer},  # full overlap: isolates the one flag under test
    )

    assert flags == ["REF_DOC_MISMATCH"]


def test_ref_dieu_mismatch_flag_when_the_answer_names_a_different_dieu():
    row = _row(
        source_unit_ids=("gas-decree-87-2018#dieu-7",),
        reference_answer="Theo Điều 5, diện tích tối thiểu là mười hai mét vuông.",
    )
    unit = row.source_unit_ids[0]

    flags = holdout_tools._flags_for_row(row, frozenset([unit]), {}, {unit: row.reference_answer})

    assert flags == ["REF_DIEU_MISMATCH"]


def test_ref_khoan_mismatch_flag_when_the_answer_names_a_different_khoan():
    row = _row(
        source_unit_ids=("gas-decree-87-2018#dieu-7#khoan-2",),
        reference_answer="Theo Khoản 5, diện tích tối thiểu là mười hai mét vuông.",
    )
    unit = row.source_unit_ids[0]

    flags = holdout_tools._flags_for_row(row, frozenset([unit]), {}, {unit: row.reference_answer})

    assert flags == ["REF_KHOAN_MISMATCH"]


def test_pattern_ref_fail_flag_when_the_rows_own_pattern_rejects_its_own_answer():
    row = _row(
        source_unit_ids=("gas-decree-87-2018#dieu-7",),
        reference_answer="Diện tích tối thiểu là mười hai mét vuông.",
        answer_must_match=["không xuất hiện ở đây"],
    )
    unit = row.source_unit_ids[0]

    flags = holdout_tools._flags_for_row(row, frozenset([unit]), {}, {unit: row.reference_answer})

    assert flags == ["PATTERN_REF_FAIL"]


def test_low_overlap_flag_when_the_answer_shares_few_words_with_its_sources():
    row = _row(
        source_unit_ids=("gas-decree-87-2018#dieu-7",),
        reference_answer="Diện tích tối thiểu là mười hai mét vuông.",
    )
    unit = row.source_unit_ids[0]

    flags = holdout_tools._flags_for_row(
        row, frozenset([unit]), {}, {unit: "Nội dung hoàn toàn khác không liên quan gì cả."}
    )

    assert flags == ["LOW_OVERLAP:0.00"]


def test_multihop_one_source_flag_on_a_multi_hop_row_with_one_source():
    row = _unvalidated_row(
        category=GoldCategory.MULTI_HOP,
        multi_hop=True,
        reference_answer="Diện tích tối thiểu là mười hai mét vuông.",
        source_unit_ids=("gas-decree-87-2018#dieu-7",),
    )
    unit = row.source_unit_ids[0]

    flags = holdout_tools._flags_for_row(row, frozenset([unit]), {}, {unit: row.reference_answer})

    assert flags == ["MULTIHOP_ONE_SOURCE"]


def test_unexpected_source_flag_on_a_refusal_row_that_names_sources():
    row = _unvalidated_row(
        category=GoldCategory.REFUSAL,
        expected_refusal=True,
        source_unit_ids=("gas-decree-87-2018#dieu-7",),
    )

    flags = holdout_tools._flags_for_row(row, frozenset(), {}, {})

    assert flags == ["UNEXPECTED_SOURCE"]


def test_not_escalated_flag_on_an_emergency_row_check_emergency_does_not_escalate():
    row = _row(
        id="emergency-row",
        category=GoldCategory.EMERGENCY,
        question="Giá gas hôm nay là bao nhiêu?",
        reference_answer="KHẨN CẤP: ra khỏi nơi này ngay và gọi 114.",
        source_unit_ids=(),
    )

    flags = holdout_tools._flags_for_row(row, frozenset(), {}, {})

    assert flags == ["NOT_ESCALATED"]


def test_refusal_flag_when_expected_refusal_disagrees_with_the_category():
    row = _unvalidated_row(
        category=GoldCategory.COMPLIANCE,
        expected_refusal=True,
        reference_answer="Diện tích tối thiểu là mười hai mét vuông.",
        source_unit_ids=("gas-decree-87-2018#dieu-7",),
    )
    unit = row.source_unit_ids[0]

    flags = holdout_tools._flags_for_row(row, frozenset([unit]), {}, {unit: row.reference_answer})

    assert flags == ["REFUSAL_FLAG"]


def test_a_clean_row_gets_no_flag_at_all():
    row = _row()
    unit = row.source_unit_ids[0]

    flags = holdout_tools._flags_for_row(row, frozenset([unit]), {}, {unit: row.reference_answer})

    assert flags == []


def test_check_rows_numbers_by_file_position_and_skips_clean_rows():
    clean = _row(id="clean-1")
    flagged = _row(
        id="emergency-row",
        category=GoldCategory.EMERGENCY,
        question="Giá gas hôm nay là bao nhiêu?",
        reference_answer="KHẨN CẤP: ra khỏi nơi này ngay và gọi 114.",
        source_unit_ids=(),
    )
    unit = clean.source_unit_ids[0]

    result = holdout_tools.check_rows(
        (clean, flagged, clean), frozenset([unit]), {}, {unit: clean.reference_answer}
    )

    assert result == [("h02", "emergency", ["NOT_ESCALATED"])]


# --- 2. drop / append --------------------------------------------------------------------------

_GOLD_YAML = """\
- id: row-one
  category: compliance
  question: "Câu hỏi một?"
  reference_answer: "Trả lời một."
  source_unit_ids: ["doc-a#dieu-1"]
- id: row-two
  category: compliance
  question: "Câu hỏi hai?"
  reference_answer: "Trả lời hai."
  source_unit_ids: ["doc-a#dieu-2"]
- id: row-three
  category: refusal
  question: "Câu hỏi ba?"
  reference_answer: "Ngoài phạm vi."
  expected_refusal: true
- id: row-four
  category: emergency
  question: "Ngửi thấy mùi gas"
  reference_answer: "KHẨN CẤP: ra khỏi nơi này ngay và gọi 114."
"""

_NEW_ROWS_YAML = """\
- id: row-five
  category: compliance
  question: "Câu hỏi năm?"
  reference_answer: "Trả lời năm."
  source_unit_ids: ["doc-a#dieu-5"]
"""


def test_drop_removes_the_given_positions_and_keeps_a_backup(tmp_path: Path):
    path = tmp_path / "gold.yaml"
    path.write_text(_GOLD_YAML, encoding="utf-8")

    code = holdout_tools.main(["drop", str(path), "h02", "h04"])

    assert code == 0
    loaded = load_gold_set(path)
    assert [row.id for row in loaded] == ["row-one", "row-three"]
    backup = path.with_name(path.name + ".bak")
    assert backup.read_text(encoding="utf-8") == _GOLD_YAML


def test_drop_refuses_an_out_of_range_position_and_leaves_the_file_untouched(tmp_path: Path):
    path = tmp_path / "gold.yaml"
    path.write_text(_GOLD_YAML, encoding="utf-8")

    code = holdout_tools.main(["drop", str(path), "h99"])

    assert code == 2
    assert path.read_text(encoding="utf-8") == _GOLD_YAML
    assert not path.with_name(path.name + ".bak").exists()


def test_append_adds_new_rows_and_revalidates_the_whole_file(tmp_path: Path):
    path, new_rows = tmp_path / "gold.yaml", tmp_path / "new_rows.yaml"
    path.write_text(_GOLD_YAML, encoding="utf-8")
    new_rows.write_text(_NEW_ROWS_YAML, encoding="utf-8")

    code = holdout_tools.main(["append", str(path), str(new_rows)])

    assert code == 0
    loaded = load_gold_set(path)
    assert [row.id for row in loaded] == ["row-one", "row-two", "row-three", "row-four", "row-five"]


def test_append_of_a_duplicate_id_leaves_the_file_byte_identical(tmp_path: Path):
    path, new_rows = tmp_path / "gold.yaml", tmp_path / "new_rows.yaml"
    path.write_text(_GOLD_YAML, encoding="utf-8")
    new_rows.write_text(
        '- id: row-one\n  category: compliance\n  question: "dup?"\n'
        '  reference_answer: "dup."\n  source_unit_ids: ["doc-a#dieu-9"]\n',
        encoding="utf-8",
    )
    before = path.read_bytes()

    code = holdout_tools.main(["append", str(path), str(new_rows)])

    assert code == 2
    assert path.read_bytes() == before
    assert not path.with_name(path.name + ".bak").exists()


def test_append_of_a_row_that_fails_the_schema_leaves_the_file_byte_identical(tmp_path: Path):
    path, new_rows = tmp_path / "gold.yaml", tmp_path / "new_rows.yaml"
    path.write_text(_GOLD_YAML, encoding="utf-8")
    new_rows.write_text(
        "- id: row-bad\n  category: nonsense\n  question: q\n  reference_answer: a\n",
        encoding="utf-8",
    )
    before = path.read_bytes()

    code = holdout_tools.main(["append", str(path), str(new_rows)])

    assert code == 2
    assert path.read_bytes() == before
    assert not path.with_name(path.name + ".bak").exists()


# --- 3. export rendering order (no database) ----------------------------------------------


def _unit(**overrides: Any) -> "holdout_tools._UnitRow":
    fields: dict[str, Any] = {
        "id": "doc#dieu-1",
        "parent_id": None,
        "ordinal": 1,
        "unit_type": "article",
        "heading": None,
        "body": "",
        "chapter_no": None,
        "dieu_no": None,
        "khoan_no": None,
        "diem": None,
        "appendix_no": None,
        "item_no": None,
        "appendix_section": None,
    }
    fields.update(overrides)
    return holdout_tools._UnitRow(**fields)


def test_export_rendering_orders_siblings_by_ordinal_not_by_id_text():
    article = _unit(
        id="doc#dieu-1",
        parent_id=None,
        ordinal=1,
        unit_type="article",
        dieu_no=1,
        heading="Tiêu đề",
    )
    khoan_10 = _unit(
        id="doc#dieu-1#khoan-10",
        parent_id="doc#dieu-1",
        ordinal=3,
        unit_type="clause",
        dieu_no=1,
        khoan_no=10,
        body="Nội dung khoản 10.",
    )
    khoan_2 = _unit(
        id="doc#dieu-1#khoan-2",
        parent_id="doc#dieu-1",
        ordinal=2,
        unit_type="clause",
        dieu_no=1,
        khoan_no=2,
        body="Nội dung khoản 2.",
    )

    lines = holdout_tools._ordered_unit_lines([khoan_10, article, khoan_2])  # deliberately shuffled

    assert lines == [
        "[doc#dieu-1] Điều 1 — Tiêu đề: ",
        "[doc#dieu-1#khoan-2] Khoản 2: Nội dung khoản 2.",
        "[doc#dieu-1#khoan-10] Khoản 10: Nội dung khoản 10.",
    ]


def test_export_rendering_still_lines_a_unit_with_a_dangling_parent_reference():
    orphan = _unit(id="doc#dieu-9", parent_id="doc#chuong-missing", ordinal=1, dieu_no=9)

    lines = holdout_tools._ordered_unit_lines([orphan])

    assert lines == ["[doc#dieu-9] Điều 9: "]


def test_every_unit_id_appears_exactly_once_even_with_many_siblings():
    rows = [_unit(id="doc#dieu-1", parent_id=None, ordinal=1, dieu_no=1)]
    rows.extend(
        _unit(
            id=f"doc#dieu-1#khoan-{n}",
            parent_id="doc#dieu-1",
            ordinal=n + 1,
            unit_type="clause",
            dieu_no=1,
            khoan_no=n,
            body=f"body {n}",
        )
        for n in range(1, 12)
    )

    lines = holdout_tools._ordered_unit_lines(rows)

    assert len(lines) == len(rows) == len(set(lines))
    assert [line.split("]")[0] for line in lines] == [f"[{row.id}" for row in rows]


@pytest.mark.parametrize(
    ("unit_type_kwargs", "expected"),
    [
        (
            {"unit_type": "chapter", "chapter_no": 1, "heading": "QUY ĐỊNH CHUNG"},
            "Chương 1 — QUY ĐỊNH CHUNG",
        ),
        ({"unit_type": "point", "diem": "a"}, "Điểm a"),
        ({"unit_type": "appendix", "appendix_no": 6}, "Phụ lục VI"),
        ({"unit_type": "appendix_item", "item_no": 5}, "Mục 5"),
        ({"unit_type": "appendix_point", "item_no": 2, "diem": "b"}, "Mục 2, Điểm b"),
        (
            {"unit_type": "appendix_subitem", "id": "doc#phu-luc-6#muc-14#muc-2", "item_no": 14},
            "Mục 14.2",
        ),
        ({"unit_type": "appendix_form", "id": "doc#phu-luc-1#mau-02"}, "Mẫu 02"),
    ],
)
def test_unit_label_formats_each_structural_type(unit_type_kwargs: dict[str, Any], expected: str):
    assert holdout_tools._unit_label(_unit(**unit_type_kwargs)) == expected


# --- 4. privacy ---------------------------------------------------------------------------------

_MALFORMED_WITH_MARKER = f"- id: [zqx-unclosed\n  question: {LEAK}-line\n"
_LEAKY_GOLD_YAML = f"""\
- id: leaky-row
  category: compliance
  question: "{LEAK}-question: diện tích tối thiểu là bao nhiêu?"
  reference_answer: "{LEAK}-answer: 12 m2."
  source_unit_ids: ["{LEAK}-doc#dieu-7"]
"""


def test_check_on_malformed_yaml_prints_only_the_type_name_and_no_marker(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    path = tmp_path / "bad.yaml"
    path.write_text(_MALFORMED_WITH_MARKER, encoding="utf-8")

    code = holdout_tools.main(["check", "--yaml", str(path)])

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert code == 2
    assert "failed:" in output
    assert LEAK not in output


def test_drop_on_malformed_yaml_prints_only_the_type_name_and_no_marker(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    path = tmp_path / "bad.yaml"
    path.write_text(_MALFORMED_WITH_MARKER, encoding="utf-8")

    code = holdout_tools.main(["drop", str(path), "h01"])

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert code == 2
    assert LEAK not in output


def test_append_on_malformed_new_rows_prints_only_the_type_name_and_no_marker(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    path, new_rows = tmp_path / "gold.yaml", tmp_path / "bad.yaml"
    path.write_text(_GOLD_YAML, encoding="utf-8")
    new_rows.write_text(_MALFORMED_WITH_MARKER, encoding="utf-8")

    code = holdout_tools.main(["append", str(path), str(new_rows)])

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert code == 2
    assert LEAK not in output


def test_drop_of_a_row_with_a_marker_prints_only_counts_and_no_marker(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    path = tmp_path / "leaky.yaml"
    path.write_text(_LEAKY_GOLD_YAML, encoding="utf-8")

    code = holdout_tools.main(["drop", str(path), "h01"])

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert code == 0  # an empty list still loads through parse_gold_set
    assert LEAK not in output


def test_append_of_a_row_with_a_marker_prints_only_counts_and_no_marker(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    path, new_rows = tmp_path / "gold.yaml", tmp_path / "leaky.yaml"
    path.write_text(_GOLD_YAML, encoding="utf-8")
    new_rows.write_text(_LEAKY_GOLD_YAML, encoding="utf-8")

    code = holdout_tools.main(["append", str(path), str(new_rows)])

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert code == 0
    assert LEAK not in output


# --- usage errors: a missing database URL, or check's own bad arguments -----------------------
#
# Both are fixed sentences with no row data (PR #190 review), unlike every other failure here,
# which prints only the exception's type name -- so these print the message in full, and are not
# privacy tests. ``get_settings()`` is process-wide cached (``lru_cache``), so clearing the env
# var alone is not enough once it has been read once already (as ``tests/test_ingest_cli.py``'s
# own ``_no_ambient_database`` fixture notes) -- the cache has to be dropped too, in both
# directions, so a database-backed test running earlier or later in the same session is
# unaffected.


def test_check_without_a_database_url_prints_the_fixed_usage_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    path = tmp_path / "gold.yaml"
    path.write_text(_GOLD_YAML, encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here, unlike a shell that happens to have one
    get_settings.cache_clear()

    try:
        code = holdout_tools.main(["check", "--yaml", str(path)])
    finally:
        get_settings.cache_clear()

    captured = capfd.readouterr()
    assert code == 2
    assert captured.out == "failed: no database URL: pass --database-url or set DATABASE_URL\n"
    assert captured.err == ""


def test_check_with_both_encrypted_and_yaml_prints_the_fixed_usage_sentence(
    capfd: pytest.CaptureFixture[str],
):
    code = holdout_tools.main(
        ["check", "--yaml", "gold.yaml", "--encrypted", "x.p7m", "--key", "x.key"]
    )

    captured = capfd.readouterr()
    assert code == 2
    assert captured.out == "failed: --yaml cannot be combined with --encrypted/--key\n"


# --- database-backed: check's own DB read, and export-corpus -----------------------------------


async def _seed(
    database_url: str, documents: Sequence[dict[str, Any]], units: Sequence[dict[str, Any]]
) -> None:
    """Insert ``documents``/``units`` rows directly -- the only two tables these tools read."""
    engine = create_engine(database_url)
    try:
        async with engine.begin() as conn:
            for document in documents:
                await conn.execute(
                    text(
                        "INSERT INTO documents (id, number, title_vi, doc_type, status) "
                        "VALUES (:id, :number, :title_vi, 'decree', :status)"
                    ),
                    {"title_vi": document.get("number", ""), **document},
                )
            for unit in units:
                await conn.execute(
                    text(
                        "INSERT INTO units (id, document_id, unit_type, parent_id, heading, body, "
                        "ordinal, chapter_no, dieu_no, khoan_no, diem, appendix_no, item_no, "
                        "appendix_section) VALUES (:id, :document_id, :unit_type, :parent_id, "
                        ":heading, :body, :ordinal, :chapter_no, :dieu_no, :khoan_no, :diem, "
                        ":appendix_no, :item_no, :appendix_section)"
                    ),
                    {
                        "chapter_no": None,
                        "dieu_no": None,
                        "khoan_no": None,
                        "diem": None,
                        "appendix_no": None,
                        "item_no": None,
                        "appendix_section": None,
                        "parent_id": None,
                        "heading": None,
                        **unit,
                    },
                )
    finally:
        await engine.dispose()


def test_check_with_a_marker_row_reads_the_database_and_prints_no_marker(
    migrated_db: str, tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    asyncio.run(
        _seed(
            migrated_db,
            [{"id": f"{LEAK}-doc", "number": "87/2018/NĐ-CP", "status": "in_force"}],
            [
                {
                    "id": f"{LEAK}-doc#dieu-7",
                    "document_id": f"{LEAK}-doc",
                    "unit_type": "article",
                    "body": f"{LEAK}-clause text",
                    "ordinal": 1,
                }
            ],
        )
    )
    path = tmp_path / "gold.yaml"
    path.write_text(_LEAKY_GOLD_YAML, encoding="utf-8")

    code = holdout_tools.main(["check", "--yaml", str(path), "--database-url", migrated_db])

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert code == 0
    assert "rows: 1; flagged:" in output
    assert LEAK not in output


def test_export_corpus_writes_the_index_and_one_file_per_document_with_units(
    migrated_db: str, tmp_path: Path
):
    asyncio.run(
        _seed(
            migrated_db,
            [
                {"id": "doc-a", "number": "87/2018/NĐ-CP", "status": "in_force"},
                {"id": "doc-stub", "number": "99/2099/NĐ-CP", "status": "unknown"},
            ],
            [
                {
                    "id": "doc-a#chuong-1",
                    "document_id": "doc-a",
                    "unit_type": "chapter",
                    "heading": "QUY ĐỊNH CHUNG",
                    "body": "",
                    "ordinal": 1,
                    "parent_id": None,
                },
                {
                    "id": "doc-a#dieu-1",
                    "document_id": "doc-a",
                    "unit_type": "article",
                    "dieu_no": 1,
                    "heading": "Phạm vi",
                    "body": "nội dung điều một",
                    "ordinal": 2,
                    "parent_id": "doc-a#chuong-1",
                },
                {
                    "id": "doc-a#dieu-1#khoan-1",
                    "document_id": "doc-a",
                    "unit_type": "clause",
                    "dieu_no": 1,
                    "khoan_no": 1,
                    "body": "nội dung khoản một",
                    "ordinal": 3,
                    "parent_id": "doc-a#dieu-1",
                },
            ],
        )
    )
    out_dir = tmp_path / "corpus-export"

    code = holdout_tools.main(["export-corpus", str(out_dir), "--database-url", migrated_db])

    assert code == 0
    index = (out_dir / "00-index.txt").read_text(encoding="utf-8")
    assert "doc-a" in index and "có nội dung" in index
    assert "doc-stub" in index and "KHÔNG có nội dung trong hệ thống" in index
    assert len(index.strip().splitlines()) == 2
    assert not (out_dir / "doc-stub.txt").exists()
    document_text = (out_dir / "doc-a.txt").read_text(encoding="utf-8")
    assert document_text.count("[doc-a#chuong-1]") == 1
    assert document_text.count("[doc-a#dieu-1]") == 1
    assert document_text.count("[doc-a#dieu-1#khoan-1]") == 1
    assert document_text.count("nội dung điều một") == 1
    assert document_text.count("nội dung khoản một") == 1
    # parent before child
    assert document_text.index("[doc-a#dieu-1]") < document_text.index("[doc-a#dieu-1#khoan-1]")
