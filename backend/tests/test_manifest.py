"""Tests for the corpus manifest loader."""

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas.document import DocStatus, DocType, Document
from ingest.manifest import MANIFEST_FILENAME, ManifestError, load_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "manifest_1doc.json"
DOCUMENT_ID = "gas-decree-87-2018"
DOCUMENT_RELATIVE_PATH = "corpus/gas-decree-87-2018.pdf"


def _build_corpus(root: Path, *, with_source_file: bool = True) -> Path:
    """Lay out ``<root>/corpus/`` from the fixture and return the manifest path."""
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus / MANIFEST_FILENAME
    shutil.copyfile(FIXTURE, manifest_path)
    if with_source_file:
        (root / DOCUMENT_RELATIVE_PATH).write_bytes(b"%PDF-1.4 stub")
    return manifest_path


def _write_manifest(root: Path, documents: list[dict[str, Any]]) -> Path:
    """Write a manifest with the given document entries and return its path."""
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps({"documents": documents}), encoding="utf-8")
    return manifest_path


def _fixture_entry(**overrides: Any) -> dict[str, Any]:
    """Return the fixture document entry with optional field overrides."""
    entry: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))["documents"][0]
    entry.update(overrides)
    return entry


def test_loads_the_fixture_into_typed_documents(tmp_path: Path) -> None:
    manifest_path = _build_corpus(tmp_path)

    manifest = load_manifest(manifest_path)

    assert len(manifest.documents) == 1
    document = manifest.documents[0]
    assert document.id == DOCUMENT_ID
    assert document.number == "87/2018/NĐ-CP"
    assert document.doc_type is DocType.DECREE
    assert document.status is DocStatus.IN_FORCE
    assert document.issued == date(2018, 6, 15)
    assert document.effective == date(2018, 8, 1)
    assert document.meta["note"].startswith("Replacement decree not confirmed")


def test_paths_resolve_against_the_repository_root(tmp_path: Path) -> None:
    manifest_path = _build_corpus(tmp_path)

    manifest = load_manifest(manifest_path)

    assert manifest.root == tmp_path.resolve()
    assert manifest.path_of(manifest.documents[0]) == (tmp_path / DOCUMENT_RELATIVE_PATH).resolve()


def test_explicit_root_overrides_the_default(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    manifest_path = _build_corpus(elsewhere)
    (tmp_path / "corpus").mkdir(exist_ok=True)
    (tmp_path / DOCUMENT_RELATIVE_PATH).write_bytes(b"%PDF-1.4 stub")

    manifest = load_manifest(manifest_path, root=tmp_path)

    assert manifest.path_of(manifest.documents[0]) == (tmp_path / DOCUMENT_RELATIVE_PATH).resolve()


def test_missing_source_file_raises_a_clear_error(tmp_path: Path) -> None:
    manifest_path = _build_corpus(tmp_path, with_source_file=False)

    with pytest.raises(ManifestError) as error:
        load_manifest(manifest_path)

    message = str(error.value)
    assert "1 missing source file" in message
    assert DOCUMENT_ID in message
    assert "SOURCES.md" in message


def test_a_not_acquired_document_loads_fine_with_no_file_on_disk(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_fixture_entry(acquired=False)])

    manifest = load_manifest(manifest_path)

    assert manifest.documents[0].acquired is False


def test_acquired_defaults_to_true_and_still_requires_the_file(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_fixture_entry()])

    with pytest.raises(ManifestError, match="1 missing source file"):
        load_manifest(manifest_path)


def test_a_not_acquired_document_does_not_mask_an_acquired_ones_missing_file(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            _fixture_entry(),
            _fixture_entry(id="pccc-law-2024", path="corpus/pccc-law-2024.pdf", acquired=False),
        ],
    )

    with pytest.raises(ManifestError) as error:
        load_manifest(manifest_path)

    message = str(error.value)
    assert "1 missing source file" in message
    assert DOCUMENT_ID in message
    assert "pccc-law-2024" not in message


def test_every_missing_file_is_listed_at_once(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            _fixture_entry(),
            _fixture_entry(id="pccc-law-2024", path="corpus/pccc-law-2024.pdf"),
        ],
    )

    with pytest.raises(ManifestError) as error:
        load_manifest(manifest_path)

    assert "2 missing source file" in str(error.value)


def test_missing_manifest_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="Cannot read manifest"):
        load_manifest(tmp_path / "corpus" / MANIFEST_FILENAME)


def test_malformed_json_raises(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    manifest_path = corpus / MANIFEST_FILENAME
    manifest_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(manifest_path)


@pytest.mark.parametrize("payload", [{}, [], {"docs": []}])
def test_payload_without_documents_key_raises(tmp_path: Path, payload: Any) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    manifest_path = corpus / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="must be an object with a 'documents' key"):
        load_manifest(manifest_path)


def test_empty_document_list_raises(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [])

    with pytest.raises(ManifestError, match="lists no documents"):
        load_manifest(manifest_path)


def test_duplicate_document_ids_are_rejected(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_fixture_entry(), _fixture_entry()])

    with pytest.raises(ManifestError, match=f"duplicate document ids: {DOCUMENT_ID}"):
        load_manifest(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "Gas Decree 87"),
        ("doc_type", "ordinance"),
        ("status", "in_force / amending"),
        ("number", ""),
        ("title_vi", ""),
        ("path", "/etc/passwd"),
        ("path", "../../secrets.pdf"),
    ],
)
def test_invalid_field_values_are_rejected(tmp_path: Path, field: str, value: str) -> None:
    manifest_path = _write_manifest(tmp_path, [_fixture_entry(**{field: value})])

    with pytest.raises(ManifestError, match="document #0"):
        load_manifest(manifest_path)


def test_unknown_fields_are_rejected(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_fixture_entry(replaced_by="pccc-decree-105-2025")])

    with pytest.raises(ManifestError, match="document #0"):
        load_manifest(manifest_path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2017", date(2017, 1, 1)), ("2013-05", date(2013, 5, 1)), ("2012-01-31", date(2012, 1, 31))],
)
def test_partial_dates_widen_to_their_earliest_day(
    tmp_path: Path, raw: str, expected: date
) -> None:
    entry = _fixture_entry(effective=raw, path="corpus/lpg-store-tcvn-6223-2017.pdf")
    manifest_path = _write_manifest(tmp_path, [entry])
    (tmp_path / entry["path"]).write_bytes(b"stub")

    manifest = load_manifest(manifest_path)

    assert manifest.documents[0].effective == expected


def test_optional_dates_may_be_absent(tmp_path: Path) -> None:
    entry = _fixture_entry()
    for optional_field in ("issued", "effective", "fetched", "source_url"):
        entry.pop(optional_field)
    manifest_path = _write_manifest(tmp_path, [entry])
    (tmp_path / DOCUMENT_RELATIVE_PATH).write_bytes(b"stub")

    manifest = load_manifest(manifest_path)

    document = manifest.documents[0]
    assert document.issued is None
    assert document.effective is None
    assert document.fetched is None
    assert document.source_url is None


def test_documents_are_immutable(tmp_path: Path) -> None:
    manifest = load_manifest(_build_corpus(tmp_path))

    with pytest.raises(ValidationError):
        manifest.documents[0].status = DocStatus.EXPIRED  # type: ignore[misc]


def test_document_accepts_native_date_objects() -> None:
    document = Document.model_validate(_fixture_entry(effective=date(2018, 8, 1)))

    assert document.effective == date(2018, 8, 1)


# --- multi-part documents (issue #126) -------------------------------------------


_PART_ONE = "corpus/gas-decree-87-2018.pdf"
_PART_TWO = "corpus/gas-decree-87-2018-part2.pdf"


def _multi_part_entry(**overrides: Any) -> dict[str, Any]:
    """Return the fixture entry rewritten to declare ``parts`` instead of ``path``."""
    entry = _fixture_entry(**overrides)
    del entry["path"]
    entry["parts"] = [{"path": _PART_ONE}, {"path": _PART_TWO}]
    return entry


def test_a_multi_part_document_loads_with_its_parts_in_order(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_multi_part_entry()])
    (tmp_path / _PART_ONE).write_bytes(b"part one")
    (tmp_path / _PART_TWO).write_bytes(b"part two")

    manifest = load_manifest(manifest_path)

    document = manifest.documents[0]
    assert document.path is None
    assert [part.path for part in document.parts] == [_PART_ONE, _PART_TWO]
    assert manifest.parts_of(document) == (
        (tmp_path / _PART_ONE).resolve(),
        (tmp_path / _PART_TWO).resolve(),
    )


def test_a_multi_part_documents_source_url_is_recorded_per_part(tmp_path: Path) -> None:
    entry = _multi_part_entry()
    entry["parts"] = [
        {"path": _PART_ONE, "source_url": "https://example.com/part1.pdf"},
        {"path": _PART_TWO, "source_url": "https://example.com/part2.pdf"},
    ]
    manifest_path = _write_manifest(tmp_path, [entry])
    (tmp_path / _PART_ONE).write_bytes(b"part one")
    (tmp_path / _PART_TWO).write_bytes(b"part two")

    manifest = load_manifest(manifest_path)

    parts = manifest.documents[0].parts
    assert parts[0].source_url == "https://example.com/part1.pdf"
    assert parts[1].source_url == "https://example.com/part2.pdf"


def test_path_and_parts_together_is_rejected(tmp_path: Path) -> None:
    entry = _fixture_entry(parts=[{"path": _PART_TWO}])
    manifest_path = _write_manifest(tmp_path, [entry])

    with pytest.raises(ManifestError, match="document #0"):
        load_manifest(manifest_path)


def test_neither_path_nor_parts_is_rejected(tmp_path: Path) -> None:
    entry = _fixture_entry()
    del entry["path"]
    manifest_path = _write_manifest(tmp_path, [entry])

    with pytest.raises(ManifestError, match="document #0"):
        load_manifest(manifest_path)


def test_an_empty_parts_list_is_rejected(tmp_path: Path) -> None:
    entry = _multi_part_entry()
    entry["parts"] = []
    manifest_path = _write_manifest(tmp_path, [entry])

    with pytest.raises(ManifestError, match="document #0"):
        load_manifest(manifest_path)


def test_a_missing_part_is_rejected_and_named(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_multi_part_entry()])
    (tmp_path / _PART_ONE).write_bytes(b"part one")
    # _PART_TWO deliberately left missing.

    with pytest.raises(ManifestError) as error:
        load_manifest(manifest_path)

    message = str(error.value)
    assert "1 missing source file" in message
    assert _PART_TWO in message


def test_single_path_documents_are_unaffected_by_the_parts_feature(tmp_path: Path) -> None:
    manifest_path = _build_corpus(tmp_path)

    manifest = load_manifest(manifest_path)

    document = manifest.documents[0]
    assert document.path == DOCUMENT_RELATIVE_PATH
    assert document.parts is None
    assert manifest.parts_of(document) == (manifest.path_of(document),)
    assert document.storage_path == DOCUMENT_RELATIVE_PATH


def test_a_multi_part_documents_storage_path_joins_its_parts(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_multi_part_entry()])
    (tmp_path / _PART_ONE).write_bytes(b"part one")
    (tmp_path / _PART_TWO).write_bytes(b"part two")

    manifest = load_manifest(manifest_path)

    assert manifest.documents[0].storage_path == f"{_PART_ONE};{_PART_TWO}"
