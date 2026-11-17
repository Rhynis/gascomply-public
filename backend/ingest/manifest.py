"""Load and validate ``corpus/manifest.json``.

The manifest is the machine-readable index of the legal corpus. Source files are
never committed, so loading fails loudly when a document declared ``acquired``
(the default) is missing: every later phase joins on these documents, and a
silently absent statute would show up as a wrong answer rather than an error.

Document paths are written relative to the **repository root** (for example
``corpus/gas-decree-87-2018.pdf``), so the same manifest works from the repository
root and from ``backend/`` via ``--corpus ../corpus``.

A document can also be declared with ``"acquired": false`` (``app.schemas.document
.Document.acquired``, issue #59): part of the corpus, but its text is not on disk
-- permanently for a copyrighted standard that must be purchased, or until someone
downloads it. Loading such a document never requires its file to exist; ``ingest.run``
still gives it a ``documents`` row (so an edge naming it, e.g. ``REPLACED_BY``, still
resolves) but skips it in the text pipeline entirely.

A document whose source spans more than one downloaded file (issue #126 --
decree 105/2025 was published across two issues of the Official Gazette)
declares ``"parts"`` instead of a single ``"path"``: an ordered list of
``{"path": ..., "source_url": ...}`` entries. Exactly one of ``path``/``parts``
is required (``Document._validate_path_xor_parts``); every part's file is
required to exist on disk under the same ``acquired`` rule as a single-``path``
document. Use ``Manifest.parts_of`` to resolve either shape to its ordered
list of absolute paths.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from app.schemas.document import Document

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
DOCUMENTS_KEY = "documents"


class ManifestError(Exception):
    """Raised when a manifest is unreadable, invalid, or references a missing file."""


class Manifest(BaseModel):
    """A validated corpus manifest together with the root its paths resolve against."""

    model_config = ConfigDict(frozen=True)

    root: Path
    documents: tuple[Document, ...]

    def path_of(self, document: Document) -> Path:
        """Return the absolute path of a single-``path`` document's source file.

        Args:
            document: A document belonging to this manifest, declared with
                ``path`` (not ``parts``).

        Returns:
            The document path resolved against the manifest root.

        Raises:
            ValueError: If ``document`` has no single ``path`` -- it was
                declared with ``parts`` instead (issue #126); use
                ``parts_of`` for that case.
        """
        if document.path is None:
            msg = f"{document.id} has no single 'path' (declared with 'parts' instead)"
            raise ValueError(msg)
        return (self.root / document.path).resolve()

    def parts_of(self, document: Document) -> tuple[Path, ...]:
        """Return the absolute paths of a document's source file(s), in reading order.

        A single-``path`` document returns a one-element tuple, exactly
        ``(self.path_of(document),)``. A multi-part document (``parts``,
        issue #126) returns each part's path in the order the manifest
        lists them -- the same order ``ingest.extract.extract_text_parts``
        concatenates their extracted text.

        Args:
            document: A document belonging to this manifest.

        Returns:
            One or more absolute paths, in reading order.
        """
        if document.parts is not None:
            return tuple((self.root / part.path).resolve() for part in document.parts)
        return (self.path_of(document),)


def load_manifest(manifest_path: Path, *, root: Path | None = None) -> Manifest:
    """Read, validate and return the corpus manifest.

    Args:
        manifest_path: Path to ``manifest.json``.
        root: Directory the document paths resolve against. Defaults to the
            manifest's grandparent, i.e. the repository root for
            ``<repo>/corpus/manifest.json``.

    Returns:
        The validated manifest.

    Raises:
        ManifestError: If the file is missing, is not valid JSON, does not match
            the document schema, contains duplicate identifiers, or an
            ``acquired`` document (the default) references a source file that
            does not exist. A document declared ``acquired=False`` never
            raises for a missing file.
    """
    resolved_root = (root or manifest_path.parent.parent).resolve()
    payload = _read_json(manifest_path)
    documents = _parse_documents(payload, manifest_path)
    _reject_duplicate_ids(documents, manifest_path)

    manifest = Manifest(root=resolved_root, documents=tuple(documents))
    _require_source_files(manifest, manifest_path)

    logger.info(
        "Loaded manifest %s: %d documents, root=%s",
        manifest_path,
        len(manifest.documents),
        resolved_root,
    )
    return manifest


def _read_json(manifest_path: Path) -> Any:  # noqa: ANN401 - arbitrary JSON payload
    """Return the parsed JSON payload of the manifest file."""
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        msg = f"Cannot read manifest {manifest_path}: {error}"
        raise ManifestError(msg) from error

    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        msg = f"Manifest {manifest_path} is not valid JSON: {error}"
        raise ManifestError(msg) from error


def _parse_documents(payload: Any, manifest_path: Path) -> list[Document]:  # noqa: ANN401
    """Validate the payload shape and return its documents."""
    if not isinstance(payload, dict) or DOCUMENTS_KEY not in payload:
        msg = f"Manifest {manifest_path} must be an object with a {DOCUMENTS_KEY!r} key"
        raise ManifestError(msg)

    entries = payload[DOCUMENTS_KEY]
    if not isinstance(entries, list) or not entries:
        msg = f"Manifest {manifest_path} lists no documents"
        raise ManifestError(msg)

    documents: list[Document] = []
    for index, entry in enumerate(entries):
        try:
            documents.append(Document.model_validate(entry))
        except ValidationError as error:
            msg = f"Manifest {manifest_path}, document #{index}: {error}"
            raise ManifestError(msg) from error
    return documents


def _reject_duplicate_ids(documents: list[Document], manifest_path: Path) -> None:
    """Raise when two entries share a document identifier."""
    counts = Counter(document.id for document in documents)
    duplicates = sorted(doc_id for doc_id, count in counts.items() if count > 1)
    if duplicates:
        msg = f"Manifest {manifest_path} has duplicate document ids: {', '.join(duplicates)}"
        raise ManifestError(msg)


def _require_source_files(manifest: Manifest, manifest_path: Path) -> None:
    """Raise when an acquired document's source file is missing, listing every offender.

    A document declared ``acquired=False`` (issue #59) is exempt: it is part
    of the corpus but its text is known not to be on disk, so a missing file
    there is expected, not an error.
    """
    missing = [
        f"{document.id} -> {path}"
        for document in manifest.documents
        if document.acquired
        for path in manifest.parts_of(document)
        if not path.is_file()
    ]
    if missing:
        listed = "\n  ".join(missing)
        msg = (
            f"Manifest {manifest_path} references {len(missing)} missing source file(s).\n"
            f"  {listed}\n"
            "Download the documents listed in SOURCES.md into corpus/ before ingesting."
        )
        raise ManifestError(msg)
