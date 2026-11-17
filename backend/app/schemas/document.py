"""Typed corpus documents, mirroring the ``documents`` table (build plan A5.3)."""

import re
from datetime import date
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.ids import document_id, stub_document_id

_YEAR_PATTERN = re.compile(r"^(?P<year>\d{4})$")
_YEAR_MONTH_PATTERN = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})$")


class DocType(StrEnum):
    """Type of legal-normative document, as used by the corpus manifest.

    ``UNKNOWN`` is the type of a ``DocumentStub`` (issue #67): a document
    known only from a citation, whose type has not been read. The number's own
    suffix does not settle it -- ``QH15`` is a law *or* a National Assembly
    resolution -- so nothing is guessed from it.
    """

    LAW = "law"
    DECREE = "decree"
    CIRCULAR = "circular"
    TCVN = "tcvn"
    QCVN = "qcvn"
    UNKNOWN = "unknown"


class DocStatus(StrEnum):
    """In-force state of a document.

    ``AMENDING`` marks a document that is in force but known to be under
    amendment; the effective-date filter treats anything other than
    ``IN_FORCE`` as superseded (A5.2).

    ``UNKNOWN`` (issue #67) is the status of a ``DocumentStub``: nobody has
    read the document, so nobody knows whether it is in force. It is not
    ``IN_FORCE``, so the same filter excludes it and it is never reported as
    current law. Equally not ``EXPIRED``: that would be a claim about a
    document nobody has read either.
    """

    IN_FORCE = "in_force"
    AMENDING = "amending"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


def _validate_relative_path(value: str) -> str:
    """Reject an absolute path or a parent-directory escape (Part C3).

    Shared by ``Document.path`` and ``DocumentPart.path`` (issue #126): both
    name a location "relative to the repository root", so both must pass the
    same containment check.
    """
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        msg = f"path must stay inside the repository, got {value!r}"
        raise ValueError(msg)
    return value


class DocumentPart(BaseModel):
    """One ordered part of a multi-file document (issue #126).

    Decree 105/2025 was published across two issues of the Official Gazette,
    so its source text lives in two separate downloaded files that must be
    read in order and concatenated before parsing -- the parser itself
    always sees one document's worth of plain text, never a part boundary.
    A single-file document keeps using ``Document.path`` instead; ``parts``
    exists only for a document whose source genuinely spans more than one
    file.

    Attributes:
        path: Location of this part's downloaded file, relative to the
            repository root -- the same rule as ``Document.path``.
        source_url: Page this specific part was downloaded from. Recorded
            per part, not only once at the document level, because each
            part of a multi-issue Gazette publication has its own URL.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    source_url: str | None = None

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _validate_relative_path(value)


class Document(BaseModel):
    """One entry of ``corpus/manifest.json``.

    Attributes:
        id: Stable document identifier, e.g. ``gas-decree-87-2018``.
        number: Official document number, e.g. ``87/2018/NĐ-CP``.
        title_vi: Official Vietnamese title.
        doc_type: Document type.
        issued: Signature date, when known.
        effective: Date the document entered into force, when known.
        status: In-force state driving the retrieval date filter.
        source_url: Page the document was downloaded from.
        fetched: Date the file was downloaded.
        path: Location of the downloaded file, relative to the repository root
            -- required even when ``acquired`` is ``False``: it still names
            where the file belongs once it is. Mutually exclusive with
            ``parts``; exactly one of the two must be set.
        parts: Ordered list of a multi-file document's own parts (issue
            #126), e.g. decree 105/2025's two Gazette issues. Mutually
            exclusive with ``path``. ``ingest.extract.extract_text_parts``
            concatenates each part's extracted text in this order, stripping
            every part after the first's own continuation preamble first.
        meta: Free-form notes, e.g. an unresolved verification flag.
        acquired: ``False`` marks a document that is part of the declared
            corpus but whose text is not on disk -- permanently for a
            copyrighted standard that must be purchased (see SOURCES.md §
            Licensing), or until someone downloads it. ``load_manifest``
            (issue #59) only requires ``path``/``parts`` to exist when this
            is ``True`` (the default); a document with ``acquired=False``
            still becomes a ``documents`` row and a graph node -- so an
            edge naming it (e.g. ``REPLACED_BY``) still resolves -- but is
            skipped by the text pipeline: no units, no chunks, no
            embeddings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    number: str = Field(min_length=1)
    title_vi: str = Field(min_length=1)
    doc_type: DocType
    issued: date | None = None
    effective: date | None = None
    status: DocStatus
    source_url: str | None = None
    fetched: date | None = None
    path: str | None = Field(default=None, min_length=1)
    parts: tuple[DocumentPart, ...] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    acquired: bool = True

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        """Enforce the locked identifier scheme (A5.1)."""
        return document_id(value)

    @field_validator("issued", "effective", "fetched", mode="before")
    @classmethod
    def _widen_partial_date(cls, value: Any) -> Any:  # noqa: ANN401 - pydantic pre-validator
        """Widen a ``YYYY`` or ``YYYY-MM`` value to the first day it can mean.

        Several standards in the corpus (TCVN/QCVN) publish a year only. The
        earliest possible day is used so an in-force check never reports a
        document as effective later than it really is.
        """
        if not isinstance(value, str):
            return value
        if match := _YEAR_PATTERN.match(value.strip()):
            return date(int(match.group("year")), 1, 1)
        if match := _YEAR_MONTH_PATTERN.match(value.strip()):
            return date(int(match.group("year")), int(match.group("month")), 1)
        return value

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str | None) -> str | None:
        """Reject absolute paths and parent-directory escapes (Part C3)."""
        if value is None:
            return None
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def _validate_path_xor_parts(self) -> "Document":
        """Require exactly one of ``path``/``parts`` (issue #126).

        Neither set leaves nothing to read; both set leaves two disagreeing
        answers to "where is this document's text" with no rule for which
        one wins -- so both are rejected the same way a missing one is,
        rather than picking a silent precedence.
        """
        if (self.path is None) == (self.parts is None):
            msg = "a document must declare exactly one of 'path' or 'parts', never both or neither"
            raise ValueError(msg)
        if self.parts is not None and len(self.parts) == 0:
            msg = "'parts' must list at least one part"
            raise ValueError(msg)
        return self

    @property
    def storage_path(self) -> str:
        """The single string persisted in the ``documents.path`` column.

        That column is one text field, not a list, so a multi-part
        document's ordered paths are joined with ``;`` for display/audit
        purposes only -- the ingest pipeline itself always resolves a
        document's real files through ``ingest.manifest.Manifest.parts_of``,
        reading the manifest, never by re-splitting this string back apart.
        """
        if self.path is not None:
            return self.path
        assert self.parts is not None  # guaranteed by _validate_path_xor_parts
        return ";".join(part.path for part in self.parts)


class DocumentStub(BaseModel):
    """A document known only because another document cites it (issue #67).

    A citation such as "Nghị định số 136/2020/NĐ-CP" names a document that is
    not in the manifest. Without a node for it the graph has nothing for that
    edge to point at, so the citation is dropped -- and the history question
    it answers ("is NĐ 136/2020 still in force?") cannot even be asked. A stub
    is that node, and nothing more: no text, no units, no chunks, exactly as
    an ``acquired=False`` document (#62), except discovered from a citation
    rather than declared in the manifest. Persisted as a ``documents`` row.

    Everything except ``number`` is fixed, because nothing else is known:
    ``status`` is ``DocStatus.UNKNOWN`` (never reported as current law),
    ``doc_type`` is ``DocType.UNKNOWN``, and ``title_vi`` is the number itself
    -- the one true thing to display, where inventing a title would be a
    claim about a document nobody has read. ``documents.path`` is ``NULL``:
    there is no file. Stubs are identified in the table by ``status =
    'unknown'``.

    Attributes:
        number: The canonical document number, e.g. ``136/2020/NĐ-CP``, as
            ``ingest.references.find_document_number`` returns it. Whether it
            is *complete* enough to deserve a node is decided by the caller
            (``ingest.graph``), not here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    number: str = Field(min_length=1)

    @property
    def id(self) -> str:
        """The stub's stable document id, derived from ``number`` alone."""
        return stub_document_id(self.number)

    @property
    def title_vi(self) -> str:
        """The number, standing in for a title nobody has read."""
        return self.number

    @property
    def doc_type(self) -> DocType:
        """Always ``DocType.UNKNOWN``."""
        return DocType.UNKNOWN

    @property
    def status(self) -> DocStatus:
        """Always ``DocStatus.UNKNOWN`` -- never reported as current law."""
        return DocStatus.UNKNOWN
