"""``GET /doc/{id}`` -- one document and its unit tree (build plan A6 Phase 3.7, issue #90).

A citation's ``unit_id``/``document_id`` (``app/schemas/ask.py``) is what a frontend
link points at; this is what that link resolves to.
"""

import logging
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from app.api.deps import SessionDep
from app.repositories.documents import DocumentRepository
from app.repositories.units import UnitRepository
from app.schemas.unit import Unit

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

RATE_LIMIT_PER_MINUTE = 60


def _rate_limit(request: Request) -> None:
    """Enforce this process's ``/doc`` limiter.

    See ``app.api.ask``'s own ``_rate_limit`` for why this reads ``app.state`` on every
    call instead of closing over a module-level ``RateLimiter``.
    """
    request.app.state.doc_rate_limiter(request)


class DocumentDetail(BaseModel):
    """One document's metadata, with its full unit tree in reading order.

    ``doc_type``/``status`` are the raw stored strings (``"unknown"`` for a stub,
    which ``GET /doc/{id}`` serves the same as any other document -- a citation's
    ``document_id`` can legitimately name one, and a frontend still needs something
    to show at that link) rather than the ingest-side enums, so a status this
    endpoint has not been taught about yet is served, not rejected.

    Attributes:
        id: Document id.
        number: Official document number, or the number itself for a stub.
        title_vi: Vietnamese title.
        doc_type: Raw stored document type.
        status: Raw stored in-force status.
        issued: Signature date, when known.
        effective: Date the document took effect, when known.
        source_url: Page the document was downloaded from.
        units: Every unit of the document, in reading order; empty for a stub or a
            metadata-only document (neither has text).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    number: str
    title_vi: str
    doc_type: str
    status: str
    issued: date | None = None
    effective: date | None = None
    source_url: str | None = None
    units: tuple[Unit, ...] = ()


@router.get("/doc/{document_id}", dependencies=[Depends(_rate_limit)])
async def get_document(document_id: str, session: SessionDep) -> DocumentDetail:
    """Return one document's metadata and its full unit tree.

    Args:
        document_id: Document id -- a manifest document or a stub alike.
        session: Database session (dependency; 503 if no database is configured).

    Returns:
        The document, with its units in reading order.

    Raises:
        HTTPException: 404, if no document has this id.
    """
    row = await DocumentRepository(session).get(document_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No document {document_id!r}."
        )
    units = await UnitRepository(session).list_by_document(document_id)
    return DocumentDetail(
        id=row.id,
        number=row.number,
        title_vi=row.title_vi,
        doc_type=row.doc_type,
        status=row.status,
        issued=row.issued,
        effective=row.effective,
        source_url=row.source_url,
        units=tuple(units),
    )
