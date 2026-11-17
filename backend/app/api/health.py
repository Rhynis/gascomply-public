"""Health router: liveness probe for the platform and for smoke tests."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app import __version__
from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", summary="Liveness probe")
def read_health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report that the process is up, with its version and environment.

    Args:
        settings: Injected application settings.

    Returns:
        The liveness payload.
    """
    return HealthResponse(status="ok", version=__version__, environment=settings.app_env)
