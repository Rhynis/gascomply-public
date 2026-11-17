"""Schemas for the health endpoint."""

from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import AppEnv


class HealthResponse(BaseModel):
    """Liveness payload returned by ``GET /health``."""

    status: Literal["ok"] = Field(description="Liveness marker; always 'ok' when served.")
    version: str = Field(description="Backend package version.")
    environment: AppEnv = Field(description="Deployment environment of the process.")
