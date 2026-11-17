"""Tests for the health endpoint and the baseline response hardening."""

from fastapi import Response
from fastapi.testclient import TestClient

from app import __version__
from app.core.config import AppEnv, Settings
from app.core.middleware import SECURITY_HEADERS
from app.main import create_app


def test_health_returns_ok_with_version_and_environment(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "environment": AppEnv.CI.value,
    }


def test_health_response_carries_security_headers(client: TestClient) -> None:
    response = client.get("/health")

    for header, value in SECURITY_HEADERS.items():
        assert response.headers[header] == value
    assert "server" not in {name.lower() for name in response.headers}


def test_unknown_path_returns_404(client: TestClient) -> None:
    assert client.get("/no-such-endpoint").status_code == 404


def test_docs_are_served_outside_production(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200


def test_docs_are_disabled_in_production() -> None:
    production_app = create_app(
        Settings(app_env=AppEnv.PRODUCTION, log_level="WARNING", _env_file=None)
    )

    with TestClient(production_app) as production_client:
        assert production_client.get("/openapi.json").status_code == 404
        assert production_client.get("/health").status_code == 200


def test_server_banner_is_stripped_from_responses() -> None:
    banner_app = create_app(Settings(app_env=AppEnv.CI, log_level="WARNING", _env_file=None))

    @banner_app.get("/banner")
    def read_banner() -> Response:
        return Response(content="", headers={"Server": "uvicorn"})

    with TestClient(banner_app) as banner_client:
        response = banner_client.get("/banner")

    assert response.status_code == 200
    assert "server" not in {name.lower() for name in response.headers}
