"""Tests for settings and logging configuration."""

import logging
import re
from pathlib import Path

import pytest

from app.core.config import AppEnv, Settings
from app.core.logging import configure_logging

ENV_EXAMPLE = Path(__file__).parent.parent / ".env.example"


def test_defaults_are_local_and_expose_docs() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env is AppEnv.LOCAL
    assert settings.log_level == "INFO"
    assert settings.docs_enabled is True


def test_production_hides_docs() -> None:
    assert Settings(app_env=AppEnv.PRODUCTION, _env_file=None).docs_enabled is False


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.app_env is AppEnv.PRODUCTION
    assert settings.log_level == "DEBUG"


def test_configure_logging_sets_the_root_level() -> None:
    configure_logging("WARNING")

    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_rejects_an_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("LOUD")


def test_env_example_documents_every_settings_field() -> None:
    """Every ``Settings`` field has an upper-cased line in ``.env.example`` (issue #204).

    A field added to ``Settings`` without a matching ``.env.example`` line silently falls
    back to its Python default for anyone copying the file, the same gap issue #204 found
    for ``DATABASE_URL``, ``LLM_PROVIDER``, ``OLLAMA_MODEL`` and ``GROUNDING_CHECK``. This
    guards against that gap reopening, rather than relying on a reviewer noticing by eye.
    """
    env_text = ENV_EXAMPLE.read_text(encoding="utf-8")
    documented = {match.group(1) for match in re.finditer(r"(?m)^([A-Z_]+)=", env_text)}

    missing = {name.upper() for name in Settings.model_fields} - documented
    assert not missing, f"backend/.env.example is missing a line for: {sorted(missing)}"
