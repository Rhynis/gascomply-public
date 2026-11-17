"""Structured logging configuration.

Business code never calls ``print``; it uses a module-level logger obtained from
``logging.getLogger(__name__)`` (Part C6).
"""

import logging
from logging.config import dictConfig

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Install the process-wide logging configuration.

    Args:
        level: Root log level name, e.g. ``"INFO"`` or ``"DEBUG"``.

    Raises:
        ValueError: If ``level`` is not a known log level name.
    """
    if not isinstance(logging.getLevelName(level.upper()), int):
        msg = f"Unknown log level: {level}"
        raise ValueError(msg)

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"default": {"format": _LOG_FORMAT}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"handlers": ["console"], "level": level.upper()},
        }
    )
