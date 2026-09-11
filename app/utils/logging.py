"""Structured logging setup.

Provides JSON-line or human-readable logging depending on configuration. A
small set of secret-looking keys is redacted defensively so API keys never
reach the log stream even if a caller passes them by mistake.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from typing import Any, Dict

_SECRET_KEYS = {
    "openai_api_key",
    "api_key",
    "langsmith_api_key",
    "authorization",
    "password",
    "secret",
    "token",
}

_CONFIGURED = False


def _redact(value: Any) -> Any:
    """Recursively redact values whose keys look like secrets."""
    if isinstance(value, dict):
        return {
            k: ("***redacted***" if k.lower() in _SECRET_KEYS else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(_redact(extra))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    """Configure the root logger once for the whole process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Retrieved textbook text contains Unicode math symbols (minus sign, Greek
    # letters). On Windows the console defaults to cp1252, which raises
    # UnicodeEncodeError on those. Force UTF-8 and never fail on an odd glyph.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a reconfigurable stream
        pass

    handler = logging.StreamHandler(sys.stdout)
    if as_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log a structured event with arbitrary (redacted) fields."""
    logger.log(level, message, extra={"extra_fields": _redact(fields)})


def new_request_id() -> str:
    """Return a short unique request identifier."""
    return uuid.uuid4().hex[:12]
