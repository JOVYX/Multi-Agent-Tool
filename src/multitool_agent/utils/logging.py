"""Logging configuration and utilities."""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

# Sensitive keys to scrub from logs
_SECRET_PATTERN = re.compile(
    r"(sk-ant-[a-zA-Z0-9_\-]+|tvly-[a-zA-Z0-9_\-]+|BSA[a-zA-Z0-9_\-]+)",
    re.IGNORECASE,
)


class SanitizedFormatter(logging.Formatter):
    """Logging formatter that scrubs API keys and secrets from output messages."""

    def format(self, record: logging.LogRecord) -> str:
        orig = super().format(record)
        return _SECRET_PATTERN.sub("[REDACTED_SECRET]", orig)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with clean formatting and sanitized output."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if reconfigured
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            SanitizedFormatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)
    # Silence noisy HTTP client connection logs unless explicitly in DEBUG mode
    if log_level != logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)



def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)


def sanitize_dict_for_logging(data: dict[str, Any], max_len: int = 500) -> str:
    """Format dictionary as JSON string, truncating long fields to keep logs readable."""
    try:
        sanitized: dict[str, Any] = {}
        for k, v in data.items():
            if any(secret_key in k.lower() for secret_key in ["key", "secret", "token", "password"]):
                sanitized[k] = "[REDACTED]"
            elif isinstance(v, str) and len(v) > max_len:
                sanitized[k] = f"{v[:max_len]}... (truncated {len(v)} chars)"
            else:
                sanitized[k] = v
        return json.dumps(sanitized)
    except Exception:
        return str(data)
