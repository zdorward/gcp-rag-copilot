"""Structured JSON logging for Cloud Run compatibility."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Context variable for request-scoped data
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestContext:
    """Context manager for request-scoped logging context."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.token = None

    def __enter__(self):
        self.token = _request_id.set(self.request_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _request_id.reset(self.token)
        return False


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for Cloud Logging."""

    LEVEL_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": self.LEVEL_MAP.get(record.levelno, "INFO"),
            "message": record.getMessage(),
            "component": record.name,
        }

        # Add request_id if available
        request_id = _request_id.get()
        if request_id:
            log_entry["request_id"] = request_id

        # Add extra fields (metrics, context, etc.)
        if hasattr(record, "metrics"):
            log_entry["metrics"] = record.metrics
        if hasattr(record, "context"):
            log_entry["context"] = record.context

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add JSON handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
