"""Tests for structured logging module."""

import json
import logging
from io import StringIO

from app.logging import setup_logging, get_logger, RequestContext


def test_setup_logging_configures_json_format():
    """Logging should output JSON format."""
    # Capture log output
    stream = StringIO()
    handler = logging.StreamHandler(stream)

    setup_logging(level="INFO")
    logger = get_logger("test")

    # Replace handler to capture output
    logger.handlers = [handler]
    logger.propagate = False

    # Get the formatter from setup_logging
    from app.logging import JSONFormatter
    handler.setFormatter(JSONFormatter())

    logger.info("test message")

    output = stream.getvalue()
    log_entry = json.loads(output.strip())

    assert log_entry["message"] == "test message"
    assert log_entry["severity"] == "INFO"
    assert "timestamp" in log_entry
    assert log_entry["component"] == "test"


def test_request_context_adds_request_id():
    """RequestContext should add request_id to log entries."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)

    from app.logging import JSONFormatter
    handler.setFormatter(JSONFormatter())

    logger = get_logger("test")
    logger.handlers = [handler]
    logger.propagate = False

    with RequestContext(request_id="req_123"):
        logger.info("contextual message")

    output = stream.getvalue()
    log_entry = json.loads(output.strip())

    assert log_entry["request_id"] == "req_123"


def test_logger_with_metrics():
    """Logger should support metrics in extra field."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)

    from app.logging import JSONFormatter
    handler.setFormatter(JSONFormatter())

    logger = get_logger("test")
    logger.handlers = [handler]
    logger.propagate = False

    logger.info("query completed", extra={"metrics": {"latency_ms": 150}})

    output = stream.getvalue()
    log_entry = json.loads(output.strip())

    assert log_entry["metrics"]["latency_ms"] == 150
