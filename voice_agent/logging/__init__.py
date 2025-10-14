"""Structured logging utilities for the voice agent."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict

LOG_SCHEMA_FIELDS = [
    "filename",
    "timestamp",
    "classname",
    "function",
    "system_section",
    "line_num",
    "error",
    "db_phase",
    "method",
    "message",
    "derived_message",
]

REVIEW_PROMPT = """Quality review checklist\n* Did we change only the systems we intended to touch?\n* Are all dependencies accounted for without hidden couplings?\n* Which edge cases or failure modes still need coverage?\n* If blocked, what outcome are we working backward from?"""


class StructuredFormatter(logging.Formatter):
    """Emit structured JSON logs plus the mandated human readable prompt."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {field: getattr(record, field, None) for field in LOG_SCHEMA_FIELDS}
        payload["timestamp"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload["message"] = record.getMessage()
        payload.setdefault("filename", record.pathname)
        payload.setdefault("line_num", record.lineno)
        payload.setdefault("function", record.funcName)
        payload.setdefault("classname", record.__dict__.get("classname"))
        payload.setdefault("system_section", record.__dict__.get("system_section"))
        payload.setdefault("db_phase", record.__dict__.get("db_phase", "none"))
        payload.setdefault("method", record.__dict__.get("method", "NONE"))
        payload.setdefault("error", record.__dict__.get("error"))
        payload.setdefault("derived_message", record.__dict__.get("derived_message"))

        json_line = json.dumps(payload, ensure_ascii=False)
        derived = record.__dict__.get("derived_message")
        if derived:
            return f"{json_line}\n{derived}\n{REVIEW_PROMPT}"
        return f"{json_line}\n{REVIEW_PROMPT}"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with the structured formatter."""

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(StructuredFormatter())
            handler.setLevel(level)
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the structured formatter."""

    configure_logging()
    return logging.getLogger(name)
