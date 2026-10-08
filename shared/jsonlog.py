"""JSON logging for model/ and eval/ (stdlib only, no dependency).

Never pass contract text or user questions in a log call: log `fingerprint(text)` instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        row: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Anything passed via `extra={...}` becomes a top-level field.
        row.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            row["exc"] = self.formatException(record.exc_info)
        return json.dumps(row, default=str)


def setup(level: int = logging.INFO, path: str | None = None) -> None:
    """Route the root logger to stdout (and optionally a .jsonl file) as one JSON object per line."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if path:
        handlers.append(logging.FileHandler(path))
    for h in handlers:
        h.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=handlers, force=True)


def fingerprint(text: str) -> dict[str, Any]:
    """What to log instead of user/contract text: a short hash and the length."""
    return {"sha256": hashlib.sha256(text.encode()).hexdigest()[:16], "len": len(text)}
