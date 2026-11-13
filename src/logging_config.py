import json
import logging
import os
import sys
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")

_STANDARD_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


def set_request_id(request_id: str) -> Token[str]:
    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    _REQUEST_ID.reset(token)


def get_request_id() -> str:
    return _REQUEST_ID.get()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and key != "request_id":
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_logging(service_name: str = "finance-decision-assistant") -> None:
    """
    Configure root logging once per process.

    Env vars:
        LOG_LEVEL   INFO (default), DEBUG, WARNING, ...
        LOG_FORMAT  "text" (default) or "json" -- format of the stdout handler
        LOG_FILE    optional path; if set, ALSO writes every record there as JSON
                    lines (always JSON, so it can be parsed with jq/pandas)
    """
    logger = logging.getLogger()
    if getattr(logger, "_finance_logging_configured", False):
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    log_file = os.getenv("LOG_FILE", "").strip()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] [request_id=%(request_id)s] %(message)s"
            )
        )

    logger.handlers.clear()
    logger.addHandler(handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.addFilter(RequestIdFilter())
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

    # httpx logs every HTTP request at INFO (incl. each 429 the SDK then retries), which
    # floods the log. Retries stay visible via the "groq._base_client"/"openai._base_client"
    # "Retrying request ..." INFO lines; a request that finally fails raises and is logged.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger._finance_logging_configured = True  # type: ignore[attr-defined]

    logging.getLogger(__name__).info(
        "Logging configured",
        extra={
            "service": service_name,
            "log_level": log_level,
            "log_format": log_format,
            "log_file": log_file or None,
        },
    )

