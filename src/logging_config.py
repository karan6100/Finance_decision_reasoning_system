import json
import logging
import os
import sys
from contextvars import ContextVar, Token
from datetime import datetime, timezone
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
    logger = logging.getLogger()
    if getattr(logger, "_finance_logging_configured", False):
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "text").lower()
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
    logger._finance_logging_configured = True  # type: ignore[attr-defined]

    logging.getLogger(__name__).info(
        "Logging configured",
        extra={
            "service": service_name,
            "log_level": log_level,
            "log_format": log_format,
        },
    )

