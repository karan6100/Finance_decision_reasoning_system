import logging
import time
import uuid
from typing import Optional

from logging_config import get_request_id, reset_request_id, set_request_id
from observability import summarize_node_metrics
from schema.types import OverallState
from nodes.manager_node import manager_node
from nodes.validation_node import validation_node
from nodes.summarizer_node import summarizer_node

logger = logging.getLogger(__name__)


def run_finance_pipeline(query: str, request_id: Optional[str] = None) -> OverallState:
    """
    Run the full finance reasoning pipeline and return the final state.

    Every log line emitted during the run carries the same `request_id`, so the
    per-node `node.metrics` lines of one query can be grouped. Uses, in order:
    the given `request_id`, an id already set by the caller, or a new short UUID.
    """
    if request_id is None:
        current = get_request_id()
        request_id = current if current != "-" else uuid.uuid4().hex[:12]
    token = set_request_id(request_id)
    try:
        return _run(query)
    finally:
        reset_request_id(token)


def _log_completed(state: OverallState, started: float) -> None:
    totals = summarize_node_metrics(state.get("node_metrics", []))
    logger.info(
        "pipeline.completed",
        extra={
            "route": state.get("route"),
            "attempt_counter": state.get("attempt_counter"),
            "reasons_count": len(state.get("reasons", [])),
            "validation_status": (state.get("validation_meta") or {}).get("status"),
            "request_duration_ms": round((time.perf_counter() - started) * 1000, 2),
            **{f"request_{k}": v for k, v in totals.items()},
        },
    )


def _run(query: str) -> OverallState:
    started = time.perf_counter()
    state: OverallState = {
        "query": query,
        "attempt_counter": 0,
        "max_attempts": 5,
        "reasons": [],
        "node_metrics": [],
    }
    logger.debug("pipeline.start", extra={"query_length": len(query)})

    state = manager_node(state)
    logger.info(
        "pipeline.node.completed",
        extra={
            "node": "manager",
            "final": bool(state.get("final")),
            "route": state.get("route"),
            "attempt_counter": state.get("attempt_counter"),
        },
    )
    if state.get("final"):
        _log_completed(state, started)
        return state

    state = validation_node(state)
    logger.info(
        "pipeline.node.completed",
        extra={
            "node": "validation",
            "final": bool(state.get("final")),
            "route": state.get("route"),
            "attempt_counter": state.get("attempt_counter"),
        },
    )
    if state.get("final"):
        _log_completed(state, started)
        return state

    state = summarizer_node(state)
    logger.info(
        "pipeline.node.completed",
        extra={
            "node": "summarizer",
            "final": bool(state.get("final")),
            "route": state.get("route"),
            "attempt_counter": state.get("attempt_counter"),
        },
    )
    _log_completed(state, started)
    return state
