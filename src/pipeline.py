import logging

from logging_config import get_request_id
from schema.types import OverallState
from nodes.manager_node import manager_node
from nodes.validation_node import validation_node
from nodes.summarizer_node import summarizer_node

logger = logging.getLogger(__name__)


def run_finance_pipeline(query: str) -> OverallState:
    """
    Run the full finance reasoning pipeline and return the final state.
    """
    state: OverallState = {
        "query": query,
        "attempt_counter": 0,
        "max_attempts": 5,
        "reasons": [],
    }
    logger.debug(
        "pipeline.start",
        extra={"query_length": len(query), "request_id": get_request_id()},
    )

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
        logger.info(
            "pipeline.completed",
            extra={
                "route": state.get("route"),
                "attempt_counter": state.get("attempt_counter"),
                "reasons_count": len(state.get("reasons", [])),
            },
        )
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
        logger.info(
            "pipeline.completed",
            extra={
                "route": state.get("route"),
                "attempt_counter": state.get("attempt_counter"),
                "reasons_count": len(state.get("reasons", [])),
            },
        )
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
    logger.info(
        "pipeline.completed",
        extra={
            "route": state.get("route"),
            "attempt_counter": state.get("attempt_counter"),
            "reasons_count": len(state.get("reasons", [])),
        },
    )
    return state
