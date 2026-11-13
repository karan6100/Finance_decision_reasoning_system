"""
Per-node observability: duration and LLM token usage for every pipeline node.

HOW IT WORKS
    - `@track_node("reasoning")` on a node function times it and opens a metrics
      "frame" for the duration of the call.
    - `usage_callback` is attached to every LLM built by `LLMFactory`. After each
      LLM call it adds that call's token usage to the INNERMOST open frame.
    - When the node returns, one JSON-friendly log line is emitted:
          message="node.metrics", node, parent_node, duration_ms, self_ms,
          llm_calls, input_tokens, output_tokens, total_tokens
      and the same dict is appended to `state["node_metrics"]`.

NESTING
    `validation_node` calls `reason_llm_node` / `basic_agent_node` inside its retry
    loop, so frames nest:
    - tokens are credited to the innermost node only -> summing `total_tokens`
      over a request's lines never double-counts;
    - `duration_ms` is INCLUSIVE (validation includes its nested regenerations);
      `self_ms` excludes time spent in nested tracked nodes.

CAVEAT
    Durations include rate-limiter waits (see `LLMs/llm_factory.py`), so they are
    good for before/after comparisons, not raw model latency.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Optional

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Stack of open frames for the current request/thread (innermost last).
_FRAMES: ContextVar[Optional[tuple[dict[str, Any], ...]]] = ContextVar("node_frames", default=None)


def _extract_usage(response: Any) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from a LangChain LLMResult."""
    input_tokens = output_tokens = 0
    found = False
    for generation_list in getattr(response, "generations", None) or []:
        for generation in generation_list:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if usage:
                found = True
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                output_tokens += int(usage.get("output_tokens", 0) or 0)

    if not found:  # older providers only report usage in llm_output
        token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        input_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(token_usage.get("completion_tokens", 0) or 0)
    return input_tokens, output_tokens


class NodeUsageCallback(BaseCallbackHandler):
    """LangChain callback: credits each LLM call's tokens to the node currently running."""

    run_inline = True  # run in the caller's context so the ContextVar is visible

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        frames = _FRAMES.get()
        if not frames:
            return  # LLM call outside any tracked node (e.g. the eval judge)
        frame = frames[-1]
        input_tokens, output_tokens = _extract_usage(response)
        frame["llm_calls"] += 1
        frame["input_tokens"] += input_tokens
        frame["output_tokens"] += output_tokens

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        frames = _FRAMES.get()
        if frames:
            frames[-1]["llm_errors"] += 1


# One shared instance; LLMFactory attaches it to every model it builds.
usage_callback = NodeUsageCallback()


def track_node(node_name: str) -> Callable:
    """Decorator for pipeline node functions `fn(state, ...) -> state`."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(state, *args, **kwargs):
            parent_frames = _FRAMES.get() or ()
            frame: dict[str, Any] = {
                "node": node_name,
                "parent_node": parent_frames[-1]["node"] if parent_frames else None,
                "llm_calls": 0,
                "llm_errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "_child_ms": 0.0,
            }
            token = _FRAMES.set(parent_frames + (frame,))
            started = time.perf_counter()
            result = state
            try:
                result = fn(state, *args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - started) * 1000
                _FRAMES.reset(token)
                if parent_frames:
                    parent_frames[-1]["_child_ms"] += duration_ms

                metrics = {
                    "node": frame["node"],
                    "parent_node": frame["parent_node"],
                    "duration_ms": round(duration_ms, 2),
                    "self_ms": round(duration_ms - frame["_child_ms"], 2),
                    "llm_calls": frame["llm_calls"],
                    "llm_errors": frame["llm_errors"],
                    "input_tokens": frame["input_tokens"],
                    "output_tokens": frame["output_tokens"],
                    "total_tokens": frame["input_tokens"] + frame["output_tokens"],
                }
                target = result if isinstance(result, dict) else state
                if isinstance(target, dict):
                    target.setdefault("node_metrics", []).append(metrics)
                logger.info("node.metrics", extra=metrics)

        return wrapper

    return decorator


def summarize_node_metrics(node_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-request totals from `state["node_metrics"]` (tokens never double-counted)."""
    return {
        "llm_calls": sum(m.get("llm_calls", 0) for m in node_metrics),
        "input_tokens": sum(m.get("input_tokens", 0) for m in node_metrics),
        "output_tokens": sum(m.get("output_tokens", 0) for m in node_metrics),
        "total_tokens": sum(m.get("total_tokens", 0) for m in node_metrics),
    }
