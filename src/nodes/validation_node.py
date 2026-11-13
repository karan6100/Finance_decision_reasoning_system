from core.validation_agent import ValidationAgent
from observability import track_node
from schema.types import OverallState, ValidationInput
from nodes import basic_node, reason_node
import json
import logging
import re
from typing import Any

validation_agent = ValidationAgent()
logger = logging.getLogger(__name__)

# Validator agent parsing error counter
MAX_PARSE_RETRIES = 2



def _strip_code_fences(text: str) -> str:
    '''removes surrounding triple backticks from a string, if they exist.'''
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_object(text: str) -> str:
    ''' pull out the first substring that looks like a JSON object using regex.'''
    match = re.search(r"\{[\s\S]*\}", text) #finds a chunk starting from { and ending at }
    return match.group(0) if match else text


def _safe_parse_validation_result(raw_result: Any) -> dict:
    if isinstance(raw_result, dict):
        decision = str(raw_result.get("decision", "")).strip().lower()
        reason = str(raw_result.get("reason", "")).strip() or "No reason provided."
        if decision in {"pass", "fail"}:
            return {"decision": decision, "reason": reason}

    if isinstance(raw_result, list): # if the model output in chunks
        parts = []
        for item in raw_result:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        raw_text = "\n".join(parts).strip()
    else:
        raw_text = str(raw_result or "").strip()

    if not raw_text:
        return {"decision": "fail", 
                "reason": "Validation model returned empty output.",
                "parse_error": True,
                }

    candidates = [
        raw_text,
        _strip_code_fences(raw_text),
        _extract_json_object(_strip_code_fences(raw_text)),
    ]

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            decision = str(parsed.get("decision", "")).strip().lower()
            reason = str(parsed.get("reason", "")).strip() or "No reason provided."

            if decision not in {"pass", "fail"}:
                return {
                    "decision": "fail",
                    "reason": f"Invalid validation decision value: {decision!r}.",
                    "parse_error": True,
                }
            return {"decision": decision, "reason": reason}
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue

    preview = raw_text[:200].replace("\n", " ")
    return {
        "decision": "fail",
        "reason": f"Validation output was not valid JSON. Raw output preview: {preview}",
        "parse_error": True,
    }


@track_node("validation")
def validation_node(state: OverallState):
    parse_retries = 0 # the only counter: resets for each new answer
    while True:
        attempts = state.get("attempt_counter", 0)
        max_attempts = state.get("max_attempts", 5)

        # Pick source + candidate
        if state.get("facts_response"):
            source_agent = "basic"
            candidate_response = state["facts_response"]
        elif state.get("reasoning_response"):
            source_agent = "reasoning"
            candidate_response = state["reasoning_response"]
        else:
            raise ValueError("No upstream response found for validation.")

        inp = ValidationInput(
            query=state["query"],
            source_agent=source_agent,
            candidate_response=candidate_response,
            intent=state["profile"].intent,
            risk=state["profile"].risk,
            personalized=state["profile"].personalized,
        )

        result = _safe_parse_validation_result(validation_agent.run(inp))
        state["valid_status"] = result.get("decision")
        state["valid_reviews"] = result
        state["validation_input"] = inp

        if state["valid_status"] == "pass":
            state["validation_meta"] = {
                "status": "passed",
                "attempts": attempts,
                "max_attempts": max_attempts,
            }
            return state

        # Parse error count check
        if result.get("parse_error"):
            parse_retries += 1
            if parse_retries <= MAX_PARSE_RETRIES: # cap separately so a broken validator can't loop forever
                logger.warning("Validator output unparseable, re-validating | retry=%s", parse_retries)
                continue                # same answer, validate again
            # retries exhausted for this answer
            # The validator itself is failing; a new answer won't help.
            logger.error("Validator unavailable after %s parse failures | reason=%r",
                         parse_retries, result.get("reason"))
            state["validation_meta"] = {
                "status": "validator_unavailable",
                "attempts": attempts,
                "parse_retries": parse_retries,
                "last_reason": result.get("reason"),
            }
            state["final_response"] = (
                "We couldn't verify this answer right now. Please try again shortly."
            )
            state["final"] = True
            return state

        # No. of attempts reached max attempts
        if attempts >= max_attempts:
            last_reason = (
                state.get("valid_reviews", {}) or {}
            ).get("reason", "No validation reason provided.")
            logger.warning(
                "Validation max attempts reached | attempts=%s max_attempts=%s route=%s last_reason=%r query=%r",
                attempts,
                max_attempts,
                state.get("route"),
                last_reason,
                state.get("query"),
            )
            state["validation_meta"] = {
                "status": "max_attempts_reached",
                "attempts": attempts,
                "max_attempts": max_attempts,
                "last_reason": last_reason,
            }
            state["final_response"] = f"Could not provide the output for your query. Reason: {last_reason}"
            state["final"] = True
            return state

        logger.info(
            "Validation failed | attempt=%s/%s source=%s reason=%r",
            attempts,
            max_attempts,
            source_agent,
            state.get("valid_reviews", {}).get("reason"),
        )

        parse_retries = 0                  # new answer coming -> fresh budget
        # fail -> regenerate (these nodes increment attempt_counter)
        if source_agent == "basic":
            # state["facts_response"] = "" 
            state = basic_node.basic_agent_node(state)
        else:
            # state["reasoning_response"] = ""  
            state = reason_node.reason_llm_node(state)

        if state.get("final"):
            return state
