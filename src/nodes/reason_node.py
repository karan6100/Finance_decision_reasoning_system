from core.reasoning_agent import ReasonAgent
from schema.types import ReasoningInput, OverallState


agent = ReasonAgent()

def reason_llm_node(state: OverallState):

    max_retries = state.get("max_attempts", 5)
    attempts = state.get("attempt_counter", 0)
    if attempts >= max_retries:
        state["final_response"] = (
            f"Stopped after {attempts} reasoning attempts to avoid an infinite loop."
        )
        state["final"] = True
        return state

    inp = ReasoningInput(
        query=state['query'],
        intent=state['profile'].intent,
        risk= state['profile'].risk
    )

    out = agent.run(inp= inp)

    state["attempt_counter"] = attempts + 1
    state["reasoning_response"] = out.reasoning
    state["reasons"] = state.get("reasons", []) + [out.reasoning]

    return state
