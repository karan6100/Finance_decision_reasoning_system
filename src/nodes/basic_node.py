from core.basic_agent import BasicAgent
from schema.types import BasicInput, OverallState


basic_agent = BasicAgent()

def basic_agent_node(state: OverallState):

    attempts = state.get("attempt_counter", 0)
    max_attempts = state.get("max_attempts", 5)
    if attempts >= max_attempts:
        state["final_response"] = f"Stopped after {attempts} attempts."
        state["final"] = True
        return state
    # Basic Agent
    inp = BasicInput(query=state["query"])
    out = basic_agent.run(inp=inp)
    state["facts_response"] = out.output
    state["attempt_counter"] = attempts + 1

    return state 


