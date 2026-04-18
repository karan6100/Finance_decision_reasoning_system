from core.summarizer_agent import SummarizerAgent
from schema.types import OverallState

summarizer_agent = SummarizerAgent()


def summarizer_node(state: OverallState):
    """
    Summarizer node that beautifies and summarizes validated responses.
    Only processes if validation passed.
    """
    
    # Only summarize if validation passed
    if state.get("valid_status") != "pass":
        state["final_response"] = (
            f"Validation failed with reason: {state.get('valid_reviews', {}).get('reason', 'No reason provided')}."
        )
        state["final"] = True
        return state
    
    valid_inp = state.get('validation_input')
    summarized_response = summarizer_agent.run(valid_inp, state)

    state["final_response"] = summarized_response
    state["final"] = True
    return state