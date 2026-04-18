from core.decision_gate import DecisionGate
from core.manager_agent import ManagerAgent
from nodes.basic_node import basic_agent_node
from nodes.reason_node import reason_llm_node
from schema.types import OverallState

manager_class = ManagerAgent()
DEFAULT_MAX_ATTEMPTS = 5

SAFE_EDU_LINKS = [
    ("Investopedia", "https://www.investopedia.com/"),
    ("Khan Academy - Personal Finance", "https://www.khanacademy.org/college-careers-more/personal-finance"),
    ("SEC Investor.gov", "https://www.investor.gov/"),
]


def _unsafe_resources_reply(query: str) -> str:
    links = "\n".join([f"- {name}: {url}" for name, url in SAFE_EDU_LINKS])
    return (
        "I cannot help with that request safely. "
        "Here are educational resources related to finance to continue learning:\n"
        f"{links}\n\n"
        f"Topic focus from your query: {query}"
    )

def require_clarification(query: str) -> str:

    return (
         f"""Please provide more information as it seems to be personalized financial query.\n
         Refrain from asking queries which expect guranteed returns.
         Query: {query}
         """
    )


def manager_node(state: OverallState):

    profile = state.get("profile", None)
    if profile is None:
        profile = DecisionGate.decision_func(state["query"])

    decision = manager_class.manager(profile=profile)

    state["profile"] = decision.profile
    state["decision"] = decision
    state["route"] = decision.route
    state["confidence_cap"] = decision.confidence_cap
    state["reasons"] = state.get("reasons", []) + decision.reasons
    state["attempt_counter"] = state.get("attempt_counter", 0)
    state["max_attempts"] = state.get(
        "max_attempts", DEFAULT_MAX_ATTEMPTS
    )

    # unsafe then deny
    if decision.route == "unsafe":
        state["final_response"] = _unsafe_resources_reply(state["query"])
        state["final"] = True
        return state

    # Low risk - allow basic
    if decision.profile.risk == "low" and decision.route == "allow_basic":
        return basic_agent_node(state)
    
    if decision.route == "require_clarification":
        state["final_response"] = require_clarification(state["query"])
        state["final"] = True
        return state

    return reason_llm_node(state)
