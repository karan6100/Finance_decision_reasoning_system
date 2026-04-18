from core.decision_gate import DecisionGate
from schema.types import DecisionProfile

def analyze_financial_query(query: str) -> DecisionProfile:
    """
    Analyzes a financial query and returns a decision profile with intent, risk level,
    personalization flag, danger claim detection, and reasons.

    Args:
        query (str): The user's financial question or statement

    Returns:
        DecisionProfile: Contains intent, risk, personalized, danger_claim, and reasons
    """
    return DecisionGate.decision_func(query)

if __name__ == "__main__":
    # Example usage for testing
    test_queries = [
        "What is a mutual fund?",
        "Should I invest in gold right now?",
        "I earn 15 LPA, how much should I invest in stocks?",
        "Which is better, FD or equity?",
        "How to get guaranteed returns in trading?",
        "My savings are 50000. How can I get returns of 100000 by investing it?",
        "I want to invest in gold"
    ]

    for query in test_queries:
        print(f"Query: {query}")
        profile = analyze_financial_query(query)
        print(f"Intent: {profile.intent}")
        print(f"Risk: {profile.risk}")
        print(f"Personalized: {profile.personalized}")
        print(f"Danger Claim: {profile.danger_claim}")
        print(f"Reasons: {profile.reasons}")
        print("-" * 50)