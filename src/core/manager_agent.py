from schema.types import DecisionProfile, Decision

class ManagerAgent:

    def __init__(self):
        pass

    def manager(self, profile: DecisionProfile)-> Decision:
        
        # Risk Escalation (hard overide)
        if profile.danger_claim and profile.personalized:
            return Decision(
                profile= profile,
                route="unsafe",
                confidence_cap=0.0,
                reasons= profile.reasons+ ["Escalation threshold crossed!!"]
            )
        
        # Routing Logic
        if profile.risk=="high":
            return Decision(
                profile= profile,
                route="require_clarification",
                confidence_cap=0.2,
                reasons = profile.reasons+[f" {profile.risk}-risk personalized financial query"]
            )

        if profile.risk=="medium":
            return Decision(
                profile= profile,
                route="allow_reasoning",
                confidence_cap=0.5,
                reasons = profile.reasons+[f" {profile.risk}-risk query. Reasoning allowed with caution"]
            )

        if profile.risk=="low":
            return Decision(
                profile= profile,
                route="allow_basic",
                confidence_cap=0.8,
                reasons = profile.reasons+[f" {profile.risk}-risk informational query"]
            )
        