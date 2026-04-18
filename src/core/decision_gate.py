from dataclasses import dataclass
from typing import Literal
from schema.types import DecisionProfile
from core import utils


class DecisionGate:
    
    @staticmethod
    def decision_func(query: str) -> DecisionProfile:
        """ Classifies the given query into 
            Intents(what kind of help is asked):
                1. Factual (low-risk).
                2. Explanation/Educational.
                3. Scenario analysis.
                4. Recommendation/Advice.
                5. Action Execution.
            and Risks (what damage could a wrong answer do):
                1. low risk (definitions/history)
                2. meduim  (general strategy, trade-offs)
                3. high (personal finance decisions, timing, amounts)
            """
        reasons = []
        query = utils.normalize_text(query) 

        # Intents classification
        signals = utils.extract_signals(query)
        intent = signals.intent
        reasons.append(f"Intent Detected: {intent}")

        # personalized = is_personalized(query)
        personalized = signals.personalized
        if personalized:
            reasons.append(f"Personalization Detected")

        # Detect dangerous language
        danger_claim = utils.detect_danger_claim(query)
        if danger_claim:
            reasons.append("Dangerous financial language detected")
        
        # Risk Assessment
        risk = utils.assess_risk(signals)
        reasons.append(f"Assessed risk level: {risk}")
        
        return DecisionProfile(intent= intent,
                            risk=risk,
                            personalized= personalized,
                            danger_claim= danger_claim,
                            reasons= reasons
            
        )


    

    
