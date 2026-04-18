from typing import Literal, TypedDict, Optional, Any
from dataclasses import dataclass

Intent= Literal["factual", "education", "analysis", "recommendation"]
RiskLevel = Literal["low", "medium", "high"]
Route = Literal["allow_basic", "allow_reasoning", "require_clarification", "unsafe"]




@dataclass
class DecisionProfile:
    intent: Intent
    risk: RiskLevel
    personalized: bool
    danger_claim: bool
    reasons: list[str]


@dataclass
class Decision:
    profile: DecisionProfile
    route: Route
    confidence_cap: float
    reasons: list[str]
    


@dataclass
class QuerySignals:
    intent: Intent
    personalized: bool

@dataclass
class ValidationInput:
    query: str
    source_agent: Literal["basic", "reasoning"]
    candidate_response: str
    intent: str
    risk: RiskLevel
    personalized: bool 


class OverallState(TypedDict, total=False):

    query: str
    # Governance
    profile: DecisionProfile
    decision: Decision
    route: Route
    confidence_cap: float

    # Basic-path response only
    facts_response: str

    # Reasoning-path response only
    reasoning_response: str

    # Unified output across all routes
    final_response: str

    # Reason
    reasons: list[str]

    # Validation 
    valid_status: Literal["pass", "fail"]
    valid_reviews: dict[str, Any]
    validation_input: ValidationInput  # Store for downstream nodes
    validation_meta: dict[str, Any]

    # Attempt Counters
    attempt_counter: int
    max_attempts: int

    #Final op
    final: bool


@dataclass
class BasicInput:
    query: str

@dataclass
class BasicOutput:
    output: str

@dataclass
class ReasoningInput:
    query: str
    intent: Intent
    risk: RiskLevel

@dataclass
class ReasonOutput:
    reasoning: str

