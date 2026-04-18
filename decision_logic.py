from dataclasses import dataclass
from typing import Literal
import re
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # collapse spaces
    return text



def fuzzy_match(word: str, target: str, threshold: float = 0.7) -> bool:
    score = SequenceMatcher(None, word, target).ratio()
    return score >= threshold


CERTAINTY_WORDS = ["guaranteed", "assured", "sure", "fixed"]
PROFIT_WORDS = ["profit", "return", "gain", "money"]
AMPLIFICATION_WORDS = ["double", "triple", "times"]

def contains_family(words, family):
    for w in words:
        for f in family:
            if fuzzy_match(w, f):
                return True
    return False

def contains_numeric_amplification(words) -> bool:
    patterns = [
        r"\b\d+x\b",        # 10x, 5x
        r"\b\d+\s*times\b", # 10 times
        r"\b\d+%\b"         # 100%
    ]
    for w in words:
        if any(re.search(p, w) for p in patterns):
            return True
    return False


def detect_danger_claim(normalized_query: str) -> bool:

    words = normalized_query.split()

    # Check for words from families
    certain_words = contains_family(words, CERTAINTY_WORDS)
    profit_words = contains_family(words, PROFIT_WORDS)
    amp_words = contains_family(words, AMPLIFICATION_WORDS)
    amp_num = contains_numeric_amplification(normalized_query)

    # Any two words together
    signals = sum([certain_words, profit_words, amp_words, amp_num])
    return signals>=2


def escalate_risk(intent: str, personalized: bool, danger_claim:bool)-> bool:
    signals = [
        intent=="recommendation",
        personalized,
        danger_claim 
    ]
    return sum(signals)>=2


