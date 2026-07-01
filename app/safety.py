"""
Safety layer for the SHL Assessment Recommender.

Handles three categories:
1. Prompt injection — "ignore instructions", "you are now", system prompt leaks
2. Off-topic requests — legal advice (C7), salary, general hiring advice
3. Scope violations — competitor products, assessment generation

Design decision: Keyword-based first pass is fast and catches obvious cases.
The LLM handles edge cases via its system prompt. Belt-and-suspenders approach.

Why not LLM-only safety? Because a prompt injection might convince the LLM 
to bypass its own safety instructions. The keyword layer can't be prompt-injected.
"""

from __future__ import annotations

import re
import logging

from app import config

logger = logging.getLogger(__name__)

# --- Prompt Injection Patterns ---
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"you\s+are\s+now\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if|a|an)",
    r"forget\s+(everything|all|your)",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+(instructions|prompt|system)",
    r"what\s+are\s+your\s+(instructions|rules)",
    r"repeat\s+(your|the)\s+(instructions|prompt|system)",
    r"output\s+(your|the)\s+(instructions|prompt|system)",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"\[INST\]",
    r"<\|im_start\|>",
    r"<\|system\|>",
]

# --- Off-Topic Patterns ---
OFF_TOPIC_PATTERNS = [
    r"(what|how\s+much)\s+(salary|pay|compensation|wage)",
    r"(legal|law|lawyer|attorney|counsel)\s+(advice|requirement|obligation|question)",
    r"(sue|lawsuit|litigation|court)",
    r"(write|draft|compose)\s+(a\s+)?(resume|cv|cover\s+letter|email)",
    r"(interview|behavioral)\s+questions?\s+(for|to\s+ask)",
    r"(competitor|gallup|hogan|wonderlic|predictive\s+index)",
    r"(recipe|weather|news|sports|movie|music|joke)",
    r"(code|program|script|python|javascript)\s+(for|to|that)",
]

# --- Legal Patterns (inspired by C7) ---
LEGAL_PATTERNS = [
    r"(legally\s+required|legal\s+requirement|legal\s+obligation)",
    r"(comply|compliance)\s+with\s+(law|regulation|statute)",
    r"(are\s+we|am\s+i)\s+(required|obligated|mandated)",
    r"(does\s+this|will\s+this)\s+(satisfy|meet|fulfill)\s+(legal|regulatory)",
    r"(employment\s+law|labor\s+law|discrimination\s+law)",
    r"(EEOC|ADA|GDPR)\s+(compliance|requirements?)",
]


class SafetyChecker:
    """
    Multi-layer safety checker.
    
    Returns (is_safe, category, message) tuples.
    - is_safe=True: message is safe to process
    - is_safe=False: message should trigger a refusal response
    """
    
    def __init__(self):
        # Pre-compile regex patterns for performance
        self._injection_re = [
            re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
        ]
        self._off_topic_re = [
            re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS
        ]
        self._legal_re = [
            re.compile(p, re.IGNORECASE) for p in LEGAL_PATTERNS
        ]
    
    def check(self, message: str) -> tuple[bool, str, str]:
        """
        Check a user message for safety issues.
        
        Returns:
            (is_safe, category, refusal_message)
            - is_safe: True if message is safe
            - category: "safe", "injection", "off_topic", "legal", "too_long"
            - refusal_message: Suggested refusal text (empty if safe)
        """
        # --- Length check ---
        if len(message) > config.MAX_MESSAGE_LENGTH:
            return (
                False,
                "too_long",
                "That message is unusually long. Could you summarize what you're looking for in assessment recommendations?"
            )
        
        # --- Prompt injection ---
        for pattern in self._injection_re:
            if pattern.search(message):
                logger.warning(f"Prompt injection detected: {pattern.pattern}")
                return (
                    False,
                    "injection",
                    "I'm designed to help you find SHL assessments. How can I help you with assessment selection?"
                )
        
        # --- Legal questions ---
        for pattern in self._legal_re:
            if pattern.search(message):
                logger.info(f"Legal question detected: {pattern.pattern}")
                return (
                    False,
                    "legal",
                    "That's a legal compliance question outside what I can advise on — "
                    "I can help you select assessments, but not interpret regulatory "
                    "obligations. Your legal or compliance team is the right resource "
                    "for that. Is there anything else I can help with regarding "
                    "assessment selection?"
                )
        
        # --- Off-topic ---
        for pattern in self._off_topic_re:
            if pattern.search(message):
                logger.info(f"Off-topic detected: {pattern.pattern}")
                return (
                    False,
                    "off_topic",
                    "I specialize in SHL assessment recommendations. "
                    "I'm not able to help with that particular request, but I'd be happy to "
                    "help you find the right assessments for your hiring needs. "
                    "What role are you looking to assess?"
                )
        
        return (True, "safe", "")
    
    def check_recommendations_grounded(
        self, 
        recommendations: list[dict],
        valid_urls: set[str],
    ) -> list[dict]:
        """
        Post-process: ensure all recommendation URLs come from the catalog.
        
        This is the final grounding check before sending the response.
        Removes any recommendation with an invalid URL.
        """
        grounded = []
        for rec in recommendations:
            url = rec.get("url", "")
            if url in valid_urls:
                grounded.append(rec)
            else:
                logger.error(f"Removed recommendation with invalid URL: {url}")
        return grounded


# --- Singleton ---
_checker = None

def get_safety_checker() -> SafetyChecker:
    global _checker
    if _checker is None:
        _checker = SafetyChecker()
    return _checker
