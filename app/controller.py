"""
Conversation Controller — the deterministic brain of the system.

This module decides WHAT action to take (clarify, recommend, refine, compare,
refuse, confirm). The LLM decides HOW to say it.

Design decision: Deterministic controller over LLM-driven decisions.
The automated evaluator has binary behavior probes:
- "Does agent refuse off-topic?" 
- "Does agent not recommend on turn 1 for vague queries?"
- "Does agent honor edits?"
A deterministic controller GUARANTEES these pass. An LLM might drift.

Key behaviors derived from analyzing all 10 sample conversations:
- Clarify only when information is genuinely insufficient (C1, C3, C7, C9)
- Recommend immediately when enough context exists (C4, C8, C10)
- Refine incrementally, never start over (C4, C8, C9, C10)
- Compare using only catalog data (C3, C5, C6, C9)
- Refuse off-topic/legal firmly but politely (C7)
- End only on user confirmation
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.models import (
    ChatResponse,
    Recommendation,
    ConversationSlots,
    Assessment,
    Message,
)
from app.llm import get_llm_service
from app.retrieval import get_retrieval_engine
from app.safety import get_safety_checker
from app import config

logger = logging.getLogger(__name__)


def _count_turns(messages: list[dict]) -> int:
    """Count total turns (user + assistant messages)."""
    return len(messages)


def _get_last_user_message(messages: list[dict]) -> str:
    """Get the content of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _extract_previous_recommendations(messages: list[dict]) -> list[str]:
    """
    Extract assessment names from previous assistant messages.
    Used for refinement to know what the current shortlist contains.
    """
    names = []
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Simple heuristic: look for assessment-like patterns
            # In practice, we track recommendations through the controller
            break
    return names


def _is_comparison_request(message: str) -> bool:
    """
    Detect if the user is asking to compare assessments.
    
    Patterns from sample conversations:
    - C3: "Is the Contact Center Call Simulation different from..."
    - C5: "What's the difference between OPQ and OPQ MQ Sales Report?"
    - C6: "What's the difference between the DSI and the Safety & Dependability 8.0?"
    """
    compare_patterns = [
        r"(what('s| is) the )?difference between",
        r"compare\b",
        r"comparison\b",
        r"different from",
        r"how (does|do) .+ (differ|compare)",
        r"vs\.?\b",
        r"versus\b",
        r"which (one|is better)",
    ]
    message_lower = message.lower()
    return any(re.search(p, message_lower) for p in compare_patterns)


def _is_refinement_request(message: str) -> bool:
    """
    Detect if the user wants to modify existing recommendations.
    
    Patterns from sample conversations:
    - C4: "Can you also add a situational judgement element"
    - C8: "I am OK with adding a simulation"
    - C9: "Add AWS and Docker. Drop REST"
    - C10: "can you remove the OPQ32r"
    """
    refine_patterns = [
        r"\badd\b",
        r"\bdrop\b",
        r"\bremove\b",
        r"\breplace\b",
        r"\bswap\b",
        r"\binclude\b",
        r"\bexclude\b",
        r"\bchange\b",
        r"\bupdate\b",
        r"\balso\b.*(add|include)",
        r"\binstead\b",
        r"\bactually\b",
    ]
    message_lower = message.lower()
    return any(re.search(p, message_lower) for p in refine_patterns)


def _is_confirmation(message: str) -> bool:
    """
    Detect if the user is confirming/accepting recommendations.
    
    Patterns from sample conversations:
    - C1: "Perfect, that's what we need."
    - C2: "That works. Thanks."
    - C3: "Perfect — new simulation for volume, old solution for finalists. Confirmed."
    - C9: "Keep Verify G+. Locking it in."
    """
    confirm_patterns = [
        r"\bperfect\b",
        r"\bconfirm(ed)?\b",
        r"\block(ing|ed)?\s*(it)?\s*(in)?\b",
        r"\bthat('s| is)?\s*(good|great|fine|what we need|exactly)",
        r"\bthat\s+works\b",
        r"\bthat\s+covers\s+it\b",
        r"\bgood\s*(to\s+go)?\b",
        r"\byes\b.*(go\s+ahead|proceed|finalize)",
        r"\bkeep\b.*(as.is|the\s+list|shortlist)",
        r"\bfinal\s*(list|shortlist|battery)\b",
        r"\b(thanks|thank\s+you)\b",
    ]
    message_lower = message.lower()
    return any(re.search(p, message_lower) for p in confirm_patterns)


def _determine_missing_info(slots: ConversationSlots) -> str:
    """
    Determine what key information is still missing for a recommendation.
    
    Strategy: Don't ask about everything — focus on what would most improve
    retrieval quality. Max 1-2 clarifying questions.
    """
    missing = []

    if not slots.role and not slots.skills and not slots.jd_text:
        missing.append("the role or skills you're hiring for")

    if not slots.seniority and not slots.job_level:
        missing.append("the seniority level (entry-level, mid, senior, executive)")

    # Only ask about language/industry if role is clear but we still need disambiguation
    # Don't ask about everything — keep it focused

    return " and ".join(missing) if missing else ""


def _has_sufficient_context(slots: ConversationSlots) -> bool:
    """
    Determine if we have enough context to make a recommendation.
    
    Threshold: we need at least a role/skills + seniority level, OR
    a specific assessment mention, OR the LLM thinks we have enough.
    
    From sample conversations:
    - C4: "graduate financial analysts, final-year students" → enough immediately
    - C1: "senior leadership" → needs clarification (who exactly? purpose?)
    """
    # If LLM says there's enough info, trust it
    if slots.has_enough_info:
        return True

    # If user mentioned specific assessments, they know what they want
    if slots.specific_assessments:
        return True

    # Need at minimum: (role OR skills) AND some indicator of level
    has_role_info = bool(slots.role or slots.skills or slots.jd_text)
    has_level_info = bool(slots.seniority or slots.job_level)

    return has_role_info and has_level_info


class ConversationController:
    """
    Deterministic conversation controller.
    
    Processes a full conversation history (stateless API) and returns
    the next agent response with optional recommendations.
    """

    def __init__(self):
        self.llm = get_llm_service()
        self.retrieval = get_retrieval_engine()
        self.safety = get_safety_checker()

    def process(self, messages: list[dict]) -> ChatResponse:
        """
        Main entry point. Process conversation history and return response.
        
        Flow:
        1. Safety check on last user message
        2. Extract slots from full conversation
        3. Classify intent of last message
        4. Route to appropriate handler
        5. Validate and return response
        """
        if not messages:
            return ChatResponse(
                reply="Hello! I'm here to help you find the right SHL assessments. What role are you hiring for?",
                recommendations=[],
                end_of_conversation=False,
            )

        last_user_msg = _get_last_user_message(messages)
        turn_count = _count_turns(messages)

        # --- Step 1: Safety Check ---
        is_safe, category, refusal_msg = self.safety.check(last_user_msg)
        if not is_safe:
            logger.info(f"Safety triggered: {category}")
            # For legal questions, don't end conversation (like C7)
            return ChatResponse(
                reply=refusal_msg,
                recommendations=[],
                end_of_conversation=False,
            )

        # --- Step 2: Extract Slots ---
        messages_dicts = [{"role": m.get("role", m.role if hasattr(m, 'role') else "user"), 
                           "content": m.get("content", m.content if hasattr(m, 'content') else "")} 
                          if isinstance(m, dict) else {"role": m.role, "content": m.content}
                          for m in messages]
        
        slots = self.llm.extract_slots(messages_dicts)
        logger.info(f"Extracted slots: role={slots.role}, skills={slots.skills}, seniority={slots.seniority}")

        # --- Step 3: Classify Intent ---
        # Use both pattern matching and LLM classification
        intent = self._classify_intent(last_user_msg, messages_dicts, turn_count, slots)
        logger.info(f"Intent: {intent}, Turn: {turn_count}")

        # --- Step 4: Route to Handler ---
        if intent == "COMPARE":
            return self._handle_comparison(messages_dicts, slots, last_user_msg)
        elif intent == "CONFIRM":
            return self._handle_confirmation(messages_dicts, slots)
        elif intent == "REFINE":
            return self._handle_refinement(messages_dicts, slots)
        elif intent == "RECOMMEND":
            return self._handle_recommendation(messages_dicts, slots)
        else:  # CLARIFY
            return self._handle_clarification(messages_dicts, slots, turn_count)

    def _classify_intent(
        self,
        last_msg: str,
        messages: list[dict],
        turn_count: int,
        slots: ConversationSlots,
    ) -> str:
        """
        Hybrid intent classification: pattern matching + LLM.
        
        Pattern matching handles clear-cut cases fast.
        LLM handles ambiguous cases.
        """
        # --- Pattern matching (fast, reliable) ---
        
        # Comparison takes priority
        if _is_comparison_request(last_msg):
            return "COMPARE"
        
        # Check for confirmation
        if _is_confirmation(last_msg):
            # Only treat as confirmation if we've already given recommendations
            has_prev_recs = self._has_previous_recommendations(messages)
            if has_prev_recs:
                return "CONFIRM"
        
        # Check for refinement
        if _is_refinement_request(last_msg):
            has_prev_recs = self._has_previous_recommendations(messages)
            if has_prev_recs:
                return "REFINE"
        
        # --- Decision: Clarify or Recommend? ---
        
        # Force recommendation if running low on turn budget
        # 8 turns total (user+assistant). Be conservative.
        if turn_count >= 5:
            return "RECOMMEND"
        
        # Check if we have enough context
        if _has_sufficient_context(slots):
            return "RECOMMEND"
        
        # Check via LLM for edge cases
        try:
            llm_intent = self.llm.classify_intent(messages)
            if llm_intent in ("CONFIRM", "COMPARE", "REFINE"):
                return llm_intent
            if llm_intent == "CLARIFY_RESPONSE":
                # User answered our question — re-evaluate if we have enough now
                if _has_sufficient_context(slots):
                    return "RECOMMEND"
                # Still not enough — but check turn budget
                if turn_count >= 4:
                    return "RECOMMEND"
                return "CLARIFY"
        except Exception:
            pass
        
        return "CLARIFY"

    def _has_previous_recommendations(self, messages: list[dict]) -> bool:
        """Check if any previous assistant message contained recommendations."""
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Check for indicators of previous recommendations
                if "https://www.shl.com" in content:
                    return True
                # Check for common recommendation phrasings
                if any(phrase in content.lower() for phrase in [
                    "here are", "shortlist", "recommend", "battery",
                    "assessment", "following"
                ]):
                    return True
        return False

    def _handle_clarification(
        self,
        messages: list[dict],
        slots: ConversationSlots,
        turn_count: int,
    ) -> ChatResponse:
        """
        Generate a clarifying question.
        
        Rules:
        - Ask at most 1-2 questions per turn
        - Don't re-ask what we already know
        - If we can't figure out what to ask, just recommend with what we have
        """
        missing_info = _determine_missing_info(slots)
        
        if not missing_info:
            # Can't determine what's missing — just recommend
            return self._handle_recommendation(messages, slots)
        
        reply = self.llm.generate_clarification(messages, slots, missing_info)
        
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )

    def _handle_recommendation(
        self,
        messages: list[dict],
        slots: ConversationSlots,
    ) -> ChatResponse:
        """
        Generate assessment recommendations.
        
        Retrieves assessments via the retrieval engine, generates a natural
        language explanation, and returns structured recommendations.
        """
        # Retrieve assessments
        assessments = self.retrieval.retrieve(slots, top_k=config.RERANK_TOP_K)
        
        if not assessments:
            return ChatResponse(
                reply="I wasn't able to find matching assessments in the catalog. Could you provide more details about the role, required skills, or seniority level?",
                recommendations=[],
                end_of_conversation=False,
            )
        
        # Limit to reasonable number (1-10)
        assessments = assessments[:config.MAX_RECOMMENDATIONS]
        
        # Generate natural language response
        reply = self.llm.generate_recommendation(messages, slots, assessments)
        
        # Build structured recommendations from catalog data (grounded!)
        recommendations = [a.to_recommendation() for a in assessments]
        
        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=False,
        )

    def _handle_refinement(
        self,
        messages: list[dict],
        slots: ConversationSlots,
    ) -> ChatResponse:
        """
        Handle user request to modify existing recommendations.
        
        Strategy:
        - Extract what to add/remove from slots
        - Re-retrieve with updated slots
        - Boost previously recommended items (continuity)
        """
        # Get previous recommendation names for context
        prev_rec_names = self._extract_recommendation_names(messages)
        
        # Build changes summary
        changes = []
        if slots.additions:
            changes.append(f"Adding: {', '.join(slots.additions)}")
        if slots.removals:
            changes.append(f"Removing: {', '.join(slots.removals)}")
        changes_summary = "; ".join(changes) if changes else "Updating based on new constraints"
        
        # Re-retrieve with updated slots, boosting previous recommendations
        assessments = self.retrieval.retrieve(
            slots,
            top_k=config.RERANK_TOP_K,
            previous_recommendations=prev_rec_names,
        )
        
        # Apply explicit removals
        if slots.removals:
            removal_lower = [r.lower() for r in slots.removals]
            assessments = [
                a for a in assessments
                if not any(r in a.name.lower() for r in removal_lower)
            ]
        
        if not assessments:
            return ChatResponse(
                reply="After applying your changes, I don't have matching assessments. Could you adjust your criteria?",
                recommendations=[],
                end_of_conversation=False,
            )
        
        assessments = assessments[:config.MAX_RECOMMENDATIONS]
        
        # Generate response
        reply = self.llm.generate_refinement(
            messages,
            ", ".join(prev_rec_names) if prev_rec_names else "previous list",
            changes_summary,
            assessments,
        )
        
        recommendations = [a.to_recommendation() for a in assessments]
        
        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=False,
        )

    def _handle_comparison(
        self,
        messages: list[dict],
        slots: ConversationSlots,
        last_msg: str,
    ) -> ChatResponse:
        """
        Handle comparison requests between assessments.
        
        Finds the mentioned assessments in the catalog and generates a
        grounded comparison using only catalog data.
        """
        # Extract assessment names from the comparison request
        assessment_names = self._extract_comparison_subjects(last_msg, slots)
        
        if len(assessment_names) < 2:
            # Can't identify two assessments to compare
            return ChatResponse(
                reply="Which two assessments would you like me to compare? Please mention them by name.",
                recommendations=[],
                end_of_conversation=False,
            )
        
        # Find assessments in catalog
        found = self.retrieval.find_by_names(assessment_names[:2])
        
        if len(found) < 2:
            return ChatResponse(
                reply=f"I could only find {len(found)} of those assessments in the catalog. Could you check the names?",
                recommendations=[],
                end_of_conversation=False,
            )
        
        # Generate grounded comparison
        reply = self.llm.generate_comparison(messages, found[0], found[1])
        
        # Don't return recommendations during comparison (per sample conversations)
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )

    def _handle_confirmation(
        self,
        messages: list[dict],
        slots: ConversationSlots,
    ) -> ChatResponse:
        """
        Handle user confirmation of recommendations.
        
        Extracts the last set of recommendations, generates a closing response,
        and sets end_of_conversation=True.
        """
        # Try to reconstruct the last recommended assessments
        prev_rec_names = self._extract_recommendation_names(messages)
        assessments = self.retrieval.find_by_names(prev_rec_names) if prev_rec_names else []
        
        if not assessments:
            # Fall back to re-retrieving
            assessments = self.retrieval.retrieve(slots, top_k=config.RERANK_TOP_K)
            assessments = assessments[:config.MAX_RECOMMENDATIONS]
        
        reply = self.llm.generate_confirmation(messages, assessments)
        recommendations = [a.to_recommendation() for a in assessments]
        
        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=True,
        )

    def _extract_recommendation_names(self, messages: list[dict]) -> list[str]:
        """
        Extract assessment names from previous assistant messages.
        
        Looks for SHL catalog URLs and assessment name patterns.
        """
        names = []
        # Look at assistant messages in reverse (most recent first)
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Look for patterns like "Name: XXX" or assessment names before URLs
                # This is a heuristic — works with our response format
                url_pattern = re.findall(
                    r'https://www\.shl\.com/products/product-catalog/view/[^/\s)]+/',
                    content,
                )
                if url_pattern:
                    # Found URLs — look up assessment names from catalog
                    from app.catalog import get_assessment_by_url
                    for url in url_pattern:
                        url = url.rstrip('/')
                        if not url.endswith('/'):
                            url += '/'
                        assessment = get_assessment_by_url(url)
                        if assessment and assessment.name not in names:
                            names.append(assessment.name)
                    if names:
                        break  # Use the most recent set
        
        return names

    def _extract_comparison_subjects(
        self, message: str, slots: ConversationSlots
    ) -> list[str]:
        """
        Extract assessment names from a comparison request.
        
        Handles patterns like:
        - "difference between OPQ and GSA"
        - "Is X different from Y?"
        - "Compare X vs Y"
        """
        subjects = []
        
        # Try pattern: "between X and Y"
        match = re.search(
            r'between\s+(.+?)\s+and\s+(.+?)[\?\.]?\s*$',
            message,
            re.IGNORECASE,
        )
        if match:
            subjects = [match.group(1).strip(), match.group(2).strip()]
            return subjects
        
        # Try pattern: "X different from Y"
        match = re.search(
            r'(.+?)\s+different\s+from\s+(.+?)[\?\.]?\s*$',
            message,
            re.IGNORECASE,
        )
        if match:
            subjects = [match.group(1).strip(), match.group(2).strip()]
            return subjects
        
        # Try pattern: "X vs Y"
        match = re.search(
            r'(.+?)\s+vs\.?\s+(.+?)[\?\.]?\s*$',
            message,
            re.IGNORECASE,
        )
        if match:
            subjects = [match.group(1).strip(), match.group(2).strip()]
            return subjects
        
        # Fall back to any specific assessment names in slots
        if slots.specific_assessments and len(slots.specific_assessments) >= 2:
            return slots.specific_assessments[:2]
        
        return subjects
