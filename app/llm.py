"""
LLM integration layer using Google Gemini.

Responsibilities:
1. Slot extraction — structured JSON from conversation history
2. Intent classification — what does the user want THIS turn?
3. Response generation — grounded natural language replies

Design decisions:
- Gemini 2.5 Flash: free tier, fast inference, good JSON mode
- Separate calls for slot extraction vs response generation
  (could be combined but separation is cleaner and more debuggable)
- All LLM outputs are validated before use
- Fallback behavior if LLM fails: return a generic "could you rephrase?" response
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from app import config

# ---------------------------------------------------------------------------
# Safety settings — set every standard category to BLOCK_NONE so that
# legitimate HR / hiring prompts are never silently dropped by Gemini's
# content filter.  finish_reason=SAFETY (code 2) causes the SDK to raise
# an exception with no text, which makes every evaluator turn time out.
# Our own safety layer (safety.py) handles prompt-injection and off-topic
# filtering, so we don't need Gemini's filter on top of that.
# ---------------------------------------------------------------------------
_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

from app.models import ConversationSlots
from app.prompts import (
    SYSTEM_PROMPT,
    SLOT_EXTRACTION_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    CLARIFICATION_PROMPT,
    RECOMMENDATION_PROMPT,
    REFINEMENT_PROMPT,
    COMPARISON_PROMPT,
    CONFIRMATION_PROMPT,
    format_conversation,
    format_assessments_for_prompt,
    format_slots_summary,
    format_assessment_for_prompt,
)

logger = logging.getLogger(__name__)


class LLMService:
    """
    Wraps Gemini API calls with structured extraction and response generation.
    """

    def __init__(self):
        if not config.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set — LLM calls will fail")
        print(f"API KEY IN USE: {str(config.GEMINI_API_KEY)[:8]}...")
        genai.configure(api_key=config.GEMINI_API_KEY)
        # Pass the system prompt here so it's baked into every call made by
        # this model instance — previously it was accepted as a parameter but
        # never forwarded, so the model had no system context at all.
        self._model = genai.GenerativeModel(
            config.GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        """
        Make a Gemini API call.
        
        Low temperature (0.3) for consistent, predictable outputs.
        Higher temperature would make responses more creative but less reliable
        for behavior probes.
        """
        try:
            gen_config_kwargs = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if json_mode:
                gen_config_kwargs["response_mime_type"] = "application/json"
                
            response = self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(**gen_config_kwargs),
                safety_settings=_SAFETY_SETTINGS,
            )

            # Detect safety block early and raise a descriptive error so the
            # caller's except-branch returns a fallback immediately rather than
            # letting response.text raise a confusing ValueError that looks like
            # a timeout to the evaluator.
            if not response.candidates:
                raise ValueError(
                    "Gemini returned no candidates — possible safety block even "
                    "with BLOCK_NONE settings (check quota or model availability)."
                )
            finish_reason = response.candidates[0].finish_reason
            # finish_reason 2 == SAFETY in the Gemini protobuf enum
            if finish_reason == 2:
                raise ValueError(
                    f"Gemini blocked this prompt (finish_reason=SAFETY). "
                    f"Prompt prefix: {prompt[:120]!r}"
                )

            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def analyze_user_input(self, messages: list[dict]) -> dict:
        """
        Combines Intent Classification and Slot Extraction into a SINGLE API call.
        """
        history = format_conversation(messages[:-1]) if len(messages) > 1 else "None"
        user_message = messages[-1].get("content", "") if messages else ""
        
        prompt = f"""
        Analyze the user's message and the conversation history.
        1. Classify the intent as one of: CLARIFY, RECOMMEND, REFINE, COMPARE, CONFIRM, OFF_TOPIC.
        2. Extract any specific job roles, skills, or seniority levels mentioned.
        
        Return ONLY valid JSON in this exact format:
        {{
            "intent": "RECOMMEND",
            "extracted_slots": {{
                "role": "financial analyst",
                "skills": ["excel", "accounting"],
                "seniority": "graduate"
            }}
        }}
        
        History: {history}
        User: {user_message}
        """
        response_text = self._call_llm(prompt, temperature=0.1, json_mode=True)
        try:
            # Clean up response just in case SDK didn't enforce valid JSON fully
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            return json.loads(cleaned)
        except Exception as e:
            logger.error(f"Failed to parse analyze_user_input JSON: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            return {"intent": "CLARIFY", "extracted_slots": {}}

    def extract_slots(self, messages: list[dict]) -> ConversationSlots:
        """
        Extract structured slots from conversation history.
        
        Uses the LLM in JSON extraction mode. Returns a ConversationSlots object.
        Falls back to empty slots on failure.
        """
        conversation_text = format_conversation(messages)
        prompt = SLOT_EXTRACTION_PROMPT.format(conversation=conversation_text)

        try:
            response_text = self._call_llm(
                prompt,
                temperature=0.1,  # Very low temp for structured extraction
                max_tokens=1024,
            )

            # Clean up response — remove markdown code fences if present
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)

            data = json.loads(cleaned)
            return ConversationSlots(**data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse slot extraction JSON: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            return ConversationSlots()
        except Exception as e:
            logger.error(f"Slot extraction failed: {e}")
            return ConversationSlots()

    def classify_intent(self, messages: list[dict]) -> str:
        """
        Classify the intent of the last user message.
        
        Returns one of: NEW_QUERY, CLARIFY_RESPONSE, REFINE, COMPARE, 
                         CONFIRM, OFF_TOPIC, GREET
        """
        valid_intents = {
            "NEW_QUERY", "CLARIFY_RESPONSE", "REFINE", "COMPARE",
            "CONFIRM", "OFF_TOPIC", "GREET"
        }

        conversation_text = format_conversation(messages)
        prompt = INTENT_CLASSIFICATION_PROMPT.format(conversation=conversation_text)

        try:
            response = self._call_llm(
                prompt,
                temperature=0.0,
                max_tokens=50,
            )
            intent = response.strip().upper().replace(" ", "_")

            if intent in valid_intents:
                return intent

            # Fuzzy match
            for valid in valid_intents:
                if valid in intent:
                    return valid

            logger.warning(f"Unknown intent: {intent}, defaulting to NEW_QUERY")
            return "NEW_QUERY"

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "NEW_QUERY"

    def generate_clarification(
        self,
        messages: list[dict],
        slots: ConversationSlots,
        missing_info: str,
    ) -> str:
        """Generate a clarifying question response."""
        prompt = CLARIFICATION_PROMPT.format(
            conversation=format_conversation(messages),
            slots_summary=format_slots_summary(slots),
            missing_info=missing_info,
        )
        try:
            return self._call_llm(prompt)
        except Exception:
            return "Could you tell me more about the role and what you're looking for in assessments?"

    def generate_recommendation(
        self,
        messages: list[dict],
        slots: ConversationSlots,
        assessments: list,
    ) -> str:
        """Generate a recommendation response with retrieved assessments."""
        prompt = RECOMMENDATION_PROMPT.format(
            conversation=format_conversation(messages),
            slots_summary=format_slots_summary(slots),
            assessments_data=format_assessments_for_prompt(assessments),
        )
        try:
            return self._call_llm(prompt)
        except Exception:
            return "Based on your requirements, here are the assessments I'd recommend:"

    def generate_refinement(
        self,
        messages: list[dict],
        current_recommendations: str,
        changes_summary: str,
        assessments: list,
    ) -> str:
        """Generate a refinement response."""
        prompt = REFINEMENT_PROMPT.format(
            conversation=format_conversation(messages),
            current_recommendations=current_recommendations,
            changes_summary=changes_summary,
            assessments_data=format_assessments_for_prompt(assessments),
        )
        try:
            return self._call_llm(prompt)
        except Exception:
            return "Updated the recommendations based on your changes."

    def generate_comparison(
        self,
        messages: list[dict],
        assessment_a,
        assessment_b,
    ) -> str:
        """Generate a grounded comparison between two assessments."""
        prompt = COMPARISON_PROMPT.format(
            conversation=format_conversation(messages),
            assessment_a=format_assessment_for_prompt(assessment_a),
            assessment_b=format_assessment_for_prompt(assessment_b),
        )
        try:
            return self._call_llm(prompt)
        except Exception:
            return (
                f"{assessment_a.name} and {assessment_b.name} serve different purposes. "
                f"Please refer to their catalog descriptions for detailed comparison."
            )

    def generate_confirmation(
        self,
        messages: list[dict],
        assessments: list,
    ) -> str:
        """Generate a confirmation/closing response."""
        prompt = CONFIRMATION_PROMPT.format(
            conversation=format_conversation(messages),
            assessments_data=format_assessments_for_prompt(assessments),
        )
        try:
            return self._call_llm(prompt)
        except Exception:
            return "Confirmed. Your assessment shortlist is finalized."


# --- Singleton ---
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service