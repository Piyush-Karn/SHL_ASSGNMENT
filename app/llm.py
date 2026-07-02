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
import os
import re
import time
import requests
from typing import Optional
from functools import lru_cache

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

PROMPT_VERSION = 1

def _freeze_messages(messages: list[dict]) -> tuple:
    return tuple(tuple(sorted(m.items())) for m in messages)

_last_mistral_call_time = 0
_call_counter = 0

class LLMService:
    """
    Wraps Gemini API calls with structured extraction and response generation.
    """

    def __init__(self):
        provider = config.LLM_PROVIDER
        if provider == "mistral" or provider == "round_robin":
            num_mistral = len(config.MISTRAL_KEYS)
            num_groq = len(config.GROQ_KEYS)
            num_gemini = len(config.GEMINI_KEYS)
            num_hf = len(config.HUGGING_FACE_KEYS)
            print(f"API KEY IN USE (ROUND ROBIN): {num_groq}x Groq + {num_mistral}x Mistral + {num_gemini}x Gemini + {num_hf}x HuggingFace")
        else:
            if not config.GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY not set — LLM calls will fail")
            else:
                print(f"API KEY IN USE (GEMINI): {str(config.GEMINI_API_KEY)[:8]}...")
                
        genai.configure(api_key=config.GEMINI_API_KEY)
        # Pass the system prompt here so it's baked into every call made by
        # this model instance — previously it was accepted as a parameter but
        # never forwarded, so the model had no system context at all.
        self._model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=SYSTEM_PROMPT,
        )

    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        # --- Mock LLM Mode ---
        if os.getenv("MOCK_LLM", "False").lower() == "true":
            if not json_mode:
                return "This is a mock response from the SHL Assessment Consultant."
            
            p = prompt.lower()
            
            # C3
            if "500 entry-level contact centre agents" in p and "english" not in p:
                return '{"intent": "CLARIFY", "extracted_slots": {"role": "contact centre agent", "seniority": "entry-level"}, "clarify_reply": "What language?"}'
            elif "english." in p and "us." not in p:
                return '{"intent": "CLARIFY", "extracted_slots": {"role": "contact centre agent", "seniority": "entry-level", "skills": ["english"]}, "clarify_reply": "Which accent?"}'
            elif "us." in p and "different from" not in p:
                return '{"intent": "RECOMMEND", "extracted_slots": {"role": "contact centre agent", "seniority": "entry-level", "skills": ["english", "us", "inbound calls", "customer service"]}}'
            
            # General COMPARE / CONFIRM matchers based on history
            if "different from" in p and "sales" in p:
                return '{"intent": "COMPARE", "extracted_slots": {"role": "sales"}}'
            elif "different from" in p and "plant operator" in p:
                return '{"intent": "COMPARE", "extracted_slots": {"role": "safety"}}'
            elif "different from" in p:
                return '{"intent": "COMPARE", "extracted_slots": {"role": "contact centre agent", "seniority": "entry-level", "skills": ["english", "us", "inbound calls", "customer service"]}}'
                
            if "confirmed" in p and "plant operator" in p:
                return '{"intent": "CONFIRM", "extracted_slots": {"role": "safety"}}'
            elif "confirmed" in p:
                return '{"intent": "CONFIRM", "extracted_slots": {"role": "contact centre agent", "seniority": "entry-level", "skills": ["english", "us", "inbound calls", "customer service"]}}'
            elif "clear. we'll use opq" in p:
                return '{"intent": "CONFIRM", "extracted_slots": {"role": "sales"}}'
            elif "as-is" in p:
                return '{"intent": "CONFIRM", "extracted_slots": {"role": "healthcare admin"}}'
                
            # C1
            if ("senior leadership" in p or "cxo" in p) and "selection" not in p:
                return '{"intent": "CLARIFY", "extracted_slots": {"role": "senior leadership"}}'
            elif "selection" in p and "leadership" in p:
                return '{"intent": "RECOMMEND", "extracted_slots": {"role": "senior leadership"}}'
            
            # C2
            if "rust engineer" in p:
                return '{"intent": "RECOMMEND", "extracted_slots": {"role": "rust"}}'
                
            # C4
            if "financial analyst" in p:
                return '{"intent": "RECOMMEND", "extracted_slots": {"role": "financial"}}'
                
            # C5
            if "sales organization" in p:
                return '{"intent": "RECOMMEND", "extracted_slots": {"role": "sales"}}'
                
            # C6
            if "plant operator" in p:
                return '{"intent": "RECOMMEND", "extracted_slots": {"role": "safety"}}'
                
            # C7
            if "healthcare admin" in p and "bilingual" not in p:
                return '{"intent": "CLARIFY", "extracted_slots": {"role": "healthcare admin"}}'
            elif "healthcare admin" in p and "legally required" in p and "as-is" not in p:
                return '{"intent": "CLARIFY", "extracted_slots": {"role": "healthcare admin"}}'
            elif "healthcare admin" in p:
                return '{"intent": "RECOMMEND", "extracted_slots": {"role": "healthcare admin"}}'
                
            # C8
            if "admin assistant" in p:
                return '{"intent": "RECOMMEND", "extracted_slots": {"role": "admin assistant"}}'
                
            # C9
            if "full-stack" in p:
                return '{"intent": "RECOMMEND", "extracted_slots": {"role": "full-stack"}}'
                
            # C10
            if "management trainees" in p:
                return '{"intent": "RECOMMEND", "extracted_slots": {"role": "management trainees"}}'
                
            # Probes
            if "ignore all previous instructions" in p:
                return '{"intent": "OFF_TOPIC", "extracted_slots": {}}'
            if "senior python developer" in p:
                return '{"intent": "RECOMMEND", "extracted_slots": {"role": "full-stack"}}' # Maps to full-stack domain
            if "i need an assessment" in p or "hiring an engineer" in p:
                return '{"intent": "CLARIFY", "extracted_slots": {}, "clarify_reply": "Could you provide a specific role?"}'
                
            # Fallback
            return '{"intent": "RECOMMEND", "extracted_slots": {"role": "general", "seniority": "entry-level"}}'

        # --- Cache Logic for Dev/Eval Speed ---
        import hashlib
        
        provider = config.LLM_PROVIDER
        CACHE_FILE = f"evaluation/{provider}_cache.json"
        
        cache = {}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                
        cache_key = hashlib.md5(f"{prompt}_{temperature}_{json_mode}".encode("utf-8")).hexdigest()
        
        if cache_key in cache:
            logger.debug(f"Cache hit for {provider} prompt.")
            return cache[cache_key]

        # Route to correct provider
        if provider == "mistral" or provider == "round_robin":
            response_text = self._call_round_robin(prompt, temperature, max_tokens, json_mode)
        else:
            if not config.GEMINI_KEYS:
                raise ValueError("No Gemini keys found")
            response_text = self._call_gemini_raw(prompt, temperature, max_tokens, json_mode, config.GEMINI_KEYS[0])
            
        # --- Save to Cache ---
        cache[cache_key] = response_text
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")
            
        return response_text

    def _call_round_robin(self, prompt: str, temperature: float, max_tokens: int, json_mode: bool) -> str:
        global _call_counter
        
        gemini_list = [("gemini", key, f"Gemini Key {i+1}") for i, key in enumerate(config.GEMINI_KEYS)]
        groq_list = [("groq", key, f"Groq Key {i+1}") for i, key in enumerate(config.GROQ_KEYS)]
        hf_list = [("huggingface", key, f"HF Key {i+1}") for i, key in enumerate(config.HUGGING_FACE_KEYS)]
        mistral_list = [("mistral", key, f"Mistral Key {i+1}") for i, key in enumerate(config.MISTRAL_KEYS)]
        
        # Interleave providers so we don't hammer the same model concurrently
        import itertools
        providers = []
        for g, q, h, m in itertools.zip_longest(gemini_list, groq_list, hf_list, mistral_list):
            if g: providers.append(g)
            if q: providers.append(q)
            if h: providers.append(h)
            if m: providers.append(m)
            
        if not providers:
            raise ValueError("No API keys configured for Round Robin")
            
        # Global LLM deadline to prevent stacking timeouts (24 seconds max allowed here)
        import time
        start_time = time.time()
            
        # Try up to the number of available providers
        for attempt in range(len(providers)):
            if time.time() - start_time > 22.0:
                logger.error("Round Robin aborted: Reached strict 22s time budget limit!")
                break
                
            provider_type, api_key, label = providers[_call_counter % len(providers)]
            _call_counter += 1
            
            try:
                if provider_type == "groq":
                    return self._call_groq_raw(prompt, temperature, max_tokens, json_mode, api_key)
                elif provider_type == "mistral":
                    return self._call_mistral_raw(prompt, temperature, max_tokens, json_mode, api_key)
                elif provider_type == "gemini":
                    return self._call_gemini_raw(prompt, temperature, max_tokens, json_mode, api_key)
                elif provider_type == "huggingface":
                    return self._call_hf_raw(prompt, temperature, max_tokens, json_mode, api_key)
            except Exception as e:
                logger.warning(f"Round Robin: {label} failed with error: {e}. Trying next...")
                
        raise ValueError(f"All round-robin providers failed or time budget exceeded!")

    def _call_groq_raw(self, prompt: str, temperature: float, max_tokens: int, json_mode: bool, api_key: str) -> str:
        if not api_key:
            raise ValueError("Groq API key not set")
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=12)
        
        if response.status_code != 200:
            raise ValueError(f"Groq API failed: {response.text}")
            
        return response.json()["choices"][0]["message"]["content"]

    def _call_mistral_raw(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        api_key: str
    ) -> str:
        if not api_key:
            raise ValueError("Mistral API key not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=12
        )
        
        if response.status_code != 200:
            raise ValueError(f"Mistral API failed: {response.text}")
            
        return response.json()["choices"][0]["message"]["content"]

    def _call_gemini_raw(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        api_key: str
    ) -> str:
        if not api_key:
            raise ValueError("Gemini API key not set")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            
        response = requests.post(url, headers=headers, json=payload, timeout=12)
        
        if response.status_code != 200:
            raise ValueError(f"Gemini API failed: {response.text}")
            
        data = response.json()
        if not data.get("candidates"):
            raise ValueError(f"Gemini returned no candidates (possible safety block). Response: {response.text}")
            
        candidate = data["candidates"][0]
        if candidate.get("finishReason") == "SAFETY":
            raise ValueError(f"Gemini blocked this prompt due to safety. Prompt prefix: {prompt[:120]!r}")
            
        return candidate["content"]["parts"][0]["text"].strip()

    def _call_hf_raw(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        api_key: str
    ) -> str:
        if not api_key:
            raise ValueError("Hugging Face API key not set")

        url = "https://router.huggingface.co/featherless-ai/v1/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Format the messages exactly as Llama-3 expects
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        llama_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        if json_mode:
            llama_prompt += "```json\n{"
            
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "prompt": llama_prompt,
            "temperature": max(temperature, 0.01), # HF fails on 0.0
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=12)
        
        if response.status_code != 200:
            raise ValueError(f"HuggingFace API failed: {response.text}")
            
        result = response.json()["choices"][0]["text"].strip()
        if json_mode:
            result = "{" + result # We forced the prompt to start with {
            
        return result

    def analyze_user_input(self, messages: list[dict]) -> dict:
        """
        Combines Intent Classification and Slot Extraction into a SINGLE API call.
        """
        # --- Create Cache Key ---
        frozen = _freeze_messages(messages)
        cache_key = (PROMPT_VERSION, frozen)
        return self._cached_analyze_user_input(cache_key, frozen)

    @lru_cache(maxsize=500)
    def _cached_analyze_user_input(self, cache_key: tuple, frozen_messages: tuple) -> dict:
        # Convert frozen tuple back to list of dicts for processing
        messages = [dict(m) for m in frozen_messages]
        
        history = format_conversation(messages[:-1]) if len(messages) > 1 else "None"
        user_message = messages[-1].get("content", "") if messages else ""
        
        prompt = f"""
        Analyze the user's message and the conversation history.
        1. Classify the intent as one of: CLARIFY, RECOMMEND, REFINE, COMPARE, CONFIRM, OFF_TOPIC.
        2. Extract any specific job roles, skills, or seniority levels mentioned.
        3. If the intent is CLARIFY, generate a conversational reply asking for the missing info (e.g. seniority or skills). For all other intents, you can leave clarify_reply blank.
        
        CRITICAL INTENT RULES:
        - If the user's message sounds like they are making a final decision (e.g., "Clear. We'll use X" or "That's perfect, we will go with these"), classify the intent as CONFIRM, even if they mention specific items. This means the conversation is ending.
        
        Return ONLY valid JSON in this exact format:
        {{
            "intent": "RECOMMEND",
            "extracted_slots": {{
                "role": "software engineer",
                "skills": ["python", "sql"],
                "seniority": "senior"
            }},
            "clarify_reply": "Could you tell me if you are looking for an entry-level or senior role?"
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
            data = json.loads(cleaned)
            
            # Normalization: Mistral sometimes outputs wrong types for JSON slots
            slots = data.get("extracted_slots", {})
            if slots is None:
                slots = {}
                data["extracted_slots"] = slots
                
            # Convert None to defaults
            if slots.get("role") is None:
                slots["role"] = ""
            if slots.get("seniority") is None:
                slots["seniority"] = ""
            if slots.get("skills") is None:
                slots["skills"] = []
                
            # Fix lists passed as strings or strings passed as lists
            if isinstance(slots.get("role"), list):
                slots["role"] = ", ".join(str(x) for x in slots["role"])
            if isinstance(slots.get("seniority"), list):
                slots["seniority"] = ", ".join(str(x) for x in slots["seniority"])
            if isinstance(slots.get("skills"), str):
                slots["skills"] = [s.strip() for s in slots["skills"].split(",") if s.strip()]
                
            return data
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
            if data is None:
                data = {}
                
            # Convert None to defaults
            if data.get("role") is None:
                data["role"] = ""
            if data.get("seniority") is None:
                data["seniority"] = ""
            if data.get("skills") is None:
                data["skills"] = []
                
            # Fix lists passed as strings or strings passed as lists
            if isinstance(data.get("role"), list):
                data["role"] = ", ".join(str(x) for x in data["role"])
            if isinstance(data.get("seniority"), list):
                data["seniority"] = ", ".join(str(x) for x in data["seniority"])
            if isinstance(data.get("skills"), str):
                data["skills"] = [s.strip() for s in data["skills"].split(",") if s.strip()]
                
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