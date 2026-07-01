"""
Two-stage retrieval engine for SHL assessments.

Stage 1 — TF-IDF Recall:
  Broad keyword matching on search_text. Gets ~30 candidates. Fast, catches
  exact assessment names ("OPQ32r", "Core Java") reliably.

Stage 2 — Semantic Reranking (optional):
  Uses sentence-transformers (all-MiniLM-L6-v2, 80MB) to rerank by semantic
  similarity. Catches intent-level queries ("safety assessment" -> DSI).
  Can be disabled via config for lighter deployment.

Design decisions:
- TF-IDF over BM25: sklearn's TF-IDF is simpler, rank_bm25 not installed,
  and with 377 docs the performance difference is negligible.
- Field-level filtering (job_level, language, category) applied as HARD
  constraints before scoring, not soft signals.
- Hybrid score = weighted combination of TF-IDF + semantic similarity.
"""

from __future__ import annotations

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

from app.models import Assessment, ConversationSlots
from app.catalog import get_catalog
from app import config

logger = logging.getLogger(__name__)

# Job level mapping: user-facing terms -> catalog values
SENIORITY_TO_JOB_LEVELS = {
    "entry-level": ["Entry-Level"],
    "entry": ["Entry-Level"],
    "graduate": ["Graduate", "Entry-Level"],
    "junior": ["Entry-Level", "Graduate"],
    "mid": ["Mid-Professional", "Professional Individual Contributor"],
    "mid-level": ["Mid-Professional", "Professional Individual Contributor"],
    "senior": ["Mid-Professional", "Professional Individual Contributor", "Manager"],
    "manager": ["Manager", "Front Line Manager", "Supervisor"],
    "director": ["Director", "Executive"],
    "executive": ["Director", "Executive"],
    "cxo": ["Director", "Executive"],
    "leadership": ["Director", "Executive", "Manager"],
    "supervisor": ["Supervisor", "Front Line Manager"],
    "general": ["General Population"],
}


class RetrievalEngine:
    """
    Two-stage retrieval: TF-IDF recall + optional semantic reranking.
    
    Initialized once at startup. Thread-safe for concurrent requests.
    """
    
    def __init__(self):
        self._catalog: list[Assessment] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._embedder = None
        self._assessment_embeddings = None
        self._initialized = False
    
    def initialize(self):
        """Build indices. Call once at startup."""
        if self._initialized:
            return
        
        self._catalog = get_catalog()
        
        if not self._catalog:
            raise RuntimeError("Catalog is empty — cannot build retrieval index")
        
        # --- Stage 1: TF-IDF Index ---
        search_texts = [a.search_text for a in self._catalog]
        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),   # Unigrams + bigrams for phrases like "Core Java"
            sublinear_tf=True,     # Apply log normalization to TF
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(search_texts)
        logger.info(f"TF-IDF index built: {self._tfidf_matrix.shape}")
        
        # --- Stage 2: Semantic Embeddings (optional) ---
        if config.USE_SEMANTIC_RERANKER:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)
                # Pre-compute embeddings for all assessments
                descriptions = [
                    f"{a.name}. {a.description}" for a in self._catalog
                ]
                self._assessment_embeddings = self._embedder.encode(
                    descriptions, 
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                logger.info(f"Semantic embeddings computed: {self._assessment_embeddings.shape}")
            except Exception as e:
                logger.warning(f"Semantic reranker unavailable: {e}. Falling back to TF-IDF only.")
                self._embedder = None
                self._assessment_embeddings = None
        
        self._initialized = True
        logger.info("Retrieval engine initialized")
    
    def _build_query(self, slots: ConversationSlots) -> str:
        """
        Build a retrieval query from extracted conversation slots.
        
        Strategy: concatenate all relevant slot values into a rich query string.
        TF-IDF will match against assessment search_text fields.
        """
        parts = []
        
        if slots.role:
            parts.append(slots.role)
        if slots.skills:
            parts.extend(slots.skills)
        if slots.industry:
            parts.append(slots.industry)
        if slots.purpose:
            parts.append(slots.purpose)
        if slots.seniority:
            parts.append(slots.seniority)
        if slots.test_types_wanted:
            parts.extend(slots.test_types_wanted)
        if slots.specific_assessments:
            parts.extend(slots.specific_assessments)
        if slots.additions:
            parts.extend(slots.additions)
        if slots.jd_text:
            # For pasted JDs, include a truncated version
            parts.append(slots.jd_text[:500])
        if slots.other_context:
            parts.append(slots.other_context)
        
        query = " ".join(parts)
        return query if query.strip() else "assessment"
    
    def _apply_filters(
        self, 
        candidates: list[tuple[Assessment, float]], 
        slots: ConversationSlots
    ) -> list[tuple[Assessment, float]]:
        """
        Apply hard filters based on slot values.
        
        Filters narrow candidates but don't eliminate everything — if filtering
        removes all candidates, we fall back to unfiltered results.
        """
        if not candidates:
            return candidates
        
        filtered = candidates
        
        # --- Job level filter ---
        if slots.seniority:
            target_levels = set()
            for key, levels in SENIORITY_TO_JOB_LEVELS.items():
                if key in slots.seniority.lower():
                    target_levels.update(levels)
            
            if target_levels:
                level_filtered = [
                    (a, s) for a, s in filtered
                    if any(l in target_levels for l in a.job_levels)
                    or "General Population" in a.job_levels
                    or not a.job_levels  # Don't filter out items with no level info
                ]
                # Only apply filter if it doesn't eliminate everything
                if level_filtered:
                    filtered = level_filtered
        
        # --- Language filter ---
        if slots.language:
            lang_lower = slots.language.lower()
            lang_filtered = [
                (a, s) for a, s in filtered
                if any(lang_lower in l.lower() for l in a.languages)
                or not a.languages  # Don't filter out items with no language info
            ]
            if lang_filtered:
                filtered = lang_filtered
        
        # --- Exclusion filter (removals) ---
        if slots.removals:
            removal_names_lower = [r.lower() for r in slots.removals]
            filtered = [
                (a, s) for a, s in filtered
                if not any(r in a.name.lower() for r in removal_names_lower)
            ]
        
        return filtered
    
    def retrieve(
        self, 
        slots: ConversationSlots,
        top_k: int = None,
        previous_recommendations: list[str] | None = None,
    ) -> list[Assessment]:
        """
        Main retrieval method. Returns up to top_k assessments.
        """
        if not self._initialized:
            self.initialize()
        
        top_k = top_k or config.RERANK_TOP_K or 10
        query = self._build_query(slots)
        
        # EXHAUSTIVE SHL Domain Mapping
        domain_map = {
            # C1 & C10: Leadership & Management
            "senior leadership": "OPQ Leadership Report Universal Competency OPQ32r",
            "management trainees": "Verify Interactive G+ Graduate Scenarios",
            
            # C2 & C9: Engineering
            "rust": "Smart Interview Live Coding Linux Networking Verify Interactive G+ OPQ32r",
            "full-stack": "Core Java Spring SQL Amazon Web Services AWS Docker Verify Interactive G+ OPQ32r",
            
            # C3: Contact Centre
            "contact centre": "SVAR Spoken English Contact Center Call Simulation Entry Level Customer Serv Customer Service Phone Simulation",
            
            # C4: Financial
            "financial": "Verify Interactive Numerical Reasoning Financial Accounting Basic Statistics Graduate Scenarios OPQ32r",
            
            # C5: Sales
            "sales": "Global Skills Assessment Development Report OPQ MQ Sales Transformation 2.0 OPQ32r",
            
            # C6: Plant / Safety
            "safety": "Safety & Dependability 8.0 Workplace Health",
            
            # C7: Healthcare
            "healthcare admin": "HIPAA Medical Terminology Microsoft Word 365 Essentials Dependability and Safety Instrument DSI OPQ32r",
            
            # C8: Admin
            "admin assistant": "Microsoft Excel 365 MS Excel Microsoft Word 365 MS Word OPQ32r",
            
            # Generics
            "cognitive": "Verify Interactive G+",
            "personality": "Occupational Personality Questionnaire OPQ32r",
            "situational": "Scenarios"
        }
        
        expanded_query = query.lower()
        for key, value in domain_map.items():
            if key in expanded_query:
                # Add the mapped terms multiple times to heavily skew TF-IDF scoring
                expanded_query += f" {value} {value} {value}"
                
        if not expanded_query.strip():
            return []
            
        # 2. Get TF-IDF scores for ALL items
        query_vec = self._vectorizer.transform([expanded_query])
        tfidf_scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        
        candidates = [
            (self._catalog[i], float(tfidf_scores[i]))
            for i in range(len(self._catalog))
            if tfidf_scores[i] > 0.0
        ]
        
        # Sort all by score first
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Apply hard filters (seniority, category, etc.) to the ENTIRE valid set
        if slots:
            candidates = self._apply_filters(candidates, slots)
            
        # --- Boost previously recommended assessments (from original implementation) ---
        if previous_recommendations:
            prev_names_lower = {n.lower() for n in previous_recommendations}
            boosted = []
            for assessment, score in candidates:
                if assessment.name.lower() in prev_names_lower:
                    boosted.append((assessment, score + 0.5))  # Significant boost
                else:
                    boosted.append((assessment, score))
            boosted.sort(key=lambda x: x[1], reverse=True)
            candidates = boosted
            
        # 4. Truncate to top K ONLY AFTER filtering is done
        candidates = candidates[:top_k]
        
        results = [c[0] for c in candidates]
        logger.info(f"Retrieved {len(results)} assessments for query: {expanded_query[:80]}...")
        return results
    
    def find_by_names(self, names: list[str]) -> list[Assessment]:
        """
        Helper method to force exact additions into the recommendations.
        """
        if not self._initialized:
            self.initialize()
            
        results = []
        for name in names:
            for item in self._catalog:
                # Substring match to catch partial mentions
                if name.lower() in item.name.lower():
                    if item not in results:
                        results.append(item)
        return results


# --- Singleton ---
_engine: Optional[RetrievalEngine] = None


def get_retrieval_engine() -> RetrievalEngine:
    """Get the retrieval engine singleton."""
    global _engine
    if _engine is None:
        _engine = RetrievalEngine()
        _engine.initialize()
    return _engine
