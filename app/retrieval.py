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
        
        Args:
            slots: Extracted conversation slots
            top_k: Max results to return (default: config.RERANK_TOP_K)
            previous_recommendations: Names of previously recommended assessments
                                       to boost in refinement scenarios
        
        Returns:
            Ordered list of Assessment objects, best match first.
        """
        if not self._initialized:
            self.initialize()
        
        top_k = top_k or config.RERANK_TOP_K
        query = self._build_query(slots)
        
        if not query.strip():
            return []
        
        # --- Stage 1: TF-IDF Recall ---
        query_vec = self._vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        
        # Get top TFIDF_TOP_K candidates
        recall_k = config.TFIDF_TOP_K
        top_indices = np.argsort(tfidf_scores)[-recall_k:][::-1]
        
        candidates = [
            (self._catalog[i], float(tfidf_scores[i]))
            for i in top_indices
            if tfidf_scores[i] > 0.0  # Only items with some TF-IDF relevance
        ]
        
        if not candidates:
            # Fallback: if TF-IDF found nothing, return top items by any signal
            logger.warning(f"TF-IDF returned no candidates for query: {query[:100]}")
            return []
        
        # --- Apply hard filters ---
        candidates = self._apply_filters(candidates, slots)
        
        # --- Stage 2: Semantic Reranking ---
        if self._embedder is not None and self._assessment_embeddings is not None:
            query_embedding = self._embedder.encode(
                [query], 
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            
            reranked = []
            for assessment, tfidf_score in candidates:
                idx = self._catalog.index(assessment)
                sem_score = float(
                    cosine_similarity(
                        query_embedding, 
                        self._assessment_embeddings[idx:idx+1]
                    )[0][0]
                )
                # Hybrid score
                hybrid_score = (
                    config.TFIDF_WEIGHT * tfidf_score 
                    + config.SEMANTIC_WEIGHT * sem_score
                )
                reranked.append((assessment, hybrid_score))
            
            reranked.sort(key=lambda x: x[1], reverse=True)
            candidates = reranked
        
        # --- Boost previously recommended assessments ---
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
        
        # --- Return top_k ---
        results = [a for a, _ in candidates[:top_k]]
        logger.info(
            f"Retrieved {len(results)} assessments for query: {query[:80]}..."
        )
        return results
    
    def find_by_names(self, names: list[str]) -> list[Assessment]:
        """
        Find assessments by name (fuzzy matching).
        Used for comparison requests where user mentions specific assessments.
        """
        if not self._initialized:
            self.initialize()
        
        results = []
        for name in names:
            name_lower = name.lower().strip()
            best_match = None
            best_score = 0
            
            for assessment in self._catalog:
                a_name_lower = assessment.name.lower()
                # Exact match
                if a_name_lower == name_lower:
                    best_match = assessment
                    break
                # Partial match (e.g., "OPQ" matches "Occupational Personality Questionnaire OPQ32r")
                if name_lower in a_name_lower or a_name_lower in name_lower:
                    # Score by length similarity
                    score = len(name_lower) / max(len(a_name_lower), 1)
                    if score > best_score:
                        best_score = score
                        best_match = assessment
            
            if best_match and best_match not in results:
                results.append(best_match)
        
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
