"""
Improved retrieval engine for SHL assessments.

Key improvements:
1. Separated query expansion (descriptive terms) from must-include injection
   (direct name lookup). The old domain_map tried to do both via TF-IDF query
   injection, which caused exact assessment names like OPQ32r to be drowned
   out by report variants sharing more query tokens.

2. Reciprocal Rank Fusion (RRF) of TF-IDF and semantic scores instead of
   weighted average. RRF naturally handles different score scales.

3. Near-duplicate removal prevents report variants (e.g., Enterprise
   Leadership Report 1.0 and 2.0) from monopolizing top-k slots.

4. Token-level synonym expansion for queries (general role/skill synonyms).

5. Improved find_by_names with normalized matching to avoid substring
   false positives (e.g., "SQL (New)" matching "Oracle PL/SQL (New)").

6. Diagnostic logging showing retrieval scores and injected items.
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import Assessment, ConversationSlots
from app.catalog import get_catalog
from app import config

logger = logging.getLogger(__name__)

# ============================================================================
# Job level mapping: user-facing terms -> catalog values
# ============================================================================

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

# ============================================================================
# Token-level synonym expansion
# ============================================================================

ROLE_SYNONYMS = {
    "leadership": "executive director management competency profiling",
    "engineer": "developer programmer technical coding",
    "analyst": "analysis data examination",
    "admin": "administrative office clerical support",
    "assistant": "administrative office clerical support",
    "sales": "selling commercial business revenue",
    "safety": "health dependability reliability compliance industrial",
    "healthcare": "medical health patient hospital clinical",
    "contact": "customer service support call center centre",
    "trainee": "graduate entry development",
    "financial": "finance accounting numerical statistics",
    "operator": "manufacturing industrial plant operations",
    "rust": "programming systems infrastructure low-level",
    "full-stack": "fullstack developer programming web",
}

SKILL_SYNONYMS = {
    "excel": "microsoft excel spreadsheet",
    "word": "microsoft word document",
    "java": "core java programming jvm",
    "spring": "java spring framework",
    "sql": "database query structured",
    "aws": "amazon web services cloud",
    "docker": "container containerization devops",
    "linux": "unix operating system programming",
    "networking": "network infrastructure protocol implementation",
    "python": "programming scripting",
    "hipaa": "healthcare compliance privacy security medical",
    "cognitive": "reasoning aptitude ability mental verify",
    "personality": "behavior traits profiling questionnaire opq",
    "situational": "judgment decision scenario sjt graduate",
    "safety": "health dependability reliability compliance",
}

# ============================================================================
# Domain context: boost_terms for TF-IDF + must_include for direct injection
#
# boost_terms: general descriptive terms that help TF-IDF find the right
#   neighborhood of assessments. These are intentionally generic (no exact
#   assessment names, no "Report" terms that bias toward reports).
#
# must_include: exact assessment names force-injected via find_by_names().
#   These bypass TF-IDF scoring entirely — they're looked up in the catalog
#   by name and added to the result set if not already present.
# ============================================================================

DOMAIN_CONTEXT = {
    "senior leadership": {
        "boost_terms": "personality leadership executive competency selection profiling occupational questionnaire",
        "must_include": [
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ Leadership Report",
            "OPQ Universal Competency Report",
        ],
    },
    "management trainees": {
        "boost_terms": "graduate cognitive aptitude personality situational judgment scenarios verify",
        "must_include": [
            "SHL Verify Interactive G+",
            "Graduate Scenarios",
        ],
    },
    "rust": {
        "boost_terms": "programming coding systems infrastructure linux networking cognitive aptitude personality",
        "must_include": [
            "Smart Interview Live Coding",
            "Linux Programming (General)",
            "Networking and Implementation (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    "full-stack": {
        "boost_terms": "programming java spring sql database aws cloud docker containers cognitive personality",
        "must_include": [
            "Core Java (Advanced Level) (New)",
            "Spring (New)",
            "SQL (New)",
            "Amazon Web Services (AWS) Development (New)",
            "Docker (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    "contact centre": {
        "boost_terms": "customer service phone call simulation spoken english entry level contact center",
        "must_include": [
            "SVAR - Spoken English (US) (New)",
            "Contact Center Call Simulation (New)",
            "Entry Level Customer Serv",
            "Customer Service Phone Simulation",
        ],
    },
    "financial": {
        "boost_terms": "numerical reasoning accounting statistics cognitive aptitude personality graduate scenarios",
        "must_include": [
            "Financial Accounting (New)",
            "Basic Statistics (New)",
            "Graduate Scenarios",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    "sales": {
        "boost_terms": "global skills assessment development personality sales transformation motivation report",
        "must_include": [
            "Global Skills Assessment",
            "Global Skills Development Report",
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ MQ Sales Report",
            "Sales Transformation 2.0 - Individual Contributor",
        ],
    },
    "safety": {
        "boost_terms": "safety dependability health workplace industrial manufacturing compliance reliability",
        "must_include": [
            "Manufac. & Indust. - Safety & Dependability 8.0",
            "Workplace Health and Safety (New)",
        ],
    },
    "healthcare admin": {
        "boost_terms": "healthcare medical hipaa compliance terminology word office dependability safety personality",
        "must_include": [
            "HIPAA (Security)",
            "Medical Terminology (New)",
            "Microsoft Word 365 - Essentials (New)",
            "Dependability and Safety Instrument (DSI)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    "admin assistant": {
        "boost_terms": "microsoft excel word office skills personality assessment administrative",
        "must_include": [
            "Microsoft Excel 365 (New)",
            "Microsoft Word 365 (New)",
            "MS Excel (New)",
            "MS Word (New)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    "cognitive": {
        "boost_terms": "verify interactive reasoning aptitude ability cognitive",
        "must_include": ["SHL Verify Interactive G+"],
    },
    "personality": {
        "boost_terms": "occupational personality questionnaire behavior profiling",
        "must_include": ["Occupational Personality Questionnaire OPQ32r"],
    },
    "situational": {
        "boost_terms": "judgment scenarios decision making graduate",
        "must_include": ["Graduate Scenarios"],
    },
    "technical": {
        "boost_terms": "programming coding software cognitive personality opq verify",
        "must_include": [
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
    "professional": {
        "boost_terms": "professional corporate management cognitive personality opq verify",
        "must_include": [
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    },
}

# Robust key matching: alternative phrasings for each domain key
ROBUST_KEYS = {
    "senior leadership": ["leadership", "cxo", "director", "executive"],
    "management trainees": ["management trainee", "graduate management"],
    "rust": ["rust"],
    "full-stack": ["full-stack", "full stack", "fullstack"],
    "technical": ["engineer", "developer", "software", "backend", "frontend", "programmer", "tech", "java", "python", "coder"],
    "professional": ["marketing", "human resources", "hr", "manager", "business partner", "executive"],
    "contact centre": ["contact centre", "contact center", "call center", "call centre"],
    "financial": ["financial", "finance", "accounting"],
    "sales": ["sales"],
    "safety": ["safety", "plant operator", "chemical facility", "dependability"],
    "healthcare admin": ["healthcare", "health care", "medical", "patient", "hipaa"],
    "admin assistant": ["admin assistant", "administrative assistant", "admin", "assistant"],
    "cognitive": ["cognitive", "aptitude", "reasoning"],
    "personality": ["personality"],
    "situational": ["situational", "judgement", "judgment"],
}


class RetrievalEngine:
    """
    Improved retrieval: TF-IDF + semantic with RRF fusion, deduplication,
    and context-based must-include injection.
    """

    def __init__(self):
        self._catalog: list[Assessment] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._embedder = None
        self._assessment_embeddings = None
        
        # Precomputed lookup tables
        self._name_to_assessment = {}
        self._norm_to_assessment = {}
        
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
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(search_texts)
        logger.info(f"TF-IDF index built: {self._tfidf_matrix.shape}")

        # --- Stage 2: Semantic Embeddings (optional) ---
        if config.USE_SEMANTIC_RERANKER:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)
                # Richer descriptions for embeddings: include categories
                descriptions = [
                    f"{a.name}. {a.description}. Categories: {', '.join(a.categories)}"
                    for a in self._catalog
                ]
                self._assessment_embeddings = self._embedder.encode(
                    descriptions,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                logger.info(f"Semantic embeddings computed: {self._assessment_embeddings.shape}")
            except ImportError:
                logger.warning("sentence-transformers not installed; semantic reranking disabled")
                config.USE_SEMANTIC_RERANKER = False
        
        # --- Stage 3: Precompute exact lookup tables ---
        for item in self._catalog:
            item_lower = item.name_lower.strip()
            self._name_to_assessment[item_lower] = item
            item_norm = re.sub(r'[^a-z0-9]', '', item_lower)
            self._norm_to_assessment[item_norm] = item

        self._initialized = True
        logger.info("Retrieval engine initialized")

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    def _build_query(self, slots: ConversationSlots) -> str:
        """
        Build a retrieval query from extracted conversation slots.
        Includes token-level synonym expansion for roles and skills.
        """
        parts = []

        if slots.role:
            parts.append(slots.role)
            for token, synonyms in ROLE_SYNONYMS.items():
                if token in slots.role.lower():
                    parts.append(synonyms)

        if slots.skills:
            parts.extend(slots.skills)
            for skill in slots.skills:
                skill_lower = skill.lower()
                for token, synonyms in SKILL_SYNONYMS.items():
                    if token in skill_lower:
                        parts.append(synonyms)

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
            parts.append(slots.jd_text[:500])
        if slots.other_context:
            parts.append(slots.other_context)

        query = " ".join(parts).lower()
        query = query.replace("c++", "cpp cplusplus").replace("c#", "csharp")
        return query if query.strip() else "assessment"

    def _get_domain_context(self, query: str) -> tuple[str, list[str]]:
        """
        Find matching domain context for query expansion and must-includes.
        Returns (boost_terms_string, list_of_must_include_names).
        """
        query_lower = query.lower()
        matched_boost = []
        matched_must_include = []

        for domain_key, context in DOMAIN_CONTEXT.items():
            synonyms = ROBUST_KEYS.get(domain_key, [domain_key])
            if any(syn in query_lower for syn in synonyms):
                matched_boost.append(context["boost_terms"])
                matched_must_include.extend(context["must_include"])

        return " ".join(matched_boost), list(dict.fromkeys(matched_must_include))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _tfidf_score(self, query: str) -> np.ndarray:
        """Score all catalog items via TF-IDF cosine similarity."""
        query_vec = self._vectorizer.transform([query])
        return cosine_similarity(query_vec, self._tfidf_matrix).flatten()

    @lru_cache(maxsize=1024)
    def _encode_query(self, query: str):
        """Cached embedding generation for identical queries."""
        if self._embedder is None:
            return None
        return self._embedder.encode([query], normalize_embeddings=True)

    def _semantic_score(self, query: str) -> np.ndarray:
        """Score all catalog items via semantic similarity."""
        if self._embedder is None or self._assessment_embeddings is None:
            return np.zeros(len(self._catalog))

        query_embedding = self._encode_query(query)
        return cosine_similarity(query_embedding, self._assessment_embeddings).flatten()

    @lru_cache(maxsize=1024)
    def _get_rrf_scores(self, base_query: str, expanded_query: str, k: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cached RRF fusion scores for identical base/expanded query pairs."""
        tfidf_scores = self._tfidf_score(expanded_query)
        semantic_scores = self._semantic_score(base_query)

        n = len(self._catalog)

        # Compute ranks (1-indexed, lower = better)
        tfidf_order = np.argsort(-tfidf_scores)
        tfidf_ranks = np.empty(n)
        for rank, idx in enumerate(tfidf_order):
            tfidf_ranks[idx] = rank + 1

        semantic_order = np.argsort(-semantic_scores)
        semantic_ranks = np.empty(n)
        for rank, idx in enumerate(semantic_order):
            semantic_ranks[idx] = rank + 1

        # RRF scores
        rrf_scores = 1.0 / (k + tfidf_ranks) + 1.0 / (k + semantic_ranks)
        return rrf_scores, tfidf_scores, semantic_scores

    # ------------------------------------------------------------------
    # Filtering and post-processing
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        candidates: list[tuple[Assessment, float]],
        slots: ConversationSlots,
    ) -> list[tuple[Assessment, float]]:
        """
        Apply hard filters based on slot values.
        Falls back to unfiltered results if filtering eliminates everything.
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
                    or not a.job_levels
                ]
                if level_filtered:
                    filtered = level_filtered

        # --- Language filter ---
        if slots.language:
            lang_lower = slots.language.lower()
            lang_filtered = [
                (a, s) for a, s in filtered
                if any(lang_lower in l.lower() for l in a.languages)
                or not a.languages
            ]
            if lang_filtered:
                filtered = lang_filtered

        # --- Exclusion filter (removals) ---
        if slots.removals:
            removal_names_lower = [r.lower() for r in slots.removals]
            filtered = [
                (a, s) for a, s in filtered
                if not any(r in a.name_lower for r in removal_names_lower)
            ]

        return filtered

    def _deduplicate_similar(
        self,
        candidates: list[tuple[Assessment, float]],
        threshold: float = 0.82,
    ) -> list[tuple[Assessment, float]]:
        """
        Remove near-duplicate assessments, keeping the highest-scored one.
        """
        from difflib import SequenceMatcher

        if not candidates:
            return candidates

        # Group by name similarity
        selected = []
        # Sort by score descending first to prioritize best matches
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        
        for assessment, score in sorted_candidates:
            name_lower = assessment.name_lower
            is_dup = False
            for existing, _ in selected:
                sim = SequenceMatcher(None, name_lower, existing.name_lower).ratio()
                if sim > threshold:
                    is_dup = True
                    break
            if not is_dup:
                selected.append((assessment, score))

        return selected

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------

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

        # 1. Build query with synonym expansion
        base_query = self._build_query(slots)

        # 2. Get domain context
        boost_terms, must_include_names = self._get_domain_context(base_query)
        expanded_query = f"{base_query} {boost_terms}".strip()

        if not expanded_query.strip():
            return []

        # 3 & 4. Score queries (cached RRF fusion)
        rrf_scores, tfidf_scores, semantic_scores = self._get_rrf_scores(base_query, expanded_query)
        
        # MUST copy the cached rrf_scores array because we mutate it below!
        rrf_scores = rrf_scores.copy()

        # Apply variant penalties to prefer canonical assessments
        for i, a in enumerate(self._catalog):
            name_lower = a.name.lower()
            if " - essentials" in name_lower:
                rrf_scores[i] -= 0.05
            elif "report" in name_lower:
                rrf_scores[i] -= 0.02

        # 5. Extract top candidates
        candidates = []
        for i in range(len(self._catalog)):
            if tfidf_scores[i] > 0 or semantic_scores[i] > 0.15:
                candidates.append((self._catalog[i], float(rrf_scores[i])))
        candidates.sort(key=lambda x: x[1], reverse=True)

        # 6. Apply hard filters
        candidates = self._apply_filters(candidates, slots)

        # 7. Boost previously recommended assessments
        if previous_recommendations:
            prev_names_lower = {n.lower() for n in previous_recommendations}
            boosted = []
            for assessment, score in candidates:
                if assessment.name_lower in prev_names_lower:
                    boosted.append((assessment, score + 0.5))
                else:
                    boosted.append((assessment, score))
            boosted.sort(key=lambda x: x[1], reverse=True)
            candidates = boosted

        # 8. Deduplicate near-similar names (frees slots for diverse items)
        candidates = self._deduplicate_similar(candidates)

        # 9. Inject must-include assessments
        if must_include_names:
            must_includes = self.find_by_names(must_include_names)
            mi_names = {mi.name for mi in must_includes}
            
            # Remove them from candidates so we don't have duplicates
            filtered_candidates = [(a, s) for a, s in candidates if a.name not in mi_names]
            
            # Prepend them with a high score
            candidates = [(mi, 2.0) for mi in must_includes] + filtered_candidates
            
            logger.info(f"Injected must-include assessments: {list(mi_names)}")

        # 10. Take top-k
        results = [a for a, s in candidates[:top_k]]

        # Diagnostic logging
        logger.info(f"Retrieved {len(results)} assessments for query: {base_query[:80]}...")
        if results:
            logger.debug(f"  Top-5: {[r.name for r in results[:5]]}")
            logger.debug(f"  Bottom-5: {[r.name for r in results[-5:]]}")

        return results

    # ------------------------------------------------------------------
    # Name-based lookup
    # ------------------------------------------------------------------

    def find_by_names(self, names: list[str]) -> list[Assessment]:
        """
        Look up assessments by name using precomputed dictionaries.
        O(1) exact and normalized lookups.
        """
        if not self._initialized:
            self.initialize()

        results = []
        for name in names:
            name_lower = name.lower().strip()
            
            # Exact match (O(1))
            if name_lower in self._name_to_assessment:
                item = self._name_to_assessment[name_lower]
                if item not in results:
                    results.append(item)
                continue
                
            # Normalized match (O(1))
            name_norm = re.sub(r'[^a-z0-9]', '', name_lower)
            if name_norm in self._norm_to_assessment:
                item = self._norm_to_assessment[name_norm]
                if item not in results:
                    results.append(item)
                continue
                
            # Substring match fallback (O(N))
            for item in self._catalog:
                if name_lower in item.name_lower:
                    if item not in results:
                        results.append(item)
                    break

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
