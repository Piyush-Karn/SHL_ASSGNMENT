"""
Configuration for the SHL Assessment Recommender.

Design decision: All config via environment variables for deployment flexibility.
No hardcoded secrets. python-dotenv for local dev convenience.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# --- LLM Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Catalog ---
CATALOG_PATH = os.getenv(
    "CATALOG_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "shlcatalogue.json"),
)

# --- Retrieval ---
TFIDF_TOP_K = 30          # Candidates from TF-IDF stage
RERANK_TOP_K = 10         # Final candidates after reranking
SEMANTIC_WEIGHT = 0.55    # Weight for semantic score in hybrid ranking
TFIDF_WEIGHT = 0.45       # Weight for TF-IDF score in hybrid ranking

# --- Embedding model for reranker ---
# all-MiniLM-L6-v2 is 80MB — small enough for free-tier deployment
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USE_SEMANTIC_RERANKER = os.getenv("USE_SEMANTIC_RERANKER", "true").lower() == "true"

# --- Conversation ---
MAX_TURNS = 8             # Hard cap from evaluator (user + assistant turns total)
MAX_RECOMMENDATIONS = 10  # Never more than 10 recommendations
MIN_RECOMMENDATIONS = 1   # At least 1 when recommending

# --- Safety ---
MAX_MESSAGE_LENGTH = 5000  # Reject absurdly long messages (possible injection)

# --- Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
