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
MISTRAL_KEYS = [
    os.getenv("MISTRAL_API_KEY"),
    os.getenv("MISTRAL_BACKUP_KEY"),
    os.getenv("MISTRAL_THIRD_KEY"),
    os.getenv("MISTRAL_FOURTH_KEY"),
    os.getenv("MISTRAL_FIFTH_KEY"),
    os.getenv("MISTRAL_SIXTH_KEY"),
]
MISTRAL_KEYS = [k for k in MISTRAL_KEYS if k]

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_SECOND_API_KEY"),
    os.getenv("GROQ_THIRD_API_KEY"),
    os.getenv("GROQ_FOURTH_API_KEY"),
    os.getenv("GROQ_FIFTH_API_KEY"),
    os.getenv("GROQ_SIXTH_API_KEY"),
]
GROQ_KEYS = [k for k in GROQ_KEYS if k and k.startswith("gsk_")] # Filter out the mistaken paste

GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_SECOND_KEY"),
    os.getenv("GEMINI_THIRD_KEY"),
    os.getenv("GEMINI_FOURTH_KEY"),
    os.getenv("GEMINI_FIFTH_KEY"),
    os.getenv("GEMINI_SIXTH_KEY"),
    os.getenv("GEMINI_SEVENTH_KEY"),
]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

HUGGING_FACE_KEYS = [
    os.getenv("HUGGING_FACE_API"),
    os.getenv("HUGGING_FACE_API_SECOND"),
]
HUGGING_FACE_KEYS = [k for k in HUGGING_FACE_KEYS if k]

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

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
