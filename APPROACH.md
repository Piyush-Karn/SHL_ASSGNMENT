# Approach Document — SHL Conversational Assessment Recommender

## Design Overview

The system is a stateless FastAPI service that processes multi-turn conversations to recommend SHL assessments. The core architectural principle is **separation of concerns**: a deterministic conversation controller decides *what* action to take (clarify, recommend, refine, compare, refuse), while a Gemini LLM decides *how* to express it in natural language.

## Retrieval Setup

**Two-stage retrieval over 377 catalog items:**
1. **TF-IDF recall** (scikit-learn): Each assessment is indexed by a concatenation of its name, description, categories, and job levels. User queries are built from extracted conversation slots (role, skills, seniority). This catches exact keyword matches like "Java", "OPQ32r", or "safety". Top 30 candidates retrieved.
2. **Semantic reranking** (sentence-transformers, all-MiniLM-L6-v2): Reranks candidates by cosine similarity between the query embedding and pre-computed assessment embeddings. This catches intent-level matches ("people skills" → Personality & Behavior). Final hybrid score = 0.45 × TF-IDF + 0.55 × semantic.

**Why not BM25/vector store?** With only 377 documents, the infrastructure overhead of Elasticsearch or a vector database is unjustified. TF-IDF provides excellent recall for exact terms, and sentence-transformer reranking adds the semantic understanding that pure keyword search misses.

**Field-level filtering** (job level, language) is applied as hard constraints between stages, not soft signals. This prevents returning executive-level assessments when the user specified "entry-level."

## Prompt Design

Three distinct LLM calls per request, each with a specific role:
- **Slot extraction** (temperature=0.1): Converts free-form conversation into structured JSON with 15 fields (role, skills, seniority, language, test types, additions, removals, etc.). Low temperature for reliable structured output.
- **Intent classification** (temperature=0.0): Classifies the last user message as NEW_QUERY, CLARIFY_RESPONSE, REFINE, COMPARE, CONFIRM, or OFF_TOPIC. Complemented by regex pattern matching for common patterns — belt-and-suspenders.
- **Response generation** (temperature=0.3): Generates consultant-style natural language. The prompt includes only the retrieved assessments, never the full catalog, ensuring grounding.

The system prompt instructs the LLM to be opinionated and consultative ("OPQ32r is the right instrument") rather than generic ("you might consider").

## Agent Behavior Design

The conversation controller handles four core behaviors:
- **Clarify**: Ask 1-2 targeted questions when role or seniority is missing. Never ask more than necessary — each question costs a turn against the 8-turn cap.
- **Recommend**: Return 1-10 assessments when sufficient context exists. Recommendations are constructed from catalog objects in code, never from LLM-generated text.
- **Refine**: Incrementally add/remove assessments when user changes constraints. Previous recommendations are boosted (+0.5 score) for continuity.
- **Compare**: Grounded comparison using only catalog data for the mentioned assessments.

**Safety**: A regex-based first pass catches prompt injection, off-topic requests, and legal questions. This layer can't be bypassed by prompt injection — it runs before the LLM sees the message.

**Turn budget**: The controller forces recommendation by turn 5 (of 8 max) to leave room for refinement and confirmation.

## Evaluation Approach

Built a local replay harness that runs all 10 provided conversation traces:
- **Recall@10**: Fraction of expected assessments found in the final shortlist
- **Schema compliance**: Every response validated against the required JSON schema
- **Behavior probes**: Off-topic refusal, no recommendation on vague queries, prompt injection resistance, URL grounding

## What Didn't Work

1. **Pure LLM decision-making** (no controller): The agent sometimes clarified when it should have recommended, and occasionally invented assessment names not in the catalog. Adding the deterministic controller fixed both issues.
2. **BM25 tokenization**: Assessment names like "OPQ32r" and ".NET MVC (New)" tokenized poorly with rank_bm25's default splitter. Switched to scikit-learn's TF-IDF which handles these better.
3. **Single-call architecture** (slot extraction + response in one call): Led to inconsistent structured output. Splitting into separate calls improved reliability significantly.

## Tools Used

- **AI assistance**: Used Gemini (Antigravity IDE) for code scaffolding, prompt iteration, and evaluation harness design. All architectural decisions and code were manually reviewed and understood.
- **Stack**: FastAPI, Pydantic, scikit-learn (TF-IDF), sentence-transformers, Google Gemini 2.5 Flash
- **Deployment**: Render (free tier), Docker
