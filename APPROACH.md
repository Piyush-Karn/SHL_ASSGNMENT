# SHL Assessment Recommendation System
## Approach Document

### 1. Problem Understanding
The goal is to build a conversational AI agent capable of recommending relevant SHL assessments based on a user's hiring needs through a multi-turn dialogue. The system must operate within strict constraints: conversations must conclude within 8 turns, all responses must adhere to a strict JSON schema, and recommendations must be 100% grounded in the provided SHL catalog. To prevent critical failures, the system must aggressively minimize hallucinations, actively refuse off-topic prompts, and ask for clarification when user queries are too vague to yield accurate results.

### 2. System Architecture
Our architecture isolates non-deterministic LLM calls behind a robust, deterministic state machine to guarantee strict adherence to behavior probes and constraints.

```text
User
   │
   ▼
FastAPI
   │
   ▼
Safety Layer
   │
   ▼
Conversation Controller
   │
   ├── Intent Detection
   ├── Slot Tracking
   └── State Machine
   │
   ▼
Hybrid Retriever
   ├── TF-IDF
   ├── Semantic Search
   └── Constraint Filtering
   │
   ▼
LLM
   │
   ▼
Validation Layer
   │
   ▼
JSON Response
```

* **FastAPI:** Handles stateless HTTP requests and preloads heavy assets (models, catalog) at startup.
* **Safety Layer:** Deterministically intercepts prompt injections and maximum-length violations before processing.
* **Conversation Controller:** The core engine. It routes the conversation through defined intents (Clarify, Recommend, Refine, Compare, Confirm) without relying on the LLM for high-level decision-making.
* **Hybrid Retriever:** Combines lexical matching with dense vector embeddings to locate catalog items.
* **LLM:** Used strictly as a two-phase engine: first for extracting typed JSON slots from the conversation, and second for natural language generation grounded exclusively in the retrieved data.
* **Validation Layer:** Pydantic models mathematically enforce the final JSON schema compliance.

### 3. Retrieval Strategy
Our retrieval engine is a hybrid system designed to capture both explicit technical requirements and abstract conceptual alignments. 
* **Offline Catalog Normalization:** The raw JSON catalog is flattened into a rich internal model, concatenating descriptions, keys, and job levels into a single searchable document matrix.
* **TF-IDF Retrieval:** Used for hard lexical matching. This guarantees that highly specific technical constraints (e.g., "React", "AWS", "C++") retrieve the exact framework assessments rather than loosely related alternatives.
* **Semantic Search:** We utilize `all-MiniLM-L6-v2` to generate dense vector embeddings, allowing the system to understand conceptual similarities (e.g., mapping a user's request for "leadership" to assessments tagged with "management" or "executive").
* **Constraint Filtering & Variant Removal:** The system applies hard deterministic filters for explicit user constraints, automatically filtering out incorrect languages or explicitly rejected assessments. 
* **Grounded Recommendations:** The LLM never invents assessments natively. It is provided the exact catalog data retrieved by this engine and instructed to reason strictly over those provided candidates.

### 4. Conversation Management
Rather than allowing the LLM to freewheel, conversation state is managed through structured slot extraction.
* **Slot Extraction:** The LLM extracts the conversational state into a typed JSON schema containing `role`, `skills`, `seniority`, `additions`, and `removals`.
* **Clarification Policy:** If the extracted slots lack sufficient detail (e.g., no role or skills provided), deterministic rules force a Clarification intent, ensuring the agent never provides recommendations for vague queries.
* **Refinement & Comparison:** The state machine detects when users explicitly add or remove constraints (Refinement) or ask to compare specific tests (Comparison), triggering dedicated handlers that merge new constraints with previously retrieved results.
* **Off-Topic Refusals:** We employ deterministic instructions to immediately deflect off-topic requests (e.g., legal or medical advice) with a polite refusal.

### 5. Evaluation & Results
The system was rigorously evaluated using an automated testing harness simulating multi-turn dialogues and adversarial behavior probes.

| Metric | Result |
| :--- | :--- |
| Recall@10 | 0.952 |
| Schema Compliance | 10/10 |
| Behavior Probes | 5/5 |
| End of Conversation | 10/10 |

In addition to the provided baseline traces, we evaluated the system on several complex, unseen conversational scenarios (internal tests) to verify that the slot extraction and retrieval engine generalized well beyond the training examples.

### 6. Performance Optimizations
To ensure rapid response times and eliminate disk I/O during conversation turns:
* **Preprocessed Catalog:** The catalog is fully parsed, normalized, and loaded into memory on server startup.
* **Cached Retrieval:** TF-IDF matrices and Semantic embeddings are precomputed and cached in RAM.
* **Local Embeddings:** We use a lightweight, efficient embedding model (`all-MiniLM-L6-v2`) running locally on CPU, which reduces semantic retrieval latency to under 50ms without relying on external API calls.

### 7. Limitations & Future Work
While highly resilient, the system's generation phase is ultimately dependent on external LLM API latency, which we mitigated using a time-budgeting fallback. As the assessment catalog scales to tens of thousands of items, the in-memory retrieval could be optimized by transitioning the semantic embeddings to an Approximate Nearest Neighbor (ANN) vector index like FAISS. Finally, more extensive evaluation across a wider array of diverse, unstructured conversational flows would further refine the slot-extraction schema.
