# SHL AI Intern Assignment -- Implementation Plan

## Goal

Build a grounded conversational assessment recommender using the SHL
catalog as the only knowledge source.

## Architecture

-   FastAPI (`GET /health`, `POST /chat`)
-   Conversation Controller (Python)
-   Gemini 2.5 Flash (reasoning only)
-   Local SHL Catalog
-   Retrieval Engine
-   Safety Layer

## Phase 1 -- Analyze the Dataset

-   Read all 10 conversation traces.
-   Extract deterministic behavior rules.
-   Document when to clarify, recommend, compare, refuse, and end
    conversations.

## Phase 2 -- Catalog Preparation

-   Use the provided `shlcatalogue.json`.
-   Normalize fields (name, URL, description, job levels, languages,
    duration, categories).
-   Never scrape at runtime.

## Phase 3 -- Retrieval

-   Hybrid retrieval (BM25/TF-IDF + optional embeddings).
-   Build queries from accumulated conversation slots.
-   Retrieve \~20 candidates, rerank, return ≤10.

## Phase 4 -- Conversation Controller

Maintain slots: - Role - Skills - Seniority - Language - Test
preference - Exclusions

Rules: - Ask clarification if required. - Retrieve when enough
information exists. - Update recommendations when constraints change. -
Compare only retrieved assessments. - Refuse legal/off-topic requests. -
Finish before the 8-turn limit.

## Phase 5 -- LLM Responsibilities

Gemini should: - Extract structured slots. - Explain recommendations. -
Compare assessments. - Produce grounded responses.

Gemini should never invent assessments or URLs.

## Phase 6 -- Grounding

Only retrieved catalog entries are passed to the LLM.

## Phase 7 -- FastAPI

Implement: - `GET /health` - `POST /chat`

Return: - `reply` - `recommendations` - `end_of_conversation`

Use Pydantic models.

## Phase 8 -- Safety

Handle: - Prompt injection - Off-topic requests - Legal advice

## Phase 9 -- Evaluation

Build a local harness using the 10 supplied conversations. Measure: -
Recall@10 - Behavior correctness - Conversation success

## Phase 10 -- Deployment

Deploy to Render/Railway/Fly/Modal/HF Spaces. Keep embeddings
precomputed.

## Deliverables

-   Public API
-   Working FastAPI service
-   Approach document
-   Grounded conversational recommender
