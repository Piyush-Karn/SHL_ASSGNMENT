---
title: SHL Assessment Recommender
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
# SHL Conversational Assessment Recommender

A conversational AI agent that helps hiring managers go from vague hiring intent to a grounded shortlist of SHL assessments through multi-turn dialogue.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the server
python -m app.main

# Or with uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`
Stateless conversation endpoint.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are assessments that fit a mid-level Java developer.",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Architecture

```
FastAPI Service
  └── Conversation Controller (deterministic decision logic)
        ├── Safety Layer (injection/off-topic/legal detection)
        ├── Slot Extractor (LLM-powered structured extraction)
        ├── Intent Classifier (pattern matching + LLM)
        ├── Retrieval Engine (TF-IDF + semantic reranking)
        └── Response Generator (grounded LLM responses)
```

## Evaluation

```bash
# Run evaluation harness
python -m evaluation.eval_harness --api-url http://localhost:8000

# Run unit tests
python tests/test_api.py

# Run a single trace
python -m evaluation.eval_harness --api-url http://localhost:8000 --trace C1 -v
```

## Deployment

### Render (recommended)
1. Push to GitHub
2. Connect repo on Render
3. Set `GEMINI_API_KEY` as environment variable
4. Deploy using the included `render.yaml`

### Docker
```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key shl-recommender
```

## Project Structure

```
app/
  main.py         # FastAPI application
  models.py       # Pydantic request/response schemas
  config.py       # Configuration
  catalog.py      # SHL catalog loader/normalizer
  retrieval.py    # TF-IDF + semantic retrieval
  controller.py   # Conversation controller
  llm.py          # Gemini LLM integration
  safety.py       # Safety/guardrails layer
  prompts.py      # Prompt templates
data/
  shlcatalogue.json
evaluation/
  eval_harness.py # Automated evaluation
tests/
  test_api.py     # Unit tests
```
