"""
FastAPI application for the SHL Assessment Recommender.

Endpoints:
- GET /health — readiness check (returns {"status": "ok"})
- POST /chat — stateless conversation endpoint

Design decisions:
- Startup event preloads catalog and retrieval index (cold start < 2 min)
- CORS enabled for potential frontend integration
- Request validation via Pydantic (automatic 422 on bad input)
- Logging configured at startup for production debugging
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import ChatRequest, ChatResponse, Message
from app.controller import ConversationController
from app.catalog import get_catalog
from app.retrieval import get_retrieval_engine
from app import config

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --- Startup / Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Preload catalog and build retrieval index at startup.
    This ensures the first /chat request doesn't pay the cold-start cost.
    """
    logger.info("Starting up — loading catalog and building indices...")
    start = time.time()

    try:
        catalog = get_catalog()
        logger.info(f"Catalog loaded: {len(catalog)} assessments")
    except Exception as e:
        logger.error(f"Failed to load catalog: {e}")
        raise

    try:
        engine = get_retrieval_engine()
        logger.info("Retrieval engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize retrieval engine: {e}")
        raise

    elapsed = time.time() - start
    logger.info(f"Startup complete in {elapsed:.1f}s")

    yield  # App runs here

    logger.info("Shutting down")


# --- App ---
app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL assessments "
        "based on hiring needs through multi-turn dialogue."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for the evaluation harness
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize THIS globally so the models and catalog load only once!
# This single change will make your API instantly respond.
controller = ConversationController()


# --- Endpoints ---

@app.get("/health")
async def health():
    """
    Readiness check. Returns {"status": "ok"} with HTTP 200.
    
    Per the assignment: the evaluator allows up to 2 minutes for cold start.
    Our lifespan handler ensures the service is ready before accepting requests.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless conversation endpoint.
    
    Every call carries the full conversation history. The service stores
    no per-conversation state.
    
    Returns:
    - reply: natural language response
    - recommendations: empty when clarifying, 1-10 items when recommending
    - end_of_conversation: true only when task is complete
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.messages:
            return ChatResponse(
                reply="Hello! I'm here to help you find the right SHL assessments. What role are you hiring for?",
                recommendations=[],
                end_of_conversation=False,
            )

        # Check turn limits before doing ANY LLM work
        total_turns = len(request.messages)
        if total_turns >= config.MAX_TURNS:
            logger.warning(f"Turn count {total_turns} exceeds max {config.MAX_TURNS}")
            return ChatResponse(
                reply="We've reached the maximum number of messages for this session. Please review the assessments recommended above, or start a new chat!",
                recommendations=[],
                end_of_conversation=True
            )

        # Convert to dicts for processing
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Process through the global controller
        response = controller.process(messages)

        elapsed = time.time() - start_time
        logger.info(
            f"Chat processed in {elapsed:.2f}s — "
            f"reply_len={len(response.reply)}, "
            f"recs={len(response.recommendations)}, "
            f"eoc={response.end_of_conversation}"
        )

        return response

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Chat failed after {elapsed:.2f}s: {e}", exc_info=True)
        # Return a graceful error response rather than crashing
        return ChatResponse(
            reply=f"SERVER ERROR: {str(e)}",
            recommendations=[],
            end_of_conversation=False,
        )


# --- Run directly ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
    )
