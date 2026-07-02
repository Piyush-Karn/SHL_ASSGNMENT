import time
import logging
from app.retrieval import get_retrieval_engine
from app.models import ConversationSlots

logging.basicConfig(level=logging.INFO)

engine = get_retrieval_engine()
print("Engine initializing...")
t0 = time.time()
engine.initialize()
print(f"Initialized in {time.time() - t0:.2f}s")

slots = ConversationSlots(role="Senior Marketing Executive", skills=["marketing"], seniority="senior")

print("Testing first retrieval...")
t1 = time.time()
engine.retrieve(slots)
print(f"Retrieval 1 took {time.time() - t1:.4f}s")

print("Testing identical retrieval...")
t2 = time.time()
engine.retrieve(slots)
print(f"Retrieval 2 took {time.time() - t2:.4f}s")

print("Testing slightly different retrieval (cache miss for RRF but maybe embedding hit?)...")
slots.role = "Senior Marketing Manager"
t3 = time.time()
engine.retrieve(slots)
print(f"Retrieval 3 took {time.time() - t3:.4f}s")
