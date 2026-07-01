import asyncio
from app.catalog import get_catalog
from app.retrieval import RetrievalEngine
from app.models import ConversationSlots

async def main():
    get_catalog()  # Ensure catalog is loaded
    engine = RetrievalEngine()
    
    slots = ConversationSlots(role="contact center agent", skills=["english", "US"], seniority="entry-level")
    
    candidates = engine.retrieve(slots, top_k=15)
    for i, c in enumerate(candidates):
        print(f"{i+1}. {c.name}")

if __name__ == "__main__":
    asyncio.run(main())
