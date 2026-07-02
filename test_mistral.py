import os
from app.llm import get_llm_service
from app.models import Message
import time

def test_mistral_call():
    print("Testing Mistral API Integration...")
    llm = get_llm_service()
    
    # 1. Test slot extraction (JSON mode)
    print("\n--- Test 1: JSON Slot Extraction ---")
    messages = [{"role": "user", "content": "We need to hire a Senior Full-Stack Engineer who knows Python, AWS, and React."}]
    start_time = time.time()
    slots = llm.analyze_user_input(messages)
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f}s")
    print(f"Extracted Slots: {slots}")
    
    # 2. Test confirmation response (Natural Language)
    print("\n--- Test 2: Natural Language Generation ---")
    start_time = time.time()
    reply = llm.generate_confirmation(messages, [])
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f}s")
    print(f"Confirmation Reply: {reply}")

if __name__ == "__main__":
    test_mistral_call()
