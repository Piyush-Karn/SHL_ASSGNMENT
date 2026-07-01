import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

print(f"Testing API Key (first 10 chars): {api_key[:10]}...")
print(f"Configured Model: {model_name}")

try:
    genai.configure(api_key=api_key)
    # Use gemini-1.5-flash as it is most standard, but also try gemini-2.5-flash
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content('Hello, is this API key working?')
    print("SUCCESS (with gemini-1.5-flash)!")
    print(f"Response: {response.text.strip()}")
except Exception as e:
    print(f"FAILED (gemini-1.5-flash): {e}")

try:
    model2 = genai.GenerativeModel('gemini-2.5-flash')
    response2 = model2.generate_content('Hello, is this API key working?')
    print("SUCCESS (with gemini-2.5-flash)!")
    print(f"Response: {response2.text.strip()}")
except Exception as e:
    print(f"FAILED (gemini-2.5-flash): {e}")
