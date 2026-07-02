import requests
import sys

API_KEY = "MTMkprUCxmGVxxV5g1XOUDMlaoNloucu"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "mistral-large-latest",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 10
}

try:
    resp = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    sys.exit(0 if resp.status_code == 200 else 1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
