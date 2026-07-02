import json
import os

CACHE_FILE = "evaluation/mistral_cache.json"

keywords = [
    "management trainee", 
    "it helpdesk", "helpdesk",
    "backend developer", "node.js", "mongodb",
    "quantitative trading", "c++", "low-latency",
    "marketing",
    "retail sales",
    "human resources", "hr business partner",
    "devops", "kubernetes", "aws", "docker"
]

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
        
    initial_len = len(cache)
    keys_to_delete = []
    
    for key, value in cache.items():
        val_lower = value.lower()
        if any(kw in val_lower for kw in keywords):
            keys_to_delete.append(key)
            
    for key in keys_to_delete:
        del cache[key]
        
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
        
    print(f"Removed {len(keys_to_delete)} entries from cache. Cache size is now {len(cache)}.")
else:
    print("Cache file not found.")
