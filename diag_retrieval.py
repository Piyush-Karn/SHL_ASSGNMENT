"""Diagnostic: what does retrieval return for each trace's final slots?"""
import json
from app.catalog import get_catalog
from app.retrieval import RetrievalEngine
from app.models import ConversationSlots

engine = RetrievalEngine()
engine.initialize()

# Each trace's final accumulated slots (what the mock LLM would extract)
trace_slots = {
    "C1": ConversationSlots(role="senior leadership", purpose="selection"),
    "C2": ConversationSlots(role="rust engineer", seniority="senior", skills=["rust", "networking", "high-performance"]),
    "C3": ConversationSlots(role="contact centre agent", seniority="entry-level", skills=["english", "us", "inbound calls", "customer service"]),
    "C4": ConversationSlots(role="financial analyst", seniority="graduate", skills=["numerical reasoning", "finance"]),
    "C5": ConversationSlots(role="sales", purpose="re-skilling"),
    "C6": ConversationSlots(role="plant operator", industry="chemical", skills=["safety", "reliability", "procedure compliance"]),
    "C7": ConversationSlots(role="healthcare admin", skills=["hipaa", "patient records", "bilingual", "spanish"]),
    "C8": ConversationSlots(role="admin assistant", skills=["excel", "word"]),
    "C9": ConversationSlots(role="full-stack engineer", seniority="senior", skills=["java", "spring", "sql", "aws", "docker"]),
    "C10": ConversationSlots(role="management trainee", seniority="graduate", skills=["cognitive", "personality"]),
}

expected = {
    "C1": ["Occupational Personality Questionnaire OPQ32r", "OPQ Universal Competency Report 2.0", "OPQ Leadership Report"],
    "C2": ["Smart Interview Live Coding", "Linux Programming (General)", "Networking and Implementation (New)", "SHL Verify Interactive G+", "Occupational Personality Questionnaire OPQ32r"],
    "C3": ["SVAR Spoken English (US) (New)", "Contact Center Call Simulation (New)", "Entry Level Customer Serv - Retail & Contact Center", "Customer Service Phone Simulation"],
    "C4": ["SHL Verify Interactive - Numerical Reasoning", "Financial Accounting (New)", "Basic Statistics (New)", "Graduate Scenarios", "Occupational Personality Questionnaire OPQ32r"],
    "C5": ["Global Skills Assessment", "Global Skills Development Report", "Occupational Personality Questionnaire OPQ32r", "OPQ MQ Sales Report", "Sales Transformation 2.0 - Individual Contributor"],
    "C6": ["Manufac. & Indust. - Safety & Dependability 8.0", "Workplace Health and Safety (New)"],
    "C7": ["HIPAA (Security)", "Medical Terminology (New)", "Microsoft Word 365 - Essentials (New)", "Dependability and Safety Instrument (DSI)", "Occupational Personality Questionnaire OPQ32r"],
    "C8": ["Microsoft Excel 365 (New)", "Microsoft Word 365 (New)", "MS Excel (New)", "MS Word (New)", "Occupational Personality Questionnaire OPQ32r"],
    "C9": ["Core Java (Advanced Level) (New)", "Spring (New)", "SQL (New)", "Amazon Web Services (AWS) Development (New)", "Docker (New)", "SHL Verify Interactive G+", "Occupational Personality Questionnaire OPQ32r"],
    "C10": ["SHL Verify Interactive G+", "Graduate Scenarios"],
}

import re
def normalize(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

for tid, slots in trace_slots.items():
    results = engine.retrieve(slots, top_k=10)
    result_names = [r.name for r in results]
    exp = expected[tid]
    
    exp_norm = {normalize(e) for e in exp}
    found_norm = {normalize(r) for r in result_names}
    hits = exp_norm & found_norm
    recall = len(hits) / len(exp_norm) if exp_norm else 0
    
    missing = []
    for e in exp:
        if normalize(e) not in found_norm:
            missing.append(e)
    
    print(f"\n{'='*60}")
    print(f"Trace {tid}: Recall@10 = {recall:.2f} ({len(hits)}/{len(exp_norm)})")
    print(f"  Retrieved: {result_names}")
    if missing:
        print(f"  MISSING: {missing}")
