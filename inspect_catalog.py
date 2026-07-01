"""Inspect catalog entries matching expected assessments from all 10 traces."""
import json

data = json.loads(open('data/shlcatalogue.json','r',encoding='utf-8').read(), strict=False)

targets = [
    'OPQ32r', 'OPQ Universal', 'OPQ Leadership', 'Graduate Scenarios',
    'Verify Interactive', 'Global Skills Assessment', 'Global Skills Development',
    'Sales Transformation', 'Safety & Dependability', 'Dependability and Safety Instrument',
    'Workplace Health and Safety', 'HIPAA', 'Medical Terminology',
    'Microsoft Word 365', 'Microsoft Excel 365', 'MS Excel', 'MS Word',
    'Core Java', 'Spring (New)', 'Docker', 'Smart Interview', 'Linux Programming',
    'Networking and Implementation', 'Financial Accounting', 'Basic Statistics',
    'Numerical Reasoning', 'Contact Center Call', 'Customer Service Phone',
    'SVAR', 'Entry Level Customer Serv', 'OPQ MQ Sales',
    'Amazon Web Services', 'SQL (New)',
]

for t in targets:
    matches = [d for d in data if t.lower() in d['name'].lower()]
    for m in matches:
        name = m['name']
        keys = m.get('keys', [])
        jl = m.get('job_levels', [])
        desc = m.get('description', '')[:120]
        print(f"  [{t}] -> {name}")
        print(f"    keys={keys}")
        print(f"    job_levels={jl}")
        print(f"    desc={desc}")
        print()
    if not matches:
        print(f"  NO MATCH for: {t}")
        print()
