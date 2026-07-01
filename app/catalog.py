"""
SHL Catalog loader and normalizer.

Loads the raw shlcatalogue.json, normalizes fields, infers test type codes,
and builds the search index text. This runs ONCE at startup.

Design decisions:
- Test type codes are inferred from the 'keys' (categories) field, matching
  the pattern observed in the 10 sample conversations.
- Duration is parsed from strings like "30 minutes" into integers.
- search_text concatenates name + description + categories for TF-IDF indexing.
- The catalog is loaded with strict=False to handle control characters in JSON.
"""

from __future__ import annotations

import json
import re
import os
import logging
from typing import Optional

from app.models import Assessment
from app import config

logger = logging.getLogger(__name__)

# Mapping from catalog 'keys' (categories) to single-letter test type codes.
# Derived from analyzing how the sample conversations label test types.
CATEGORY_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


def _parse_duration(duration_str: str) -> Optional[int]:
    """
    Parse duration string to minutes.
    
    Examples:
        "30 minutes" -> 30
        "Approximate Completion Time in minutes = 30" -> 30
        "" -> None
        "Variable" -> None
        "Untimed" -> None
    """
    if not duration_str or duration_str.lower() in ("", "untimed", "variable"):
        return None
    # Try to find a number in the string
    match = re.search(r"(\d+)", duration_str)
    if match:
        return int(match.group(1))
    return None


def _infer_test_type_code(categories: list[str]) -> str:
    """
    Convert category list to comma-separated test type codes.
    
    Examples:
        ["Knowledge & Skills"] -> "K"
        ["Knowledge & Skills", "Simulations"] -> "K,S"
        ["Personality & Behavior"] -> "P"
    """
    codes = []
    for cat in categories:
        code = CATEGORY_TO_CODE.get(cat)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"  # Default to K if unknown


def _build_search_text(assessment: dict) -> str:
    """
    Build a rich text field for TF-IDF indexing.
    
    Concatenates name, description, categories, job levels, and languages.
    This gives TF-IDF the best chance of matching user queries.
    """
    parts = [
        assessment.get("name", ""),
        assessment.get("description", ""),
        " ".join(assessment.get("keys", [])),
        " ".join(assessment.get("job_levels", [])),
    ]
    return " ".join(parts).strip()


def load_catalog(path: Optional[str] = None) -> list[Assessment]:
    """
    Load and normalize the SHL catalog from JSON.
    
    Returns a list of Assessment objects ready for indexing and retrieval.
    """
    catalog_path = path or config.CATALOG_PATH
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(
            f"Catalog not found at {catalog_path}. "
            f"Ensure shlcatalogue.json is in the data/ directory."
        )
    
    with open(catalog_path, "r", encoding="utf-8") as f:
        raw_data = json.loads(f.read(), strict=False)
    
    assessments = []
    seen_ids = set()
    
    for item in raw_data:
        entity_id = item.get("entity_id", "")
        
        # Skip duplicates (shouldn't happen, but defensive)
        if entity_id in seen_ids:
            logger.warning(f"Duplicate entity_id: {entity_id}, skipping")
            continue
        seen_ids.add(entity_id)
        
        # Skip items with bad status
        if item.get("status", "ok") != "ok":
            logger.warning(f"Skipping assessment {entity_id} with status: {item.get('status')}")
            continue
        
        categories = item.get("keys", [])
        
        assessment = Assessment(
            entity_id=entity_id,
            name=item.get("name", ""),
            url=item.get("link", ""),
            job_levels=item.get("job_levels", []),
            languages=item.get("languages", []),
            duration_minutes=_parse_duration(item.get("duration", "")),
            duration_raw=item.get("duration", ""),
            remote=item.get("remote", "yes") == "yes",
            adaptive=item.get("adaptive", "no") == "yes",
            description=item.get("description", ""),
            categories=categories,
            test_type_code=_infer_test_type_code(categories),
            search_text=_build_search_text(item),
        )
        
        assessments.append(assessment)
    
    logger.info(f"Loaded {len(assessments)} assessments from catalog")
    return assessments


# --- Singleton pattern: load catalog once at import time ---
_catalog: Optional[list[Assessment]] = None


def get_catalog() -> list[Assessment]:
    """Get the loaded catalog, loading it on first access."""
    global _catalog
    if _catalog is None:
        _catalog = load_catalog()
    return _catalog


def get_assessment_by_name(name: str) -> Optional[Assessment]:
    """Look up an assessment by exact name (case-insensitive)."""
    catalog = get_catalog()
    name_lower = name.lower().strip()
    for a in catalog:
        if a.name.lower().strip() == name_lower:
            return a
    return None


def get_assessment_by_url(url: str) -> Optional[Assessment]:
    """Look up an assessment by URL."""
    catalog = get_catalog()
    for a in catalog:
        if a.url == url:
            return a
    return None


def validate_url(url: str) -> bool:
    """Check if a URL exists in the catalog. Grounding check."""
    return get_assessment_by_url(url) is not None
