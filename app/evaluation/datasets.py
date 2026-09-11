"""Dataset loaders for the evaluation pipeline.

Loads the custom Socratic pedagogy dataset, the red-team safety dataset, and
any processed benchmark records produced by the data-acquisition scripts.
Loaders are tolerant: a missing file yields an empty list with a warning rather
than raising, so the pipeline degrades gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Repository-relative default locations.
DATA_DIR = Path("data")
SOCRATIC_TESTS = DATA_DIR / "raw" / "tutoring" / "socratic_tests.json"
SAFETY_TESTS = DATA_DIR / "raw" / "red_team" / "safety_tests.json"
RAG_INJECTION_TESTS = DATA_DIR / "raw" / "red_team" / "rag_injection_tests.json"
GSM8K_PROCESSED = DATA_DIR / "processed" / "evaluation" / "gsm8k_eval.jsonl"
MATH_PROCESSED = DATA_DIR / "processed" / "evaluation" / "math_eval.jsonl"


def _load_json(path: Path) -> List[Dict]:
    if not path.exists():
        logger.warning("Dataset not found: %s", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return []
    if isinstance(data, dict) and "examples" in data:
        return data["examples"]
    if isinstance(data, list):
        return data
    logger.warning("Unexpected dataset shape in %s", path)
    return []


def _load_jsonl(path: Path, limit: int | None = None) -> List[Dict]:
    if not path.exists():
        logger.warning("Dataset not found: %s", path)
        return []
    rows: List[Dict] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_socratic_tests() -> List[Dict]:
    """Load the custom pedagogy dataset."""
    return _load_json(SOCRATIC_TESTS)


def load_safety_tests() -> List[Dict]:
    """Load the red-team safety dataset."""
    return _load_json(SAFETY_TESTS)


def load_rag_injection_tests() -> List[Dict]:
    """Load RAG prompt-injection cases (malicious text inside documents)."""
    return _load_json(RAG_INJECTION_TESTS)


def load_gsm8k(limit: int | None = 50) -> List[Dict]:
    """Load processed GSM8K evaluation records if present."""
    return _load_jsonl(GSM8K_PROCESSED, limit=limit)


def load_math(limit: int | None = 50) -> List[Dict]:
    """Load processed MATH evaluation records if present."""
    return _load_jsonl(MATH_PROCESSED, limit=limit)
