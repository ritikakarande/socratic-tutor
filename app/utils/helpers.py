"""Small shared helpers with no external dependencies."""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from typing import Iterator, List


@contextmanager
def timer() -> Iterator[callable]:
    """Context manager yielding a function that returns elapsed milliseconds.

    Example::

        with timer() as elapsed:
            do_work()
        latency_ms = elapsed()
    """
    start = time.perf_counter()
    stop = None

    def elapsed() -> float:
        end = stop if stop is not None else time.perf_counter()
        return round((end - start) * 1000.0, 2)

    try:
        yield elapsed
    finally:
        stop = time.perf_counter()


def truncate(text: str, max_chars: int = 4000) -> str:
    """Truncate text to a maximum length, appending an ellipsis marker."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [...]"


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip ends."""
    return re.sub(r"\s+", " ", text or "").strip()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric value into the inclusive range [low, high]."""
    return max(low, min(high, value))


_MATH_WORDS = [
    (re.compile(r"\bmultiplied\s+by\b", re.I), " * "),
    (re.compile(r"\btimes\b", re.I), " * "),
    (re.compile(r"\bdivided\s+by\b", re.I), " / "),
    (re.compile(r"\bplus\b", re.I), " + "),
    (re.compile(r"\bminus\b", re.I), " - "),
]


def normalize_math_words(text: str) -> str:
    """Turn common arithmetic words into operators (e.g. "6 times 7" -> "6 * 7").

    Also normalizes the caret exponent ``^`` to Python's ``**`` so extracted
    expressions parse in SymPy.
    """
    out = text or ""
    for pattern, repl in _MATH_WORDS:
        out = pattern.sub(repl, out)
    return out.replace("^", "**")


def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter used for lightweight response analysis."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p]
