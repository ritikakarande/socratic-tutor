"""Tool registry: a small controlled catalogue of tools the agent may call.

The registry exists so that tool access is explicit and restricted (part of the
defense in depth described in the README). The agent cannot invoke an arbitrary
callable; it can only request a tool by name from this registry, and each tool
validates its own input.
"""

from __future__ import annotations

from typing import Optional

from app.rag.retriever import Retriever
from app.tools.calculator import CalculationResult, calculate
from app.tools.retrieval import RetrievalResult, retrieve


class ToolRegistry:
    """Holds the concrete tool dependencies and exposes safe call methods."""

    def __init__(self, retriever: Optional[Retriever] = None) -> None:
        self._retriever = retriever

    @property
    def has_retriever(self) -> bool:
        return self._retriever is not None

    def calculator(
        self, operation: str, expression: str, variable: Optional[str] = None
    ) -> CalculationResult:
        """Run the safe SymPy calculator."""
        return calculate(operation=operation, expression=expression, variable=variable)

    def retrieval(
        self, query: str, subject: Optional[str] = None, top_k: Optional[int] = None
    ) -> RetrievalResult:
        """Run retrieval, or return a structured error if none is configured."""
        if self._retriever is None:
            return RetrievalResult(success=False, query=query, error="Retriever not configured.")
        return retrieve(query=query, retriever=self._retriever, subject=subject, top_k=top_k)
