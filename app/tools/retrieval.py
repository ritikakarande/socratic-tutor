"""Retrieval tool wrapper.

This is a thin adapter that lets the agent call retrieval through the same
registry interface as the calculator. The heavy lifting lives in
:mod:`app.rag.retriever`; this module keeps the tool surface small and returns
structured, guard-friendly output.

Retrieved documents are treated as untrusted data. This wrapper never marks
retrieved text as instructions; it only carries it as reference material.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.rag.retriever import Retriever, RetrievedChunk


class RetrievalResult(BaseModel):
    """Structured retrieval output."""

    success: bool
    query: str
    subject_filter: Optional[str] = None
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.chunks


def retrieve(
    query: str,
    retriever: Retriever,
    subject: Optional[str] = None,
    top_k: Optional[int] = None,
) -> RetrievalResult:
    """Retrieve reference chunks for a query with optional subject filtering."""
    if not query or not query.strip():
        return RetrievalResult(success=False, query=query, error="Empty query.")
    try:
        chunks = retriever.search(query, subject=subject, top_k=top_k)
        return RetrievalResult(
            success=True,
            query=query,
            subject_filter=subject,
            chunks=chunks,
        )
    except Exception as exc:  # noqa: BLE001 - retrieval failure must not crash agent
        return RetrievalResult(success=False, query=query, subject_filter=subject, error=str(exc))
