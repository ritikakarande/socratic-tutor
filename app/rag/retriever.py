"""Retriever: embeds a query and searches the vector store.

The retriever is the only place the agent touches the RAG stack. It applies
optional subject-based metadata filtering and returns Pydantic models so the
result flows cleanly through the tool layer and into evaluation.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.config.settings import Settings, get_settings
from app.rag.embeddings import Embedder, build_embedder
from app.rag.vector_store import VectorStore, build_vector_store


class RetrievedChunk(BaseModel):
    """A single retrieved reference chunk."""

    text: str
    score: float
    source: str = "unknown"
    subject: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None

    def citation(self) -> str:
        """Human-readable citation used in the response context block."""
        parts = [self.source]
        if self.chapter:
            parts.append(self.chapter)
        if self.page is not None:
            parts.append(f"p.{self.page}")
        return ", ".join(str(p) for p in parts if p)


class Retriever:
    """Coordinates embedding and vector search."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        default_top_k: int = 4,
        min_score: float = 0.0,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._default_top_k = default_top_k
        self._min_score = min_score

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def search(
        self,
        query: str,
        subject: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Embed the query and return matching chunks above the score floor."""
        k = top_k or self._default_top_k
        vector = self._embedder.embed_query(query)
        where = {"subject": subject} if subject else None
        hits = self._store.query(vector, top_k=k, where=where)
        chunks: List[RetrievedChunk] = []
        for hit in hits:
            if hit.score < self._min_score:
                continue
            meta = hit.metadata or {}
            page = meta.get("page")
            chunks.append(
                RetrievedChunk(
                    text=hit.text,
                    score=round(hit.score, 4),
                    source=meta.get("source", "unknown"),
                    subject=meta.get("subject"),
                    chapter=meta.get("chapter"),
                    section=meta.get("section"),
                    page=int(page) if page not in (None, "") else None,
                )
            )
        return chunks


def build_retriever(settings: Optional[Settings] = None, auto_seed: bool = True) -> Retriever:
    """Factory constructing a retriever from settings.

    When ``auto_seed`` is set and the store is an empty in-memory store (the
    fallback used when a persistent ChromaDB is unavailable), the built-in seed
    corpus is loaded so RAG works out of the box. Persistent stores are left
    untouched; ingest them with the scripts instead.
    """
    settings = settings or get_settings()
    embedder = build_embedder(settings)
    store = build_vector_store(settings)
    if auto_seed:
        from app.rag.seed_corpus import seed_if_empty
        from app.rag.vector_store import InMemoryVectorStore

        if isinstance(store, InMemoryVectorStore):
            seed_if_empty(embedder, store)
    return Retriever(embedder=embedder, store=store, default_top_k=settings.rag_top_k)
