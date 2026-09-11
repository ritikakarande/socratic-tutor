"""Vector store abstraction.

An abstract :class:`VectorStore` defines the small surface the retriever needs.
Two implementations ship today:

* :class:`ChromaVectorStore` - persistent local ChromaDB (default).
* :class:`InMemoryVectorStore` - dependency-free store used for tests and as a
  fallback when Chroma cannot be initialised.

A Qdrant implementation can be added by subclassing :class:`VectorStore`
without touching the retriever or agent. The store works in "bring your own
embeddings" mode: callers pass precomputed vectors, so the embedding backend
and the vector database are independently swappable.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config.settings import Settings, VectorDB
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StoredDocument:
    """A single stored chunk with its vector and metadata."""

    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SearchHit:
    """A retrieval result with a similarity score in [0, 1]."""

    id: str
    text: str
    score: float
    metadata: Dict[str, str] = field(default_factory=dict)


class VectorStore(ABC):
    """Minimal vector store interface."""

    @abstractmethod
    def add(self, documents: List[StoredDocument]) -> None:
        """Insert or upsert documents."""

    @abstractmethod
    def query(
        self,
        embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, str]] = None,
    ) -> List[SearchHit]:
        """Return the top_k most similar documents, optionally metadata filtered."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored documents."""


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore(VectorStore):
    """Simple, exact, dependency-free vector store."""

    def __init__(self) -> None:
        self._docs: Dict[str, StoredDocument] = {}

    def add(self, documents: List[StoredDocument]) -> None:
        for doc in documents:
            self._docs[doc.id] = doc

    def query(
        self,
        embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, str]] = None,
    ) -> List[SearchHit]:
        candidates = self._docs.values()
        if where:
            candidates = [
                d for d in candidates if all(d.metadata.get(k) == v for k, v in where.items())
            ]
        scored = [
            SearchHit(
                id=d.id,
                text=d.text,
                score=(_cosine(embedding, d.embedding) + 1.0) / 2.0,
                metadata=d.metadata,
            )
            for d in candidates
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._docs)


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB-backed store using externally supplied embeddings."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        import chromadb  # imported lazily
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, documents: List[StoredDocument]) -> None:
        if not documents:
            return
        self._collection.upsert(
            ids=[d.id for d in documents],
            embeddings=[d.embedding for d in documents],
            documents=[d.text for d in documents],
            metadatas=[d.metadata or {"source": "unknown"} for d in documents],
        )

    def query(
        self,
        embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, str]] = None,
    ) -> List[SearchHit]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where or None,
        )
        hits: List[SearchHit] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids):
            # Chroma cosine distance is in [0, 2]; convert to a [0, 1] similarity.
            distance = dists[i] if i < len(dists) else 1.0
            score = max(0.0, 1.0 - distance / 2.0)
            hits.append(
                SearchHit(
                    id=doc_id,
                    text=docs[i] if i < len(docs) else "",
                    score=score,
                    metadata=metas[i] if i < len(metas) else {},
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()


def build_vector_store(settings: Settings) -> VectorStore:
    """Factory returning the configured vector store, with a safe fallback."""
    if settings.vector_db == VectorDB.CHROMA:
        try:
            return ChromaVectorStore(settings.chroma_persist_dir, settings.chroma_collection)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma unavailable (%s); falling back to in-memory store.", exc)
            return InMemoryVectorStore()
    if settings.vector_db == VectorDB.QDRANT:
        raise NotImplementedError(
            "Qdrant backend is planned. Implement a QdrantVectorStore(VectorStore) "
            "and wire it here; the retriever and agent need no changes."
        )
    return InMemoryVectorStore()
