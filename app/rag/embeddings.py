"""Configurable embedding backends.

Two backends are provided:

* ``openai`` uses the OpenAI embeddings API (needs an API key).
* ``local`` is a deterministic hashing embedder with no external dependency.

The local backend is not competitive with a trained model, but it is fully
deterministic and offline, which lets the RAG pipeline, ingestion script and
the test suite run on any machine without credentials. The backend is selected
through :class:`~app.config.settings.Settings`, so swapping providers is a
configuration change, not a code change.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List

from app.config.settings import EmbeddingBackend, Settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_LOCAL_DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    """Abstract embedding interface."""

    dimension: int

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class LocalHashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder.

    Each token is hashed into a fixed number of buckets and its contribution is
    signed by a second hash. The resulting vector is L2 normalized. Semantically
    related texts that share vocabulary land near each other, which is enough
    for demos and tests without any network access.
    """

    def __init__(self, dimension: int = _LOCAL_DIM) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall((text or "").lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        return _normalize(vec)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings backend."""

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None) -> None:
        from openai import OpenAI  # imported lazily so offline use never needs it

        self._client = OpenAI(api_key=api_key or "not-required", base_url=base_url or None)
        self._model = model
        # text-embedding-3-small is 1536 dims; queried lazily on first call.
        self.dimension = 1536

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=self._model, input=[text])
        return resp.data[0].embedding


def build_embedder(settings: Settings) -> Embedder:
    """Factory that returns the configured embedder.

    Falls back to the local embedder if the OpenAI backend is requested without
    a key, logging a warning rather than crashing.
    """
    if settings.embedding_backend == EmbeddingBackend.OPENAI:
        if settings.has_openai_key:
            logger.info("Using OpenAI-compatible embedder: %s", settings.embedding_model)
            return OpenAIEmbedder(
                settings.embedding_model,
                settings.openai_api_key or "",
                base_url=settings.openai_base_url,
            )
        logger.warning("OpenAI embeddings requested but no API key found; using local embedder.")
    return LocalHashingEmbedder()
