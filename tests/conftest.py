"""Shared pytest fixtures.

All fixtures build the system with offline components (no API key required) so
the suite is hermetic and deterministic.
"""

from __future__ import annotations

import pytest

from app.agents.graph import TutorAgent, build_dependencies
from app.agents.llm import OfflineTutorClient
from app.config.settings import Settings, TutorMode
from app.rag.embeddings import LocalHashingEmbedder
from app.rag.ingestion import ingest_documents
from app.rag.retriever import Retriever
from app.rag.vector_store import InMemoryVectorStore


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key=None, tutor_mode=TutorMode.STRICT)


@pytest.fixture
def offline_llm() -> OfflineTutorClient:
    return OfflineTutorClient()


@pytest.fixture
def seeded_retriever(tmp_path) -> Retriever:
    """A retriever backed by an in-memory store with a few science docs."""
    embedder = LocalHashingEmbedder()
    store = InMemoryVectorStore()
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    samples = {
        "newton.md": ("Newton's second law states that force equals mass times "
                      "acceleration, F = m a.", "physics"),
        "photosynthesis.md": ("Photosynthesis converts light energy into chemical "
                              "energy in the chloroplasts of plant cells.", "biology"),
        "mole.md": ("A mole is a fixed number of particles; water has a molar mass "
                    "of about 18 grams per mole.", "chemistry"),
    }
    documents = []
    for name, (text, subject) in samples.items():
        path = docs_dir / name
        path.write_text(text, encoding="utf-8")
        documents.append({"path": str(path), "subject": subject, "source": name})
    ingest_documents(documents, embedder=embedder, store=store, chunk_size=500, overlap=50)
    return Retriever(embedder=embedder, store=store, default_top_k=3)


@pytest.fixture
def agent(settings, offline_llm, seeded_retriever) -> TutorAgent:
    deps = build_dependencies(settings=settings, retriever=seeded_retriever, llm=offline_llm)
    return TutorAgent(deps)
