"""General RAG ingestion of the built-in seed corpus.

Populates the configured vector store (persistent ChromaDB when available) with
the self-authored science notes in ``app/rag/seed_corpus.py``. This gives the
retrieval pipeline usable content with no external downloads.

For the full OpenStax corpus use ``scripts/ingest_openstax.py`` instead.

    python scripts/ingest.py
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

from app.config.settings import get_settings
from app.rag.embeddings import build_embedder
from app.rag.seed_corpus import SEED, seed_documents
from app.rag.vector_store import build_vector_store

SEED_DIR = Path("data/raw/seed_corpus")


def _write_reference_copies() -> None:
    """Write the seed notes to disk as Markdown for transparency."""
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    for key, meta in SEED.items():
        (SEED_DIR / f"{key}.md").write_text(
            f"# {meta['chapter']} ({meta['subject']})\n\n{meta['text']}\n", encoding="utf-8"
        )


def main() -> None:
    settings = get_settings()
    print("Building embedder and vector store...")
    embedder = build_embedder(settings)
    store = build_vector_store(settings)

    _write_reference_copies()
    print(f"Ingesting {len(SEED)} seed documents...")
    chunks = seed_documents(
        embedder, store, chunk_size=settings.rag_chunk_size, overlap=settings.rag_chunk_overlap
    )
    print(f"Ingested {chunks} chunks. Vector store now holds {store.count()} documents.")


if __name__ == "__main__":
    main()
