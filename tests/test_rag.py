"""Tests for the RAG pipeline: embeddings, ingestion, retrieval, filtering."""

from __future__ import annotations

from app.rag.embeddings import LocalHashingEmbedder
from app.rag.ingestion import chunk_text, ingest_documents
from app.rag.vector_store import InMemoryVectorStore, StoredDocument


def test_local_embedder_deterministic():
    emb = LocalHashingEmbedder()
    a = emb.embed_query("Newton's second law")
    b = emb.embed_query("Newton's second law")
    assert a == b
    assert len(a) == emb.dimension


def test_chunking_produces_overlap():
    text = "\n\n".join(f"Paragraph number {i} with several words in it." for i in range(20))
    chunks = chunk_text(text, chunk_size=120, overlap=30, base_metadata={"source": "t"})
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "t" for c in chunks)


def test_oversized_paragraph_is_split():
    """Regression: a single paragraph longer than chunk_size must be split.

    Extracted PDF text often yields one very long paragraph per page. Before the
    fix these were emitted whole, so chunk_size was silently ignored (chunks up
    to 5300 chars were observed with chunk_size=1000).
    """
    page = "word " * 900  # ~4500 chars, no blank lines
    chunks = chunk_text(page, chunk_size=1000, overlap=150, base_metadata={"source": "t"})
    assert len(chunks) > 1
    assert max(len(c.text) for c in chunks) <= 1000


def test_chunk_size_is_a_hard_maximum_with_overlap():
    """Regression: the carried overlap must not push a chunk past chunk_size."""
    mixed = "\n\n".join(["short para here."] * 5 + ["x" * 3000])
    chunks = chunk_text(mixed, chunk_size=1000, overlap=150, base_metadata={"source": "t"})
    assert max(len(c.text) for c in chunks) <= 1000


def test_tiny_fragments_are_dropped():
    """Page numbers and stray glyphs should not become embedded chunks."""
    assert chunk_text("42", chunk_size=1000, overlap=150, base_metadata={"source": "t"}) == []


def test_in_memory_store_similarity():
    emb = LocalHashingEmbedder()
    store = InMemoryVectorStore()
    store.add([
        StoredDocument("1", "force equals mass times acceleration", emb.embed_query("force equals mass times acceleration"), {"subject": "physics"}),
        StoredDocument("2", "photosynthesis occurs in chloroplasts", emb.embed_query("photosynthesis occurs in chloroplasts"), {"subject": "biology"}),
    ])
    hits = store.query(emb.embed_query("what is force and acceleration"), top_k=1)
    assert hits
    assert hits[0].id == "1"


def test_metadata_filtering(seeded_retriever):
    physics = seeded_retriever.search("energy and motion", subject="physics")
    assert physics
    assert all(c.subject == "physics" for c in physics)
    biology = seeded_retriever.search("energy and motion", subject="biology")
    assert all(c.subject == "biology" for c in biology)


def test_retriever_returns_citations(seeded_retriever):
    chunks = seeded_retriever.search("Newton second law force")
    assert chunks
    assert chunks[0].citation()


def test_ingestion_skips_missing_files():
    emb = LocalHashingEmbedder()
    store = InMemoryVectorStore()
    report = ingest_documents(
        [{"path": "does/not/exist.txt", "subject": "math"}],
        embedder=emb,
        store=store,
    )
    assert report.files == 0
    assert report.skipped
