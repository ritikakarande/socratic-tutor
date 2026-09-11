"""Ingest permitted OpenStax PDFs into the vector database.

Reads ``data/source_manifest.json`` and ingests only the RAG sources whose
``ai_use_status`` permits it and whose local PDF is present and valid. Each
chunk keeps source, subject, chapter, section and page metadata.

    python scripts/ingest_openstax.py

If no permitted PDFs are present, the script explains what to do and exits
without error. Use ``scripts/ingest.py`` for the offline seed corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401

from app.config.settings import get_settings
from app.rag.embeddings import build_embedder
from app.rag.ingestion import ingest_documents
from app.rag.vector_store import build_vector_store
from scripts.download_openstax import PERMITTED_STATUSES, validate_pdf

MANIFEST = Path("data/source_manifest.json")


def main() -> None:
    settings = get_settings()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = manifest.get("rag_sources", [])

    documents = []
    for src in sources:
        status = src.get("ai_use_status", "verify_before_ingestion")
        local = Path(src.get("local_path", ""))
        if status == "restricted":
            print(f"skip (restricted): {src.get('name')}")
            continue
        if status not in PERMITTED_STATUSES:
            print(f"skip (verify license first, set ai_use_status=permitted): {src.get('name')}")
            continue
        ok, detail = validate_pdf(local)
        if not ok:
            print(f"skip (invalid PDF: {detail}): {src.get('name')}")
            continue
        documents.append({
            "path": str(local),
            "subject": src.get("subject", "science"),
            "source": src.get("name"),
            "publisher": src.get("publisher", "OpenStax"),
        })
        print(f"queued: {src.get('name')} ({detail})")

    if not documents:
        print("\nNo permitted, present PDFs to ingest.")
        print("Run scripts/download_openstax.py, verify licenses, place PDFs, and set")
        print("ai_use_status to 'permitted' in data/source_manifest.json, then rerun.")
        print("For an offline demo corpus instead, run: python scripts/ingest.py")
        return

    print("\nBuilding embedder and vector store...")
    embedder = build_embedder(settings)
    store = build_vector_store(settings)
    report = ingest_documents(
        documents,
        embedder=embedder,
        store=store,
        chunk_size=settings.rag_chunk_size,
        overlap=settings.rag_chunk_overlap,
        progress=True,
    )
    print(f"Ingested files={report.files} chunks={report.chunks} skipped={len(report.skipped)}")
    print(f"Vector store now holds {store.count()} documents.")


if __name__ == "__main__":
    main()
