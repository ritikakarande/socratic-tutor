"""Document ingestion pipeline for the RAG knowledge base.

Supports TXT, Markdown and PDF. The pipeline is:

    load -> extract text (per page for PDFs) -> clean -> chunk with overlap
    -> enrich metadata -> embed -> store

Chunking respects paragraph boundaries where possible instead of slicing at
arbitrary character offsets, and every chunk keeps its source, subject,
chapter, section and page metadata so retrieval can filter and cite.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app.rag.embeddings import Embedder
from app.rag.vector_store import StoredDocument, VectorStore
from app.utils.helpers import normalize_whitespace
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk plus its metadata prior to embedding."""

    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class IngestionReport:
    """Summary returned after ingesting one or more documents."""

    files: int = 0
    chunks: int = 0
    skipped: List[str] = field(default_factory=list)


_HEADING_RE = re.compile(r"^(chapter|section|unit)\s+[\w.\-]+", re.IGNORECASE)


def _clean(text: str) -> str:
    """Remove control noise and collapse excessive blank lines."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_text_file(path: Path) -> List[tuple[str, Optional[int]]]:
    """Load a TXT or Markdown file as a single (text, page=None) unit."""
    return [(path.read_text(encoding="utf-8", errors="ignore"), None)]


def load_pdf(path: Path) -> List[tuple[str, Optional[int]]]:
    """Load a PDF as a list of (page_text, page_number) tuples using PyMuPDF."""
    import fitz  # PyMuPDF, imported lazily

    pages: List[tuple[str, Optional[int]]] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            pages.append((page.get_text("text"), i + 1))
    return pages


def load_document(path: Path) -> List[tuple[str, Optional[int]]]:
    """Dispatch to the correct loader by file extension."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return load_text_file(path)
    if suffix == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _detect_heading(line: str) -> Optional[str]:
    line = line.strip()
    if _HEADING_RE.match(line) and len(line) < 120:
        return line
    return None


def _split_oversized(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Hard-split a single paragraph that is longer than ``chunk_size``.

    Extracted PDF text frequently yields one very long "paragraph" per page.
    Without this, such a paragraph would be emitted as a single oversized chunk
    and the configured chunk size would be silently ignored. Splits prefer a
    word boundary and carry ``overlap`` characters into the next piece.
    """
    if len(text) <= chunk_size:
        return [text]
    pieces: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Back off to the last space in the latter part of the window so we
            # do not cut mid-word, but never collapse the chunk to nothing.
            boundary = text.rfind(" ", start + int(chunk_size * 0.6), end)
            if boundary != -1:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    base_metadata: Dict[str, str],
    page: Optional[int] = None,
    min_chunk_chars: int = 40,
) -> List[Chunk]:
    """Split text into overlapping chunks along paragraph boundaries.

    Paragraphs are accumulated until adding the next would exceed ``chunk_size``.
    A trailing overlap of ``overlap`` characters is carried into the next chunk
    to preserve context across boundaries. Simple heading lines update the
    running ``section`` metadata.
    """
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # Break any paragraph that exceeds the chunk size before accumulating.
    paragraphs: List[str] = []
    for para in raw_paragraphs:
        paragraphs.extend(_split_oversized(para, chunk_size, overlap))

    chunks: List[Chunk] = []
    buffer = ""
    current_section = base_metadata.get("section")

    def flush() -> None:
        nonlocal buffer
        content = normalize_whitespace(buffer)
        # Drop fragments too small to carry meaning (page numbers, stray glyphs).
        if content and len(content) >= min_chunk_chars:
            meta = dict(base_metadata)
            if current_section:
                meta["section"] = current_section
            if page is not None:
                meta["page"] = str(page)
            chunks.append(Chunk(text=content, metadata=meta))
        buffer = ""

    for para in paragraphs:
        heading = _detect_heading(para)
        if heading:
            current_section = heading
        if len(buffer) + len(para) + 1 > chunk_size and buffer:
            tail = buffer[-overlap:] if overlap > 0 else ""
            flush()
            # Carry the overlap only when it still fits, so chunk_size stays a
            # hard maximum. Oversized paragraphs already carry their own overlap
            # from _split_oversized.
            if tail and len(tail) + 1 + len(para) <= chunk_size:
                buffer = tail + " " + para
            else:
                buffer = para
        else:
            buffer = f"{buffer}\n{para}" if buffer else para
    flush()
    return chunks


def ingest_documents(
    documents: List[Dict],
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 1000,
    overlap: int = 150,
    batch_size: int = 512,
    progress: bool = False,
) -> IngestionReport:
    """Ingest a list of document descriptors into the vector store.

    Each descriptor is a dict with keys ``path`` and metadata such as
    ``subject``, ``source``, ``chapter``, ``grade``. Missing files are skipped
    and recorded in the report rather than raising.

    Chunks are embedded and flushed to the store in batches of ``batch_size``
    rather than accumulated in memory, so book-scale corpora (thousands of
    pages) ingest with bounded memory and avoid oversized upserts.
    """
    report = IngestionReport()
    batch: List[StoredDocument] = []

    def flush() -> None:
        """Write the pending batch to the store and clear it."""
        nonlocal batch
        if batch:
            store.add(batch)
            batch = []

    for descriptor in documents:
        path = Path(descriptor["path"])
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            report.skipped.append(str(path))
            continue

        base_meta = {
            k: str(v)
            for k, v in descriptor.items()
            if k != "path" and v is not None
        }
        base_meta.setdefault("source", path.stem)

        try:
            units = load_document(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", path, exc)
            report.skipped.append(str(path))
            continue

        file_chunks: List[Chunk] = []
        for raw_text, page in units:
            cleaned = _clean(raw_text)
            if not cleaned:
                continue
            file_chunks.extend(chunk_text(cleaned, chunk_size, overlap, base_meta, page))

        if not file_chunks:
            report.skipped.append(str(path))
            continue

        # Embed and flush in batches to bound memory and upsert size.
        for start in range(0, len(file_chunks), batch_size):
            window = file_chunks[start : start + batch_size]
            embeddings = embedder.embed_documents([c.text for c in window])
            for chunk, vector in zip(window, embeddings):
                batch.append(
                    StoredDocument(
                        id=uuid.uuid4().hex,
                        text=chunk.text,
                        embedding=vector,
                        metadata=chunk.metadata,
                    )
                )
            if len(batch) >= batch_size:
                flush()
            if progress:
                done = min(start + batch_size, len(file_chunks))
                print(f"    {path.name}: {done}/{len(file_chunks)} chunks", flush=True)

        flush()
        report.files += 1
        report.chunks += len(file_chunks)
        logger.info("Ingested %s (%d chunks)", path.name, len(file_chunks))

    flush()
    return report
