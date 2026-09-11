"""OpenStax source acquisition helper.

This script does NOT scrape or bypass any access control. It reads
``data/source_manifest.json``, and for each RAG textbook source it:

  1. Reports the recorded license and AI-use status.
  2. Skips any source marked ``restricted``.
  3. Checks whether a local PDF already exists at the manifest path.
  4. Validates any present PDF (openable, non-empty, extractable text, not an
     HTML error page saved as .pdf).
  5. For sources not yet present, prints the official page URL and instructs the
     user to download the PDF manually from OpenStax.

Because a Creative Commons license does not automatically grant permission to
ingest a work into a generative AI system, the ``ai_use_status`` must be set to
a permitting value in the manifest (after you verify the current terms) before
``scripts/ingest_openstax.py`` will ingest a book.

    python scripts/download_openstax.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import _bootstrap  # noqa: F401

MANIFEST = Path("data/source_manifest.json")

# Statuses in the manifest that permit ingestion. "verify_before_ingestion" is
# intentionally NOT permitting: the user must confirm terms and update it.
PERMITTED_STATUSES = {"permitted", "permission_obtained"}


def validate_pdf(path: Path) -> tuple[bool, str]:
    """Validate a PDF file. Returns (ok, detail)."""
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty file"
    head = path.read_bytes()[:1024].lstrip()
    if head[:5] != b"%PDF-":
        if head[:1] in (b"<", b"{"):
            return False, "looks like HTML/JSON saved as .pdf"
        return False, "not a PDF header"
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return True, "header ok (install pymupdf to fully validate)"
    try:
        with fitz.open(path) as doc:
            pages = doc.page_count
            if pages == 0:
                return False, "zero pages"
            sample = "".join(doc[i].get_text("text") for i in range(min(3, pages)))
            if len(sample.strip()) < 20:
                return False, f"{pages} pages but no extractable text"
            return True, f"{pages} pages, text extraction OK"
    except Exception as exc:  # noqa: BLE001
        return False, f"could not open: {exc}"


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"Manifest not found at {MANIFEST}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def main() -> None:
    manifest = load_manifest()
    sources = manifest.get("rag_sources", [])
    print("=" * 40)
    print("OpenStax Source Check")
    print("=" * 40)

    for src in sources:
        name = src.get("name", "unknown")
        status = src.get("ai_use_status", "verify_before_ingestion")
        local = Path(src.get("local_path", ""))
        print(f"\n- {name}")
        print(f"  license: {src.get('license')}")
        print(f"  ai_use_status: {status}")

        if status == "restricted":
            print("  RESTRICTED: this source will not be ingested. Obtain permission or replace it.")
            continue

        if local.exists():
            ok, detail = validate_pdf(local)
            mark = "OK" if ok else "INVALID"
            print(f"  [{mark}] {local} ({detail})")
            if ok and status not in PERMITTED_STATUSES:
                print("  NOTE: file present but ai_use_status is not permitting.")
                print("        Verify the current terms and set ai_use_status to 'permitted'")
                print("        in the manifest before ingestion.")
        else:
            # Create the target directory so the download has somewhere to land
            # (git does not track empty directories, so a fresh clone lacks it).
            if local.parent != Path("."):
                local.parent.mkdir(parents=True, exist_ok=True)
            print(f"  not present locally. Download the PDF from the official page:")
            print(f"    {src.get('url')}")
            print(f"  and save it to: {local}")

    print("\nDone. Run scripts/ingest_openstax.py after verifying licenses and placing PDFs.")


if __name__ == "__main__":
    main()
