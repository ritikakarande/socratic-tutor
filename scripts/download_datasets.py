"""Benchmark and custom dataset acquisition.

Entry point for obtaining evaluation data:

    python scripts/download_datasets.py

Steps:
  1. Download GSM8K (Hugging Face) and write a normalized eval copy.
  2. Download Hendrycks MATH (Hugging Face) and write a normalized eval copy.
  3. Prepare the custom Socratic pedagogy dataset.
  4. Prepare the custom red-team safety datasets.

The script is idempotent: existing outputs are not re-downloaded unless
``--force`` is passed. Hugging Face downloads degrade gracefully: if the
``datasets`` library or network is unavailable, the step is skipped with clear
instructions rather than crashing, and the custom datasets (steps 3-4) still
run so the pipeline remains usable offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (sets sys.path)

from app.evaluation.dataset_builders import (
    build_rag_injection_dataset,
    build_safety_dataset,
    build_socratic_dataset,
)

RAW = Path("data/raw")
PROCESSED = Path("data/processed/evaluation")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def download_gsm8k(force: bool) -> bool:
    out_dir = RAW / "gsm8k"
    eval_path = PROCESSED / "gsm8k_eval.jsonl"
    if eval_path.exists() and not force:
        print("  already present, skipping (use --force to redownload)")
        return True
    try:
        from datasets import load_dataset
    except ImportError:
        print("  'datasets' not installed. Install it (pip install datasets) to fetch GSM8K.")
        return False
    try:
        ds = load_dataset("openai/gsm8k", "main")
    except Exception as exc:  # noqa: BLE001
        print(f"  download failed: {exc}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = []
    for split in ("train", "test"):
        if split not in ds:
            continue
        raw_rows = [dict(r) for r in ds[split]]
        _write_jsonl(out_dir / f"{split}.jsonl", raw_rows)
        if split == "test":
            for i, r in enumerate(raw_rows):
                answer = r["answer"].split("####")[-1].strip() if "####" in r["answer"] else r["answer"].strip()
                normalized.append({
                    "id": f"gsm8k_{i:06d}",
                    "question": r["question"],
                    "expected_answer": answer,
                    "subject": "math",
                    "source": "GSM8K",
                    "split": "evaluation" if i % 5 == 0 else "development",
                })
    _write_jsonl(eval_path, normalized)
    print(f"  wrote {len(normalized)} normalized eval records")
    return True


def download_math(force: bool) -> bool:
    out_dir = RAW / "math"
    eval_path = PROCESSED / "math_eval.jsonl"
    if eval_path.exists() and not force:
        print("  already present, skipping (use --force to redownload)")
        return True
    try:
        from datasets import load_dataset
    except ImportError:
        print("  'datasets' not installed. Install it (pip install datasets) to fetch MATH.")
        return False
    # This dataset exposes one config per category rather than an "all" config,
    # so we load each category and preserve it in the metadata.
    categories = [
        "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
        "number_theory", "prealgebra", "precalculus",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = []
    index = 0
    loaded_any = False
    for category in categories:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", category)
        except Exception as exc:  # noqa: BLE001
            print(f"  {category}: failed ({exc})")
            continue
        loaded_any = True
        split = "test" if "test" in ds else list(ds.keys())[0]
        rows = [dict(r) for r in ds[split]]
        _write_jsonl(out_dir / f"{category}.jsonl", rows)
        for r in rows:
            normalized.append({
                "id": f"math_{index:06d}",
                "question": r.get("problem", ""),
                "solution": r.get("solution", ""),
                "subject": "math",
                "category": category,
                "source": "MATH",
                "split": "evaluation" if index % 5 == 0 else "development",
            })
            index += 1
    if not loaded_any:
        print("  download failed for all categories.")
        return False
    _write_jsonl(eval_path, normalized)
    print(f"  wrote {len(normalized)} normalized eval records")
    return True


def prepare_tutoring(force: bool) -> bool:
    out = RAW / "tutoring" / "socratic_tests.json"
    if out.exists() and not force:
        print("  already present, regenerating anyway (deterministic)")
    rows = build_socratic_dataset()
    _write_json(out, rows)
    print(f"  wrote {len(rows)} Socratic pedagogy examples")

    # Khan dataset: never auto-redistribute; document how to obtain it.
    khan_dir = RAW / "tutoring" / "khan"
    if not khan_dir.exists():
        print("  Khan Academy Tutoring Accuracy Dataset is NOT auto-downloaded.")
        print("  Review its license at https://github.com/Khan/tutoring-accuracy-dataset")
        print(f"  and place a permitted local copy under {khan_dir} if you choose to use it.")
    return True


def prepare_red_team(force: bool) -> bool:
    safety = RAW / "red_team" / "safety_tests.json"
    rag_inj = RAW / "red_team" / "rag_injection_tests.json"
    rows = build_safety_dataset()
    _write_json(safety, rows)
    _write_json(rag_inj, build_rag_injection_dataset())
    print(f"  wrote {len(rows)} red-team safety examples + RAG-injection cases")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare datasets.")
    parser.add_argument("--force", action="store_true", help="Re-download benchmarks.")
    args = parser.parse_args()

    print("=" * 40)
    print("Socratic Tutor Data Acquisition")
    print("=" * 40)
    print()
    print("[1/4] Downloading GSM8K...")
    ok1 = download_gsm8k(args.force)
    print("  " + ("done" if ok1 else "skipped"))
    print()
    print("[2/4] Downloading MATH...")
    ok2 = download_math(args.force)
    print("  " + ("done" if ok2 else "skipped"))
    print()
    print("[3/4] Preparing tutoring evaluation data...")
    prepare_tutoring(args.force)
    print()
    print("[4/4] Preparing red-team evaluation data...")
    prepare_red_team(args.force)
    print()
    print("Data acquisition complete.")


if __name__ == "__main__":
    main()
