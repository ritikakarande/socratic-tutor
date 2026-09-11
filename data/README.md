# Data

This directory holds three clearly separated data categories. The downloaded
contents are never committed to git (see the repository `.gitignore`); only the
manifest, this README, and the hand-authored custom datasets are tracked.

## Categories

| Category | Purpose | Location | Committed? |
| --- | --- | --- | --- |
| Benchmark / evaluation | Measure tutor performance (correctness, tool accuracy) | `data/raw/gsm8k/`, `data/raw/math/` | No |
| RAG knowledge base | Ground responses through retrieval | `data/raw/openstax/`, `data/chroma/` | No |
| Custom safety and pedagogy | Measure Socratic behavior and robustness | `data/raw/tutoring/`, `data/raw/red_team/` | Yes (authored) |

These categories are never mixed. Benchmarks feed the evaluation pipeline, the
educational corpus feeds RAG ingestion, and the custom datasets feed the
pedagogy and guardrail evaluations.

## Directory layout

```
data/
  raw/
    gsm8k/        GSM8K benchmark (downloaded)
    math/         Hendrycks MATH benchmark (downloaded)
    tutoring/     Custom Socratic tests + any licensed tutoring data
    openstax/     OpenStax PDFs (downloaded manually where required)
    red_team/     Custom adversarial safety tests
  processed/
    evaluation/   Normalized benchmark records (JSONL)
    rag/          Intermediate ingestion artefacts
    safety/       Processed safety records
  chroma/         Persisted ChromaDB vector store
  source_manifest.json
  README.md
```

## How to obtain the data

```bash
python scripts/download_datasets.py    # benchmarks + custom datasets
python scripts/download_openstax.py     # OpenStax sources (with license checks)
python scripts/ingest_openstax.py       # build the vector database
```

## Dataset details

### GSM8K
- Source: `openai/gsm8k` on Hugging Face (config `main`).
- License: MIT. Used for evaluation only. Original benchmark data is never modified;
  a separate normalized copy is written to `data/processed/evaluation/gsm8k_eval.jsonl`.

### Hendrycks MATH
- Source: `EleutherAI/hendrycks_math` on Hugging Face.
- License: MIT. Categories (algebra, counting_and_probability, geometry, intermediate_algebra,
  number_theory, prealgebra, precalculus) are preserved. Normalized copy in
  `data/processed/evaluation/math_eval.jsonl`.

### Khan Academy Tutoring Accuracy Dataset
- Source: https://github.com/Khan/tutoring-accuracy-dataset
- Redistribution is NOT assumed. `scripts/download_datasets.py` inspects for a local copy
  and, if absent, prints instructions to obtain it and where to place it
  (`data/raw/tutoring/khan/`). Review the repository LICENSE before use. This project does
  not bypass any licensing restriction.

### OpenStax textbooks (RAG corpus) - ingested

Five books, reviewed and marked `permitted` in the manifest on 2026-09-11:

| Book | Subject | Pages | PDF size |
| --- | --- | --- | --- |
| College Algebra 2e | math | 896 | 55 MB |
| Algebra 1 | math | 1535 | 79 MB |
| University Physics Volume 1 | physics | 959 | 78 MB |
| Chemistry 2e | chemistry | 1203 | 208 MB |
| Biology 2e | biology | 1475 | 383 MB |
| **Total** | | **6068** | **803 MB** |

Ingested into ChromaDB as **16,837 chunks** (chunk size 1000, overlap 150),
each carrying source, subject, chapter, section and page metadata. The
persisted store lives in `data/chroma` (about 169 MB) and is not committed.
Attribution for the CC BY source is preserved in every chunk's metadata and
surfaced as citations in the UI.

### OpenStax licensing policy
- A Creative Commons license does NOT automatically permit ingestion into a generative AI
  system. For every book, verify the current official license and any AI/LLM restriction and
  record the result in `source_manifest.json` (`ai_use_status`). Only ingest a book whose
  current terms permit the intended use, or for which you have obtained permission. A book
  marked `restricted` is skipped by the ingestion script.

### Custom datasets (authored, tracked)
- `raw/tutoring/socratic_tests.json` (>= 50 examples): pedagogy behavior across math,
  physics, chemistry and biology, including misconceptions and hint/direct-answer requests.
- `raw/red_team/safety_tests.json` (>= 50 examples): adversarial prompts across
  instruction_override, system_prompt_extraction, role_manipulation, jailbreak_attempt,
  tool_manipulation, rag_injection, secret_extraction, off_topic_request, unsafe_request and
  policy_confusion.
- `raw/red_team/rag_injection_tests.json`: benign questions paired with poisoned documents.

Regenerate the custom datasets deterministically at any time:

```bash
python scripts/download_datasets.py   # steps 3 and 4 rewrite the custom JSON
```

## Data splitting

Prompts are tuned only against the `development` split. The `evaluation` split is held out
and is what `scripts/evaluate.py` reports on. The red-team set is never used to tune prompts.

## Versioning

When refreshing a downloaded dataset, record the name, version/config, source URL, download
date, license and example count in `source_manifest.json`. Never silently modify benchmark
data; write a new processed representation instead.
