"""Run the benchmark evaluation and produce a report.

    python scripts/evaluate.py [--limit N] [--out eval_report]

Evaluates three suites through the tutor:
  * Custom Socratic pedagogy (correctness, relevance, faithfulness, socratic).
  * Red-team safety (safety score, resistance to injection/extraction).
  * A held-out slice of GSM8K if it has been downloaded.

Writes both a JSON and a Markdown report and prints a summary table plus failed
cases. Runs fully offline with the deterministic tutor client; supply an
OPENAI_API_KEY to evaluate the live model instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401

from app.agents.graph import build_agent
from app.agents.state import TutorState
from app.evaluation import datasets
from app.evaluation.evaluators import BenchmarkEvaluator, BenchmarkReport


def _make_runner(agent):
    def runner(question: str, example: Dict) -> TutorState:
        # RAG-injection cases feed the poisoned document as retrieved context to
        # prove the agent treats it as data, not instructions.
        history = None
        return agent.run(
            question=question,
            final_answer_requested="direct" in str(example.get("expected_behavior", "")),
            history=history,
        )

    return runner


def _print_report(title: str, report: BenchmarkReport) -> None:
    print(f"\n=== {title} ===")
    print(f"total: {report.total}")
    for key, val in report.aggregates.items():
        print(f"  {key}: {val}")
    failed = report.failed_cases()
    if failed:
        print(f"  failed cases: {len(failed)}")
        for case in failed[:10]:
            print(f"    - [{case.id}] {case.detail}: {case.question[:60]}")


def _report_to_dict(report: BenchmarkReport) -> Dict:
    return {
        "total": report.total,
        "aggregates": report.aggregates,
        "failed": [
            {"id": c.id, "category": c.category, "detail": c.detail, "question": c.question}
            for c in report.failed_cases()
        ],
        "results": [
            {
                "id": c.id,
                "category": c.category,
                "passed": c.passed,
                "scores": c.scores,
                "latency_ms": c.latency_ms,
            }
            for c in report.results
        ],
    }


def _write_markdown(path: Path, reports: Dict[str, BenchmarkReport]) -> None:
    lines = ["# Evaluation Report", ""]
    for name, report in reports.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- total: {report.total}")
        for key, val in report.aggregates.items():
            lines.append(f"- {key}: {val}")
        failed = report.failed_cases()
        lines.append(f"- failed: {len(failed)}")
        if failed:
            lines.append("")
            lines.append("| id | detail | question |")
            lines.append("| --- | --- | --- |")
            for c in failed[:20]:
                q = c.question.replace("|", "/")[:70]
                lines.append(f"| {c.id} | {c.detail} | {q} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per suite.")
    parser.add_argument("--out", type=str, default="eval_report", help="Output file stem.")
    parser.add_argument("--eval-split-only", action="store_true", help="Only use held-out eval split.")
    args = parser.parse_args()

    agent = build_agent()
    runner = _make_runner(agent)
    evaluator = BenchmarkEvaluator(runner)

    def maybe_limit(rows: List[Dict]) -> List[Dict]:
        return rows[: args.limit] if args.limit else rows

    def eval_split(rows: List[Dict]) -> List[Dict]:
        if args.eval_split_only:
            return [r for r in rows if r.get("split", "evaluation") == "evaluation"]
        return rows

    socratic = maybe_limit(eval_split(datasets.load_socratic_tests()))
    safety = maybe_limit(datasets.load_safety_tests())
    gsm8k = maybe_limit(datasets.load_gsm8k(limit=args.limit or 20))

    reports: Dict[str, BenchmarkReport] = {}
    if socratic:
        reports["Socratic pedagogy"] = evaluator.evaluate_dataset(socratic)
    if safety:
        reports["Red-team safety"] = evaluator.evaluate_dataset(safety)
    if gsm8k:
        reports["GSM8K"] = evaluator.evaluate_dataset(gsm8k)

    if not reports:
        print("No datasets found. Run: python scripts/download_datasets.py")
        return

    for title, report in reports.items():
        _print_report(title, report)

    json_path = Path(f"{args.out}.json")
    md_path = Path(f"{args.out}.md")
    json_path.write_text(
        json.dumps({k: _report_to_dict(v) for k, v in reports.items()}, indent=2),
        encoding="utf-8",
    )
    _write_markdown(md_path, reports)
    print(f"\nReports written to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
