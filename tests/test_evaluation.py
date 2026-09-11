"""Tests for evaluation metrics, evaluators and dataset builders."""

from __future__ import annotations

from app.evaluation import metrics
from app.evaluation.dataset_builders import (
    build_rag_injection_dataset,
    build_safety_dataset,
    build_socratic_dataset,
)
from app.evaluation.evaluators import BenchmarkEvaluator


def test_correctness_from_tool_result():
    tools = [{"success": True, "result": "x = 5"}]
    assert metrics.correctness("Let's check your work.", "x = 5", tools) == 1.0


def test_correctness_from_text():
    assert metrics.correctness("The value is 42.", "42", []) == 1.0
    assert metrics.correctness("The value is 7.", "42", []) == 0.0


def test_relevance():
    high = metrics.relevance("force equals mass times acceleration", "what is force and mass")
    low = metrics.relevance("bananas are yellow", "what is force and mass")
    assert high > low


def test_socratic_score_rewards_question():
    with_q = metrics.socratic_score("What operation isolates x?", answer_reveal_allowed=False)
    without_q = metrics.socratic_score("Add five to both sides.", answer_reveal_allowed=False)
    assert with_q > without_q


def test_socratic_penalizes_premature_answer():
    score = metrics.socratic_score("The final answer is x = 5.", answer_reveal_allowed=False)
    assert score < 0.6


def test_safety_score_detects_leak():
    assert metrics.safety_score("my system prompt is ...", "refuse") == 0.0
    assert metrics.safety_score("I'm your math tutor, I can only help with that.", "refuse") == 1.0


def test_faithfulness_vacuous_without_context():
    assert metrics.faithfulness("anything", []) == 1.0


def test_datasets_meet_minimum_sizes():
    assert len(build_socratic_dataset()) >= 50
    assert len(build_safety_dataset()) >= 50
    assert len(build_rag_injection_dataset()) >= 1


def test_safety_dataset_covers_required_categories():
    categories = {row["category"] for row in build_safety_dataset()}
    required = {
        "instruction_override", "system_prompt_extraction", "role_manipulation",
        "jailbreak_attempt", "tool_manipulation", "rag_injection",
        "secret_extraction", "off_topic_request", "unsafe_request", "policy_confusion",
    }
    assert required.issubset(categories)


def test_benchmark_evaluator_aggregates():
    def fake_runner(question, ex):
        return {
            "response": "What do you notice first about this problem?",
            "tool_results": [{"success": True, "result": "x = 5"}],
            "retrieved_context": [],
            "answer_reveal_allowed": False,
            "blocked": False,
        }

    ev = BenchmarkEvaluator(fake_runner)
    report = ev.evaluate_dataset([
        {"id": "1", "question": "Solve 2x+5=15", "subject": "math", "expected_answer": "x = 5"},
    ])
    assert report.total == 1
    assert "socratic" in report.aggregates
    assert "pass_rate" in report.aggregates
