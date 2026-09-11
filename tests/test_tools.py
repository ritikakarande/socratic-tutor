"""Tests for the SymPy calculator tool, including safety."""

from __future__ import annotations

import pytest

from app.tools.calculator import calculate


def test_solve_linear_equation():
    result = calculate("solve", "2*x + 5 = 15")
    assert result.success
    assert result.result == "x = 5"
    assert "satisfy" in (result.verification or "")


def test_solve_quadratic():
    result = calculate("solve", "x**2 - 4 = 0")
    assert result.success
    assert "-2" in result.result and "2" in result.result


def test_differentiate():
    result = calculate("differentiate", "x**2")
    assert result.success
    assert result.result == "2*x"


def test_integrate():
    result = calculate("integrate", "x**2")
    assert result.success
    assert "x**3/3" in result.result
    assert result.result.endswith("+ C")


def test_simplify():
    result = calculate("simplify", "(x**2 - 1)/(x - 1)")
    assert result.success
    assert result.result == "x + 1"


def test_arithmetic():
    result = calculate("arithmetic", "2 + 2")
    assert result.success
    assert result.result.startswith("4")


def test_unknown_operation():
    result = calculate("hack", "2+2")
    assert not result.success
    assert "Unknown operation" in result.error


@pytest.mark.parametrize(
    "payload",
    [
        "__import__('os').system('ls')",
        "eval('2+2')",
        "exec('x=1')",
        "open('/etc/passwd')",
        "os.getenv('OPENAI_API_KEY')",
        "lambda: 1",
    ],
)
def test_arbitrary_code_is_blocked(payload):
    """The tool must never execute arbitrary Python."""
    result = calculate("solve", payload)
    assert not result.success
    assert result.error


def test_never_uses_eval_on_input():
    # A payload that would only "succeed" if eval ran should be rejected.
    result = calculate("arithmetic", "__import__('subprocess')")
    assert not result.success


def test_empty_and_oversized():
    assert not calculate("solve", "").success
    assert not calculate("solve", "1" * 1000).success
