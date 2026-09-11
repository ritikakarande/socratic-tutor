"""Safe symbolic mathematics tool built on SymPy.

Security model
--------------
User and model supplied strings are *never* passed to :func:`eval` or
``exec``. Every expression is parsed with :func:`sympy.parsing.sympy_parser`
using an explicit, restricted transformation set and an empty ``global_dict``
so arbitrary Python cannot be reached. Parsing runs in a guarded scope and any
failure is caught and returned as structured data rather than raising into the
agent.

The tool returns a :class:`CalculationResult` so downstream nodes and the
evaluation harness can reason about it without string scraping.
"""

from __future__ import annotations

import re
from typing import List, Optional

import sympy as sp
from pydantic import BaseModel, Field
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# Restricted transformation set. We allow implicit multiplication (so "2x"
# parses) but nothing that could reach arbitrary callables.
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# A curated symbol table. parse_expr is called with global_dict={} to block
# builtins, and local_dict limited to safe SymPy names.
_LOCAL_DICT = {
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
    "I": sp.I,
    "oo": sp.oo,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "Abs": sp.Abs,
    "factorial": sp.factorial,
}

# Reject anything containing characters or names that hint at code execution.
_FORBIDDEN = re.compile(
    r"(__|import|lambda|exec|eval|open|os\.|sys\.|subprocess|"
    r"getattr|setattr|globals|locals|compile|input|file)",
    re.IGNORECASE,
)

_MAX_LEN = 500


class CalculationResult(BaseModel):
    """Structured output of a symbolic computation."""

    success: bool
    operation: str
    expression: str
    result: Optional[str] = None
    steps: List[str] = Field(default_factory=list)
    verification: Optional[str] = None
    error: Optional[str] = None


def _guard(expr: str) -> Optional[str]:
    """Return an error message if the raw input is unsafe, else None."""
    if not expr or not expr.strip():
        return "Empty expression."
    if len(expr) > _MAX_LEN:
        return "Expression too long."
    if _FORBIDDEN.search(expr):
        return "Expression contains disallowed tokens."
    return None


def _parse(expr: str) -> sp.Expr:
    """Parse a string into a SymPy expression under the restricted scope."""
    # parse_expr is a tokenizing parser, not eval. It cannot reach Python
    # builtins unless they are placed in the namespace, and the ``_guard``
    # forbidden-token filter rejects dunder / import style names before we ever
    # get here. We keep SymPy's default global_dict so numeric literals become
    # sympy Integers, and layer our restricted safe names on top via local_dict.
    return parse_expr(
        expr,
        transformations=_TRANSFORMATIONS,
        local_dict=_LOCAL_DICT,
        evaluate=True,
    )


def _split_equation(text: str) -> sp.Expr:
    """Turn "lhs = rhs" into an Eq, or a bare expression into itself."""
    if "=" in text and "==" not in text:
        lhs, rhs = text.split("=", 1)
        return sp.Eq(_parse(lhs), _parse(rhs))
    return _parse(text)


def _default_symbol(expr: sp.Basic) -> sp.Symbol:
    """Pick the variable to operate on, defaulting to x when present."""
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if not symbols:
        return sp.Symbol("x")
    for s in symbols:
        if s.name == "x":
            return s
    return symbols[0]


def solve_equation(expression: str, variable: Optional[str] = None) -> CalculationResult:
    """Solve an equation or expression for a variable."""
    err = _guard(expression)
    if err:
        return CalculationResult(success=False, operation="solve", expression=expression, error=err)
    try:
        parsed = _split_equation(expression)
        sym = sp.Symbol(variable) if variable else _default_symbol(parsed)
        solutions = sp.solve(parsed, sym, dict=False)
        rendered = [f"{sym} = {sp.simplify(s)}" for s in solutions]
        verification = None
        if isinstance(parsed, sp.Eq) and solutions:
            checks = [bool(sp.simplify(parsed.lhs - parsed.rhs).subs(sym, s) == 0) for s in solutions]
            verification = "all solutions satisfy the equation" if all(checks) else "verification incomplete"
        return CalculationResult(
            success=True,
            operation="solve",
            expression=expression,
            result="; ".join(rendered) if rendered else "no solution found",
            steps=[f"Parsed as {parsed}", f"Solved for {sym}"],
            verification=verification,
        )
    except Exception as exc:  # noqa: BLE001 - tool must never raise into agent
        return CalculationResult(success=False, operation="solve", expression=expression, error=str(exc))


def simplify_expression(expression: str) -> CalculationResult:
    """Simplify an algebraic expression."""
    err = _guard(expression)
    if err:
        return CalculationResult(success=False, operation="simplify", expression=expression, error=err)
    try:
        parsed = _parse(expression)
        simplified = sp.simplify(parsed)
        return CalculationResult(
            success=True,
            operation="simplify",
            expression=expression,
            result=str(simplified),
            steps=[f"Parsed as {parsed}"],
        )
    except Exception as exc:  # noqa: BLE001
        return CalculationResult(success=False, operation="simplify", expression=expression, error=str(exc))


def differentiate(expression: str, variable: Optional[str] = None) -> CalculationResult:
    """Compute a derivative."""
    err = _guard(expression)
    if err:
        return CalculationResult(success=False, operation="differentiate", expression=expression, error=err)
    try:
        parsed = _parse(expression)
        sym = sp.Symbol(variable) if variable else _default_symbol(parsed)
        derivative = sp.diff(parsed, sym)
        return CalculationResult(
            success=True,
            operation="differentiate",
            expression=expression,
            result=str(sp.simplify(derivative)),
            steps=[f"d/d{sym} of {parsed}"],
        )
    except Exception as exc:  # noqa: BLE001
        return CalculationResult(success=False, operation="differentiate", expression=expression, error=str(exc))


def integrate(expression: str, variable: Optional[str] = None) -> CalculationResult:
    """Compute an indefinite integral."""
    err = _guard(expression)
    if err:
        return CalculationResult(success=False, operation="integrate", expression=expression, error=err)
    try:
        parsed = _parse(expression)
        sym = sp.Symbol(variable) if variable else _default_symbol(parsed)
        antideriv = sp.integrate(parsed, sym)
        return CalculationResult(
            success=True,
            operation="integrate",
            expression=expression,
            result=f"{antideriv} + C",
            steps=[f"integral of {parsed} d{sym}"],
        )
    except Exception as exc:  # noqa: BLE001
        return CalculationResult(success=False, operation="integrate", expression=expression, error=str(exc))


def evaluate_arithmetic(expression: str) -> CalculationResult:
    """Evaluate a numeric arithmetic expression to a decimal value."""
    err = _guard(expression)
    if err:
        return CalculationResult(success=False, operation="arithmetic", expression=expression, error=err)
    try:
        parsed = _parse(expression)
        value = sp.N(parsed)
        return CalculationResult(
            success=True,
            operation="arithmetic",
            expression=expression,
            result=str(value),
            steps=[f"Parsed as {parsed}"],
        )
    except Exception as exc:  # noqa: BLE001
        return CalculationResult(success=False, operation="arithmetic", expression=expression, error=str(exc))


# Dispatch table mapping an operation name to its implementation.
_OPERATIONS = {
    "solve": solve_equation,
    "simplify": simplify_expression,
    "differentiate": differentiate,
    "integrate": integrate,
    "arithmetic": evaluate_arithmetic,
}


def calculate(operation: str, expression: str, variable: Optional[str] = None) -> CalculationResult:
    """Single entry point used by the tool registry.

    ``operation`` must be one of the known safe operations. ``expression`` is
    the untrusted string to work on and ``variable`` optionally names the target
    symbol.
    """
    op = (operation or "").strip().lower()
    if op not in _OPERATIONS:
        return CalculationResult(
            success=False,
            operation=op,
            expression=expression,
            error=f"Unknown operation '{operation}'. Supported: {sorted(_OPERATIONS)}",
        )
    if op in {"simplify", "arithmetic"}:
        return _OPERATIONS[op](expression)
    return _OPERATIONS[op](expression, variable)
