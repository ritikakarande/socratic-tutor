"""Intent router.

Classifies subject, difficulty and intent, and decides whether calculation or
retrieval is needed. A fast heuristic classifier runs by default so routing is
deterministic and offline. When a live client is available and configured, an
LLM classifier can refine the result, but the heuristic remains the floor.

The router deliberately does NOT retrieve for everything: pure arithmetic and
simple conceptual questions skip RAG, matching the RAG-quality requirement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.agents.llm import LLMClient, OpenAIChatClient
from app.agents.prompts import ROUTER_SYSTEM
from app.utils.helpers import normalize_math_words
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RouteDecision:
    subject: str
    difficulty: str
    intent: str
    needs_calculation: bool
    needs_retrieval: bool
    final_answer_requested: bool


_SUBJECT_KEYWORDS = {
    "physics": r"(newton|force|velocity|acceleration|kinematic|momentum|energy|"
    r"gravity|friction|voltage|current|circuit|ohm|magnetic|wave|optics|thermodynamic)",
    "chemistry": r"(atom|molecule|mole|stoichiometry|reaction|acid|base|ph\b|"
    r"periodic|electron|bond|compound|reagent|oxidation|valence|molar)",
    "biology": r"(cell|dna|rna|gene|protein|enzyme|photosynthesis|mitosis|"
    r"organism|evolution|ecosystem|chromosome|membrane)",
    "math": r"(solve|equation|derivative|integral|factor|simplify|algebra|"
    r"geometry|calculus|probability|fraction|percent|polynomial|matrix|theorem|=)",
}

_MATH_EXPR = re.compile(
    r"[0-9]\s*[\+\-\*/\^=]|[\+\-\*/\^=]\s*[0-9x]|"
    r"\bderivative\b|\bdifferentiate\b|\bintegral\b|\bintegrate\b|"
    r"\bsolve\b|\bsimplify\b|\bfactor\b|\bevaluate\b"
)

_KNOWLEDGE = re.compile(r"(what\s+is|define|explain|describe|state\s+the|why\s+does|how\s+does)", re.I)
_RAG_EXPLICIT = re.compile(r"(textbook|from\s+my\s+book|reference|source|chapter|openstax)", re.I)
_PURE_ARITHMETIC = re.compile(r"^\s*[-+*/().\d\s^]+\s*=?\s*\??\s*$")
_ANSWER_REQUEST = re.compile(
    r"(just\s+(give|tell)\s+me\s+the\s+answer|what'?s\s+the\s+answer|"
    r"final\s+answer|show\s+me\s+the\s+solution|solve\s+it\s+for\s+me)",
    re.I,
)

_ELEMENTARY = re.compile(r"(mystery\s+box|simple\s+terms|i'?m\s+in\s+(grade|elementary))", re.I)
_ADVANCED = re.compile(r"(prove|rigorous|manifold|eigen|hamiltonian|quantum|tensor|lagrangian)", re.I)
_COLLEGE = re.compile(r"(integral|derivative|calculus|stoichiometry|vector|matrix)", re.I)


def heuristic_route(question: str) -> RouteDecision:
    """Rule-based routing that always produces a decision."""
    q = question or ""
    # Normalize arithmetic words so "6 times 7" is recognised as a calculation.
    lower = normalize_math_words(q).lower()

    # Subject
    subject = "unknown"
    best = 0
    for name, pattern in _SUBJECT_KEYWORDS.items():
        hits = len(re.findall(pattern, lower))
        if hits > best:
            best = hits
            subject = name
    if subject == "unknown" and _MATH_EXPR.search(lower):
        subject = "math"

    # Difficulty
    if _ADVANCED.search(lower):
        difficulty = "advanced"
    elif _COLLEGE.search(lower):
        difficulty = "college"
    elif _ELEMENTARY.search(lower):
        difficulty = "elementary"
    else:
        difficulty = "high_school"

    # Calculation vs knowledge
    needs_calculation = bool(_MATH_EXPR.search(lower)) and subject in {"math", "physics", "chemistry", "unknown"}
    is_knowledge = bool(_KNOWLEDGE.search(lower))

    # Retrieval: explicit request, or a knowledge question in a science subject.
    needs_retrieval = bool(_RAG_EXPLICIT.search(lower)) or (
        is_knowledge and subject in {"physics", "chemistry", "biology", "science"}
    )
    # Pure arithmetic never needs retrieval.
    if _PURE_ARITHMETIC.match(q.strip()):
        needs_retrieval = False
        needs_calculation = True
        subject = "math"

    if needs_calculation and is_knowledge:
        intent = "mixed"
    elif needs_calculation:
        intent = "math_calculation"
    elif needs_retrieval or is_knowledge:
        intent = "knowledge"
    else:
        intent = "conceptual"

    return RouteDecision(
        subject=subject,
        difficulty=difficulty,
        intent=intent,
        needs_calculation=needs_calculation,
        needs_retrieval=needs_retrieval,
        final_answer_requested=bool(_ANSWER_REQUEST.search(lower)),
    )


class IntentRouter:
    """Heuristic router with an optional LLM refinement layer."""

    def __init__(self, llm: Optional[LLMClient] = None, use_llm: bool = False) -> None:
        self._llm = llm
        self._use_llm = use_llm and isinstance(llm, OpenAIChatClient)

    def route(self, question: str) -> RouteDecision:
        decision = heuristic_route(question)
        if not self._use_llm:
            return decision
        try:
            data = self._llm.complete_json(ROUTER_SYSTEM, question)  # type: ignore[union-attr]
            if data:
                decision = RouteDecision(
                    subject=data.get("subject", decision.subject) or decision.subject,
                    difficulty=data.get("difficulty", decision.difficulty) or decision.difficulty,
                    intent=data.get("intent", decision.intent) or decision.intent,
                    needs_calculation=bool(data.get("needs_calculation", decision.needs_calculation)),
                    needs_retrieval=bool(data.get("needs_retrieval", decision.needs_retrieval)),
                    final_answer_requested=decision.final_answer_requested,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM router failed, using heuristic: %s", exc)
        return decision
