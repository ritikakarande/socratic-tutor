"""Output guardrail.

Validates a generated response before it reaches the student. Checks for:

* system / developer prompt leakage
* echoed hidden instructions
* secret or environment-variable disclosure
* direct final-answer leakage when the pedagogy policy forbids it
* a required guiding question when the policy requires one

Returns a structured result. The caller (the graph) decides whether to accept,
trigger a bounded regeneration, or fall back to a controlled safe message.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field

from app.guardrails.policies import TutorPolicy
from app.utils.helpers import clamp


class OutputInspection(BaseModel):
    """Structured output-guard result."""

    approved: bool
    violations: List[str] = Field(default_factory=list)
    pedagogy_score: float = 1.0
    reason: str = ""


_PROMPT_LEAK = re.compile(
    r"(my\s+system\s+prompt\s+is|here\s+(is|are)\s+my\s+(system\s+)?instructions|"
    r"i\s+was\s+instructed\s+to|the\s+developer\s+told\s+me\s+to|"
    r"as\s+an\s+ai\s+language\s+model,?\s+my\s+instructions)",
    re.IGNORECASE,
)

_SECRET_LEAK = re.compile(
    r"(sk-[a-zA-Z0-9]{12,}|api[_\s-]?key\s*[:=]\s*\S+|OPENAI_API_KEY\s*[:=]|"
    r"environment\s+variable[s]?\s+are\s*[:=])",
    re.IGNORECASE,
)

_QUESTION = re.compile(r"\?")

# Heuristic markers that the model is stating a final numeric/closed answer.
_FINAL_ANSWER = re.compile(
    r"(the\s+(final\s+)?answer\s+is|therefore,?\s+x\s*=|so\s+x\s*=|"
    r"the\s+solution\s+is\s+x?\s*=)",
    re.IGNORECASE,
)


class OutputGuard:
    """Inspects generated text against safety and pedagogy rules."""

    def __init__(self, tutor_policy: TutorPolicy) -> None:
        self._policy = tutor_policy

    def inspect(
        self,
        response: str,
        answer_reveal_allowed: bool,
        require_question: Optional[bool] = None,
    ) -> OutputInspection:
        """Inspect a response.

        ``answer_reveal_allowed`` comes from the pedagogy policy applied to the
        current conversation state. ``require_question`` overrides the policy's
        default guiding-question requirement when provided.
        """
        response = response or ""
        violations: List[str] = []
        score = 1.0

        # ---- Safety checks (always enforced) ----
        if _PROMPT_LEAK.search(response):
            violations.append("system_prompt_leak")
        if _SECRET_LEAK.search(response):
            violations.append("secret_leak")

        # ---- Pedagogy checks ----
        needs_question = self._policy.require_guiding_question if require_question is None else require_question
        has_question = bool(_QUESTION.search(response))
        if needs_question and not has_question:
            violations.append("missing_guiding_question")
            score -= 0.4

        if not answer_reveal_allowed and _FINAL_ANSWER.search(response):
            violations.append("premature_answer_disclosure")
            score -= 0.5

        # Reward encouraging, question-led phrasing lightly.
        if has_question:
            score += 0.1

        safety_violations = {"system_prompt_leak", "secret_leak"}
        pedagogy_violations = set(violations) - safety_violations

        # Safety violations are never approved. Pedagogy violations block only
        # when they concern the answer-disclosure or required-question rules.
        approved = not (set(violations) & safety_violations)
        if pedagogy_violations & {"missing_guiding_question", "premature_answer_disclosure"}:
            approved = False

        return OutputInspection(
            approved=approved,
            violations=violations,
            pedagogy_score=round(clamp(score), 3),
            reason="approved" if approved else f"violations: {', '.join(violations)}",
        )
