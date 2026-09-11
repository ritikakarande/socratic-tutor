"""Prompt-injection detection with layered defenses.

Layer 1 - Heuristics: fast regex patterns for well-known attack phrasing.
Layer 2 - Structured classification: an optional LLM classifier returning a
          small controlled schema. Disabled by default so the system runs and
          tests offline; enabled with ``INJECTION_LLM_CLASSIFIER=true``.
Layer 3 - Instruction hierarchy: enforced elsewhere by how prompts are built
          (system > developer > user > retrieved documents). Retrieved
          documents are never treated as instructions.

The detector reports a structured result; it does not itself act. The input
guard decides what to do with the result under the active guardrail policy.
"""

from __future__ import annotations

import re
from typing import List, Optional, Protocol

from pydantic import BaseModel, Field

from app.guardrails.policies import RiskLevel
from app.utils.logging import get_logger

logger = get_logger(__name__)


class InjectionSignal(BaseModel):
    """Structured output of the injection detector."""

    is_injection: bool = False
    risk: RiskLevel = RiskLevel.NONE
    categories: List[str] = Field(default_factory=list)
    reason: str = ""
    matched_patterns: List[str] = Field(default_factory=list)


# Each pattern maps to (category, risk). Patterns are intentionally readable.
_HEURISTICS: list[tuple[re.Pattern, str, RiskLevel]] = [
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)", re.I),
     "instruction_override", RiskLevel.HIGH),
    (re.compile(r"disregard\s+(?:all\s+|your\s+|prior\s+|previous\s+|the\s+)*(instructions|rules|guidelines|prompts?)", re.I),
     "instruction_override", RiskLevel.HIGH),
    (re.compile(r"(reveal|show|print|repeat|display|output|leak|give\s+me)\s+(me\s+)?(your\s+)?(the\s+)?(?:(system|hidden|initial|developer|original)\s+){1,3}(prompt|instructions?|message|configuration)", re.I),
     "system_prompt_extraction", RiskLevel.HIGH),
    (re.compile(r"(message|instructions?|prompt)\s+that\s+(configured|set\s+up|initialized)\s+you", re.I),
     "system_prompt_extraction", RiskLevel.HIGH),
    (re.compile(r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions|rules)", re.I),
     "system_prompt_extraction", RiskLevel.MEDIUM),
    (re.compile(r"repeat\s+everything\s+(above|before)", re.I),
     "system_prompt_extraction", RiskLevel.HIGH),
    (re.compile(r"you\s+are\s+(no\s+longer|not)\s+a\s+tutor", re.I),
     "role_manipulation", RiskLevel.HIGH),
    (re.compile(r"you\s+are\s+now\s+['\"\w].*(no\s+rules|unrestricted|no\s+restrictions|god)", re.I),
     "role_manipulation", RiskLevel.HIGH),
    (re.compile(r"(confirm\s+and\s+proceed|acknowledge\s+and\s+comply)", re.I),
     "role_manipulation", RiskLevel.MEDIUM),
    (re.compile(r"(pretend|act\s+as\s+if|imagine|suppose)\s+.{0,30}(safety|rules|guidelines|restrictions|policies)\s+.{0,15}(don'?t|do\s+not|no\s+longer)\s+(exist|apply)", re.I),
     "jailbreak_attempt", RiskLevel.HIGH),
    (re.compile(r"\b(dan\s+mode|jailbreak|developer\s+mode|do\s+anything\s+now)\b", re.I),
     "jailbreak_attempt", RiskLevel.HIGH),
    (re.compile(r"(the\s+)?developer\s+(says|told|instructs|wants)\s+you\s+to", re.I),
     "role_manipulation", RiskLevel.MEDIUM),
    (re.compile(r"(disable|turn\s+off|bypass|override|ignore)\s+(your\s+)?(safety|guardrails|filters|protections)", re.I),
     "jailbreak_attempt", RiskLevel.HIGH),
    (re.compile(r"(print|show|return|reveal|give\s+me|list|expose|dump)\s+.{0,25}(secret|environment|env|api|access)\s+\w*\s*(variables|keys?|tokens?|vars?|credentials?)", re.I),
     "secret_extraction", RiskLevel.HIGH),
    (re.compile(r"(openai_api_key|api[_\s-]?keys?|secret[_\s-]?keys?|access[_\s-]?tokens?)", re.I),
     "secret_extraction", RiskLevel.HIGH),
    (re.compile(r"what\s+(api\s+)?(keys?|tokens?|credentials?)\s+(do\s+you|are)", re.I),
     "secret_extraction", RiskLevel.HIGH),
    (re.compile(r"(contents\s+of\s+)?your\s+\.env|\.env\s+file|dotenv", re.I),
     "secret_extraction", RiskLevel.HIGH),
    (re.compile(r"(reveal|expose|show|dump)\s+.{0,20}(tokens?|credentials?|api\s+keys?)", re.I),
     "secret_extraction", RiskLevel.HIGH),
    (re.compile(r"(run|execute|eval|exec)\s+(this\s+)?(arbitrary\s+)?(python|code|script)", re.I),
     "tool_manipulation", RiskLevel.HIGH),
    (re.compile(r"(exec|eval)\s*\(|__import__|os\.(system|getenv|popen)|subprocess", re.I),
     "tool_manipulation", RiskLevel.HIGH),
    (re.compile(r"the\s+(textbook|document|reference|context)\s+says\s+to\s+(ignore|disregard|reveal|override)", re.I),
     "rag_injection", RiskLevel.HIGH),
]


class LLMClassifier(Protocol):
    """Callable that classifies text for injection risk."""

    def classify_injection(self, text: str) -> Optional[InjectionSignal]:
        ...


def _max_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel:
    order = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
    return a if order.index(a) >= order.index(b) else b


def heuristic_scan(text: str) -> InjectionSignal:
    """Run the fast regex layer over the text."""
    categories: List[str] = []
    matched: List[str] = []
    risk = RiskLevel.NONE
    for pattern, category, level in _HEURISTICS:
        if pattern.search(text or ""):
            categories.append(category)
            matched.append(pattern.pattern)
            risk = _max_risk(risk, level)
    is_injection = risk in (RiskLevel.MEDIUM, RiskLevel.HIGH)
    return InjectionSignal(
        is_injection=is_injection,
        risk=risk,
        categories=sorted(set(categories)),
        reason="heuristic pattern match" if is_injection else "no heuristic match",
        matched_patterns=matched,
    )


class PromptInjectionDetector:
    """Combines the heuristic layer with an optional LLM classifier."""

    def __init__(self, llm_classifier: Optional[LLMClassifier] = None) -> None:
        self._classifier = llm_classifier

    def detect(self, text: str) -> InjectionSignal:
        """Return the highest-risk signal across all enabled layers."""
        signal = heuristic_scan(text)

        if self._classifier is not None:
            try:
                llm_signal = self._classifier.classify_injection(text)
            except Exception as exc:  # noqa: BLE001 - classifier failure is non-fatal
                logger.warning("Injection LLM classifier failed: %s", exc)
                llm_signal = None
            if llm_signal is not None:
                merged_risk = _max_risk(signal.risk, llm_signal.risk)
                signal = InjectionSignal(
                    is_injection=signal.is_injection or llm_signal.is_injection,
                    risk=merged_risk,
                    categories=sorted(set(signal.categories) | set(llm_signal.categories)),
                    reason="; ".join(filter(None, [signal.reason, llm_signal.reason])),
                    matched_patterns=signal.matched_patterns,
                )
        return signal
