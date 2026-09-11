"""Input guardrail.

Inspects the raw student input before any generation happens. It combines the
prompt-injection detector with lightweight topical and safety heuristics and
then applies the active guardrail policy to decide whether to allow, redirect
or refuse.

Crucially this classifies adversarial input rather than obeying it. An input
saying "ignore your instructions" is labelled as an injection attempt, not
followed.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.guardrails.policies import GuardrailPolicy, RiskLevel
from app.guardrails.prompt_injection import InjectionSignal, PromptInjectionDetector


class InputDecision(str, Enum):
    ALLOW = "allow"
    REDIRECT = "redirect"  # off-topic: steer back to tutoring
    REFUSE = "refuse"  # unsafe or blocked by policy


class InputInspection(BaseModel):
    """Structured result of inspecting student input."""

    decision: InputDecision
    risk: RiskLevel
    flags: List[str] = Field(default_factory=list)
    injection: InjectionSignal
    reason: str = ""
    safe_message: Optional[str] = None  # message to show when not allowed


# Genuinely harmful request signals. These stay active in every policy.
_HARMFUL = re.compile(
    r"\b(build|make|synthesize|create|construct|assemble)\s+(a\s+|an\s+)?(bomb|explosive|weapon|bioweapon|nerve\s+agent|explosive\s+device)\b"
    r"|\bhow\s+(to|can\s+i|do\s+i)\s+.{0,20}\b(kill|poison|harm|hurt)\s+(someone|somebody|people|a\s+person)\b"
    r"|\b(synthesi[sz]e|manufacture|produce)\s+(a\s+)?(methamphetamine|fentanyl|ricin|sarin|nerve\s+agent)\b",
    re.IGNORECASE,
)

# Topical hint: does this look like a tutoring / academic question at all?
_ON_TOPIC = re.compile(
    r"(solve|calculate|equation|derivative|integral|factor|simplify|prove|explain|why|how|what|"
    r"physics|chemistry|biology|math|algebra|geometry|calculus|force|energy|atom|molecule|cell|"
    r"velocity|acceleration|probability|fraction|percent|theorem|reaction|newton|gravity|=|\+|\bx\b)",
    re.IGNORECASE,
)

_OFF_TOPIC_MARKERS = re.compile(
    r"(write\s+me\s+a\s+poem|write\s+a\s+poem|tell\s+me\s+a\s+joke|joke\s+about|"
    r"stock\s+tip|stock\s+should\s+i\s+buy|should\s+i\s+(buy|invest)|invest\s+in|"
    r"dating\s+advice|advice\s+for\s+my\s+crush|who\s+will\s+win|next\s+election|"
    r"weather\s+forecast|book\s+(me\s+)?a\s+flight|recipe\s+for)",
    re.IGNORECASE,
)

_MAX_INPUT_LEN = 6000


class InputGuard:
    """Applies input inspection under a guardrail policy."""

    def __init__(self, policy: GuardrailPolicy, detector: PromptInjectionDetector) -> None:
        self._policy = policy
        self._detector = detector

    def inspect(self, text: str) -> InputInspection:
        flags: List[str] = []
        text = text or ""

        if len(text.strip()) == 0:
            return InputInspection(
                decision=InputDecision.REFUSE,
                risk=RiskLevel.LOW,
                flags=["empty_input"],
                injection=InjectionSignal(),
                reason="Empty input.",
                safe_message="It looks like your message was empty. What would you like help with?",
            )

        if len(text) > _MAX_INPUT_LEN:
            flags.append("oversized_input")
            text = text[:_MAX_INPUT_LEN]

        # Harmful content is refused regardless of policy (safety floor).
        if _HARMFUL.search(text):
            return InputInspection(
                decision=InputDecision.REFUSE,
                risk=RiskLevel.HIGH,
                flags=flags + ["harmful_request"],
                injection=InjectionSignal(),
                reason="Request matches a harmful-content pattern.",
                safe_message=(
                    "I can't help with that. I'm a tutor for math and science. "
                    "Ask me about a homework problem or a concept you're studying."
                ),
            )

        # Prompt-injection layer.
        injection = self._detector.detect(text)
        risk = injection.risk
        if injection.is_injection:
            flags.append("prompt_injection")
            flags.extend(injection.categories)

        # Off-topic detection (only when not already an injection attempt).
        if not injection.is_injection:
            if _OFF_TOPIC_MARKERS.search(text) or not _ON_TOPIC.search(text):
                flags.append("off_topic")
                return InputInspection(
                    decision=InputDecision.REDIRECT,
                    risk=RiskLevel.LOW,
                    flags=flags,
                    injection=injection,
                    reason="Input does not look like a math or science tutoring question.",
                    safe_message=(
                        "I'm your math and science tutor, so I'll stick to those. "
                        "What problem or concept can I help you work through?"
                    ),
                )

        # Policy decision for injection attempts.
        if self._policy.blocks(risk):
            return InputInspection(
                decision=InputDecision.REFUSE,
                risk=risk,
                flags=flags,
                injection=injection,
                reason=f"Blocked by guardrail policy '{self._policy.name}' at risk {risk.value}.",
                safe_message=(
                    "I noticed that message is trying to change my instructions, which I can't do. "
                    "My job is to help you learn. What math or science question can I help with?"
                ),
            )

        # Lower-risk injection attempts are allowed through but flagged, so the
        # generation node knows to stay on guard and the response is still safe.
        return InputInspection(
            decision=InputDecision.ALLOW,
            risk=risk,
            flags=flags,
            injection=injection,
            reason="Allowed." if not flags else "Allowed with flags.",
        )
