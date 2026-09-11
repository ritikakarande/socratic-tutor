"""LLM client abstraction.

A tiny interface (:class:`LLMClient`) with two implementations:

* :class:`OpenAIChatClient` - real chat completions via the OpenAI SDK.
* :class:`OfflineTutorClient` - a deterministic, rule-based stand-in that
  produces plausible Socratic responses with no network. It keeps the whole
  system (and the demo UI) usable without an API key and makes the agent tests
  hermetic.

The offline client is not a language model; it uses the structured state
(subject, tool results, retrieved context, policy) to assemble a short guiding
response. This keeps behavior deterministic and safe for tests while the real
client is used in production.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Protocol

from app.config.settings import Settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LLMClient(Protocol):
    """Minimal chat interface used by the agent."""

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Return a completion string for a system + user prompt."""

    @property
    def is_live(self) -> bool:
        """True when backed by a real model."""


class OpenAIChatClient:
    """Chat completions client for OpenAI or any OpenAI-compatible provider.

    Passing ``base_url`` points the same client at Groq, OpenRouter, Together,
    a local Ollama server, Gemini's compatibility endpoint, and so on. Nothing
    else in the agent changes.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: int = 45,
        base_url: Optional[str] = None,
    ) -> None:
        from openai import OpenAI

        # Local runtimes need no credential, but the SDK requires a non-empty
        # string, so supply a harmless placeholder.
        self._client = OpenAI(
            api_key=api_key or "not-required",
            timeout=timeout,
            base_url=base_url or None,
        )
        self._model = model

    @property
    def is_live(self) -> bool:
        return True

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_json(self, system: str, user: str) -> dict:
        """Request a JSON object response (used by classifiers)."""
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        try:
            return json.loads(resp.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return {}


class OfflineTutorClient:
    """Deterministic Socratic response generator for offline / test use.

    It parses a lightweight instruction block embedded by the prompt builder to
    know the subject, whether an answer may be revealed, tool results and
    context. It always asks a guiding question unless direct solutions are
    permitted, so it satisfies the pedagogy guard in strict modes.
    """

    @property
    def is_live(self) -> bool:
        return False

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        ctx = self._parse_context(user)
        question = ctx.get("question", "your question")
        subject = ctx.get("subject", "this topic")
        may_reveal = ctx.get("may_reveal", "false") == "true"
        tool_result = ctx.get("tool_result")
        reference = ctx.get("reference")

        lead = self._subject_lead(subject)
        parts: List[str] = [lead]

        if reference:
            parts.append(
                f"Using the reference material on {subject}, here is the key idea: "
                f"{self._first_sentence(reference)}"
            )

        if may_reveal and tool_result:
            parts.append(
                f"Working it through, we get {tool_result}. "
                "Can you explain in your own words why each step works?"
            )
        elif tool_result and not may_reveal:
            parts.append(
                "Let's take the first step together rather than jumping to the end. "
                f"For a problem like \"{question}\", what operation would isolate the unknown? "
                "Try that step and tell me what you get."
            )
        else:
            parts.append(self._guiding_question(subject, question))

        return " ".join(parts).strip()

    def complete_json(self, system: str, user: str) -> dict:
        """Conservative offline classifier: never claims injection on its own."""
        return {"is_injection": False, "risk": "none", "reason": "offline classifier"}

    @staticmethod
    def _parse_context(user: str) -> dict:
        ctx: dict = {}
        for line in user.splitlines():
            m = re.match(r"\[([A-Z_]+)\]\s*(.*)", line.strip())
            if m:
                ctx[m.group(1).lower()] = m.group(2)
        return ctx

    @staticmethod
    def _first_sentence(text: str) -> str:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return parts[0] if parts else text[:160]

    @staticmethod
    def _subject_lead(subject: str) -> str:
        leads = {
            "math": "Great, let's reason through this step by step.",
            "physics": "Good physics question. Let's build the intuition first.",
            "chemistry": "Let's think about what's happening at the molecular level.",
            "biology": "Let's connect this to the underlying biological principle.",
            "science": "Let's reason from the underlying principle.",
        }
        return leads.get(subject, "Let's work through this together.")

    @staticmethod
    def _guiding_question(subject: str, question: str) -> str:
        return (
            f"Before I give anything away: what do you already know that relates to \"{question}\"? "
            "Identifying the first concept usually points to the next step."
        )


def build_llm_client(settings: Settings) -> LLMClient:
    """Return a live client when a key is present, else the offline client."""
    if settings.has_openai_key:
        logger.info(
            "Using live chat client: model=%s endpoint=%s",
            settings.model_name,
            settings.openai_base_url or "api.openai.com (default)",
        )
        return OpenAIChatClient(
            api_key=settings.openai_api_key or "",
            model=settings.model_name,
            timeout=settings.llm_timeout_seconds,
            base_url=settings.openai_base_url,
        )
    logger.warning("No LLM credentials configured; using deterministic offline tutor client.")
    return OfflineTutorClient()
