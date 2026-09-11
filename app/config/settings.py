"""Central application settings, loaded from environment variables and .env.

All configuration flows through :class:`Settings`. Nothing else in the code
base should read ``os.environ`` directly. API keys are never hard coded and are
never logged.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TutorMode(str, Enum):
    """Pedagogical strictness. This is a teaching configuration only.

    Changing the tutor mode never relaxes safety protections. It only affects
    how quickly the tutor is allowed to reveal a final answer and how it phrases
    guidance.
    """

    STRICT = "STRICT"
    GUIDED = "GUIDED"
    BALANCED = "BALANCED"
    DIRECT = "DIRECT"


class GuardrailPolicyName(str, Enum):
    """Named guardrail policy. RESEARCH_DEMO relaxes pedagogy, never safety."""

    STRICT = "STRICT"
    BALANCED = "BALANCED"
    RESEARCH_DEMO = "RESEARCH_DEMO"


class EmbeddingBackend(str, Enum):
    OPENAI = "openai"
    LOCAL = "local"


class VectorDB(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"


class Settings(BaseSettings):
    """Strongly typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # model_name / model_temperature would otherwise clash with Pydantic's
        # protected "model_" namespace; opt out of the protection.
        protected_namespaces=("settings_",),
    )

    # ---- LLM provider ----
    # Any OpenAI-compatible provider works (Groq, OpenRouter, Together, Ollama,
    # Gemini's compatibility endpoint, GitHub Models). Point openai_base_url at
    # the provider and set the matching model name; no other code changes.
    openai_api_key: Optional[str] = Field(default=None)
    openai_base_url: Optional[str] = Field(default=None)
    model_name: str = Field(default="gpt-4o-mini")
    model_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    llm_timeout_seconds: int = Field(default=45, gt=0)

    # ---- Embeddings / RAG ----
    embedding_backend: EmbeddingBackend = Field(default=EmbeddingBackend.LOCAL)
    embedding_model: str = Field(default="text-embedding-3-small")
    vector_db: VectorDB = Field(default=VectorDB.CHROMA)
    chroma_persist_dir: str = Field(default="data/chroma")
    chroma_collection: str = Field(default="socratic_tutor")
    rag_chunk_size: int = Field(default=1000, gt=0)
    rag_chunk_overlap: int = Field(default=150, ge=0)
    rag_top_k: int = Field(default=4, gt=0)

    # ---- Tutoring policy ----
    tutor_mode: TutorMode = Field(default=TutorMode.STRICT)

    # ---- Guardrails ----
    guardrail_policy: GuardrailPolicyName = Field(default=GuardrailPolicyName.STRICT)
    injection_llm_classifier: bool = Field(default=False)
    max_output_retries: int = Field(default=2, ge=0, le=5)

    # ---- Observability ----
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True)
    langsmith_tracing: bool = Field(default=False)
    langsmith_api_key: Optional[str] = Field(default=None)
    langsmith_project: str = Field(default="socratic-tutor")

    # ---- API ----
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    backend_url: str = Field(default="http://localhost:8000")

    @property
    def has_openai_key(self) -> bool:
        """True when live generation is possible.

        Either a real API key, or a custom base URL (a self-hosted runtime such
        as Ollama needs no key, so a placeholder is supplied at client build
        time).
        """
        if self.openai_api_key and self.openai_api_key.strip():
            return True
        return bool(self.openai_base_url and self.openai_base_url.strip())

    def redacted(self) -> dict:
        """Return a dict of settings safe to log (secrets removed)."""
        data = self.model_dump()
        for secret in ("openai_api_key", "langsmith_api_key"):
            if data.get(secret):
                data[secret] = "***redacted***"
        return data


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using an lru_cache keeps a single source of truth without global mutable
    state and makes the dependency easy to override in tests.
    """
    return Settings()
