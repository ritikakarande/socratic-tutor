"""Pydantic request/response schemas for the API.

Response models expose only the structured information the frontend needs. They
never include the system prompt, developer rules or any hidden reasoning.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """A single tutoring turn request."""

    question: str = Field(..., min_length=1, max_length=6000)
    session_id: str = Field(default="default", max_length=128)
    # Optional per-request overrides for demonstrations.
    tutor_mode: Optional[str] = Field(default=None, description="STRICT|GUIDED|BALANCED|DIRECT")
    enable_rag: bool = Field(default=True)
    final_answer_requested: bool = Field(default=False)


class ToolUsed(BaseModel):
    name: str
    detail: str


class Citation(BaseModel):
    source: str
    chapter: Optional[str] = None
    page: Optional[int] = None
    score: float


class AskResponse(BaseModel):
    """A tutoring turn response."""

    request_id: str
    session_id: str
    response: str
    subject: str
    difficulty: str
    intent: str
    blocked: bool
    safety_flags: List[str] = Field(default_factory=list)
    tools_used: List[ToolUsed] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    evaluation: Dict = Field(default_factory=dict)
    output_guard: Dict = Field(default_factory=dict)
    hints_given: int = 0
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    llm_live: bool
    vector_documents: int
    tutor_mode: str
    guardrail_policy: str


class ResetRequest(BaseModel):
    session_id: str = Field(default="default", max_length=128)


class ResetResponse(BaseModel):
    session_id: str
    cleared: bool
