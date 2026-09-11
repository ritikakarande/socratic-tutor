"""Typed agent state shared across LangGraph nodes.

The state carries only structured information required for application behavior
and evaluation. It deliberately does NOT store chain-of-thought or hidden
reasoning; nodes produce concise structured fields instead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class Message(TypedDict):
    """A single conversation message."""

    role: str  # "student" | "tutor"
    content: str


class TutorState(TypedDict, total=False):
    """State threaded through the tutor graph.

    ``total=False`` lets nodes populate fields incrementally. See
    :func:`new_state` for the canonical initial shape.
    """

    # Request / identity
    request_id: str
    session_id: str

    # Conversation
    messages: List[Message]
    user_question: str

    # Classification
    subject: str
    difficulty: str
    intent: str  # "math_calculation" | "knowledge" | "conceptual" | "mixed"
    needs_retrieval: bool
    needs_calculation: bool

    # Working memory across turns
    hints_given: int
    attempts_made: int
    current_problem: Optional[str]
    final_answer_requested: bool

    # Tool / retrieval outputs
    retrieved_context: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    # Safety
    safety_flags: List[str]
    input_decision: str
    blocked: bool

    # Generation
    response: str
    answer_reveal_allowed: bool

    # Guard / evaluation metadata
    output_guard: Dict[str, Any]
    evaluation: Dict[str, Any]
    regeneration_count: int

    # Config snapshot for this run
    tutor_mode: str
    guardrail_policy: str


def new_state(
    user_question: str,
    session_id: str = "default",
    request_id: str = "",
    history: Optional[List[Message]] = None,
    hints_given: int = 0,
    attempts_made: int = 0,
    current_problem: Optional[str] = None,
    final_answer_requested: bool = False,
) -> TutorState:
    """Construct a fresh state for one tutoring turn."""
    return TutorState(
        request_id=request_id,
        session_id=session_id,
        messages=list(history or []),
        user_question=user_question,
        subject="unknown",
        difficulty="unknown",
        intent="conceptual",
        needs_retrieval=False,
        needs_calculation=False,
        hints_given=hints_given,
        attempts_made=attempts_made,
        current_problem=current_problem,
        final_answer_requested=final_answer_requested,
        retrieved_context=[],
        tool_results=[],
        safety_flags=[],
        input_decision="allow",
        blocked=False,
        response="",
        answer_reveal_allowed=False,
        output_guard={},
        evaluation={},
        regeneration_count=0,
        tutor_mode="STRICT",
        guardrail_policy="STRICT",
    )
