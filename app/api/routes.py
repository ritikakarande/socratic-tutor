"""API routes.

Exposes the tutor over HTTP. The agent is built once at startup and reused. A
per-request tutor-mode override is supported for demonstrations, but it only
changes pedagogy; the guardrail policy (safety) is fixed by configuration.
"""

from __future__ import annotations

import difflib
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.agents.graph import TutorAgent, build_agent
from app.agents.nodes import TutorDependencies
from app.api.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    ToolUsed,
)
from app.api.sessions import SessionStore
from app.config.settings import TutorMode, get_settings
from app.guardrails.policies import get_tutor_policy
from app.utils.logging import get_logger, log_event, new_request_id

logger = get_logger("api")
router = APIRouter()

# Built once at import; see main.py lifespan for explicit warm-up.
_AGENT: TutorAgent | None = None
_SESSIONS = SessionStore()


def get_agent() -> TutorAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent(get_settings())
    return _AGENT


def _agent_for_mode(base: TutorAgent, mode_override: str | None) -> TutorAgent:
    """Return an agent whose pedagogy matches an optional per-request mode.

    Safety (guardrail policy) is never altered here; only the tutor policy and
    the output guard's pedagogy expectations change.
    """
    if not mode_override:
        return base
    try:
        mode = TutorMode(mode_override.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown tutor_mode '{mode_override}'.")
    if mode.value == base.deps.tutor_policy.name:
        return base
    from app.agents.graph import TutorAgent as _TA
    from app.guardrails.output_guard import OutputGuard

    base_deps = base.deps
    policy = get_tutor_policy(mode)
    new_deps = TutorDependencies(
        settings=base_deps.settings,
        llm=base_deps.llm,
        router=base_deps.router,
        tools=base_deps.tools,
        input_guard=base_deps.input_guard,  # unchanged: safety stays fixed
        output_guard=OutputGuard(tutor_policy=policy),
        tutor_policy=policy,
        evaluator=base_deps.evaluator,
    )
    return _TA(new_deps)


def _is_new_problem(session_history: List, question: str) -> bool:
    """Heuristic: a message is a follow-up if it closely echoes recent context."""
    if not session_history:
        return True
    recent = " ".join(m["content"] for m in session_history[-4:])
    ratio = difflib.SequenceMatcher(None, question.lower(), recent.lower()).ratio()
    short_followup = len(question.split()) <= 8
    return not (ratio > 0.3 or short_followup)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    agent = get_agent()
    try:
        doc_count = agent.deps.tools._retriever.store.count() if agent.deps.tools.has_retriever else 0
    except Exception:  # noqa: BLE001
        doc_count = 0
    return HealthResponse(
        status="ok",
        llm_live=agent.deps.llm.is_live,
        vector_documents=doc_count,
        tutor_mode=agent.deps.tutor_policy.name,
        guardrail_policy=settings.guardrail_policy.value,
    )


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    request_id = new_request_id()
    base_agent = get_agent()
    agent = _agent_for_mode(base_agent, req.tutor_mode)

    session = _SESSIONS.get(req.session_id)
    is_new = _is_new_problem(session.history, req.question)

    # Temporarily disable retrieval for this turn if the caller opted out.
    original_retriever = agent.deps.tools._retriever
    if not req.enable_rag:
        agent.deps.tools._retriever = None

    try:
        state = agent.run(
            question=req.question,
            session_id=req.session_id,
            request_id=request_id,
            history=session.history,
            hints_given=session.hints_given,
            attempts_made=0 if is_new else session.attempts_made,
            current_problem=None if is_new else session.current_problem,
            final_answer_requested=req.final_answer_requested,
        )
    finally:
        agent.deps.tools._retriever = original_retriever

    _SESSIONS.record_turn(
        session_id=req.session_id,
        question=req.question,
        response=state.get("response", ""),
        hints_given=state.get("hints_given", 0),
        is_new_problem=is_new,
    )

    tools_used: List[ToolUsed] = []
    for raw in state.get("tool_results", []):
        if raw.get("success"):
            tools_used.append(
                ToolUsed(name="Mathematical verification", detail=f"{raw['operation']}: {raw['result']}")
            )
    citations: List[Citation] = []
    if state.get("retrieved_context"):
        tools_used.append(ToolUsed(name="Textbook retrieval", detail=f"{len(state['retrieved_context'])} passages"))
        for c in state["retrieved_context"]:
            citations.append(
                Citation(
                    source=c.get("source", "unknown"),
                    chapter=c.get("chapter"),
                    page=c.get("page"),
                    score=c.get("score", 0.0),
                )
            )

    latency = float(state.get("evaluation", {}).get("generation_ms", 0.0))

    log_event(
        logger, 20, "turn_complete",
        request_id=request_id,
        session_id=req.session_id,
        subject=state.get("subject"),
        difficulty=state.get("difficulty"),
        intent=state.get("intent"),
        tools_used=[t.name for t in tools_used],
        retrieval_used=bool(state.get("retrieved_context")),
        blocked=state.get("blocked"),
        latency_ms=latency,
        guardrail=state.get("output_guard", {}).get("approved"),
        socratic_score=state.get("evaluation", {}).get("socratic_score"),
    )

    return AskResponse(
        request_id=request_id,
        session_id=req.session_id,
        response=state.get("response", ""),
        subject=state.get("subject", "unknown"),
        difficulty=state.get("difficulty", "unknown"),
        intent=state.get("intent", "conceptual"),
        blocked=state.get("blocked", False),
        safety_flags=state.get("safety_flags", []),
        tools_used=tools_used,
        citations=citations,
        evaluation=state.get("evaluation", {}),
        output_guard=state.get("output_guard", {}),
        hints_given=state.get("hints_given", 0),
        latency_ms=latency,
    )


@router.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    existed = _SESSIONS.reset(req.session_id)
    return ResetResponse(session_id=req.session_id, cleared=existed)
