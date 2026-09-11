"""LangGraph assembly and the top-level tutor agent.

Graph shape::

    input_guard --(blocked)--> finalize_blocked --> END
                \\--(ok)--> router --> (calculate) --> (retrieve) --> generate
    generate --> output_guard --> evaluate --> END

Conditional edges implement the blocked-input short circuit and the router
branch into the tool nodes. The calculate and retrieve nodes are no-ops when
the router decides they are not needed, so pure arithmetic skips RAG and
concept questions skip the calculator. Nodes are created from an injected
:class:`TutorDependencies` so the whole graph is testable and free of global
state.
"""

from __future__ import annotations

from typing import List, Optional

from langgraph.graph import END, StateGraph

from app.agents.llm import LLMClient, OpenAIChatClient, build_llm_client
from app.agents.nodes import (
    TutorDependencies,
    make_calculation_node,
    make_evaluation_node,
    make_finalize_blocked_node,
    make_generation_node,
    make_input_guard_node,
    make_output_guard_node,
    make_retrieval_node,
    make_router_node,
)
from app.agents.router import IntentRouter
from app.agents.state import Message, TutorState, new_state
from app.config.settings import Settings, get_settings
from app.evaluation.evaluators import InlineEvaluator
from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from app.guardrails.policies import get_guardrail_policy, get_tutor_policy
from app.guardrails.prompt_injection import InjectionSignal, PromptInjectionDetector
from app.rag.retriever import Retriever, build_retriever
from app.tools.tool_registry import ToolRegistry
from app.utils.logging import get_logger

logger = get_logger(__name__)


class _LLMInjectionClassifier:
    """Adapts an OpenAI client into the injection LLMClassifier protocol."""

    def __init__(self, client: OpenAIChatClient) -> None:
        self._client = client

    def classify_injection(self, text: str) -> Optional[InjectionSignal]:
        from app.agents.prompts import INJECTION_CLASSIFIER_SYSTEM
        from app.guardrails.policies import RiskLevel

        data = self._client.complete_json(INJECTION_CLASSIFIER_SYSTEM, text)
        if not data:
            return None
        risk_str = str(data.get("risk", "none")).lower()
        risk = {
            "none": RiskLevel.NONE,
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
        }.get(risk_str, RiskLevel.NONE)
        return InjectionSignal(
            is_injection=bool(data.get("is_injection", False)),
            risk=risk,
            reason=str(data.get("reason", "")),
            categories=["llm_classified"] if data.get("is_injection") else [],
        )


def _route_after_input(state: TutorState) -> str:
    return "blocked" if state.get("blocked") else "ok"


def build_tutor_graph(deps: TutorDependencies):
    """Compile and return the LangGraph state machine."""
    graph = StateGraph(TutorState)

    graph.add_node("input_guard", make_input_guard_node(deps))
    graph.add_node("router", make_router_node(deps))
    graph.add_node("calculate", make_calculation_node(deps))
    graph.add_node("retrieve", make_retrieval_node(deps))
    graph.add_node("generate", make_generation_node(deps))
    graph.add_node("guard_output", make_output_guard_node(deps))
    graph.add_node("evaluate", make_evaluation_node(deps))
    graph.add_node("finalize_blocked", make_finalize_blocked_node(deps))

    graph.set_entry_point("input_guard")
    graph.add_conditional_edges(
        "input_guard",
        _route_after_input,
        {"blocked": "finalize_blocked", "ok": "router"},
    )
    graph.add_edge("finalize_blocked", END)

    # Linear tool chain; each tool node self-skips when not needed.
    graph.add_edge("router", "calculate")
    graph.add_edge("calculate", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "guard_output")
    graph.add_edge("guard_output", "evaluate")
    graph.add_edge("evaluate", END)

    return graph.compile()


class TutorAgent:
    """High-level entry point wrapping the compiled graph."""

    def __init__(self, deps: TutorDependencies) -> None:
        self._deps = deps
        self._graph = build_tutor_graph(deps)

    @property
    def deps(self) -> TutorDependencies:
        return self._deps

    def run(
        self,
        question: str,
        session_id: str = "default",
        request_id: str = "",
        history: Optional[List[Message]] = None,
        hints_given: int = 0,
        attempts_made: int = 0,
        current_problem: Optional[str] = None,
        final_answer_requested: bool = False,
    ) -> TutorState:
        """Run one tutoring turn and return the final state."""
        state = new_state(
            user_question=question,
            session_id=session_id,
            request_id=request_id,
            history=history,
            hints_given=hints_given,
            attempts_made=attempts_made,
            current_problem=current_problem,
            final_answer_requested=final_answer_requested,
        )
        state["tutor_mode"] = self._deps.tutor_policy.name
        state["guardrail_policy"] = self._deps.settings.guardrail_policy.value
        return self._graph.invoke(state)


def build_dependencies(
    settings: Optional[Settings] = None,
    retriever: Optional[Retriever] = None,
    llm: Optional[LLMClient] = None,
) -> TutorDependencies:
    """Assemble all agent dependencies from settings.

    Retriever and LLM can be injected (used by tests); otherwise they are built
    from configuration. Retriever construction failures degrade to no retriever
    rather than crashing.
    """
    settings = settings or get_settings()
    llm = llm or build_llm_client(settings)

    if retriever is None:
        try:
            retriever = build_retriever(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retriever unavailable: %s", exc)
            retriever = None

    # Optional LLM injection classifier layer.
    classifier = None
    if settings.injection_llm_classifier and isinstance(llm, OpenAIChatClient):
        classifier = _LLMInjectionClassifier(llm)
    detector = PromptInjectionDetector(llm_classifier=classifier)

    guardrail_policy = get_guardrail_policy(settings.guardrail_policy)
    tutor_policy = get_tutor_policy(settings.tutor_mode)

    return TutorDependencies(
        settings=settings,
        llm=llm,
        router=IntentRouter(llm=llm, use_llm=settings.injection_llm_classifier),
        tools=ToolRegistry(retriever=retriever),
        input_guard=InputGuard(policy=guardrail_policy, detector=detector),
        output_guard=OutputGuard(tutor_policy=tutor_policy),
        tutor_policy=tutor_policy,
        evaluator=InlineEvaluator(),
    )


def build_agent(settings: Optional[Settings] = None) -> TutorAgent:
    """Build a ready-to-use tutor agent from settings."""
    return TutorAgent(build_dependencies(settings))
