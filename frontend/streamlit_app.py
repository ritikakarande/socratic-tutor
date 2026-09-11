"""Streamlit frontend for the Socratic Tutor.

Kept strictly separate from backend/agent logic: it talks to the FastAPI
backend over HTTP when one is reachable, and transparently falls back to an
in-process agent otherwise, so the UI works with or without the API running.

Two pages:
  * Tutor: the chat experience with a configuration sidebar.
  * Dashboard: evaluation metrics and failed cases for developers/teachers.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st

# Allow "import app..." when run via "streamlit run frontend/streamlit_app.py".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Socratic Tutor", page_icon="🎓", layout="wide")


# --------------------------------------------------------------------------- #
# Backend access (HTTP with in-process fallback)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _local_agent():
    from app.agents.graph import build_agent

    return build_agent()


def backend_available() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def ask_backend(payload: dict) -> dict:
    r = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


def ask_local(payload: dict) -> dict:
    """Run the agent in-process and shape the result like the API response."""
    agent = _local_agent()
    history = st.session_state.get("history_msgs", [])
    from app.agents.graph import build_dependencies, TutorAgent
    from app.config.settings import TutorMode, get_settings
    from app.guardrails.output_guard import OutputGuard
    from app.guardrails.policies import get_tutor_policy

    # Apply per-request tutor mode (pedagogy only).
    base = agent
    mode = payload.get("tutor_mode")
    if mode:
        policy = get_tutor_policy(TutorMode(mode))
        d = base.deps
        base = TutorAgent(type(d)(
            settings=d.settings, llm=d.llm, router=d.router, tools=d.tools,
            input_guard=d.input_guard, output_guard=OutputGuard(tutor_policy=policy),
            tutor_policy=policy, evaluator=d.evaluator,
        ))

    if not payload.get("enable_rag", True):
        original = base.deps.tools._retriever
        base.deps.tools._retriever = None
    try:
        state = base.run(
            question=payload["question"],
            history=history,
            final_answer_requested=payload.get("final_answer_requested", False),
        )
    finally:
        if not payload.get("enable_rag", True):
            base.deps.tools._retriever = original

    tools_used = []
    for r in state.get("tool_results", []):
        if r.get("success"):
            tools_used.append({"name": "Mathematical verification", "detail": f"{r['operation']}: {r['result']}"})
    citations = []
    if state.get("retrieved_context"):
        tools_used.append({"name": "Textbook retrieval", "detail": f"{len(state['retrieved_context'])} passages"})
        for c in state["retrieved_context"]:
            citations.append({"source": c.get("source"), "chapter": c.get("chapter"), "page": c.get("page"), "score": c.get("score", 0.0)})
    return {
        "response": state.get("response", ""),
        "subject": state.get("subject"),
        "difficulty": state.get("difficulty"),
        "intent": state.get("intent"),
        "blocked": state.get("blocked", False),
        "safety_flags": state.get("safety_flags", []),
        "tools_used": tools_used,
        "citations": citations,
        "evaluation": state.get("evaluation", {}),
        "output_guard": state.get("output_guard", {}),
    }


def ask(payload: dict, use_backend: bool) -> dict:
    if use_backend:
        try:
            return ask_backend(payload)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Backend error, falling back to local agent: {exc}")
    return ask_local(payload)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> dict:
    st.sidebar.title("🎓 Socratic Tutor")
    page = st.sidebar.radio("Page", ["Tutor", "Dashboard"], index=0)
    st.sidebar.divider()

    tutor_mode = st.sidebar.selectbox(
        "Tutor Mode", ["STRICT", "GUIDED", "BALANCED", "DIRECT"], index=0,
        help="Pedagogical strictness only. Safety protections stay active in every mode.",
    )
    subject = st.sidebar.selectbox("Subject hint", ["Auto", "Math", "Physics", "Chemistry", "Science"], index=0)
    difficulty = st.sidebar.selectbox("Difficulty", ["Auto", "Elementary", "High School", "College"], index=0)
    enable_rag = st.sidebar.toggle("Enable RAG (textbook retrieval)", value=True)
    final_answer = st.sidebar.toggle("I've tried; request a direct explanation", value=False)

    st.sidebar.divider()
    use_backend = st.sidebar.toggle("Use API backend", value=backend_available(),
                                    help=f"Backend: {BACKEND_URL}")
    if st.sidebar.button("Reset conversation"):
        st.session_state["messages"] = []
        st.session_state["history_msgs"] = []
        try:
            requests.post(f"{BACKEND_URL}/reset", json={"session_id": st.session_state.get("session_id", "default")}, timeout=2)
        except Exception:  # noqa: BLE001
            pass
        st.rerun()

    return {
        "page": page, "tutor_mode": tutor_mode, "subject": subject,
        "difficulty": difficulty, "enable_rag": enable_rag,
        "final_answer": final_answer, "use_backend": use_backend,
    }


# --------------------------------------------------------------------------- #
# Tutor page
# --------------------------------------------------------------------------- #
def tutor_page(cfg: dict) -> None:
    st.title("AI Socratic Tutor")
    st.caption("A guided math and science tutor. It leads you to the answer with questions and hints.")

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history_msgs", [])
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])

    for msg in st.session_state["messages"]:
        with st.chat_message("user" if msg["role"] == "student" else "assistant"):
            st.markdown(msg["content"])
            if msg.get("meta"):
                _render_meta(msg["meta"])

    prompt = st.chat_input("Ask a question...")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "student", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "question": prompt,
        "session_id": st.session_state["session_id"],
        "tutor_mode": cfg["tutor_mode"],
        "enable_rag": cfg["enable_rag"],
        "final_answer_requested": cfg["final_answer"],
    }
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(payload, cfg["use_backend"])
        st.markdown(result["response"])
        _render_meta(result)

    st.session_state["messages"].append({"role": "tutor", "content": result["response"], "meta": result})
    st.session_state["history_msgs"].append({"role": "student", "content": prompt})
    st.session_state["history_msgs"].append({"role": "tutor", "content": result["response"]})


def _render_meta(result: dict) -> None:
    cols = st.columns(4)
    cols[0].caption(f"Subject: {result.get('subject', '-')}")
    cols[1].caption(f"Level: {result.get('difficulty', '-')}")
    cols[2].caption(f"Intent: {result.get('intent', '-')}")
    ev = result.get("evaluation", {})
    if "socratic_score" in ev:
        cols[3].caption(f"Socratic: {ev['socratic_score']:.2f}")

    if result.get("blocked"):
        st.info("This request was handled by the safety guardrail.")

    tools = result.get("tools_used", [])
    if tools:
        with st.expander("Tools Used"):
            for t in tools:
                st.markdown(f"- {t['name']}: {t['detail']}")
    cites = result.get("citations", [])
    if cites:
        with st.expander("Sources"):
            for c in cites:
                st.markdown(f"- {c.get('source')} ({c.get('chapter') or ''} p.{c.get('page') or '-'}), score {c.get('score', 0):.2f}")


# --------------------------------------------------------------------------- #
# Dashboard page
# --------------------------------------------------------------------------- #
def dashboard_page() -> None:
    import json

    st.title("Evaluation Dashboard")
    st.caption("Run `python scripts/evaluate.py` to (re)generate the report.")

    report_path = Path("eval_report.json")
    if not report_path.exists():
        st.warning("No eval_report.json found. Run: python scripts/evaluate.py")
        return

    data = json.loads(report_path.read_text(encoding="utf-8"))
    for suite, report in data.items():
        st.subheader(suite)
        agg = report.get("aggregates", {})
        metric_cols = st.columns(len(agg) or 1)
        for i, (key, val) in enumerate(agg.items()):
            metric_cols[i % len(metric_cols)].metric(key, val)
        failed = report.get("failed", [])
        st.caption(f"{len(failed)} failed cases")
        if failed:
            st.dataframe(failed, use_container_width=True)
        st.divider()


def main() -> None:
    cfg = sidebar()
    if cfg["page"] == "Tutor":
        tutor_page(cfg)
    else:
        dashboard_page()


main()
