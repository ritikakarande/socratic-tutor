"""In-memory session store for conversation memory.

Holds concise per-session state (history, hints given, attempts, current
problem). This is deliberately simple and process-local; a production system
would back it with Redis or a database. It stores structured summaries only,
never hidden chain-of-thought.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List

from app.agents.state import Message


@dataclass
class SessionState:
    history: List[Message] = field(default_factory=list)
    hints_given: int = 0
    attempts_made: int = 0
    current_problem: str = ""


class SessionStore:
    """Thread-safe dictionary of sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            return self._sessions.setdefault(session_id, SessionState())

    def reset(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._sessions
            self._sessions[session_id] = SessionState()
            return existed

    def record_turn(
        self,
        session_id: str,
        question: str,
        response: str,
        hints_given: int,
        is_new_problem: bool,
    ) -> None:
        """Update session memory after a completed turn."""
        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())
            state.history.append(Message(role="student", content=question))
            state.history.append(Message(role="tutor", content=response))
            # Trim history to keep prompts bounded.
            if len(state.history) > 20:
                state.history = state.history[-20:]
            state.hints_given = hints_given
            if is_new_problem:
                state.current_problem = question
                state.attempts_made = 0
            else:
                state.attempts_made += 1
