"""
Context Memory
==============
Requirement: "The assistant should remember conversation context within the
current session."

Design choice: an in-memory dict keyed by session_id, storing a rolling
window of chat turns in the exact {"role": ..., "content": ...} shape the
OpenAI API expects. This is intentionally simple:

- No coreference-resolution logic is hand-written (e.g. figuring out that
  "cheaper options" refers to "laptops"). Instead, the full recent history
  is passed to the LLM on every call, and the LLM resolves references
  naturally the same way it would in ChatGPT. This is the standard,
  robust approach -- hand-rolled pronoun resolution is brittle and
  unnecessary when you control the LLM call.
- Storage is process-local memory, not a database or Redis. This satisfies
  "within the current session" exactly as specified. It does NOT persist
  across server restarts -- documented as a known limitation in the README,
  with a note on how you'd swap in Redis/Postgres for production.
- Each session's history is capped at MEMORY_MAX_TURNS turns (a "turn" =
  one user message + one assistant reply) to bound token usage / cost.
"""
from typing import Dict, List
from app.config import settings

# session_id -> list of {"role": "user"|"assistant", "content": str}
_sessions: Dict[str, List[dict]] = {}


def get_history(session_id: str) -> List[dict]:
    """Return the message history for a session (empty list if new)."""
    return _sessions.setdefault(session_id, [])


def append_message(session_id: str, role: str, content: str) -> None:
    """Append a message to a session's history and trim to the max window."""
    history = _sessions.setdefault(session_id, [])
    history.append({"role": role, "content": content})

    # Trim to last N turns (N user + N assistant messages = 2N entries)
    max_entries = settings.MEMORY_MAX_TURNS * 2
    if len(history) > max_entries:
        _sessions[session_id] = history[-max_entries:]


def clear_session(session_id: str) -> None:
    """Wipe a session's memory (useful for a 'reset chat' button)."""
    _sessions.pop(session_id, None)


def list_sessions() -> List[str]:
    """Debug helper: list all active session IDs."""
    return list(_sessions.keys())