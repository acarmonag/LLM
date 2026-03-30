import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Sliding window conversation memory for a single session."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.entries: list[dict] = []

    def add(self, role: str, content: str, rag_cases_used: list[str] = None):
        entry = {
            "role": role,
            "content": content,
            "rag_cases_used": rag_cases_used or [],
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.entries.append(entry)
        # Trim to window size
        if len(self.entries) > self.window_size:
            self.entries = self.entries[-self.window_size:]

    def to_llm_messages(self, n: int = None) -> list[dict]:
        """Return the last N messages formatted for Ollama's messages API."""
        if n is None:
            n = self.window_size
        recent = self.entries[-n:]
        return [{"role": e["role"], "content": e["content"]} for e in recent]


class ConversationStore:
    """In-memory store of conversation memories keyed by session_id."""

    def __init__(self):
        self._sessions: dict[str, ConversationMemory] = {}

    def get_or_create(self, session_id: str = None) -> tuple[str, ConversationMemory]:
        """Get existing session or create new one. Returns (session_id, memory)."""
        if session_id and session_id in self._sessions:
            return session_id, self._sessions[session_id]
        if not session_id:
            session_id = str(uuid.uuid4())
        memory = ConversationMemory()
        self._sessions[session_id] = memory
        return session_id, memory
