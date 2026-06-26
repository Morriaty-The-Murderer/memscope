from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    """A single piece of memory extracted from a conversation turn."""
    content: str
    user_id: str = "default"
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryBackend(ABC):
    """Abstract interface for a pluggable agent memory backend.

    V1 contract — retrieval-focused evaluation:
      - add(): ingest a conversation (list of {role, content} messages)
      - search(): retrieve relevant memory strings for a query

    V2 roadmap (NOT implemented in V1):
      - evaluate_task_success(): given a task + memory context, measure whether
        memory helped the agent succeed at a downstream task (not just recall).
        This requires a task harness and is deferred to V2.
    """

    name: str = "abstract"

    @abstractmethod
    def add(self, messages: list[dict[str, str]], user_id: str = "default", session_id: str | None = None) -> None:
        """Ingest a conversation as a list of {role, content} message dicts."""
        ...

    @abstractmethod
    def search(self, query: str, user_id: str = "default", top_k: int = 10) -> list[str]:
        """Retrieve the top-k most relevant memory strings for the query."""
        ...

    def reset(self, user_id: str = "default") -> None:
        """Clear all memories for a user. Override for backends that support it."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Return backend statistics (memory count, index size, etc.)."""
        return {"name": self.name}
