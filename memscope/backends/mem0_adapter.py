from __future__ import annotations

from typing import Any

from .base import MemoryBackend


class Mem0Backend(MemoryBackend):
    """Adapter for Mem0 (mem0ai).

    Wraps Mem0's add/search API behind our unified MemoryBackend interface.
    Requires `pip install mem0ai` and configuration of the underlying vector store.
    """

    name = "mem0"

    def __init__(self, config: dict[str, Any] | None = None):
        try:
            from mem0 import Memory
        except ImportError as e:
            raise ImportError(
                "mem0ai is required for Mem0Backend. Install with: pip install mem0ai"
            ) from e

        if config:
            self._memory = Memory.from_config(config)
        else:
            self._memory = Memory()
        self._config = config or {}

    def add(self, messages: list[dict[str, str]], user_id: str = "default", session_id: str | None = None) -> None:
        self._memory.add(messages, user_id=user_id)

    def search(self, query: str, user_id: str = "default", top_k: int = 10) -> list[str]:
        results = self._memory.search(query, user_id=user_id, limit=top_k)
        if isinstance(results, dict) and "results" in results:
            results = results["results"]
        return [r.get("memory", "") if isinstance(r, dict) else str(r) for r in results]

    def reset(self, user_id: str = "default") -> None:
        self._memory.delete_all(user_id=user_id)

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": self._config,
        }
