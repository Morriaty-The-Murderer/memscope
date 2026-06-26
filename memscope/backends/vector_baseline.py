from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import MemoryBackend


class VectorBaselineBackend(MemoryBackend):
    """Pure vector-similarity baseline.

    Embeds each user turn as a standalone memory, retrieves by cosine similarity.
    No LLM extraction, no graph, no deduplication — the simplest possible memory
    strategy that still works. Serves as the sanity-check floor for comparison.
    """

    name = "vector-baseline"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self._model = SentenceTransformer(model_name, device=device)
        self._embeddings: list[np.ndarray] = []
        self._contents: list[str] = []
        self._user_ids: list[str] = []
        self._session_ids: list[str | None] = []
        self._dim = self._model.get_embedding_dimension()

    def add(self, messages: list[dict[str, str]], user_id: str = "default", session_id: str | None = None) -> None:
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "").strip()
            if not content:
                continue
            emb = self._model.encode(content, normalize_embeddings=True, show_progress_bar=False)
            self._embeddings.append(emb)
            self._contents.append(content)
            self._user_ids.append(user_id)
            self._session_ids.append(session_id)

    def search(self, query: str, user_id: str = "default", top_k: int = 10) -> list[str]:
        if not self._embeddings:
            return []

        mask = np.array([uid == user_id for uid in self._user_ids])
        if not mask.any():
            return []

        user_indices = np.where(mask)[0]
        user_embs = np.array(self._embeddings)[user_indices]
        user_contents = [self._contents[i] for i in user_indices]

        query_emb = self._model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        scores = user_embs @ query_emb
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [user_contents[i] for i in top_indices]

    def reset(self, user_id: str = "default") -> None:
        keep = [i for i, uid in enumerate(self._user_ids) if uid != user_id]
        self._embeddings = [self._embeddings[i] for i in keep]
        self._contents = [self._contents[i] for i in keep]
        self._user_ids = [self._user_ids[i] for i in keep]
        self._session_ids = [self._session_ids[i] for i in keep]

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self._model.model_name if hasattr(self._model, "model_name") else "unknown",
            "embedding_dim": self._dim,
            "total_memories": len(self._contents),
        }
