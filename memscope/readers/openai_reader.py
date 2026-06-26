from __future__ import annotations

import os

from .base import Reader


class OpenAIReader(Reader):
    """Reader that uses an OpenAI-compatible API to answer questions.

    Works with OpenAI, or any OpenAI-compatible endpoint (vLLM, Ollama, etc.)
    via base_url override.
    """

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 256,
    ):
        self._model = model
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._max_tokens = max_tokens

    def answer(self, question: str, memory_context: list[str]) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self._base_url, api_key=self._api_key)

        context_text = "\n\n".join(f"[Memory {i+1}] {m}" for i, m in enumerate(memory_context)) or "No relevant memories found."

        system_prompt = (
            "You are a helpful assistant answering questions based on the user's conversation history. "
            "Use only the provided memories to answer. If the memories don't contain the answer, say 'I don't know'. "
            "Be concise — answer in one sentence."
        )

        user_prompt = f"Conversation memories:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
