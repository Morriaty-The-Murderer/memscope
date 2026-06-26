from abc import ABC, abstractmethod


class Reader(ABC):
    """Abstract reader that answers a question given retrieved memory context.

    V1: used for end-to-end accuracy evaluation (does the memory help answer the question?).
    V2: will be extended to support task-success evaluation.
    """

    name: str = "abstract"

    @abstractmethod
    def answer(self, question: str, memory_context: list[str]) -> str:
        """Given a question and retrieved memory strings, produce an answer."""
        ...
