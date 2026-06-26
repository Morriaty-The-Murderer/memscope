from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

DATASET_REPO = "xiaowu0162/longmemeval-cleaned"
DATASET_FILE = "longmemeval_s_cleaned.json"


@dataclass
class LongMemEvalQuestion:
    question_id: str
    question_type: str
    question: str
    question_date: str
    answer: str
    answer_session_ids: list[str]
    haystack_dates: list[str]
    haystack_session_ids: list[str]
    haystack_sessions: list[list[dict[str, str]]]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LongMemEvalQuestion:
        return cls(
            question_id=d["question_id"],
            question_type=d["question_type"],
            question=d["question"],
            question_date=d["question_date"],
            answer=d["answer"],
            answer_session_ids=d["answer_session_ids"],
            haystack_dates=d["haystack_dates"],
            haystack_session_ids=d["haystack_session_ids"],
            haystack_sessions=d["haystack_sessions"],
        )


def load_longmemeval_s(
    cache_dir: str | Path | None = None,
) -> list[LongMemEvalQuestion]:
    """Download (if needed) and load LongMemEval-S.

    Returns 500 questions, each with ~53 haystack sessions of conversation.
    Total download ~264MB.
    """
    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    with open(local_path) as f:
        raw = json.load(f)
    return [LongMemEvalQuestion.from_dict(q) for q in raw]
