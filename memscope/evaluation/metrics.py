from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import MemoryBackend
from ..datasets.longmemeval import LongMemEvalQuestion
from ..readers.base import Reader


@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    question: str
    ground_truth_answer: str
    answer_session_ids: list[str]
    retrieved_memories: list[str]
    reader_answer: str | None = None
    recall_at_k: dict[int, float] = field(default_factory=dict)
    answer_correct: bool | None = None


@dataclass
class EvaluationReport:
    backend_name: str
    reader_name: str
    dataset_name: str
    total_questions: int
    results: list[QuestionResult]
    aggregate_recall: dict[int, float] = field(default_factory=dict)
    aggregate_recall_by_type: dict[str, dict[int, float]] = field(default_factory=dict)
    answer_accuracy: float = 0.0
    answer_accuracy_by_type: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "reader_name": self.reader_name,
            "dataset_name": self.dataset_name,
            "total_questions": self.total_questions,
            "aggregate_recall": self.aggregate_recall,
            "aggregate_recall_by_type": self.aggregate_recall_by_type,
            "answer_accuracy": self.answer_accuracy,
            "answer_accuracy_by_type": self.answer_accuracy_by_type,
            "config": self.config,
            "results": [
                {
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    "question": r.question,
                    "ground_truth_answer": r.ground_truth_answer,
                    "reader_answer": r.reader_answer,
                    "recall_at_k": r.recall_at_k,
                    "answer_correct": r.answer_correct,
                    "retrieved_memories": r.retrieved_memories[:5],
                }
                for r in self.results
            ],
        }


def compute_recall_at_k(
    retrieved: list[str],
    haystack_session_ids: list[str],
    haystack_sessions: list[list[dict[str, str]]],
    answer_session_ids: list[str],
    k_values: list[int],
) -> dict[int, float]:
    """Compute Recall@K.

    For LongMemEval-S, recall means: did the top-K retrieved memories include
    content from any of the answer_session_ids?

    We check if any retrieved memory string appears in a session that is an
    answer session. Since backends return text (not session IDs), we match
    by checking if retrieved text overlaps with answer-session content.
    """
    answer_contents: list[str] = []
    for i, sid in enumerate(haystack_session_ids):
        if sid in answer_session_ids:
            for msg in haystack_sessions[i]:
                content = msg.get("content", "").strip()
                if content:
                    answer_contents.append(content)

    if not answer_contents:
        return {k: 0.0 for k in k_values}

    def _is_hit(memory_text: str) -> bool:
        memory_text_lower = memory_text.lower()
        for ac in answer_contents:
            if ac.lower() in memory_text_lower or memory_text_lower in ac.lower():
                return True
            if _fuzzy_overlap(memory_text_lower, ac.lower()):
                return True
        return False

    hits = [i for i, m in enumerate(retrieved) if _is_hit(m)]

    result = {}
    for k in k_values:
        top_k_hits = [h for h in hits if h < k]
        result[k] = 1.0 if top_k_hits else 0.0

    return result


def _fuzzy_overlap(a: str, b: str, min_len: int = 50) -> bool:
    """Check if two strings share a substantial substring."""
    if len(a) < min_len or len(b) < min_len:
        return False
    check_len = min(100, len(a), len(b))
    for i in range(0, len(a) - check_len, check_len // 2):
        chunk = a[i : i + check_len]
        if chunk in b:
            return True
    return False


def judge_answer(reader_answer: str, ground_truth: str) -> bool:
    """Lightweight LLM-free answer judge.

    Checks if the ground truth appears in the reader's answer (case-insensitive).
    Not perfect, but reproducible and zero-cost — deliberately avoids the
    LLM-judge approach whose prompt variations cause the 45-point score gaps
    we document in the README.
    """
    if not reader_answer or not ground_truth:
        return False

    gt_lower = ground_truth.lower().strip()
    ans_lower = reader_answer.lower().strip()

    if gt_lower in ans_lower:
        return True

    gt_words = set(re.findall(r"\b\w+\b", gt_lower))
    ans_words = set(re.findall(r"\b\w+\b", ans_lower))

    if not gt_words:
        return False

    overlap = len(gt_words & ans_words) / len(gt_words)
    return overlap >= 0.6


def evaluate_backend(
    backend: MemoryBackend,
    questions: list[LongMemEvalQuestion],
    reader: Reader | None = None,
    k_values: list[int] | None = None,
    top_k_retrieval: int = 25,
    skip_answer: bool = False,
    progress_callback=None,
) -> EvaluationReport:
    """Run full evaluation of a backend on a set of questions.

    For each question:
    1. Reset backend
    2. Ingest all haystack sessions
    3. Search with the question
    4. Compute Recall@K
    5. (Optional) Use reader to answer, judge correctness

    Returns an EvaluationReport.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 25]

    results: list[QuestionResult] = []

    for idx, q in enumerate(questions):
        user_id = f"user_{q.question_id}"
        backend.reset(user_id=user_id)

        for session_idx, session in enumerate(q.haystack_sessions):
            session_id = q.haystack_session_ids[session_idx]
            backend.add(session, user_id=user_id, session_id=session_id)

        retrieved = backend.search(q.question, user_id=user_id, top_k=top_k_retrieval)

        recall = compute_recall_at_k(
            retrieved=retrieved,
            haystack_session_ids=q.haystack_session_ids,
            haystack_sessions=q.haystack_sessions,
            answer_session_ids=q.answer_session_ids,
            k_values=k_values,
        )

        result = QuestionResult(
            question_id=q.question_id,
            question_type=q.question_type,
            question=q.question,
            ground_truth_answer=q.answer,
            answer_session_ids=q.answer_session_ids,
            retrieved_memories=retrieved,
            recall_at_k=recall,
        )

        if reader and not skip_answer:
            answer = reader.answer(q.question, retrieved[:10])
            result.reader_answer = answer
            result.answer_correct = judge_answer(answer, q.answer)

        results.append(result)

        if progress_callback:
            progress_callback(idx + 1, len(questions))

    report = EvaluationReport(
        backend_name=backend.name,
        reader_name=reader.name if reader else "none",
        dataset_name="longmemeval-s",
        total_questions=len(results),
        results=results,
        config={
            "k_values": k_values,
            "top_k_retrieval": top_k_retrieval,
            "skip_answer": skip_answer,
        },
    )

    _aggregate(report, k_values)
    return report


def _aggregate(report: EvaluationReport, k_values: list[int]) -> None:
    """Compute aggregate metrics from per-question results."""
    by_type: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in report.results:
        by_type[r.question_type].append(r)

    for k in k_values:
        vals = [r.recall_at_k.get(k, 0.0) for r in report.results]
        report.aggregate_recall[k] = sum(vals) / len(vals) if vals else 0.0

    for qtype, group in by_type.items():
        report.aggregate_recall_by_type[qtype] = {}
        for k in k_values:
            vals = [r.recall_at_k.get(k, 0.0) for r in group]
            report.aggregate_recall_by_type[qtype][k] = sum(vals) / len(vals) if vals else 0.0

    judged = [r for r in report.results if r.answer_correct is not None]
    if judged:
        report.answer_accuracy = sum(1 for r in judged if r.answer_correct) / len(judged)
        for qtype, group in by_type.items():
            judged_group = [r for r in group if r.answer_correct is not None]
            if judged_group:
                report.answer_accuracy_by_type[qtype] = sum(1 for r in judged_group if r.answer_correct) / len(judged_group)
