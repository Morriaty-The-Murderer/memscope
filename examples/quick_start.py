#!/usr/bin/env python3
"""Quick start example: evaluate vector-baseline on 5 mixed-type questions."""

from memscope.backends import VectorBaselineBackend
from memscope.datasets import load_longmemeval_s
from memscope.evaluation import evaluate_backend
from memscope.report import report_to_markdown

# Load dataset (auto-downloads from HuggingFace on first run)
questions = load_longmemeval_s()

# Sample one question of each type for a quick mixed test
seen_types = set()
sampled = []
for q in questions:
    if q.question_type not in seen_types:
        seen_types.add(q.question_type)
        sampled.append(q)
    if len(sampled) >= 6:
        break

print(f"Testing with {len(sampled)} questions, types: {seen_types}")

# Create backend
backend = VectorBaselineBackend(model_name="all-MiniLM-L6-v2")

# Run evaluation (retrieval only, no reader)
report = evaluate_backend(
    backend=backend,
    questions=sampled,
    skip_answer=True,
    progress_callback=lambda done, total: print(f"  {done}/{total}"),
)

# Print results
print(f"\nBackend: {report.backend_name}")
print(f"Recall@5: {report.aggregate_recall.get(5, 0):.1%}")
for qtype, recall in report.aggregate_recall_by_type.items():
    print(f"  {qtype}: R@5={recall.get(5, 0):.1%}")

# Save report
report_to_markdown([report], "example_report.md")
print("\nReport saved to example_report.md")
