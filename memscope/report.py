from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluation.metrics import EvaluationReport


def report_to_markdown(reports: list[EvaluationReport], output_path: str | Path) -> None:
    """Generate a comparison markdown report from multiple backend evaluations."""
    lines = [
        "# memscope Evaluation Report",
        "",
        f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"> Dataset: {reports[0].dataset_name if reports else 'N/A'}",
        f"> Questions: {reports[0].total_questions if reports else 0}",
        "",
    ]

    lines.append("## Summary: Recall@K")
    lines.append("")
    header = "| Backend | R@1 | R@3 | R@5 | R@10 | R@25 |"
    sep = "|---------|-----|-----|-----|------|------|"
    lines.append(header)
    lines.append(sep)
    for r in reports:
        recall = r.aggregate_recall
        row = f"| {r.backend_name} |"
        for k in [1, 3, 5, 10, 25]:
            val = recall.get(k, 0.0)
            row += f" {val:.1%} |"
        lines.append(row)
    lines.append("")

    lines.append("## Summary: End-to-End Answer Accuracy")
    lines.append("")
    lines.append("| Backend | Reader | Answer Accuracy |")
    lines.append("|---------|--------|-----------------|")
    for r in reports:
        acc = f"{r.answer_accuracy:.1%}" if r.answer_accuracy > 0 else "N/A"
        lines.append(f"| {r.backend_name} | {r.reader_name} | {acc} |")
    lines.append("")

    for r in reports:
        lines.append(f"## {r.backend_name}: Recall@K by Question Type")
        lines.append("")
        lines.append("| Question Type | R@1 | R@5 | R@10 | R@25 | Count |")
        lines.append("|---------------|-----|-----|------|------|-------|")
        type_counts: dict[str, int] = {}
        for res in r.results:
            type_counts[res.question_type] = type_counts.get(res.question_type, 0) + 1
        for qtype in sorted(r.aggregate_recall_by_type.keys()):
            recall = r.aggregate_recall_by_type[qtype]
            count = type_counts.get(qtype, 0)
            row = f"| {qtype} |"
            for k in [1, 5, 10, 25]:
                row += f" {recall.get(k, 0.0):.1%} |"
            row += f" {count} |"
            lines.append(row)
        lines.append("")

    lines.append("## Configuration")
    lines.append("")
    for r in reports:
        lines.append(f"### {r.backend_name}")
        lines.append(f"```json")
        lines.append(json.dumps(r.config, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("- **Recall@K**: For each question, all ~53 haystack sessions are ingested into the backend,")
    lines.append("  then the question is used as the search query. Recall@K = 1 if any of the top-K retrieved")
    lines.append("  memories originate from a ground-truth answer session, 0 otherwise.")
    lines.append("- **Answer Accuracy**: A reader LLM answers the question using the top-10 retrieved memories.")
    lines.append("  Correctness is judged by substring/word-overlap match against the ground-truth answer.")
    lines.append("  This is deliberately **not** an LLM-judge — see README for why.")
    lines.append("- **Why your scores differ from vendor reports**: See README.md → 'Methodology'.")
    lines.append("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def report_to_json(reports: list[EvaluationReport], output_path: str | Path) -> None:
    """Dump raw results as JSON for programmatic comparison."""
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": [r.to_dict() for r in reports],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
