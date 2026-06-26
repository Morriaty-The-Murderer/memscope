#!/usr/bin/env python3
"""memscope CLI — one-line reproducible agent memory evaluation.

Usage:
    memscope run --backends vector-baseline --dataset longmemeval-s
    memscope run --backends vector-baseline,mem0 --dataset longmemeval-s --reader openai
    memscope run --backends vector-baseline --num-questions 10 --skip-answer
"""

from __future__ import annotations

import argparse
import sys
import time

from .backends import VectorBaselineBackend
from .backends.base import MemoryBackend
from .datasets import load_longmemeval_s
from .evaluation import evaluate_backend
from .readers import OpenAIReader, Reader
from .report import report_to_json, report_to_markdown


def create_backend(name: str, **kwargs) -> MemoryBackend:
    if name == "vector-baseline":
        return VectorBaselineBackend(**kwargs)
    elif name == "mem0":
        from .backends.mem0_adapter import Mem0Backend
        return Mem0Backend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}. Available: vector-baseline, mem0")


def create_reader(name: str, **kwargs) -> Reader:
    if name == "openai":
        return OpenAIReader(**kwargs)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown reader: {name}. Available: openai, none")


def main():
    parser = argparse.ArgumentParser(
        prog="memscope",
        description="Lightweight, vendor-neutral agent memory evaluation harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run evaluation")
    run_parser.add_argument(
        "--backends",
        type=str,
        default="vector-baseline",
        help="Comma-separated backend names (vector-baseline, mem0)",
    )
    run_parser.add_argument(
        "--dataset",
        type=str,
        default="longmemeval-s",
        help="Dataset name (currently: longmemeval-s)",
    )
    run_parser.add_argument(
        "--reader",
        type=str,
        default="none",
        help="Reader for end-to-end accuracy (openai, none)",
    )
    run_parser.add_argument(
        "--reader-model",
        type=str,
        default="gpt-4o-mini",
        help="Model for the reader",
    )
    run_parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Limit number of questions (for quick testing)",
    )
    run_parser.add_argument(
        "--skip-answer",
        action="store_true",
        help="Skip end-to-end answer evaluation (retrieval only)",
    )
    run_parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Top-K memories to retrieve per question",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for reports",
    )
    run_parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model for vector-baseline backend",
    )

    args = parser.parse_args()

    if args.command == "run":
        _run_command(args)


def _run_command(args):
    print(f"Loading dataset: {args.dataset}")
    questions = load_longmemeval_s()

    if args.num_questions:
        questions = questions[: args.num_questions]
        print(f"Limited to {len(questions)} questions")

    print(f"Total questions: {len(questions)}")

    backend_names = [b.strip() for b in args.backends.split(",")]
    reader = create_reader(args.reader, model=args.reader_model) if args.reader != "none" else None

    reports = []

    for bname in backend_names:
        print(f"\n{'='*60}")
        print(f"Evaluating backend: {bname}")
        print(f"{'='*60}")

        backend_kwargs = {}
        if bname == "vector-baseline":
            backend_kwargs["model_name"] = args.embedding_model

        backend = create_backend(bname, **backend_kwargs)

        def progress(done, total):
            sys.stdout.write(f"\r  Progress: {done}/{total} ({done/total*100:.0f}%)")
            sys.stdout.flush()

        start = time.time()
        report = evaluate_backend(
            backend=backend,
            questions=questions,
            reader=reader,
            top_k_retrieval=args.top_k,
            skip_answer=args.skip_answer,
            progress_callback=progress,
        )
        elapsed = time.time() - start
        sys.stdout.write("\n")
        print(f"  Done in {elapsed:.1f}s")
        print(f"  Recall@5: {report.aggregate_recall.get(5, 0):.1%}")
        if not args.skip_answer and report.answer_accuracy > 0:
            print(f"  Answer accuracy: {report.answer_accuracy:.1%}")

        reports.append(report)

    print(f"\n{'='*60}")
    print("Generating reports...")
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    report_to_markdown(reports, f"{args.output_dir}/report.md")
    report_to_json(reports, f"{args.output_dir}/report.json")
    print(f"  Markdown: {args.output_dir}/report.md")
    print(f"  JSON:     {args.output_dir}/report.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
