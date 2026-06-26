# Contributing to memscope

Thank you for your interest in improving memscope.

## What's most useful right now

- **New memory backend adapters** — Zep, Letta, your custom backend
- **New dataset adapters** — LoCoMo, BEAM, LongMemEval-V2
- **Bug reports** — unexpected results, incorrect Recall@K, dataset loading failures
- **Reproducibility reports** — run the benchmark and share your results

## Adding a backend

1. Subclass `MemoryBackend` from `memscope/backends/base.py`
2. Implement `add()`, `search()`, and `reset()`
3. Add it to `memscope/backends/__init__.py`
4. Register it in `memscope/cli.py` under `BACKEND_REGISTRY`
5. Add a smoke test in `tests/`

See `memscope/backends/vector_baseline.py` for a minimal example.

## Adding a dataset

1. Implement a loader that returns `list[Question]` (see `memscope/datasets/longmemeval.py`)
2. Register it in `memscope/datasets/__init__.py`
3. Add it to `memscope/cli.py` under `DATASET_REGISTRY`

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Run `python -m pytest tests/` before submitting
- If you're adding a backend or dataset, include a short note on how you tested it

## Issues

Open a GitHub issue for bugs, dataset requests, or backend requests. Include:
- Your OS and Python version
- The command you ran
- The full error output

## Code style

- Python 3.10+, type hints preferred
- No external formatters required — just keep it readable
