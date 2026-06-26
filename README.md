# memscope

> Lightweight, vendor-neutral agent memory evaluation harness.
> Compare Mem0, Zep, Letta, or your own backend on LongMemEval — fairly, reproducibly, on your laptop.

## Why does this exist?

Every agent memory vendor publishes benchmark scores showing they're the best. The problem: **the same system gets wildly different scores depending on who's measuring**.

Example: Mem0 on LongMemEval with GPT-4o:

| Source | Score |
|--------|-------|
| Mem0 official self-report (April 2026) | **94.4%** |
| Independent third-party test (June 2026) | **49.0%** |

That's a **45-percentage-point gap** on the same benchmark, same model. This isn't a rounding error — it's a systemic reproducibility crisis. The gap comes from differences in:

1. **Judge prompt** — the LLM-as-judge prompt that scores answer correctness. Small wording changes swing scores by double digits.
2. **Dataset split** — which subset of LongMemEval's 500 questions are used.
3. **Retrieval depth** — how many memories are retrieved before answering (top-5 vs top-25).
4. **Reader model** — which LLM consumes the retrieved memories to produce an answer.
5. **Insertion semantics** — how conversation sessions are chunked and fed to the memory backend.
6. **Re-ranker** — whether a re-ranking step is applied after retrieval.

No existing harness controls for all of these variables. **memscope does.**

## What it does

```
memscope run --backends vector-baseline,mem0 --dataset longmemeval-s --reader openai
```

For each question in LongMemEval-S (500 questions, each with ~53 conversation sessions):

1. **Reset** the backend (clean slate per question)
2. **Ingest** all ~53 haystack sessions into the memory backend
3. **Search** using the question as query
4. **Score Recall@K** — did the top-K retrieved memories come from the ground-truth answer sessions?
5. **(Optional) Answer** — a reader LLM answers using top-10 memories, judged for correctness

Output: a Markdown + JSON comparison report with per-backend and per-question-type breakdowns.

## Quick Start

```bash
# Clone
git clone https://github.com/yourname/memscope.git
cd memscope

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Run (retrieval-only, no API key needed, ~10 min on CPU)
memscope run --backends vector-baseline --dataset longmemeval-s --skip-answer

# Run with end-to-end accuracy (requires OPENAI_API_KEY)
memscope run --backends vector-baseline --dataset longmemeval-s --reader openai

# Quick test (10 questions only)
memscope run --backends vector-baseline --num-questions 10 --skip-answer
```

### Adding Mem0 as a backend

```bash
pip install -e ".[mem0]"

# Configure Mem0 (uses Qdrant by default, or set up local vector store)
export OPENAI_API_KEY=your_key  # Mem0 uses LLM for extraction

memscope run --backends vector-baseline,mem0 --dataset longmemeval-s --skip-answer
```

## Backends

| Backend | Package | Description |
|---------|---------|-------------|
| `vector-baseline` | built-in | Pure vector similarity. Embeds each user turn, retrieves by cosine. No LLM extraction, no graph. Sanity-check floor. |
| `mem0` | `pip install mem0ai` | Wraps Mem0's `add()`/`search()` API. Uses Mem0's internal LLM extraction + vector store. |

### Adding your own backend

```python
from memscope.backends.base import MemoryBackend

class MyBackend(MemoryBackend):
    name = "my-backend"

    def add(self, messages, user_id="default", session_id=None):
        # messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        ...

    def search(self, query, user_id="default", top_k=10) -> list[str]:
        # Return top_k memory strings relevant to query
        ...
```

Then pass it to `evaluate_backend()` directly, or register it in `cli.py`.

## Methodology: Why your scores differ from vendor reports

This is the most important section. If you run memscope and get a score that differs from a vendor's published number, here's why:

### 1. We don't use an LLM judge for answer correctness

Most vendor benchmarks use GPT-4 as an LLM judge to score answer correctness. The judge prompt is typically not published. We've seen the same Q&A pair scored as "correct" by one judge prompt and "incorrect" by another, simply by changing "Is the answer correct?" to "Does the answer contain the key information?".

**memscope V1** uses a deterministic string-matching judge: the ground-truth answer must appear as a substring, or share ≥60% word overlap with the reader's answer. This is stricter than most LLM judges — we'll report lower accuracy scores, but they're **reproducible**.

**Trade-off**: Our judge will mark some semantically-correct answers as wrong (e.g., "BA" vs "Business Administration"). This is a known limitation. V2 will offer an optional LLM judge with a **published, frozen prompt** so at least the bias is documented.

### 2. Recall@K is retrieval-only, not answer accuracy

Recall@K measures whether the *right memory* was retrieved — not whether the *right answer* was produced. A backend can have 100% Recall@5 but 0% answer accuracy if the reader LLM fails to extract the answer from the retrieved context.

Vendor benchmarks often conflate these. We report them separately.

### 3. We ingest full sessions, not pre-extracted facts

Some benchmarks pre-process conversations into "facts" before feeding them to memory backends. This gives extraction-based backends (like Mem0) an unfair advantage because the extraction step is done externally.

**memscope** feeds raw conversation sessions `[{role, content}, ...]` to each backend's `add()` method. Each backend does its own extraction (or not). This is the fair comparison.

### 4. We reset between questions

Each question gets a fresh backend. No cross-question contamination. Some benchmarks share state across questions, which inflates scores for backends that benefit from accumulated context.

### 5. Embedding model is fixed and local

We use `all-MiniLM-L6-v2` (384-dim, CPU, no API key) for the vector-baseline backend. This is deliberately small — it's the floor, not the ceiling. If your backend uses a better embedding model, that's part of your backend's value proposition, not a benchmark artifact.

### 6. We report per-question-type breakdowns

LongMemEval-S has 6 question types (single-session-user, multi-session, temporal-reasoning, knowledge-update, etc.). Aggregate scores hide where backends actually shine or fail. A backend might score 80% overall but 30% on temporal-reasoning — that's the insight that matters.

## V2 Roadmap

### Task-success evaluation (user-requested)

**The gap**: Current memory benchmarks measure *can you retrieve the right memory?* — not *does having memory make the agent better at its task?* A backend with 95% Recall@K might not improve agent task success if the retrieved memories are redundant, poorly formatted, or cause the reader to hallucinate.

**V2 plan**: Add `evaluate_task_success()` to the evaluation pipeline:
1. Run an agent on a task suite **without** memory (baseline)
2. Run the same agent **with** each memory backend
3. Measure delta in task success rate

This requires a task harness (e.g., WebArena, Terminal-Bench) and is substantially more complex. The `MemoryBackend` interface already reserves space for this via the `evaluate_task_success` method signature in the V2 roadmap.

### Additional backends
- Zep / Graphiti adapter
- Letta adapter
- RAG baseline (BM25-only, no embeddings)

### Additional datasets
- LongMemEval-V2 (100M token web agent trajectories)
- LoCoMo (1,540 questions, multi-hop + temporal)
- BEAM (1M/10M token scale)

### Optional LLM judge with frozen prompt
A documented, version-controlled judge prompt that users can optionally enable. The prompt will be published in the repo and frozen per release, so bias is at least visible.

## Architecture

```
memscope/
├── backends/
│   ├── base.py              # MemoryBackend ABC (add/search/reset)
│   ├── vector_baseline.py   # Pure vector similarity baseline
│   └── mem0_adapter.py      # Mem0 adapter
├── datasets/
│   └── longmemeval.py       # LongMemEval-S loader (HF auto-download)
├── evaluation/
│   └── metrics.py           # Recall@K + answer accuracy pipeline
├── readers/
│   ├── base.py              # Reader ABC
│   └── openai_reader.py     # OpenAI-compatible reader
├── report.py                # Markdown + JSON report generation
└── cli.py                   # CLI entry point
```

## Requirements

- Python ≥ 3.10
- No GPU required (CPU inference with all-MiniLM-L6-v2)
- ~500MB disk for dataset cache
- Optional: `OPENAI_API_KEY` for end-to-end accuracy evaluation or Mem0 backend

## License

MIT
