# memscope Design Document

> V1 implementation + V2 roadmap

## V1 Architecture (Implemented)

### Core Abstraction: MemoryBackend

The interface takes raw conversation messages `[{role, content}, ...]`, not pre-extracted facts. This ensures fair comparison — each backend does its own extraction (or not). Backends that rely on external extraction (like Mem0 with LLM) bear that cost themselves.

### Evaluation Pipeline

For each question Q in dataset:
1. `backend.reset(user_id=Q.id)` — clean slate
2. For each session S in Q.haystack_sessions: `backend.add(S, user_id=Q.id, session_id=S.id)`
3. `retrieved = backend.search(Q.question, user_id=Q.id, top_k=25)`
4. `recall_at_k = compute_recall(retrieved, Q.answer_session_ids)`
5. (optional) `answer = reader.answer(Q.question, retrieved[:10])`
6. (optional) `correct = judge_answer(answer, Q.ground_truth)`

### Recall@K Computation

For LongMemEval-S, recall is binary per question: did ANY of the top-K retrieved memories originate from a ground-truth answer session?

Since backends return text (not session IDs), we match by content overlap:
- Exact substring match (case-insensitive)
- Fuzzy overlap (50+ char shared substring)

This is intentionally strict. A retrieval counts as a hit only if the actual answer-session content surfaces in the top-K.

### Answer Judge

V1 uses deterministic string matching (no LLM judge):
1. Ground truth as substring of reader answer (case-insensitive)
2. ≥60% word overlap

**Why not LLM judge?** The README documents this in detail. Short version: LLM judge prompts are the #1 source of the 45-point score gap between vendor self-reports and independent tests. A deterministic judge is reproducible.

## V2 Roadmap: Task-Success Evaluation

### The Problem

Current memory benchmarks (including V1) measure **retrieval quality** — can the right memory be found? But the user's insight is correct: what matters in production is **task-success lift** — does having memory make the agent better at its job?

A backend can have 95% Recall@K but still hurt agent performance if:
- Retrieved memories are redundant (waste context tokens)
- Memories are poorly formatted (confuse the reader)
- The backend's extraction loses critical nuance
- Temporal ordering is lost (wrong fact surfaces for the current time)

### Proposed V2 Interface Extension

V2 adds `get_context_for_agent(self, query, user_id, max_tokens) -> str` to `MemoryBackend`. Backends control formatting within a token budget — this is where extraction quality and formatting matter beyond raw retrieval.

### V2 Evaluation Pipeline

**Phase 1: Baseline (no memory)**
- For each task T in task_suite: `agent.run(T, memory=None)` → success/fail

**Phase 2: With memory backend B**
- For each task T: pre-populate memory with T.history, then `agent.run(T, memory=B)` → success/fail

**Phase 3: Measure**
- `task_success_lift = P(success|memory) - P(success|no memory)`

### Task Suite Candidates

| Suite | Type | Why |
|-------|------|-----|
| WebArena | Web agent | Standard, multi-step, requires state tracking |
| Terminal-Bench | Terminal tasks | Long-running, state-heavy, Letta already benchmarks here |
| Custom QA suite | Conversation | Cheapest, reuses LongMemEval questions as tasks |

### V2 Design Constraints

1. **Agent-agnostic**: The task harness must work with any agent framework (LangGraph, CrewAI, raw API). Define an `AgentRunner` interface.
2. **Cost control**: Full task suites are expensive (many LLM calls). Support subsampling and stratified evaluation.
3. **Statistical rigor**: With small task samples, need confidence intervals, not point estimates.
4. **Reproducibility**: Pin agent prompts, model versions, and task seeds.

### V2 Milestones

1. Define `AgentRunner` + `TaskSuite` interfaces
2. Implement WebArena adapter (or custom conversation task suite)
3. Run baseline (no-memory) evaluation
4. Run with-memory evaluation for each backend
5. Report task-success lift with confidence intervals

## V1 Limitations (Documented)

1. **Answer judge is strict**: String matching will mark semantically-correct but lexically-different answers as wrong. Known limitation.
2. **Recall matching is approximate**: Content overlap is a proxy for session-ID matching. Very long sessions with overlapping content may cause false positives.
3. **Single dataset**: V1 only supports LongMemEval-S. LoCoMo, BEAM, LongMemEval-V2 are planned.
4. **No streaming**: Backends that support streaming retrieval can't showcase that advantage.
5. **No temporal evaluation**: V1 doesn't test point-in-time queries ("what did the user prefer in Q1?"). LongMemEval-V2 and Zep's strength is here.
