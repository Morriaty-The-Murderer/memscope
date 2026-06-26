# memscope

> **Agent Memory 领域的 MLPerf —— 本地运行、无需 GPU、厂商中立的 Agent Memory 评测工具。**
> Compare Mem0, Zep, Letta, or your own backend on LongMemEval — fairly, reproducibly, on your laptop.

[English](#english) | 中文

---

## 为什么需要它

每个 Agent Memory 厂商都发布 benchmark 证明自己最强。问题在于：**同一个系统，谁在量，分数就差多少。**

Mem0 在 LongMemEval 上用 GPT-4o 的分数：

| 来源 | 分数 |
|------|------|
| Mem0 官方自报（2026 年 4 月） | **94.4%** |
| 独立第三方测试（2026 年 6 月） | **49.0%** |

同一个 benchmark、同一个模型，分数差 **45 个百分点**。这不是误差，是系统性的可复现性危机。差异来自 6 个变量：

1. **Judge prompt** — LLM-as-judge 的评判 prompt，措辞微调就能让分数波动两位数
2. **Dataset split** — 用 LongMemEval 500 题的哪个子集
3. **Retrieval depth** — 检索 top-5 还是 top-25
4. **Reader model** — 用哪个 LLM 消费检索结果生成答案
5. **Insertion semantics** — 对话如何分块喂给记忆后端
6. **Re-ranker** — 检索后是否做重排序

现有评测工具没有一个同时控制这 6 个变量。**memscope 控制全部。**

## 它做什么

```bash
memscope run --backends vector-baseline,mem0 --dataset longmemeval-s --reader openai
```

对 LongMemEval-S 的每一道题（500 题，每题约 53 轮对话）：

1. **Reset** — 清空后端，每题干净开始
2. **Ingest** — 将约 53 轮 haystack 对话喂入记忆后端
3. **Search** — 用问题做查询检索
4. **Score Recall@K** — top-K 检索结果是否命中 ground-truth 答案所在会话
5. **(可选) Answer** — reader LLM 用 top-10 记忆回答，评判正确性

输出：Markdown + JSON 对比报告，按后端和题目类型拆分。

## 快速开始

```bash
# Clone
git clone https://github.com/Morriaty-The-Murderer/memscope.git
cd memscope

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装
pip install -e .

# 运行（仅检索，无需 API key，CPU 约 10 分钟）
memscope run --backends vector-baseline --dataset longmemeval-s --skip-answer

# 运行端到端准确率（需要 OPENAI_API_KEY）
memscope run --backends vector-baseline --dataset longmemeval-s --reader openai

# 快速测试（仅 10 题）
memscope run --backends vector-baseline --num-questions 10 --skip-answer
```

### 添加 Mem0 后端

```bash
pip install -e ".[mem0]"

# 配置 Mem0（默认用 Qdrant，或用本地向量存储）
export OPENAI_API_KEY=your_key  # Mem0 用 LLM 做 extraction

memscope run --backends vector-baseline,mem0 --dataset longmemeval-s --skip-answer
```

## 后端

| 后端 | 安装 | 说明 |
|------|------|------|
| `vector-baseline` | 内置 | 纯向量相似度。embed 每轮对话，cosine 检索。无 LLM extraction，无 graph。地板线。 |
| `mem0` | `pip install mem0ai` | 包装 Mem0 的 `add()`/`search()` API。用 Mem0 内置 LLM extraction + 向量存储。 |

### 添加你自己的后端

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

然后传给 `evaluate_backend()`，或在 `cli.py` 中注册。

## 方法论：为什么你的分数和厂商报告不同

这是最重要的 section。如果你跑 memscope 得到的分数和厂商发布的不同，原因如下：

### 1. 我们不用 LLM judge 评判答案正确性

多数厂商 benchmark 用 GPT-4 做 LLM judge 评分，评判 prompt 通常不公开。同一个 Q&A 对，把 "Is the answer correct?" 改成 "Does the answer contain the key information?"，判定结果就能从"对"变"错"。

**memscope V1** 用确定性字符串匹配：ground-truth 答案必须作为子串出现，或与 reader 答案有 ≥60% 词重叠。比大多数 LLM judge 更严格——我们会报更低的准确率，但**可复现**。

**代价**：会误判一些语义正确但词面不同的答案（如 "BA" vs "Business Administration"）。已知限制。V2 会提供可选 LLM judge，但 prompt **公开且冻结**，让偏差至少可见。

### 2. Recall@K 是检索指标，不是答案准确率

Recall@K 测的是有没有检索到正确的记忆——不是有没有答对问题。一个后端可以 Recall@5 = 100% 但答案准确率 0%，如果 reader LLM 没能从检索结果中提取答案。

厂商 benchmark 经常把两者混为一谈。我们分开报。

### 3. 我们喂原始会话，不是预提取的 facts

有些 benchmark 预处理对话成 "facts" 再喂给记忆后端。这对 extraction-based 后端（如 Mem0）不公平——extraction 步骤被外部完成了。

**memscope** 喂原始对话 `[{role, content}, ...]` 给每个后端的 `add()` 方法。每个后端自己做 extraction（或不做）。这才是公平比较。

### 4. 每题之间 reset

每道题用干净后端。无跨题污染。有些 benchmark 跨题共享状态，会 inflate 依赖累积上下文的后端分数。

### 5. Embedding 模型固定且本地

vector-baseline 用 `all-MiniLM-L6-v2`（384 维，CPU，无需 API key）。故意选小的——这是地板线，不是天花板。如果你的后端用更好的 embedding，那是你后端的价值，不是 benchmark 的 artifact。

### 6. 按题型拆分报告

LongMemEval-S 有 6 种题型（single-session-user、multi-session、temporal-reasoning、knowledge-update 等）。总分会掩盖后端在哪类题上真正发光或翻车。一个后端总分 80% 但 temporal-reasoning 只有 30%——这才是有价值的洞察。

## 架构

```
memscope/
├── backends/
│   ├── base.py              # MemoryBackend ABC (add/search/reset)
│   ├── vector_baseline.py   # 纯向量相似度 baseline
│   └── mem0_adapter.py      # Mem0 适配器
├── datasets/
│   └── longmemeval.py       # LongMemEval-S loader (HF 自动下载)
├── evaluation/
│   └── metrics.py           # Recall@K + 答案准确率 pipeline
├── readers/
│   ├── base.py              # Reader ABC
│   └── openai_reader.py     # OpenAI 兼容 reader
├── report.py                # Markdown + JSON 报告生成
└── cli.py                   # CLI 入口
```

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌──────────────┐
│  Dataset     │────▶│  Memory Backend  │────▶│  Retriever    │────▶│  Recall@K    │
│ (LongMemEval)│     │ (pluggable:      │     │ (local embed  │     │ (per-type    │
│  500 Q       │     │  mem0 / custom)  │     │  + ranking)   │     │  breakdown)  │
└─────────────┘     └──────────────────┘     └───────────────┘     └──────────────┘
                                                     │
                                                     ▼                    V2 扩展层
                                            ┌──────────────────┐     ┌──────────────┐
                                            │  Agent Runner     │────▶│ Task Success │
                                            │ (with/without     │     │ Rate Δ       │
                                            │  memory)          │     │ (extrinsic)  │
                                            └──────────────────┘     └──────────────┘
```

V1 跑上半部分：数据集 → 记忆后端 → 检索 → Recall@K（按题型拆分）。V2 接 Agent Runner，对比有/无记忆的任务成功率差异。两层共享 MemoryBackend 接口，V2 是 V1 的自然扩展。

## 路线图

- [x] **v0.1** — V1 检索评测：LongMemEval-S + 确定性评判 + Recall@K + 可插拔后端
- [ ] **v0.2** — 补充 LoCoMo / BEAM 数据集 + 更多 baseline + Zep/Letta 适配器
- [ ] **v0.3** — V2 任务成功率评测：Agent Runner 接口 + 有/无记忆对比
- [ ] **v0.4** — 可选 LLM judge（公开冻结 prompt）+ 社区贡献的后端适配

> V2 是核心差异化。当前领域的问题不是"能不能测检索"，而是"检索好不等于任务好"——MemoryArena (2026) 发现 LoCoMo 上接近满分的 Agent 在多轮依赖任务上成功率骤降。memscope V2 直接回答这个问题。

## 环境要求

- Python ≥ 3.10
- 无需 GPU（CPU 推理 all-MiniLM-L6-v2）
- 约 500MB 磁盘（数据集缓存）
- 可选：`OPENAI_API_KEY`（端到端准确率评测或 Mem0 后端需要）

## 贡献

欢迎贡献数据集适配、记忆后端插件、benchmark 结果复现。见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

MIT
