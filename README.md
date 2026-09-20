# Spark-4B: Specialized Micro-Pipeline for Zero-Regex Task Progress Guard

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-ee4c2c.svg)](https://pytorch.org/)
[![Parameter Scale](https://img.shields.io/badge/Model%20Size-%3C400M-success.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-Intel%20XPU%20%7C%20CUDA%20%7C%20CPU-blue.svg)]()

> **TL;DR**: AI Agents frequently make "premature exits" — declaring a task finished when it is only partially done, distracted by superficial keywords, or trapped in deadlocks. 
> 
> We demonstrate that **large LLMs are neither necessary nor optimal for task-closure verification**. Instead, a **Zero-Regex Two-Stage Specialized Micro-Pipeline (<400M total parameters)** decoupling **Pragmatic Speech-Act Classification (287M)** from **Contrastively Aligned Object Binding (117M)** with an **Incremental Entity Coverage State Machine** achieves **100% defense against cross-topic distractions, 100% interception of premature multi-intent exits, and 100% deadlock-free task cancellations** with **~70ms latency** on consumer edge hardware.

---

## 1. The Real Problems: Why AI Agent Progress Fails

In real-world engineering environments (e.g., automated coding and debugging), software agents fail in three systematic ways when attempting to judge their own progress:

```
❌ 1. Cross-Topic Distraction (False Positives)
   User Request:     "Optimize MySQL query performance on large tables."
   Agent Response:   "Refactored the frontend navigation bar CSS. All done!"
   Naive Classifier: Detects "All done!" -> Incorrectly clears the database task.

❌ 2. Multi-Intent Premature Exit (Partial Completion Leak)
   User Request:     "Change port to 9090 AND implement JWT auth middleware."
   Agent Response:   "Updated port to 9090 in server.py."
   Dense Embedding:  Local similarity pulls score > 0.65 -> Prematurely clears both tasks, 
                     permanently leaking the unfulfilled JWT requirement!

❌ 3. Cancellation Deadlock (Stale Blocking)
   User Command:     "Forget the JWT requirement, let's keep the existing auth."
   Naive State Machine: Does not recognize "abandon" as a legitimate release -> 
                        Task stays blocked forever, preventing the agent from returning.
```

Historically, developers attempted to patch these defects with **monolithic regular expression suites** (`INJECT_RE`, `VERIFY_RE`, commit hash regexes, token splitters). This leads to severe brittle failure modes: any phrasing variance or cross-lingual term causes either false interception or complete leakage.

---

## 2. The Solution: Two-Stage Specialized Micro-Pipeline

We replace hand-crafted regular expressions with a decoupled, specialized model pipeline where **each model focuses 100% of its capacity on a single orthogonal sub-problem**:

```
                              Incoming Dialogue Block
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │ Station 1: Pragmatic Role Classifier (287M)   │
                 │ Focus: Speech-act intent only, 0% topic bias  │
                 │ Output: REQUEST / CLOSE / PARTIAL / DROPPED   │
                 └───────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
          [ If REQUEST ]                                   [ If CLOSE ]
    Extract Target Entities                        Station 2: Dense Aligner (117M)
    & Register in Monotonic Ledger                 & Entity Coverage Verification
                 │                                               │
                 │                                               ▼
                 │                                 ┌───────────────────────────┐
                 │                                 │ Check Object Similarity   │
                 │                                 │   AND Entity Coverage:    │
                 │                                 │     |Covered| / |Target|  │
                 │                                 └─────────────┬─────────────┘
                 │                                               │
                 │                     ┌─────────────────────────┴─────────────────────────┐
                 │                     ▼                                                   ▼
                 │             [ Coverage < 100% ]                                 [ Coverage = 100% ]
                 │             Downgrade to PARTIAL                                 Resolve & Dismiss
                 │             Keep Blocking Agent                                 Safe Exit Allowed
                 └────────────────────►│◄──────────────────────────────────────────────────┘
```

### Architectural Pillars
1. **Station 1: Pragmatic Role Classification (287M mDeBERTa-v3)**:
   - Specializes strictly in speech-act roles (`REQUEST`, `CLOSE`, `PARTIAL`, `DROPPED`, `NEUTRAL`).
   - Eliminates all hardcoded question marks, polite phrasing lists, and negation assertion regexes.
2. **Station 2: Contrastively Aligned Object Binding (117M MiniLM-L12)**:
   - Fine-tuned via triplet contrastive learning on strictly held-out engineering pairs (Redis, Kafka, Prometheus, WASM, etc.).
   - Maps user target objects and completion artifacts into a shared vector space, driving cross-topic distraction similarities into deep negative values (**mean -0.1432**).
3. **Incremental Entity Coverage State Machine**:
   - Deconstructs multi-intent requests ($A + B$) into required semantic entities.
   - Enforces dual criteria: `Dense Similarity >= tau` **AND** `Entity Coverage == 100%`.
   - Partial completion ($A$ only) automatically downgrades to `PARTIAL`, retaining the block until $B$ is satisfied across subsequent turns.
4. **Deadlock-Free Abandonment (`DROPPED`)**:
   - Safely de-registers tasks when users explicitly abandon them without inflating completion metrics.

---

## 3. Empirical Benchmarks & Comparisons

### A. All-Scenario Engineering Benchmark (20 Critical Real-World Traps)
Evaluated on our balanced benchmark encompassing exact completion, cross-topic distractions, in-progress investigations, assertion failures, and implicit refactorings:

| Model / Architecture | Parameters | Latency (XPU) | Cross-Topic Defense | Premature Exit Defense | Benchmark Acc |
|---|---|---|---|---|---|
| Single 287M End-to-End | 287M | 113.9 ms | 25.0% (3/4 False Positives) | 0.0% (Bypassed) | 70.0% |
| BAAI/bge-reranker-base | 278M | 21.2 ms | 100.0% | 0.0% (Bypassed) | 60.0% |
| Zero-shot Qwen2.5-1.5B | 1500M | 394.7 ms | 100.0% | 50.0% | 90.0% |
| **Specialized Pipeline (Ours)** | **404M** | **71.2 ms** | **100.0%** | **100.0%** | **95.0%** |

### B. Adversarial Stress Suite (12 Unit Tests)
Our stress suite (`pipeline/test_entity_coverage_guard.py`) validates edge boundaries:
- **Premature Exit Interception**: **4 / 4 (100.0%)** — Sub-task completion ($A$ only) is strictly caught and held as `PARTIAL`.
- **Full Resolution Release**: **3 / 3 (100.0%)** — Monolithic and multi-turn incremental fulfillment ($A \to B$) cleanly closes.
- **Deadlock-Free Abandonment**: **3 / 3 (100.0%)** — Dynamic user cancellations release the ledger cleanly.
- **Cross-Topic Rejection**: **2 / 2 (100.0%)** — Distractions and explicit execution refusals are blocked.

### C. Subjective Playback on Production Dialogue Trace (138 Blocks)
Tested on full session logs from `pi-dag-core` (state machine, evidence bugs, and git commits):
- Intercepted **10+ premature closure claims** during intermediate debugging.
- Cleanly resolved 3 long-distance conversational intents (`Q2: State machine`, `Q3: Workflow IPC`, `Q5: Retry architecture`) with high margin ($> +0.38$).
- Correctly retained and blocked on unfulfilled objectives at session termination with **zero handcoded regex rules**.

---

## 4. Quick Start

### Installation & Environment
```bash
git clone https://github.com/Nuctori/spark-4b.git
cd spark-4b

# Requirements: Python 3.10+, PyTorch 2.1+, HuggingFace Transformers, GLiNER2
pip install -r requirements.txt
```

### Run the Specialized Micro-Pipeline
```python
from pipeline.entity_coverage_guard import EntityCoverageGuard

# Initializes both 287M pragmatic model and 117M alignment encoder
guard = EntityCoverageGuard(device="cpu") # or "xpu" / "cuda"

# 1. User registers a multi-intent request
guard.step(role="user", text="把服务端口修改为 9090，并且新增 JWT 鉴权中间件")
# State: REGISTERED (Target entities: ['9090', 'JWT'])

# 2. Assistant only completes task A
action, is_blocked = guard.step(role="assistant", text="已在 server.py 中将端口成功修改为 9090")
# Result: PARTIAL (Covered: ['9090'], Missing: ['JWT']) -> is_blocked = True!

# 3. Assistant completes task B
action, is_blocked = guard.step(role="assistant", text="新增了 auth_middleware.py 实现了 JWT 校验")
# Result: RESOLVED (Coverage: 100%) -> is_blocked = False (Safe to return)
```

### Reproduce Benchmarks
```bash
# 1. Run all 20-scenario engineering benchmark
python pipeline/eval_semantic_matcher_benchmark.py

# 2. Run adversarial stress & regression suite
python pipeline/test_entity_coverage_guard.py

# 3. Run production session subjective playback
python pipeline/run_dag_session_subjective_eval_clean.py
```

---

## 5. Scientific Exclusion Chain Archive

Before arriving at the decoupled micro-pipeline, we conducted extensive causal exclusion experiments. For full reproducibility, failure logs and causal ablations are preserved under `results/`:

1. **Backbone Replacement (Route 6 / 6b)**: Replacing full attention with linear attention (`fla + causal_conv1d` or `Qwen3.5-0.8B`) fell significantly below the Pareto frontier ($\Delta = -5.6$ to $-10.0$ residual).
2. **Two-Stage End-to-End Training (Stage 1 + 2)**: Forcing an end-to-end 287M model to judge obligation satisfaction caused severe recall collapse (recall dropped from $0.640$ to $0.400$).
3. **Convention Dependence (§218)**: Changing prompt conventions swinging positive rates from $6\%$ to $73.5\%$ proved that single-step holistic closure judgment is subjective and ill-posed.

See [`results/DISCIPLINE_FAILURE.md`](results/DISCIPLINE_FAILURE.md) and [`results/MEASUREMENT_AUDIT.md`](results/MEASUREMENT_AUDIT.md) for full audit reports.

---

## Citation

```bibtex
@software{spark4b_progress_guard_2026,
  author = {Nuctori},
  title = {Spark-4B: Specialized Micro-Pipeline for Zero-Regex Task Progress Guard},
  year = {2026},
  url = {https://github.com/Nuctori/spark-4b}
}
```

## License
Apache License 2.0. See [LICENSE](LICENSE) for details.
