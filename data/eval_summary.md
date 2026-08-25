
### 📊 Empirical Evaluation Benchmark Results

| Metric | Baseline RAG (Single-Pass) | Self-Healing RAG (With Critic Loop) | Improvement |
| :--- | :---: | :---: | :---: |
| **Total Test Queries** | 15 | 15 | — |
| **Out-of-Domain / Unsupported Queries** | 5 | 5 | — |
| **Hallucination Rate (Out-of-Domain)** | **100.0%** | **0.0%** | **-100.0% Reduction** |
| **Honest Fallback Rate** | 0.0% | **100.0%** | **+100.0% Accuracy** |
| **Average Latency per Query** | 0.02s | 0.01s | +-0.01s (Critic overhead) |

> [!NOTE]
> Evaluation run on 15 test cases (10 in-domain answerable questions, 5 out-of-domain trick questions).
