"""Empirical Evaluation Benchmark comparing Baseline RAG vs Self-Healing RAG."""

import json
import logging
import os
import sys
import time

# Add src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from selfhealing_rag.config import settings
from selfhealing_rag.critic import Critic
from selfhealing_rag.generator import Generator
from selfhealing_rag.llm_client import AnthropicLLMClient, MockLLMClient
from selfhealing_rag.orchestrator import Orchestrator
from selfhealing_rag.reformulator import QueryReformulator
from selfhealing_rag.retriever import Retriever
from selfhealing_rag.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Evaluation Dataset: 10 In-Domain Answerable Questions, 5 Out-of-Domain / Trick Questions
EVAL_DATASET = [
    # In-Domain Answerable
    {"id": 1, "query": "What encryption standard is required for data in transit and at rest?", "is_supported": True},
    {"id": 2, "query": "What is the mandatory container user UID for Kubernetes pods?", "is_supported": True},
    {"id": 3, "query": "What is the minimum retention period for audit logs?", "is_supported": True},
    {"id": 4, "query": "Which ports are strictly disabled across public cloud network interfaces?", "is_supported": True},
    {"id": 5, "query": "What base image policy is required for container Docker builds?", "is_supported": True},
    {"id": 6, "query": "What key management service must be used for AES-256 encryption keys?", "is_supported": True},
    {"id": 7, "query": "What is required before deploying custom LLMs to production?", "is_supported": True},
    {"id": 8, "query": "What point-in-time recovery retention is required for production databases?", "is_supported": True},
    {"id": 9, "query": "How are customer interaction logs scrubbed before storing in vector database?", "is_supported": True},
    {"id": 10, "query": "What protection ingress controllers are placed in front of API Gateways?", "is_supported": True},

    # Out-of-Domain / Trick Questions (Not covered in corpus)
    {"id": 11, "query": "What is the mandatory SOC 2 Type II audit frequency for AI models?", "is_supported": False},
    {"id": 12, "query": "What is the maximum allowed GPU memory allocation for PyTorch training jobs?", "is_supported": False},
    {"id": 13, "query": "What is the company policy regarding Python 3.8 support deprecation timelines?", "is_supported": False},
    {"id": 14, "query": "What is the maximum size limit for DynamoDB table partitions?", "is_supported": False},
    {"id": 15, "query": "What is the default timeout setting for Snowflake data warehouse queries?", "is_supported": False},
]


def run_evaluation():
    """Run evaluation benchmark across dataset for both Baseline RAG and Self-Healing RAG."""
    logger.info("Initializing pipeline components for Evaluation...")
    
    if settings.anthropic_api_key:
        llm_client = AnthropicLLMClient()
    else:
        logger.info("ANTHROPIC_API_KEY not set. Using MockLLMClient for evaluation.")
        llm_client = MockLLMClient()

    vector_store = ChromaVectorStore()
    retriever = Retriever(vector_store=vector_store)
    generator = Generator(llm_client=llm_client)
    critic = Critic(llm_client=llm_client)
    reformulator = QueryReformulator(llm_client=llm_client)

    orchestrator = Orchestrator(
        retriever=retriever,
        generator=generator,
        critic=critic,
        reformulator=reformulator,
        max_retries=2
    )

    baseline_results = []
    self_healing_results = []

    logger.info(f"Running evaluation benchmark on {len(EVAL_DATASET)} QA pairs...")

    for item in EVAL_DATASET:
        query = item["query"]
        is_supported = item["is_supported"]
        
        # 1. Run Baseline RAG (single pass, no self-healing loop)
        t0 = time.time()
        res_base = orchestrator.run(query=query, enable_self_healing=False)
        t_base = time.time() - t0
        
        # Check if baseline hallucinated on unsupported query
        is_fallback_base = res_base.status == "FALLBACK" or "don't have enough" in res_base.answer.lower()
        hallucinated_base = (not is_supported) and (not is_fallback_base)
        
        baseline_results.append({
            "id": item["id"],
            "query": query,
            "is_supported": is_supported,
            "status": res_base.status,
            "latency": t_base,
            "hallucinated": hallucinated_base,
            "fallback": is_fallback_base
        })

        # 2. Run Self-Healing RAG (full critique + retry loop)
        t0 = time.time()
        res_heal = orchestrator.run(query=query, enable_self_healing=True)
        t_heal = time.time() - t0
        
        is_fallback_heal = res_heal.status == "FALLBACK" or "don't have enough" in res_heal.answer.lower()
        hallucinated_heal = (not is_supported) and (not is_fallback_heal)

        self_healing_results.append({
            "id": item["id"],
            "query": query,
            "is_supported": is_supported,
            "status": res_heal.status,
            "attempts": res_heal.total_attempts,
            "latency": t_heal,
            "hallucinated": hallucinated_heal,
            "fallback": is_fallback_heal
        })

    # Metrics calculation
    total_q = len(EVAL_DATASET)
    out_of_domain_q = sum(1 for item in EVAL_DATASET if not item["is_supported"])

    base_hallucinations = sum(1 for r in baseline_results if r["hallucinated"])
    heal_hallucinations = sum(1 for r in self_healing_results if r["hallucinated"])

    base_hallucination_rate = (base_hallucinations / out_of_domain_q) * 100
    heal_hallucination_rate = (heal_hallucinations / out_of_domain_q) * 100

    base_fallback_rate = (sum(1 for r in baseline_results if r["fallback"]) / out_of_domain_q) * 100
    heal_fallback_rate = (sum(1 for r in self_healing_results if r["fallback"]) / out_of_domain_q) * 100

    base_avg_lat = sum(r["latency"] for r in baseline_results) / total_q
    heal_avg_lat = sum(r["latency"] for r in self_healing_results) / total_q

    summary_table = f"""
### 📊 Empirical Evaluation Benchmark Results

| Metric | Baseline RAG (Single-Pass) | Self-Healing RAG (With Critic Loop) | Improvement |
| :--- | :---: | :---: | :---: |
| **Total Test Queries** | {total_q} | {total_q} | — |
| **Out-of-Domain / Unsupported Queries** | {out_of_domain_q} | {out_of_domain_q} | — |
| **Hallucination Rate (Out-of-Domain)** | **{base_hallucination_rate:.1f}%** | **{heal_hallucination_rate:.1f}%** | **-{base_hallucination_rate - heal_hallucination_rate:.1f}% Reduction** |
| **Honest Fallback Rate** | {base_fallback_rate:.1f}% | **{heal_fallback_rate:.1f}%** | **+{heal_fallback_rate - base_fallback_rate:.1f}% Accuracy** |
| **Average Latency per Query** | {base_avg_lat:.2f}s | {heal_avg_lat:.2f}s | +{heal_avg_lat - base_avg_lat:.2f}s (Critic overhead) |

> [!NOTE]
> Evaluation run on {total_q} test cases (10 in-domain answerable questions, 5 out-of-domain trick questions).
"""

    print("\n" + summary_table)
    
    # Save output
    os.makedirs("./data", exist_ok=True)
    with open("./data/eval_summary.md", "w") as f:
        f.write(summary_table)

    with open("./data/eval_results.json", "w") as f:
        json.dump({
            "baseline": baseline_results,
            "self_healing": self_healing_results
        }, f, indent=2)

    logger.info("Evaluation results saved to data/eval_summary.md and data/eval_results.json")
    return summary_table


if __name__ == "__main__":
    run_evaluation()
