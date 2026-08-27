"""Interactive CLI query script for Self-Healing RAG pipeline."""

import os
import sys
import argparse

# Add src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from selfhealing_rag.config import settings
from selfhealing_rag.llm_client import AnthropicLLMClient, MockLLMClient
from selfhealing_rag.vector_store import ChromaVectorStore
from selfhealing_rag.retriever import Retriever
from selfhealing_rag.generator import Generator
from selfhealing_rag.critic import Critic
from selfhealing_rag.reformulator import QueryReformulator
from selfhealing_rag.orchestrator import Orchestrator


def run_cli_query(query: str):
    """Run a single query through the pipeline and print formatted trace output."""
    if settings.anthropic_api_key:
        llm_client = AnthropicLLMClient()
    else:
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

    print("\n" + "="*70)
    print(f"🔍 QUESTION: {query}")
    print("="*70 + "\n")

    response = orchestrator.run(query=query, enable_self_healing=True)

    for trace in response.traces:
        print(f"--- Attempt #{trace.attempt_number} [{trace.status}] ---")
        print(f"Search Query: {trace.query}")
        
        if trace.generation:
            print(f"Generated Answer: {trace.generation.answer}")
            
        if trace.verdict:
            print(f"Critic Verdict: Grounded={trace.verdict.grounded} (Confidence: {trace.verdict.confidence * 100:.0f}%)")
            print(f"Critic Rationale: {trace.verdict.reason}")
            if trace.verdict.unsupported_claims:
                print(f"Unsupported Claims: {trace.verdict.unsupported_claims}")
                
        if trace.reformulation:
            print(f"⚡ Reformulated Query -> '{trace.reformulation.reformulated_query}'")
            print(f"  Reason: {trace.reformulation.reasoning}")
        print()

    print("="*70)
    print(f"🎯 FINAL ANSWER [{response.status}] (Attempts: {response.total_attempts}, Time: {response.execution_time_seconds}s):")
    print(response.answer)
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Self-Healing RAG Pipeline")
    parser.add_argument("query", nargs="?", type=str, help="Question to ask the pipeline")
    args = parser.parse_args()

    if args.query:
        run_cli_query(args.query)
    else:
        query_input = input("\nType your question for Self-Healing RAG: ")
        if query_input.strip():
            run_cli_query(query_input.strip())
