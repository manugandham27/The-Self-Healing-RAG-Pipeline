"""Integration test deliberately forcing a hallucination-and-retry self-healing scenario."""

import json

from selfhealing_rag.critic import Critic
from selfhealing_rag.generator import Generator
from selfhealing_rag.llm_client import LLMClient
from selfhealing_rag.orchestrator import Orchestrator
from selfhealing_rag.reformulator import QueryReformulator
from selfhealing_rag.retriever import Retriever


class StepWiseStatefulMockLLM(LLMClient):
    """Stateful Mock LLM that simulates a 1st attempt hallucination failure followed by a 2nd attempt self-healing success."""

    def __init__(self):
        self.attempt = 0

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        prompt_lower = prompt.lower()
        sys_lower = system_prompt.lower()

        # 1. Reformulator check (highest priority when prompt has REFORMULATION JSON)
        if "reformulation json:" in prompt_lower or "original query:" in prompt_lower:
            return json.dumps({
                "reformulated_query": "What are the verified encryption standards for data in transit?",
                "reasoning": "Focusing query specifically on data in transit encryption rules."
            })

        # 2. Critic check (when system prompt mentions critic or evaluator)
        if "critic" in sys_lower or "evaluator" in sys_lower:
            if self.attempt <= 1:
                return json.dumps({
                    "grounded": False,
                    "unsupported_claims": ["Mandatory RSA 8192-bit quantum-proof encryption is required."],
                    "confidence": 0.95,
                    "reason": "Context mentions TLS 1.3, but RSA 8192-bit quantum encryption is not mentioned anywhere in context."
                })
            else:
                return json.dumps({
                    "grounded": True,
                    "unsupported_claims": [],
                    "confidence": 0.98,
                    "reason": "All claims are directly grounded in passage [1]."
                })

        # 3. Generator check (when user question or context passages present)
        if "user question:" in prompt_lower or "context passages:" in prompt_lower:
            self.attempt += 1
            if self.attempt == 1:
                # Attempt 1: Return a hallucinated answer mentioning unsupported RSA 8192-bit keys
                return "Data in transit uses TLS 1.3 and mandatory RSA 8192-bit quantum-proof encryption [1]."
            else:
                # Attempt 2: Return a clean grounded answer
                return "Data in transit must strictly use TLS 1.3 or higher [1]."

        return "Default response"


def test_orchestrator_forced_hallucination_and_retry(ephemeral_vector_store):
    """Test that Orchestrator detects a 1st attempt hallucination, reformulates query, and succeeds on 2nd attempt."""
    mock_llm = StepWiseStatefulMockLLM()
    
    retriever = Retriever(vector_store=ephemeral_vector_store)
    generator = Generator(llm_client=mock_llm)
    critic = Critic(llm_client=mock_llm)
    reformulator = QueryReformulator(llm_client=mock_llm)

    orchestrator = Orchestrator(
        retriever=retriever,
        generator=generator,
        critic=critic,
        reformulator=reformulator,
        max_retries=2
    )

    response = orchestrator.run(query="What encryption standards are required?", enable_self_healing=True)

    # Verification assertions
    assert response.status == "HEALED_AFTER_RETRY"
    assert response.total_attempts == 2
    assert len(response.traces) == 2

    # Attempt 1 check
    trace_1 = response.traces[0]
    assert trace_1.attempt_number == 1
    assert trace_1.status == "REJECTED"
    assert trace_1.verdict.grounded is False
    assert "RSA 8192-bit" in trace_1.verdict.unsupported_claims[0]
    assert trace_1.reformulation is not None
    assert trace_1.reformulation.reformulated_query == "What are the verified encryption standards for data in transit?"

    # Attempt 2 check
    trace_2 = response.traces[1]
    assert trace_2.attempt_number == 2
    assert trace_2.status == "ACCEPTED"
    assert trace_2.verdict.grounded is True
    assert "TLS 1.3" in response.answer
