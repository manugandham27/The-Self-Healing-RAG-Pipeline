"""Tests for Critic claim verification and verdict parsing."""

import json

from selfhealing_rag.critic import Critic
from selfhealing_rag.llm_client import MockLLMClient
from selfhealing_rag.schemas import Citation, GenerationResult


def test_critic_accepted_verdict(sample_chunks):
    """Test Critic when answer is fully grounded."""
    mock_llm = MockLLMClient()
    mock_llm.set_response(
        "CRITIQUE",
        json.dumps({
            "grounded": True,
            "unsupported_claims": [],
            "confidence": 0.95,
            "reason": "All claims directly backed by context."
        })
    )
    
    critic = Critic(llm_client=mock_llm)
    generation = GenerationResult(
        answer="Data in transit must use TLS 1.3 [1].",
        citations=[Citation(citation_index=1, chunk_id="chunk_1", source="security_policy.md", snippet="...")]
    )

    verdict = critic.evaluate(
        query="What encryption is required?",
        generation=generation,
        chunks=sample_chunks
    )

    assert verdict.grounded is True
    assert len(verdict.unsupported_claims) == 0
    assert verdict.confidence == 0.95


def test_critic_rejected_verdict(sample_chunks):
    """Test Critic when answer contains unsupported hallucination."""
    mock_llm = MockLLMClient()
    mock_llm.set_response(
        "CRITIQUE",
        json.dumps({
            "grounded": False,
            "unsupported_claims": ["Quantum key distribution is mandatory for TLS."],
            "confidence": 0.90,
            "reason": "Context mentions TLS 1.3 but says nothing about Quantum key distribution."
        })
    )

    critic = Critic(llm_client=mock_llm)
    generation = GenerationResult(
        answer="Data in transit uses TLS 1.3 and Quantum key distribution [1].",
        citations=[]
    )

    verdict = critic.evaluate(
        query="What encryption is required?",
        generation=generation,
        chunks=sample_chunks
    )

    assert verdict.grounded is False
    assert "Quantum key distribution" in verdict.unsupported_claims[0]
