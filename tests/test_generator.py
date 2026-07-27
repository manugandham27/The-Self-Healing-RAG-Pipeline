"""Tests for Generator and citation extraction."""

from selfhealing_rag.generator import Generator
from selfhealing_rag.llm_client import MockLLMClient


def test_generator_citations(sample_chunks):
    """Test inline citation extraction in Generator."""
    mock_llm = MockLLMClient()
    mock_llm.set_response(
        "USER QUESTION",
        "Data in transit must use TLS 1.3 [1]. Kubernetes pods must use non-root UID 10001 [2]."
    )

    generator = Generator(llm_client=mock_llm)
    result = generator.generate(query="What are the security standards?", chunks=sample_chunks)

    assert "[1]" in result.answer
    assert "[2]" in result.answer
    assert len(result.citations) == 2
    assert result.citations[0].chunk_id == "chunk_1"
    assert result.citations[1].chunk_id == "chunk_2"
