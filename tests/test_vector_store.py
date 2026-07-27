"""Tests for ChromaVectorStore."""

from selfhealing_rag.vector_store import ChromaVectorStore


def test_vector_store_add_and_search(ephemeral_vector_store: ChromaVectorStore):
    """Test adding documents and performing semantic search."""
    results = ephemeral_vector_store.search(query="TLS encryption standards", top_k=1)
    
    assert len(results) == 1
    assert "TLS 1.3" in results[0].content
    assert results[0].score > 0.0
    assert results[0].metadata.get("source") == "security_policy.md"
