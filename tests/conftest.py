"""Pytest configuration and shared fixtures."""

import os
import sys

import pytest

# Add src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from selfhealing_rag.llm_client import MockLLMClient
from selfhealing_rag.schemas import DocumentChunk
from selfhealing_rag.vector_store import ChromaVectorStore


@pytest.fixture
def mock_llm_client():
    """Returns a MockLLMClient instance."""
    return MockLLMClient()


@pytest.fixture
def ephemeral_vector_store():
    """Returns an in-memory ephemeral ChromaVectorStore for test isolation."""
    store = ChromaVectorStore(
        collection_name="test_collection",
        is_ephemeral=True
    )
    # Add dummy test data
    store.add_documents(
        ids=["doc_1", "doc_2"],
        documents=[
            "Data in transit must use TLS 1.3 or higher. Data at rest must use AES-256 encryption keys managed in AWS KMS.",
            "All custom LLMs must pass a third-party penetration test and SAST/DAST security scan prior to production release."
        ],
        metadatas=[
            {"source": "security_policy.md", "chunk_index": 1},
            {"source": "ai_policy.md", "chunk_index": 2}
        ]
    )
    return store


@pytest.fixture
def sample_chunks():
    """Returns sample document chunks."""
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            content="Data in transit must use TLS 1.3 or higher. Data at rest must use AES-256.",
            metadata={"source": "security_policy.md"},
            score=0.92
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            content="Kubernetes pods must run as non-root users with UID 10001.",
            metadata={"source": "k8s_policy.md"},
            score=0.88
        )
    ]
