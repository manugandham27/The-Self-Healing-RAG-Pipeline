"""Pydantic data models for every pipeline stage I/O and trace logs."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a retrieved document chunk from the vector store."""

    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="Text content of the document chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata such as source filename, title, page number")
    score: float = Field(0.0, description="Similarity or relevance score")


class Citation(BaseModel):
    """Represents an inline citation mapping an answer claim back to a chunk."""

    citation_index: int = Field(..., description="1-based index corresponding to [1], [2] in answer")
    chunk_id: str = Field(..., description="ID of the chunk cited")
    source: str = Field("Unknown", description="Source document name or path")
    snippet: str = Field("", description="Relevant text snippet from cited chunk")


class GenerationResult(BaseModel):
    """Output from the answer Generator stage."""

    answer: str = Field(..., description="Generated answer text with inline citations")
    citations: List[Citation] = Field(default_factory=list, description="Citations mapping markers to chunks")


class CriticVerdict(BaseModel):
    """Structured verdict from the Critic stage evaluating answer groundedness."""

    grounded: bool = Field(..., description="True if all claims are fully supported by cited chunks")
    unsupported_claims: List[str] = Field(default_factory=list, description="List of specific claims not backed by source context")
    confidence: float = Field(..., description="Critic confidence score between 0.0 and 1.0")
    reason: str = Field(..., description="Explanation of why answer was accepted or rejected")


class ReformulationResult(BaseModel):
    """Output from Query Reformulator stage when critic rejects an answer."""

    original_query: str = Field(..., description="Original user query or previous query")
    reformulated_query: str = Field(..., description="New reformulated search query")
    reasoning: str = Field(..., description="Rationale for why reformulation took this direction")


class AttemptTrace(BaseModel):
    """Complete trace of a single attempt in the self-healing loop."""

    attempt_number: int = Field(..., description="Attempt number (1-based)")
    query: str = Field(..., description="Search query used in this attempt")
    retrieved_chunks: List[DocumentChunk] = Field(default_factory=list, description="Chunks retrieved for query")
    generation: Optional[GenerationResult] = Field(None, description="Generated answer and citations")
    verdict: Optional[CriticVerdict] = Field(None, description="Critic evaluation verdict")
    reformulation: Optional[ReformulationResult] = Field(None, description="Query reformulation details if rejected")
    status: str = Field(..., description="Outcome status: ACCEPTED, REJECTED, or FALLBACK")


class PipelineResponse(BaseModel):
    """Final output response returned by the self-healing orchestrator."""

    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Final answer or honest fallback text")
    status: str = Field(..., description="Pipeline outcome: SUCCESS_FIRST_TRY, HEALED_AFTER_RETRY, or FALLBACK")
    total_attempts: int = Field(..., description="Total number of attempts executed")
    execution_time_seconds: float = Field(0.0, description="Total execution time in seconds")
    traces: List[AttemptTrace] = Field(default_factory=list, description="Full trace log of each attempt")
