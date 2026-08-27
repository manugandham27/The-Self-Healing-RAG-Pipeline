"""FastAPI Application layer exposing Self-Healing RAG pipeline endpoints."""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from selfhealing_rag.config import settings
from selfhealing_rag.critic import Critic
from selfhealing_rag.generator import Generator
from selfhealing_rag.llm_client import AnthropicLLMClient, MockLLMClient
from selfhealing_rag.orchestrator import Orchestrator
from selfhealing_rag.reformulator import QueryReformulator
from selfhealing_rag.retriever import Retriever
from selfhealing_rag.schemas import PipelineResponse
from selfhealing_rag.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Self-Healing RAG API",
    description="Production-grade RAG pipeline with self-critique, hallucination detection, and automatic query reformulation.",
    version="0.1.0"
)

# CORS middleware for Streamlit/React frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request schema for /query endpoint."""
    query: str = Field(..., json_schema_extra={"example": "What are the encryption requirements for enterprise AI vector databases?"})
    max_retries: Optional[int] = Field(None, json_schema_extra={"example": 2})
    enable_self_healing: bool = Field(True, description="Toggle self-healing retry loop on/off")


class IngestResponse(BaseModel):
    """Response schema for /ingest endpoint."""
    status: str
    message: str


# Helper to instantiate pipeline
def get_pipeline(max_retries: Optional[int] = None) -> Orchestrator:
    """Instantiate components and return Orchestrator."""
    if settings.anthropic_api_key:
        llm_client = AnthropicLLMClient()
    else:
        logger.info("ANTHROPIC_API_KEY not set. Falling back to MockLLMClient.")
        llm_client = MockLLMClient()

    vector_store = ChromaVectorStore()
    retriever = Retriever(vector_store=vector_store)
    generator = Generator(llm_client=llm_client)
    critic = Critic(llm_client=llm_client)
    reformulator = QueryReformulator(llm_client=llm_client)

    return Orchestrator(
        retriever=retriever,
        generator=generator,
        critic=critic,
        reformulator=reformulator,
        max_retries=max_retries
    )


@app.get("/")
def read_root():
    """Root status endpoint."""
    return {
        "name": "Self-Healing RAG Pipeline API",
        "status": "online",
        "docs": "/docs",
        "model": settings.llm_model
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/query", response_model=PipelineResponse)
def execute_query(request: QueryRequest):
    """Execute a query through the Self-Healing RAG pipeline and return answer + full reasoning trace."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        pipeline = get_pipeline(max_retries=request.max_retries)
        response = pipeline.run(
            query=request.query,
            enable_self_healing=request.enable_self_healing
        )
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
def trigger_ingest():
    """Trigger ingestion of sample documents into vector database."""
    try:
        from scripts.ingest import ingest_sample_docs
        ingest_sample_docs()
        return IngestResponse(status="success", message="Sample documentation ingested into ChromaDB successfully.")
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e!s}")
