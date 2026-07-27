"""Retriever component for vector search and chunk retrieval."""

import logging
from typing import List, Optional
from selfhealing_rag.config import settings
from selfhealing_rag.schemas import DocumentChunk
from selfhealing_rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves context document chunks from the vector store for a given query."""

    def __init__(self, vector_store: VectorStore, top_k: Optional[int] = None):
        self.vector_store = vector_store
        self.top_k = top_k or settings.top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[DocumentChunk]:
        """Search vector store and return top-k matching document chunks."""
        k = top_k or self.top_k
        logger.info(f"Retrieving top {k} chunks for query: '{query}'")
        
        chunks = self.vector_store.search(query=query, top_k=k)
        logger.info(f"Retrieved {len(chunks)} chunks from vector store.")
        return chunks
