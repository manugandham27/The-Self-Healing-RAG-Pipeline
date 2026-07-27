"""Vector store abstraction interface and ChromaDB implementation."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from selfhealing_rag.config import settings
from selfhealing_rag.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract interface for vector database implementations."""

    @abstractmethod
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents and embeddings to the store."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        """Search for top_k most similar document chunks given a query."""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of VectorStore using sentence-transformers embeddings."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        is_ephemeral: bool = False
    ):
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.embedding_model_name = embedding_model_name or settings.embedding_model
        
        import chromadb
        from chromadb.utils import embedding_functions

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )

        if is_ephemeral:
            self.client = chromadb.EphemeralClient()
        else:
            self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Insert or update document chunks in the Chroma collection."""
        if not ids or not documents:
            return
        
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Added/updated {len(documents)} documents in Chroma collection '{self.collection_name}'")

    def search(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        """Query Chroma collection and convert results to List[DocumentChunk]."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks: List[DocumentChunk] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return chunks

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i in range(len(ids)):
            doc_id = ids[i]
            doc_text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 0.0
            
            # Distance in Chroma cosine distance ranges from 0 to 2; convert to similarity score 1 / (1 + dist)
            similarity_score = round(1.0 / (1.0 + float(dist)), 4)

            chunks.append(
                DocumentChunk(
                    chunk_id=doc_id,
                    content=doc_text,
                    metadata=meta or {},
                    score=similarity_score
                )
            )

        return chunks
