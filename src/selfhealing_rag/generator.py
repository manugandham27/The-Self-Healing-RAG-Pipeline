"""Generator stage for creating answers with inline citations from retrieved context."""

import logging
import re
from typing import List

from selfhealing_rag.llm_client import LLMClient
from selfhealing_rag.schemas import Citation, DocumentChunk, GenerationResult

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """You are an accurate, objective assistant answering questions strictly based on provided context passages.

RULES:
1. Base your answer ONLY on the provided context passages. Do not use external knowledge or invent facts.
2. For EVERY statement or claim you make, include inline citations using numbers corresponding to the source context passage (e.g. [1], [2]).
3. If the context does not contain enough information to answer fully, state what is supported and explicitly note what information is missing.
4. Keep answers concise, direct, and factual.
"""


class Generator:
    """Generates answers with inline citations grounded in retrieved document chunks."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def format_context(self, chunks: List[DocumentChunk]) -> str:
        """Format retrieved chunks into a numbered context block for the LLM prompt."""
        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown Source")
            context_blocks.append(f"Passage [{idx}] (Source: {source}):\n{chunk.content}")
        return "\n\n".join(context_blocks)

    def extract_citations(self, answer: str, chunks: List[DocumentChunk]) -> List[Citation]:
        """Extract inline citation indices (e.g. [1], [2]) from answer and map to chunks."""
        matches = re.findall(r"\[(\d+)\]", answer)
        cited_indices = sorted(list(set(int(m) for m in matches)))

        citations = []
        for idx in cited_indices:
            # 1-based indexing
            chunk_pos = idx - 1
            if 0 <= chunk_pos < len(chunks):
                chunk = chunks[chunk_pos]
                source = chunk.metadata.get("source", "Unknown Source")
                snippet = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                citations.append(
                    Citation(
                        citation_index=idx,
                        chunk_id=chunk.chunk_id,
                        source=source,
                        snippet=snippet
                    )
                )
        return citations

    def generate(self, query: str, chunks: List[DocumentChunk]) -> GenerationResult:
        """Generate an answer and inline citations for a user query."""
        if not chunks:
            logger.warning("No context chunks provided to Generator.")
            return GenerationResult(
                answer="No relevant context passages were found to answer the query.",
                citations=[]
            )

        context_str = self.format_context(chunks)
        user_prompt = f"CONTEXT PASSAGES:\n{context_str}\n\nUSER QUESTION: {query}\n\nANSWER WITH INLINE CITATIONS:"

        raw_answer = self.llm_client.generate(prompt=user_prompt, system_prompt=GENERATOR_SYSTEM_PROMPT)
        citations = self.extract_citations(raw_answer, chunks)

        return GenerationResult(answer=raw_answer, citations=citations)
