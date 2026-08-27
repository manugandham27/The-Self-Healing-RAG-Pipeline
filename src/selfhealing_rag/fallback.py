"""Fallback handler for graceful failure when information cannot be grounded after max retries."""

import logging
from typing import List, Optional

from selfhealing_rag.schemas import CriticVerdict, DocumentChunk

logger = logging.getLogger(__name__)


class FallbackHandler:
    """Generates transparent, honest uncertainty responses when self-healing retries are exhausted."""

    def format_fallback_response(
        self,
        query: str,
        attempts_count: int,
        last_verdict: Optional[CriticVerdict] = None,
        last_chunks: Optional[List[DocumentChunk]] = None
    ) -> str:
        """Construct an informative fallback message acknowledging limits rather than hallucinating."""
        reason_detail = ""
        if last_verdict and last_verdict.unsupported_claims:
            claims_str = "; ".join(last_verdict.unsupported_claims[:2])
            reason_detail = f"\n\n*Validation note*: The system attempted {attempts_count} retrieval iterations, but could not ground claims such as: '{claims_str}'."

        available_sources = ""
        if last_chunks:
            sources = set(c.metadata.get("source", "Unknown") for c in last_chunks)
            if sources:
                available_sources = f"\n\n*Available sources checked*: {', '.join(sources)}."

        return (
            f"I don't have enough verified information in the knowledge base to answer your question "
            f"('{query}') confidently without hallucinating.{reason_detail}{available_sources}\n\n"
            f"Please verify if your question is within the scope of our documentation or try rephrasing with specific key terms."
        )
