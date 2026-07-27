"""Critic stage for claim verification, groundedness evaluation, and structured verdicts."""

import json
import re
import logging
from typing import List
from selfhealing_rag.schemas import DocumentChunk, GenerationResult, CriticVerdict
from selfhealing_rag.llm_client import LLMClient
from selfhealing_rag.config import settings

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a rigorous Fact-Checking Critic and Entailment Evaluator.

Your job is to critically evaluate whether an AI-generated answer is FULLY GROUNDED in the provided source context chunks.

INSTRUCTIONS:
1. Break down the answer into individual factual claims.
2. For each claim, check if it is directly supported by the provided context chunks.
3. Identify any unsupported claims, hallucinations, external facts, or logical leaps not present in context.
4. Output your final judgment strictly as a valid JSON object with the following structure:

{
    "grounded": true or false,
    "unsupported_claims": ["list of specific unsupported claims, if any"],
    "confidence": float between 0.0 and 1.0,
    "reason": "Detailed concise explanation of your verdict"
}

Do NOT wrap the JSON in markdown codeblocks or extra text. Output ONLY raw JSON.
"""


class Critic:
    """Evaluates answer groundedness and detects hallucinations using an LLM critic prompt."""

    def __init__(self, llm_client: LLMClient, confidence_threshold: float = None):
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold or settings.confidence_threshold

    def format_chunks(self, chunks: List[DocumentChunk]) -> str:
        """Format chunks into numbered context passages."""
        blocks = []
        for idx, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown")
            blocks.append(f"Passage [{idx}] ({source}):\n{chunk.content}")
        return "\n\n".join(blocks)

    def _parse_json_verdict(self, raw_output: str) -> CriticVerdict:
        """Parse raw LLM string into CriticVerdict model, handling code blocks if present."""
        text = raw_output.strip()
        # Remove ```json ... ``` tags if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
            else:
                text = text.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(text)
            grounded = bool(data.get("grounded", False))
            unsupported = data.get("unsupported_claims", [])
            if isinstance(unsupported, str):
                unsupported = [unsupported]
            confidence = float(data.get("confidence", 0.5))
            reason = str(data.get("reason", "No reason provided."))

            # If confidence is below threshold, treat as ungrounded
            if confidence < self.confidence_threshold:
                grounded = False

            return CriticVerdict(
                grounded=grounded,
                unsupported_claims=unsupported,
                confidence=confidence,
                reason=reason
            )
        except Exception as e:
            logger.error(f"Failed to parse critic JSON response: {e}. Raw text: {raw_output}")
            return CriticVerdict(
                grounded=False,
                unsupported_claims=["Failed to parse structured critic response"],
                confidence=0.0,
                reason=f"Parse error: {str(e)}"
            )

    def evaluate(
        self,
        query: str,
        generation: GenerationResult,
        chunks: List[DocumentChunk]
    ) -> CriticVerdict:
        """Evaluate generation groundedness against context chunks."""
        if not chunks:
            return CriticVerdict(
                grounded=False,
                unsupported_claims=["No context chunks available to verify answer."],
                confidence=1.0,
                reason="No context retrieved."
            )

        context_str = self.format_chunks(chunks)
        prompt = (
            f"USER QUESTION: {query}\n\n"
            f"SOURCE CONTEXT PASSAGES:\n{context_str}\n\n"
            f"GENERATED ANSWER TO CRITIQUE:\n{generation.answer}\n\n"
            "EVALUATION JSON:"
        )

        raw_output = self.llm_client.generate(prompt=prompt, system_prompt=CRITIC_SYSTEM_PROMPT)
        verdict = self._parse_json_verdict(raw_output)
        
        logger.info(f"Critic verdict - Grounded: {verdict.grounded}, Confidence: {verdict.confidence}")
        return verdict
