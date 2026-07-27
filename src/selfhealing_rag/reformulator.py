"""Query Reformulator component triggered upon Critic rejection."""

import json
import re
import logging
from selfhealing_rag.schemas import CriticVerdict, ReformulationResult
from selfhealing_rag.llm_client import LLMClient

logger = logging.getLogger(__name__)

REFORMULATOR_SYSTEM_PROMPT = """You are an Expert Search Query Optimizer and Information Retrieval Specialist.

When a RAG system fails to produce a fully grounded answer, your task is to reformulate the search query to retrieve missing context or broader/narrower evidence.

INSTRUCTIONS:
1. Analyze the original query, the rejected answer, and the specific unsupported claims/reasons provided by the critic.
2. Determine why retrieval failed (e.g. query too narrow, missing domain terminology, wrong keyword focus).
3. Reformulate the query into a more effective search string.
4. Output your response as a valid JSON object:

{
    "reformulated_query": "The new optimized search query",
    "reasoning": "Explanation of why this reformulation will retrieve better context"
}

Output ONLY valid JSON without codeblocks or extra conversational text.
"""


class QueryReformulator:
    """Reformulates search queries based on Critic failure feedback."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def reformulate(
        self,
        current_query: str,
        rejected_answer: str,
        verdict: CriticVerdict
    ) -> ReformulationResult:
        """Generate a reformulated query incorporating critic feedback."""
        prompt = (
            f"ORIGINAL QUERY: {current_query}\n"
            f"REJECTED ANSWER: {rejected_answer}\n"
            f"UNSUPPORTED CLAIMS: {json.dumps(verdict.unsupported_claims)}\n"
            f"CRITIC REASON: {verdict.reason}\n\n"
            "REFORMULATION JSON:"
        )

        raw_output = self.llm_client.generate(prompt=prompt, system_prompt=REFORMULATOR_SYSTEM_PROMPT)
        
        # Clean up JSON
        text = raw_output.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
            else:
                text = text.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(text)
            new_query = data.get("reformulated_query", f"{current_query} enterprise policy standards")
            reasoning = data.get("reasoning", "Refining search terms to broaden context.")
        except Exception as e:
            logger.warning(f"Failed to parse reformulator output ({e}). Falling back to query keyword expansion.")
            new_query = f"{current_query} requirements standards guidelines"
            reasoning = "Fallback keyword expansion after parsing error."

        logger.info(f"Reformulated query from '{current_query}' -> '{new_query}'")
        return ReformulationResult(
            original_query=current_query,
            reformulated_query=new_query,
            reasoning=reasoning
        )
