"""Orchestrator state machine controlling the self-healing retrieve-generate-critique-retry loop."""

import time
import logging
from typing import Optional, List

from selfhealing_rag.config import settings
from selfhealing_rag.schemas import (
    AttemptTrace,
    PipelineResponse,
    CriticVerdict,
)
from selfhealing_rag.retriever import Retriever
from selfhealing_rag.generator import Generator
from selfhealing_rag.critic import Critic
from selfhealing_rag.reformulator import QueryReformulator
from selfhealing_rag.fallback import FallbackHandler

logger = logging.getLogger(__name__)


class Orchestrator:
    """Controls the Self-Healing RAG pipeline execution loop with trace recording."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        critic: Critic,
        reformulator: QueryReformulator,
        fallback_handler: Optional[FallbackHandler] = None,
        max_retries: Optional[int] = None
    ):
        self.retriever = retriever
        self.generator = generator
        self.critic = critic
        self.reformulator = reformulator
        self.fallback_handler = fallback_handler or FallbackHandler()
        self.max_retries = max_retries if max_retries is not None else settings.max_retries

    def run(self, query: str, enable_self_healing: bool = True) -> PipelineResponse:
        """Run the full self-healing pipeline for a user query.
        
        Args:
            query: The user query string.
            enable_self_healing: If False, runs single-pass RAG without critique/retry (used for benchmarking).
        """
        start_time = time.time()
        traces: List[AttemptTrace] = []
        
        current_query = query
        max_attempts = (self.max_retries + 1) if enable_self_healing else 1
        
        last_verdict: Optional[CriticVerdict] = None
        last_chunks = []
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"=== Starting Attempt {attempt}/{max_attempts} for query: '{current_query}' ===")
            
            # 1. Retrieve
            chunks = self.retriever.retrieve(query=current_query)
            last_chunks = chunks

            # 2. Generate
            generation = self.generator.generate(query=current_query, chunks=chunks)

            # If self-healing is disabled, accept single-pass answer immediately
            if not enable_self_healing:
                verdict = CriticVerdict(
                    grounded=True,
                    unsupported_claims=[],
                    confidence=1.0,
                    reason="Self-healing loop disabled (baseline mode)."
                )
                trace = AttemptTrace(
                    attempt_number=attempt,
                    query=current_query,
                    retrieved_chunks=chunks,
                    generation=generation,
                    verdict=verdict,
                    reformulation=None,
                    status="ACCEPTED"
                )
                traces.append(trace)
                exec_time = round(time.time() - start_time, 3)
                return PipelineResponse(
                    query=query,
                    answer=generation.answer,
                    status="SUCCESS_FIRST_TRY",
                    total_attempts=1,
                    execution_time_seconds=exec_time,
                    traces=traces
                )

            # 3. Critique
            verdict = self.critic.evaluate(query=current_query, generation=generation, chunks=chunks)
            last_verdict = verdict

            # 4. Check verdict
            if verdict.grounded:
                logger.info(f"Attempt {attempt} ACCEPTED by Critic!")
                trace = AttemptTrace(
                    attempt_number=attempt,
                    query=current_query,
                    retrieved_chunks=chunks,
                    generation=generation,
                    verdict=verdict,
                    reformulation=None,
                    status="ACCEPTED"
                )
                traces.append(trace)
                
                status_str = "SUCCESS_FIRST_TRY" if attempt == 1 else "HEALED_AFTER_RETRY"
                exec_time = round(time.time() - start_time, 3)
                
                return PipelineResponse(
                    query=query,
                    answer=generation.answer,
                    status=status_str,
                    total_attempts=attempt,
                    execution_time_seconds=exec_time,
                    traces=traces
                )

            # 5. Handle Rejection / Retry
            logger.warning(f"Attempt {attempt} REJECTED by Critic. Reason: {verdict.reason}")
            
            reformulation_result = None
            if attempt < max_attempts:
                # Reformulate query for next iteration
                reformulation_result = self.reformulator.reformulate(
                    current_query=current_query,
                    rejected_answer=generation.answer,
                    verdict=verdict
                )
                
                trace = AttemptTrace(
                    attempt_number=attempt,
                    query=current_query,
                    retrieved_chunks=chunks,
                    generation=generation,
                    verdict=verdict,
                    reformulation=reformulation_result,
                    status="REJECTED"
                )
                traces.append(trace)
                
                # Update query for next loop iteration
                current_query = reformulation_result.reformulated_query
            else:
                # Max retries exhausted
                trace = AttemptTrace(
                    attempt_number=attempt,
                    query=current_query,
                    retrieved_chunks=chunks,
                    generation=generation,
                    verdict=verdict,
                    reformulation=None,
                    status="FALLBACK"
                )
                traces.append(trace)

        # 6. Fallback after max retries
        logger.warning(f"Exhausted all {max_attempts} attempts. Triggering Fallback Handler.")
        fallback_answer = self.fallback_handler.format_fallback_response(
            query=query,
            attempts_count=len(traces),
            last_verdict=last_verdict,
            last_chunks=last_chunks
        )
        exec_time = round(time.time() - start_time, 3)

        return PipelineResponse(
            query=query,
            answer=fallback_answer,
            status="FALLBACK",
            total_attempts=len(traces),
            execution_time_seconds=exec_time,
            traces=traces
        )
