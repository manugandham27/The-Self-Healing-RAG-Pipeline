"""LLM client interface and implementations (Anthropic API & Mock Client)."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from selfhealing_rag.config import settings

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract interface for LLM calls."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text from a prompt."""


class AnthropicLLMClient(LLMClient):
    """Anthropic Claude LLM client implementation."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.llm_model
        
        if not self.api_key:
            logger.warning("No Anthropic API key provided. AnthropicLLMClient will fail unless API key is set.")
            self.client = None
        else:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("anthropic package is not installed.")
                self.client = None

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Call Anthropic API to generate a response."""
        if not self.client:
            raise ValueError(
                "Anthropic client is not initialized. Please set ANTHROPIC_API_KEY environment variable "
                "or pass an explicit api_key."
            )
        
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class MockLLMClient(LLMClient):
    """Mock LLM Client for testing, offline demos, and deterministic retry scenarios."""

    def __init__(self, predefined_responses: Optional[Dict[str, str]] = None):
        self.predefined_responses = predefined_responses or {}
        self.call_count = 0

    def set_response(self, prompt_keyword: str, response: str):
        """Register a mock response when prompt contains prompt_keyword."""
        self.predefined_responses[prompt_keyword] = response

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Return a predefined response or default structured/text mock."""
        self.call_count += 1
        
        # Check registered keywords
        for keyword, resp in self.predefined_responses.items():
            if keyword.lower() in prompt.lower() or keyword.lower() in system_prompt.lower():
                return resp

        # Default fallback responses based on system prompt intent
        if "critic" in system_prompt.lower() or "evaluator" in system_prompt.lower():
            # If prompt contains out-of-domain queries (e.g. SOC 2, PyTorch, Python 3.8, DynamoDB, Snowflake), return ungrounded
            out_of_domain_terms = ["soc 2", "pytorch", "python 3.8", "dynamodb", "snowflake", "quantum"]
            if any(term in prompt.lower() for term in out_of_domain_terms):
                return json.dumps({
                    "grounded": False,
                    "unsupported_claims": ["Query topic is outside the knowledge base context."],
                    "confidence": 0.95,
                    "reason": "The retrieved context passages do not contain information regarding this topic."
                })

            return json.dumps({
                "grounded": True,
                "unsupported_claims": [],
                "confidence": 0.95,
                "reason": "All claims are supported by the provided context chunks."
            })
        elif "reformulate" in system_prompt.lower() or "query reformulator" in system_prompt.lower():
            return "What are the specific governance requirements for enterprise AI model deployment?"
        else:
            return (
                "Based on the provided documentation, enterprise AI models must undergo "
                "security review prior to deployment [1]. "
                "All data processing must comply with encryption standards [2]."
            )
