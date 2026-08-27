"""Configuration settings using Pydantic Settings."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables or .env file."""

    anthropic_api_key: Optional[str] = None
    llm_model: str = "claude-3-5-sonnet-20241022"

    chroma_persist_directory: str = "./data/chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "self_healing_rag_docs"

    top_k: int = 3
    max_retries: int = 2
    confidence_threshold: float = 0.7

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Singleton settings instance
settings = Settings()
