"""
Configuration models for the application.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class AzureOpenAIConfig(BaseModel):
    """Configuration for Azure OpenAI."""
    deployment_name: str
    endpoint: str
    api_version: str
    api_key: str
    max_tokens: int = 1024
    temperature: float = 0.7


class LLMConfig(BaseModel):
    """Configuration for LLM models."""
    model_path: str
    model_type: str = "mistral"
    model_file: Optional[str] = None
    context_length: int = 8192
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    gpu_layers: int = 0


class RAGConfig(BaseModel):
    """Configuration for RAG system."""
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    vector_dimension: int = 384
    top_k: int = 3


class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    name: str
    type: str
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    """Main application configuration."""
    llm: LLMConfig
    rag: RAGConfig
    agents: list[AgentConfig]
    log_level: str = "INFO"
