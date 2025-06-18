"""
GraphRAG application entrypoint with Milvus Vector DB and Azure OpenAI.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

from src.orchestrator.agent_factory import AgentFactory
from src.agents.graph_rag_agent import GraphRAGAgent
from src.orchestrator.agent_factory import RAGStrategy
from src.orchestrator.orchestrator import Orchestrator


class ConsoleLogger:
    """Console logger implementation."""
    
    def __init__(self, name: str = "graph_rag_orchestrator"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)


def create_graph_rag_orchestrator(
    milvus_uri: str,
    milvus_token: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_deployment: Optional[str] = None,
    context_dir: str = "context",
    embedding_model: str = "all-MiniLM-L6-v2",
    temperature: float = 0.0,
    use_llm_extraction: bool = True
) -> Orchestrator:
    """
    Create and configure the GraphRAG orchestrator.

    Args:
        milvus_uri: Milvus connection URI
        milvus_token: Milvus authentication token
        azure_endpoint: Azure OpenAI endpoint
        azure_api_key: Azure OpenAI API key
        azure_deployment: Azure OpenAI deployment name
        context_dir: Directory containing documents
        embedding_model: Sentence transformer model
        temperature: Model temperature
        use_llm_extraction: Whether to use LLM for triplet extraction

    Returns:
        Configured Orchestrator instance
    """
    logger = ConsoleLogger()

    # Get Azure OpenAI configuration from environment if not provided
    azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    if not all([azure_endpoint, azure_api_key, azure_deployment]):
        raise ValueError(
            "Azure OpenAI configuration is incomplete. Please provide azure_endpoint, "
            "azure_api_key, and azure_deployment, or set the environment variables "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME."
        )

    try:
        logger.info("Initializing GraphRAG system...")

        # Register GraphRAG agent type
        AgentFactory.register("graph_rag", GraphRAGAgent)

        # Create GraphRAG agent
        graph_rag_agent = AgentFactory.create(
            "graph_rag",
            name="graph_rag",
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_deployment=azure_deployment,
            context_dir=context_dir,
            embedding_model=embedding_model,
            temperature=temperature,
            use_llm_extraction=use_llm_extraction
        )

        # Set up strategies
        strategies = {
            graph_rag_agent.name: RAGStrategy(),
        }

        # Create orchestrator
        orchestrator = Orchestrator(
            agents=[graph_rag_agent],
            strategies=strategies
        )

        logger.info("GraphRAG orchestrator created successfully")
        return orchestrator

    except Exception as e:
        logger.error(f"Failed to create GraphRAG orchestrator: {e}")
        raise


def create_graph_rag_orchestrator_from_env(
    context_dir: str = "context",
    embedding_model: str = "all-MiniLM-L6-v2",
    temperature: float = 0.0,
    use_llm_extraction: bool = True
) -> Orchestrator:
    """
    Create GraphRAG orchestrator using environment variables.
    
    Required environment variables:
    - MILVUS_URI: Milvus connection URI
    - MILVUS_TOKEN: Milvus authentication token (optional for local)
    - AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint
    - AZURE_OPENAI_API_KEY: Azure OpenAI API key
    - AZURE_OPENAI_DEPLOYMENT_NAME: Azure OpenAI deployment name
    
    Args:
        context_dir: Directory containing documents
        embedding_model: Sentence transformer model
        temperature: Model temperature
        use_llm_extraction: Whether to use LLM for triplet extraction
        
    Returns:
        Configured Orchestrator instance
    """
    milvus_uri = os.getenv("MILVUS_URI")
    if not milvus_uri:
        raise ValueError("MILVUS_URI environment variable is required")
    
    milvus_token = os.getenv("MILVUS_TOKEN")  # Optional for local Milvus
    
    return create_graph_rag_orchestrator(
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
        context_dir=context_dir,
        embedding_model=embedding_model,
        temperature=temperature,
        use_llm_extraction=use_llm_extraction
    ) 