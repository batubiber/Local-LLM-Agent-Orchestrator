# src/orchestrator/app.py
"""
Application entrypoint: initialize and configure agents and orchestrator.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

from src.orchestrator.agent_factory import AgentFactory
from src.agents.rag_agent import RAGAgent
from src.agents.summary_agent import SummaryAgent
from src.data.vector_store import VectorStore
from src.llm_wrapper import AzureOpenAIClient
from src.orchestrator.agent_factory import RAGStrategy, SummaryStrategy
from src.orchestrator.orchestrator import Orchestrator
from src.models import AzureOpenAIConfig, RAGConfig

# Set up a console logger implementing ILogger
class ConsoleLogger:
    def __init__(self, name: str = "agent_orchestrator"):
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


def create_orchestrator(
    model_path: Optional[str] = None,
    temperature: float = 0.7,
    embedding_dimension: int = 384,
    vector_db_path: Optional[str] = None
) -> Orchestrator:
    """
    Create and configure the application orchestrator with default agents.

    Args:
        model_path: Path to local LLM model (not used with Azure OpenAI)
        temperature: Model temperature
        embedding_dimension: Dimension of embeddings
        vector_db_path: Path to vector store index

    Returns:
        Configured Orchestrator instance
    """
    logger = ConsoleLogger()

    # Configure Azure OpenAI
    azure_config = AzureOpenAIConfig(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        temperature=temperature
    )

    # Configure RAG
    rag_config = RAGConfig(
        embedding_model="all-MiniLM-L6-v2",
        vector_dimension=embedding_dimension
    )

    # Initialize components
    try:
        # Initialize vector store with proper index path
        index_path = os.path.join(vector_db_path, "faiss.index") if vector_db_path else None
        vector_store = VectorStore(
            dimension=rag_config.vector_dimension,
            index_path=index_path,
            embedding_model=rag_config.embedding_model
        )
        logger.info("Vector store initialized successfully")

        # Initialize Azure OpenAI client
        if not all([azure_config.deployment_name, azure_config.endpoint, azure_config.api_key]):
            raise ValueError("Azure OpenAI configuration is incomplete. Please set all required environment variables.")
            
        llm = AzureOpenAIClient(
            config=azure_config,
            logger=logger
        )
        logger.info("Azure OpenAI client initialized successfully")

        # Register agent types
        AgentFactory.register("rag", RAGAgent)
        AgentFactory.register("summary", SummaryAgent)

        # Create agents
        rag_agent = AgentFactory.create(
            "rag",
            name="rag",
            vector_store=vector_store,
            llm=llm,
            context_dir=vector_db_path
        )
        
        summary_agent = AgentFactory.create(
            "summary",
            name="summary",
            llm=llm
        )

        # Set up strategies
        strategies = {
            rag_agent.name: RAGStrategy(),
            summary_agent.name: SummaryStrategy(),
        }

        # Create orchestrator
        orchestrator = Orchestrator(
            agents=[rag_agent, summary_agent],
            strategies=strategies
        )
        
        logger.info("Orchestrator created successfully")
        return orchestrator

    except Exception as e:
        logger.error(f"Failed to create orchestrator: {e}")
        raise
