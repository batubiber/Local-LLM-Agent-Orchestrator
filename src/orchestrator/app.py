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
from src.llm.local_llm import LocalLLM
from src.orchestrator.agent_factory import RAGStrategy, SummaryStrategy
from src.orchestrator.orchestrator import Orchestrator
from src.models import LLMConfig, RAGConfig

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
        model_path: Path to local LLM model
        temperature: Model temperature
        embedding_dimension: Dimension of embeddings
        vector_db_path: Path to vector store index

    Returns:
        Configured Orchestrator instance
    """
    logger = ConsoleLogger()

    # Configure LLM
    llm_config = LLMConfig(
        model_path=model_path or os.getenv("LLM_MODEL_PATH", ""),
        temperature=temperature,
        model_type="mistral"  # or "llama" depending on your model
    )

    # Configure RAG
    rag_config = RAGConfig(
        embedding_model="all-MiniLM-L6-v2",
        vector_dimension=embedding_dimension
    )

    # Initialize components
    try:
        # Initialize vector store
        vector_store = VectorStore(
            dimension=rag_config.vector_dimension,
            index_path=vector_db_path,
            embedding_model=rag_config.embedding_model
        )
        logger.info("Vector store initialized successfully")

        # Initialize LLM
        if not llm_config.model_path:
            raise ValueError("Model path not provided and LLM_MODEL_PATH environment variable not set")
            
        llm = LocalLLM(
            model_path=llm_config.model_path,
            logger=logger,
            model_type=llm_config.model_type,
            temperature=llm_config.temperature
        )
        logger.info("LLM initialized successfully")

        # Register agent types
        AgentFactory.register("rag", RAGAgent)
        AgentFactory.register("summary", SummaryAgent)

        # Create agents
        rag_agent = AgentFactory.create(
            "rag",
            name="rag",
            vector_store=vector_store,
            llm=llm
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
