"""
RAGAgent implementation using Retrieval-Augmented Generation.
"""
from __future__ import annotations

from typing import Any
from src.agents.base_agent import BaseAgent
from src.orchestrator.agent_factory import RAGStrategy
from src.data.vector_store import VectorStore


class RAGAgent(BaseAgent):
    """Agent that retrieves relevant documents and generates answers."""

    def __init__(
        self,
        name: str,
        vector_store: VectorStore,
        llm: Any
    ) -> None:
        """
        :param name: Unique name for the agent.
        :param vector_store: Vector store for document retrieval.
        :param llm: Language model for generation.
        """
        self._name = name
        self._vector_store = vector_store
        self._llm = llm
        self._strategy = RAGStrategy()

    @property
    def name(self) -> str:
        return self._name

    def can_handle(self, request: str) -> bool:
        # Example condition: handle if keyword 'search' in request
        return 'search' in request.lower()

    def process(self, request: str) -> dict:
        """Process the request via RAG strategy."""
        return self._strategy.execute(self, request)

    def retrieve_and_generate(self, request: str) -> dict:
        """Retrieve documents and generate a response using LLM."""
        # Retrieve top k contexts
        contexts = self._vector_store.query(request)
        # Prepare prompt
        prompt = f"Context: {contexts}\nQuestion: {request}"
        # Generate answer
        answer = self._llm.generate(prompt)
        return {'answer': answer, 'contexts': contexts}

    def summarize(self, request: str) -> dict:
        raise NotImplementedError("Summarization not supported by RAGAgent.")