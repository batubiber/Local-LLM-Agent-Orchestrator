"""
SummaryAgent implementation using LLM summarization.
"""
from __future__ import annotations

from typing import Any
from src.agents.base_agent import BaseAgent
from src.orchestrator.agent_factory import SummaryStrategy


class SummaryAgent(BaseAgent):
    """Agent that summarizes provided text or documents."""

    def __init__(
        self,
        name: str,
        llm: Any
    ) -> None:
        """
        :param name: Unique name for the agent.
        :param llm: Language model for summarization.
        """
        self._name = name
        self._llm = llm
        self._strategy = SummaryStrategy()

    @property
    def name(self) -> str:
        return self._name

    def can_handle(self, request: str) -> bool:
        # Example condition: handle if keyword 'summarize' in request
        return 'summarize' in request.lower() or len(request.split()) > 50

    def process(self, request: str) -> dict:
        """Process the request via summarization strategy."""
        return self._strategy.execute(self, request)

    def retrieve_and_generate(self, request: str) -> dict:
        raise NotImplementedError("RAG not supported by SummaryAgent.")

    def summarize(self, request: str) -> dict:
        """Generate summary using LLM."""
        prompt = f"Summarize the following text: {request}"
        summary = self._llm.generate(prompt)
        return {'summary': summary}