"""
BaseAgent interface defining required methods for all agents.
Agents must declare a unique name, determine if they can handle a request,
process the request, and expose RAG and summarization methods.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseAgent(Protocol):
    """Protocol for agent implementations."""

    @property
    def name(self) -> str:
        """Unique name of the agent."""
        ...

    def can_handle(self, request: str) -> bool:
        """Determine if this agent can process the given request."""
        ...

    def process(self, request: str) -> dict:
        """Process the request and return a result dictionary."""
        ...

    def retrieve_and_generate(self, request: str) -> dict:
        """Perform Retrieval-Augmented Generation on the request."""
        ...

    def summarize(self, request: str) -> dict:
        """Generate a summary for the given request."""
        ...
