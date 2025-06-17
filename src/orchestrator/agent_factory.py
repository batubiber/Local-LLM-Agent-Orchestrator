"""Agent factory and processing strategy implementations.

This module provides the Strategy design pattern for processing requests within agents
and the Factory Method design pattern for creating agent instances dynamically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

from src.agents.base_agent import BaseAgent


class ProcessingStrategy(ABC):
    """Interface for processing strategies within agents using the Strategy pattern."""

    @abstractmethod
    def execute(self, agent: BaseAgent, request: str) -> dict:
        """
        Execute processing logic on the given agent.

        :param agent: The agent instance to process the request.
        :param request: The input request string.
        :return: A dictionary containing the agent's response.
        """


class RAGStrategy(ProcessingStrategy):
    """Retrieval-Augmented Generation (RAG) strategy implementation."""

    def execute(self, agent: BaseAgent, request: str) -> dict:
        """Delegate request to the agent's RAG implementation."""
        return agent.retrieve_and_generate(request)


class SummaryStrategy(ProcessingStrategy):
    """Summarization strategy implementation."""

    def execute(self, agent: BaseAgent, request: str) -> dict:
        """Delegate request to the agent's summarization implementation."""
        return agent.summarize(request)


class AgentFactory:
    """Factory for creating agent instances by key using the Factory Method pattern."""

    _registry: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, key: str, agent_cls: Type[BaseAgent]) -> None:
        """
        Register an agent class with a key.

        :param key: Unique identifier for the agent type.
        :param agent_cls: The agent class to register.
        """
        cls._registry[key] = agent_cls

    @classmethod
    def create(cls, key: str, **kwargs) -> BaseAgent:
        """
        Create an agent instance by its registered key.

        :param key: Unique identifier of the agent type.
        :param kwargs: Keyword arguments to pass to the agent constructor.
        :return: An instance of BaseAgent.
        :raises ValueError: If the agent key is not registered.
        """
        try:
            agent_cls = cls._registry[key]
        except KeyError as error:
            raise ValueError(f"Agent type '{key}' is not registered.") from error

        return agent_cls(**kwargs)
