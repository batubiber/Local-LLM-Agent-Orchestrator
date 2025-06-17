"""Main orchestrator coordinating registered agents using Factory and Strategy patterns."""
from __future__ import annotations

from typing import List, Dict
from src.agents.base_agent import BaseAgent
from src.orchestrator.agent_factory import AgentFactory, ProcessingStrategy


class Orchestrator:
    """Coordinates request handling across multiple agents."""

    def __init__(
        self,
        agents: List[BaseAgent],
        strategies: Dict[str, ProcessingStrategy]
    ) -> None:
        """
        :param agents: List of agent instances to register.
        :param strategies: Mapping of agent name to its processing strategy.
        """
        self._agents = agents
        self._strategies = strategies

    def register_agent(self, agent_key: str, **kwargs) -> None:
        """Register a new agent by key via the AgentFactory."""
        agent = AgentFactory.create(agent_key, **kwargs)
        self._agents.append(agent)

    def handle(self, request: str) -> Dict[str, dict]:
        """Dispatch the request to capable agents and collect their responses."""
        responses: Dict[str, dict] = {}
        for agent in self._agents:
            if agent.can_handle(request):
                strategy = self._strategies.get(agent.name)
                if not strategy:
                    continue
                responses[agent.name] = strategy.execute(agent, request)
        return responses

    def get_agent_status(self) -> Dict[str, dict]:
        """Get status of all registered agents."""
        return {
            agent.name: {
                "type": agent.__class__.__name__,
                "enabled": True,
                "can_handle_requests": bool(self._strategies.get(agent.name))
            }
            for agent in self._agents
        }
