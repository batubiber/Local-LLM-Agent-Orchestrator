"""
Interfaces for LLM client and logger abstraction.
"""
from __future__ import annotations

from typing import Protocol, Any


class ILLMClient(Protocol):
    """Protocol for LLM client implementations."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text based on a prompt.

        :param prompt: Prompt text for generation.
        :param kwargs: Additional generation parameters.
        :return: Generated text as string.
        """
        ...


class ILogger(Protocol):
    """Protocol for logger implementations."""

    def info(self, message: str) -> None:
        """Log informational message."""
        ...

    def warning(self, message: str) -> None:
        """Log warning message."""
        ...

    def error(self, message: str) -> None:
        """Log error message."""
        ...
