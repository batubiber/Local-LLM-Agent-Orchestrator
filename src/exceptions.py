"""
Custom exceptions for Azure OpenAI client errors.
"""
from __future__ import annotations


class AzureOpenAIError(Exception):
    """Base exception for Azure OpenAI errors."""
    ...


class RateLimitError(AzureOpenAIError):
    """Raised when Azure OpenAI rate limit is exceeded."""
    ...


class ModelTimeoutError(AzureOpenAIError):
    """Raised when Azure OpenAI request times out or service is unavailable."""
    ...


class HealthCheckError(AzureOpenAIError):
    """Raised when health check of Azure OpenAI service fails."""
    ...
