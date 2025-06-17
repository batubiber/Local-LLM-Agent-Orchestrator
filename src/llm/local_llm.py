"""
Local LLM integration using CTransformers for efficient inference.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from ctransformers import AutoModelForCausalLM
from src.interfaces import ILLMClient, ILogger


class LocalLLM(ILLMClient):
    """Local LLM implementation using CTransformers."""

    def __init__(
        self,
        model_path: str,
        logger: ILogger,
        model_type: str = "mistral",
        model_file: Optional[str] = None,
        context_length: int = 8192,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        batch_size: int = 1,
        gpu_layers: int = 0
    ) -> None:
        """
        Initialize local LLM.

        Args:
            model_path: Path to model directory
            logger: Logger instance
            model_type: Model architecture type
            model_file: Specific model file name (optional)
            context_length: Maximum context length
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            batch_size: Batch size for inference
            gpu_layers: Number of layers to offload to GPU (0 for CPU-only)
        """
        self._logger = logger
        self._model_path = model_path
        self._model_file = model_file
        self._context_length = context_length
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        
        try:
            model_path = os.path.join(model_path, model_file) if model_file else model_path
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type=model_type,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                gpu_layers=gpu_layers
            )
            self._logger.info(f"Local LLM loaded successfully from {model_path}")
        except Exception as e:
            self._logger.error(f"Failed to load local LLM: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using local LLM.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate (overrides instance setting)
            temperature: Sampling temperature (overrides instance setting)

        Returns:
            Generated text
        """
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            response = self._model(
                full_prompt,
                max_new_tokens=max_tokens or self._max_new_tokens,
                temperature=temperature or self._temperature,
                top_p=kwargs.get("top_p", self._top_p),
                stop=kwargs.get("stop", None)
            )
            
            return response.strip()
            
        except Exception as e:
            self._logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}") from e

    def get_model_info(self) -> Dict[str, str]:
        """Return model information."""
        return {
            "model_path": self._model_path,
            "model_file": self._model_file or "N/A",
            "context_length": str(self._context_length),
            "max_new_tokens": str(self._max_new_tokens)
        }

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "_model"):
            del self._model 