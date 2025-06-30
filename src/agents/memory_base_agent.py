"""
Enhanced base agent with memory support.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import time
import logging

from ..memory.memory_manager import MemoryManager
from ..memory.memory_models import Message

logger = logging.getLogger(__name__)


class MemoryBaseAgent(ABC):
    """Base agent class with integrated memory support."""
    
    def __init__(
        self,
        name: str,
        memory_manager: Optional[MemoryManager] = None,
        use_memory: bool = True
    ):
        """Initialize the agent with optional memory support."""
        self._name = name
        self.memory_manager = memory_manager
        self.use_memory = use_memory and memory_manager is not None
        self._processing_start_time: Optional[float] = None
    
    @property
    def name(self) -> str:
        """Unique name of the agent."""
        return self._name
    
    @abstractmethod
    def can_handle(self, request: str) -> bool:
        """Determine if this agent can process the given request."""
        pass
    
    def process(self, request: str) -> dict:
        """Process the request with memory support."""
        self._processing_start_time = time.time()
        
        try:
            # Save user message to memory if enabled
            if self.use_memory and self.memory_manager:
                asyncio.run(self._save_user_message(request))
            
            # Get memory context if available
            context = []
            if self.use_memory and self.memory_manager:
                context = asyncio.run(self._get_memory_context())
            
            # Process the request with context
            result = self._process_with_context(request, context)
            
            # Save assistant response to memory if enabled
            if self.use_memory and self.memory_manager and result.get("response"):
                processing_time = time.time() - self._processing_start_time
                asyncio.run(self._save_assistant_response(
                    result["response"],
                    result.get("metadata", {}),
                    result.get("tokens_used"),
                    processing_time,
                    result.get("confidence_score")
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request in {self.name}: {e}")
            return {
                "error": str(e),
                "agent": self.name,
                "success": False
            }
    
    @abstractmethod
    def _process_with_context(self, request: str, context: List[Message]) -> dict:
        """Process the request with memory context. To be implemented by subclasses."""
        pass
    
    def retrieve_and_generate(self, request: str) -> dict:
        """Perform RAG with memory support."""
        return self.process(request)
    
    def summarize(self, request: str) -> dict:
        """Generate a summary with memory support."""
        # Default implementation - can be overridden
        return self.process(f"Please summarize: {request}")
    
    # Memory helper methods
    async def _save_user_message(self, content: str) -> None:
        """Save user message to memory."""
        try:
            await self.memory_manager.add_user_message(content)
        except Exception as e:
            logger.warning(f"Failed to save user message to memory: {e}")
    
    async def _save_assistant_response(
        self,
        content: str,
        metadata: Dict[str, Any],
        tokens_used: Optional[int] = None,
        processing_time: Optional[float] = None,
        confidence_score: Optional[float] = None
    ) -> None:
        """Save assistant response to memory."""
        try:
            await self.memory_manager.add_assistant_message(
                content=content,
                agent_used=self.name,
                metadata=metadata,
                tokens_used=tokens_used,
                processing_time=processing_time,
                confidence_score=confidence_score
            )
        except Exception as e:
            logger.warning(f"Failed to save assistant response to memory: {e}")
    
    async def _get_memory_context(self, message_count: int = 5) -> List[Message]:
        """Get recent messages from memory for context."""
        try:
            return await self.memory_manager.get_recent_context(message_count)
        except Exception as e:
            logger.warning(f"Failed to get memory context: {e}")
            return []
    
    async def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search through memory for relevant information."""
        if not self.use_memory or not self.memory_manager:
            return []
        
        try:
            results = await self.memory_manager.search_memory(query, limit=limit)
            return [
                {
                    "content": result.message.content,
                    "role": result.message.role.value,
                    "timestamp": result.message.timestamp.isoformat(),
                    "relevance_score": result.relevance_score,
                    "conversation_title": result.conversation.title
                }
                for result in results
            ]
        except Exception as e:
            logger.warning(f"Failed to search memory: {e}")
            return []
    
    def format_context_prompt(self, context: List[Message], max_tokens: int = 1000) -> str:
        """Format memory context into a prompt string."""
        if not context:
            return ""
        
        context_parts = []
        total_length = 0
        
        for msg in reversed(context):  # Start with most recent
            msg_text = f"{msg.role.value.upper()}: {msg.content}"
            msg_length = len(msg_text.split())  # Rough token estimate
            
            if total_length + msg_length > max_tokens:
                break
            
            context_parts.insert(0, msg_text)
            total_length += msg_length
        
        if context_parts:
            return "Previous conversation context:\n" + "\n".join(context_parts) + "\n\n"
        return ""


# Import asyncio at the module level
import asyncio 