"""
High-level memory manager for the orchestrator.
"""
from __future__ import annotations

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .memory_store import MemoryStore
from .sqlite_memory_store import SQLiteMemoryStore
from .memory_models import (
    Conversation,
    Message,
    Context,
    UserProfile,
    MessageRole,
    ContextStatus,
    MemorySearchResult
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """High-level interface for managing persistent memory."""
    
    def __init__(self, store: Optional[MemoryStore] = None):
        """Initialize the memory manager."""
        self.store = store or SQLiteMemoryStore()
        self._current_conversation: Optional[Conversation] = None
        self._current_context: Optional[Context] = None
        self._current_user_id: Optional[str] = None
    
    async def initialize(self) -> None:
        """Initialize the memory store."""
        await self.store.initialize()
        logger.info("Memory manager initialized")
    
    async def close(self) -> None:
        """Close the memory store."""
        await self.store.close()
    
    # User management
    async def set_user(self, user_id: str, create_if_not_exists: bool = True) -> UserProfile:
        """Set the current user and load their profile."""
        self._current_user_id = user_id
        
        profile = await self.store.get_user_profile(user_id)
        if not profile and create_if_not_exists:
            profile = UserProfile(user_id=user_id)
            profile = await self.store.create_or_update_user_profile(profile)
        
        # Load active context
        if profile:
            context = await self.store.get_active_context(user_id)
            if context:
                self._current_context = context
        
        return profile
    
    async def get_current_user_id(self) -> Optional[str]:
        """Get the current user ID."""
        return self._current_user_id
    
    # Context management
    async def create_context(
        self,
        name: str,
        description: Optional[str] = None,
        set_as_active: bool = True
    ) -> Context:
        """Create a new context."""
        if not self._current_user_id:
            raise ValueError("No user set. Call set_user() first.")
        
        context = Context(
            user_id=self._current_user_id,
            name=name,
            description=description
        )
        context = await self.store.create_context(context)
        
        if set_as_active:
            self._current_context = context
            # Update user's default context
            profile = await self.store.get_user_profile(self._current_user_id)
            if profile:
                profile.default_context_id = context.id
                await self.store.create_or_update_user_profile(profile)
        
        return context
    
    async def switch_context(self, context_id: int) -> Context:
        """Switch to a different context."""
        context = await self.store.get_context(context_id)
        if not context:
            raise ValueError(f"Context {context_id} not found")
        
        if context.user_id != self._current_user_id:
            raise ValueError("Cannot switch to another user's context")
        
        self._current_context = context
        
        # Update user's default context
        profile = await self.store.get_user_profile(self._current_user_id)
        if profile:
            profile.default_context_id = context.id
            await self.store.create_or_update_user_profile(profile)
        
        return context
    
    async def get_current_context(self) -> Optional[Context]:
        """Get the current context."""
        return self._current_context
    
    async def list_contexts(self, include_archived: bool = False) -> List[Context]:
        """List all contexts for the current user."""
        if not self._current_user_id:
            raise ValueError("No user set. Call set_user() first.")
        
        return await self.store.list_contexts(self._current_user_id, include_archived)
    
    # Conversation management
    async def start_conversation(
        self,
        title: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Conversation:
        """Start a new conversation."""
        if not self._current_user_id:
            raise ValueError("No user set. Call set_user() first.")
        
        conversation = Conversation(
            user_id=self._current_user_id,
            session_id=session_id or str(uuid.uuid4()),
            context_id=self._current_context.id if self._current_context else None,
            title=title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        self._current_conversation = await self.store.create_conversation(conversation)
        return self._current_conversation
    
    async def get_current_conversation(self) -> Optional[Conversation]:
        """Get the current conversation."""
        return self._current_conversation
    
    async def resume_conversation(self, conversation_id: int) -> Conversation:
        """Resume an existing conversation."""
        conversation = await self.store.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        if conversation.user_id != self._current_user_id:
            raise ValueError("Cannot resume another user's conversation")
        
        self._current_conversation = conversation
        
        # Load the conversation's context if different
        if conversation.context_id and (
            not self._current_context or 
            self._current_context.id != conversation.context_id
        ):
            self._current_context = await self.store.get_context(conversation.context_id)
        
        return conversation
    
    async def list_conversations(
        self,
        context_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Conversation]:
        """List conversations for the current user."""
        if not self._current_user_id:
            raise ValueError("No user set. Call set_user() first.")
        
        return await self.store.list_conversations(
            self._current_user_id,
            context_id,
            limit,
            offset
        )
    
    # Message management
    async def add_user_message(self, content: str) -> Message:
        """Add a user message to the current conversation."""
        if not self._current_conversation:
            await self.start_conversation()
        
        message = Message(
            conversation_id=self._current_conversation.id,
            role=MessageRole.USER,
            content=content
        )
        
        message = await self.store.add_message(message)
        self._current_conversation.total_messages += 1
        
        return message
    
    async def add_assistant_message(
        self,
        content: str,
        agent_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tokens_used: Optional[int] = None,
        processing_time: Optional[float] = None,
        confidence_score: Optional[float] = None
    ) -> Message:
        """Add an assistant message to the current conversation."""
        if not self._current_conversation:
            raise ValueError("No active conversation. Start or resume a conversation first.")
        
        message = Message(
            conversation_id=self._current_conversation.id,
            role=MessageRole.ASSISTANT,
            content=content,
            agent_used=agent_used,
            metadata=metadata,
            tokens_used=tokens_used,
            processing_time=processing_time,
            confidence_score=confidence_score
        )
        
        message = await self.store.add_message(message)
        
        # Update conversation statistics
        self._current_conversation.total_messages += 1
        if tokens_used:
            self._current_conversation.total_tokens += tokens_used
        if agent_used and agent_used not in self._current_conversation.agents_used:
            self._current_conversation.agents_used.append(agent_used)
        
        await self.store.update_conversation(self._current_conversation)
        
        return message
    
    async def get_conversation_messages(
        self,
        conversation_id: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get messages from a conversation."""
        conv_id = conversation_id or (
            self._current_conversation.id if self._current_conversation else None
        )
        
        if not conv_id:
            raise ValueError("No conversation specified or active")
        
        return await self.store.get_messages(conv_id, limit)
    
    # Search functionality
    async def search_memory(
        self,
        query: str,
        context_id: Optional[int] = None,
        limit: int = 20
    ) -> List[MemorySearchResult]:
        """Search through all conversations and messages."""
        if not self._current_user_id:
            raise ValueError("No user set. Call set_user() first.")
        
        return await self.store.search_messages(
            self._current_user_id,
            query,
            context_id,
            limit
        )
    
    async def search_by_entities(
        self,
        entities: List[str],
        context_id: Optional[int] = None,
        limit: int = 20
    ) -> List[MemorySearchResult]:
        """Search for messages mentioning specific entities."""
        if not self._current_user_id:
            raise ValueError("No user set. Call set_user() first.")
        
        return await self.store.search_by_entities(
            self._current_user_id,
            entities,
            context_id,
            limit
        )
    
    # Context helpers
    async def add_document_to_context(
        self,
        document_id: str,
        context_id: Optional[int] = None
    ) -> Context:
        """Add a document to a context."""
        context = await self._get_context(context_id)
        
        if document_id not in context.document_ids:
            context.document_ids.append(document_id)
            context = await self.store.update_context(context)
        
        return context
    
    async def set_context_knowledge_graph(
        self,
        knowledge_graph_id: str,
        context_id: Optional[int] = None
    ) -> Context:
        """Set the knowledge graph ID for a context."""
        context = await self._get_context(context_id)
        
        context.knowledge_graph_id = knowledge_graph_id
        context = await self.store.update_context(context)
        
        return context
    
    # Utility methods
    async def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for the current user."""
        if not self._current_user_id:
            raise ValueError("No user set. Call set_user() first.")
        
        return await self.store.get_user_statistics(self._current_user_id)
    
    async def export_data(self) -> Dict[str, Any]:
        """Export all data for the current user."""
        if not self._current_user_id:
            raise ValueError("No user set. Call set_user() first.")
        
        return await self.store.export_user_data(self._current_user_id)
    
    async def _get_context(self, context_id: Optional[int] = None) -> Context:
        """Get a context by ID or return the current context."""
        if context_id:
            context = await self.store.get_context(context_id)
            if not context:
                raise ValueError(f"Context {context_id} not found")
            if context.user_id != self._current_user_id:
                raise ValueError("Cannot access another user's context")
            return context
        
        if not self._current_context:
            raise ValueError("No active context. Create or switch to a context first.")
        
        return self._current_context
    
    # Convenience methods for quick operations
    async def quick_save(
        self,
        user_message: str,
        assistant_response: str,
        agent_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[Message, Message]:
        """Quickly save a user message and assistant response."""
        # Ensure we have a conversation
        if not self._current_conversation:
            await self.start_conversation()
        
        # Add user message
        user_msg = await self.add_user_message(user_message)
        
        # Add assistant response
        assistant_msg = await self.add_assistant_message(
            assistant_response,
            agent_used=agent_used,
            metadata=metadata
        )
        
        return user_msg, assistant_msg
    
    async def get_recent_context(
        self,
        message_count: int = 10
    ) -> List[Message]:
        """Get recent messages from the current conversation for context."""
        if not self._current_conversation:
            return []
        
        messages = await self.get_conversation_messages(limit=message_count)
        return messages[-message_count:] if len(messages) > message_count else messages 