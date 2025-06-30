"""
Abstract interface for memory storage backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from .memory_models import (
    Conversation,
    Message,
    Context,
    UserProfile,
    KnowledgeSnapshot,
    MemorySearchResult
)


class MemoryStore(ABC):
    """Abstract base class for memory storage implementations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (create tables, indices, etc.)."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup resources."""
        pass
    
    # Conversation methods
    @abstractmethod
    async def create_conversation(self, conversation: Conversation) -> Conversation:
        """Create a new conversation."""
        pass
    
    @abstractmethod
    async def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Get a conversation by ID."""
        pass
    
    @abstractmethod
    async def update_conversation(self, conversation: Conversation) -> Conversation:
        """Update an existing conversation."""
        pass
    
    @abstractmethod
    async def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation."""
        pass
    
    @abstractmethod
    async def list_conversations(
        self,
        user_id: str,
        context_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Conversation]:
        """List conversations for a user, optionally filtered by context."""
        pass
    
    # Message methods
    @abstractmethod
    async def add_message(self, message: Message) -> Message:
        """Add a message to a conversation."""
        pass
    
    @abstractmethod
    async def get_messages(
        self,
        conversation_id: int,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """Get messages from a conversation."""
        pass
    
    @abstractmethod
    async def update_message(self, message: Message) -> Message:
        """Update a message."""
        pass
    
    @abstractmethod
    async def delete_message(self, message_id: int) -> bool:
        """Delete a message."""
        pass
    
    # Context methods
    @abstractmethod
    async def create_context(self, context: Context) -> Context:
        """Create a new context."""
        pass
    
    @abstractmethod
    async def get_context(self, context_id: int) -> Optional[Context]:
        """Get a context by ID."""
        pass
    
    @abstractmethod
    async def update_context(self, context: Context) -> Context:
        """Update a context."""
        pass
    
    @abstractmethod
    async def delete_context(self, context_id: int) -> bool:
        """Delete a context."""
        pass
    
    @abstractmethod
    async def list_contexts(
        self,
        user_id: str,
        include_archived: bool = False
    ) -> List[Context]:
        """List contexts for a user."""
        pass
    
    @abstractmethod
    async def get_active_context(self, user_id: str) -> Optional[Context]:
        """Get the active context for a user."""
        pass
    
    # User profile methods
    @abstractmethod
    async def create_or_update_user_profile(self, profile: UserProfile) -> UserProfile:
        """Create or update a user profile."""
        pass
    
    @abstractmethod
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user profile."""
        pass
    
    @abstractmethod
    async def delete_user_profile(self, user_id: str) -> bool:
        """Delete a user profile."""
        pass
    
    # Knowledge snapshot methods
    @abstractmethod
    async def create_knowledge_snapshot(self, snapshot: KnowledgeSnapshot) -> KnowledgeSnapshot:
        """Create a knowledge graph snapshot."""
        pass
    
    @abstractmethod
    async def get_knowledge_snapshot(self, snapshot_id: int) -> Optional[KnowledgeSnapshot]:
        """Get a knowledge snapshot by ID."""
        pass
    
    @abstractmethod
    async def list_knowledge_snapshots(
        self,
        context_id: int,
        limit: int = 10
    ) -> List[KnowledgeSnapshot]:
        """List knowledge snapshots for a context."""
        pass
    
    @abstractmethod
    async def get_latest_knowledge_snapshot(self, context_id: int) -> Optional[KnowledgeSnapshot]:
        """Get the latest knowledge snapshot for a context."""
        pass
    
    # Search methods
    @abstractmethod
    async def search_messages(
        self,
        user_id: str,
        query: str,
        context_id: Optional[int] = None,
        limit: int = 20
    ) -> List[MemorySearchResult]:
        """Search through messages using full-text search."""
        pass
    
    @abstractmethod
    async def search_by_entities(
        self,
        user_id: str,
        entities: List[str],
        context_id: Optional[int] = None,
        limit: int = 20
    ) -> List[MemorySearchResult]:
        """Search messages that mention specific entities."""
        pass
    
    @abstractmethod
    async def get_conversation_summary(
        self,
        conversation_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get a summary of a conversation."""
        pass
    
    # Analytics methods
    @abstractmethod
    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get usage statistics for a user."""
        pass
    
    @abstractmethod
    async def get_context_statistics(self, context_id: int) -> Dict[str, Any]:
        """Get statistics for a context."""
        pass
    
    @abstractmethod
    async def get_agent_usage_stats(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get agent usage statistics."""
        pass
    
    # Maintenance methods
    @abstractmethod
    async def cleanup_old_conversations(
        self,
        days_to_keep: int = 90
    ) -> int:
        """Clean up old conversations. Returns number of conversations deleted."""
        pass
    
    @abstractmethod
    async def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user (for GDPR compliance)."""
        pass
    
    @abstractmethod
    async def import_user_data(self, user_id: str, data: Dict[str, Any]) -> bool:
        """Import user data from an export."""
        pass 