"""
Pydantic models for the memory system.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class MessageRole(str, Enum):
    """Role of the message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContextStatus(str, Enum):
    """Status of a context."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Message(BaseModel):
    """Represents a single message in a conversation."""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    conversation_id: int
    role: MessageRole
    content: str
    agent_used: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Additional fields for enhanced memory
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    confidence_score: Optional[float] = None
    referenced_messages: Optional[List[int]] = Field(default_factory=list)


class Conversation(BaseModel):
    """Represents a conversation session."""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    user_id: str
    session_id: str
    context_id: Optional[int] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Conversation analytics
    total_messages: int = 0
    total_tokens: int = 0
    agents_used: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class Context(BaseModel):
    """Represents a working context (project, topic, etc.)."""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    user_id: str
    name: str
    description: Optional[str] = None
    status: ContextStatus = ContextStatus.ACTIVE
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Context-specific data
    document_ids: List[str] = Field(default_factory=list)
    knowledge_graph_id: Optional[str] = None
    default_agents: List[str] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)


class UserProfile(BaseModel):
    """Represents a user's profile and preferences."""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    user_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # User-specific settings
    default_context_id: Optional[int] = None
    preferred_agents: List[str] = Field(default_factory=list)
    language: str = "en"
    timezone: str = "UTC"
    
    # Usage statistics
    total_conversations: int = 0
    total_messages: int = 0
    total_contexts: int = 0
    last_active: Optional[datetime] = None


class KnowledgeSnapshot(BaseModel):
    """Represents a snapshot of the knowledge graph at a point in time."""
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    context_id: int
    snapshot_data: Dict[str, Any]  # Serialized graph data
    entity_count: int = 0
    relationship_count: int = 0
    document_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    description: Optional[str] = None
    
    # Snapshot metadata
    version: str = "1.0"
    compression: Optional[str] = None
    size_bytes: Optional[int] = None


class MemorySearchResult(BaseModel):
    """Result from searching through memory."""
    message: Message
    conversation: Conversation
    context: Optional[Context] = None
    relevance_score: float
    highlights: List[str] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    """Summary of a conversation for quick reference."""
    conversation_id: int
    title: str
    summary: str
    key_points: List[str]
    entities_mentioned: List[str]
    action_items: List[str]
    created_at: datetime
    message_count: int
    duration_minutes: float 