"""
SQLite implementation of the memory store.
"""
from __future__ import annotations

import json
import sqlite3
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging

from .memory_store import MemoryStore
from .memory_models import (
    Conversation,
    Message,
    Context,
    UserProfile,
    KnowledgeSnapshot,
    MemorySearchResult,
    MessageRole,
    ContextStatus
)

logger = logging.getLogger(__name__)


class SQLiteMemoryStore(MemoryStore):
    """SQLite implementation of the memory store."""
    
    def __init__(self, db_path: str = "memory.db"):
        """Initialize the SQLite memory store."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def _get_connection(self):
        """Get a database connection with async context manager."""
        async with self._lock:
            if self._connection is None:
                self._connection = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    isolation_level=None
                )
                self._connection.row_factory = sqlite3.Row
                # Enable foreign keys
                self._connection.execute("PRAGMA foreign_keys = ON")
            yield self._connection
    
    async def initialize(self) -> None:
        """Initialize the database schema."""
        async with self._get_connection() as conn:
            # Create tables
            conn.executescript("""
                -- User profiles table
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    name TEXT,
                    email TEXT,
                    preferences TEXT,
                    default_context_id INTEGER,
                    preferred_agents TEXT,
                    language TEXT DEFAULT 'en',
                    timezone TEXT DEFAULT 'UTC',
                    total_conversations INTEGER DEFAULT 0,
                    total_messages INTEGER DEFAULT 0,
                    total_contexts INTEGER DEFAULT 0,
                    last_active TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Contexts table
                CREATE TABLE IF NOT EXISTS contexts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'active',
                    metadata TEXT,
                    document_ids TEXT,
                    knowledge_graph_id TEXT,
                    default_agents TEXT,
                    settings TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                );
                
                -- Conversations table
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    context_id INTEGER,
                    title TEXT,
                    summary TEXT,
                    metadata TEXT,
                    total_messages INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    agents_used TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id),
                    FOREIGN KEY (context_id) REFERENCES contexts(id)
                );
                
                -- Messages table
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent_used TEXT,
                    metadata TEXT,
                    tokens_used INTEGER,
                    processing_time REAL,
                    confidence_score REAL,
                    referenced_messages TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                
                -- Knowledge snapshots table
                CREATE TABLE IF NOT EXISTS knowledge_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id INTEGER NOT NULL,
                    snapshot_data TEXT NOT NULL,
                    entity_count INTEGER DEFAULT 0,
                    relationship_count INTEGER DEFAULT 0,
                    document_count INTEGER DEFAULT 0,
                    description TEXT,
                    version TEXT DEFAULT '1.0',
                    compression TEXT,
                    size_bytes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (context_id) REFERENCES contexts(id)
                );
                
                -- Create indices for better performance
                CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_context_id ON conversations(context_id);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
                CREATE INDEX IF NOT EXISTS idx_contexts_user_id ON contexts(user_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_snapshots_context_id ON knowledge_snapshots(context_id);
                
                -- Create full-text search virtual table for messages
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    message_id,
                    content,
                    tokenize='porter'
                );
                
                -- Trigger to keep FTS table in sync
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(message_id, content) VALUES (new.id, new.content);
                END;
                
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    DELETE FROM messages_fts WHERE message_id = old.id;
                END;
                
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    UPDATE messages_fts SET content = new.content WHERE message_id = new.id;
                END;
            """)
            
            logger.info("SQLite memory store initialized successfully")
    
    async def close(self) -> None:
        """Close the database connection."""
        async with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None 

    # Conversation methods
    async def create_conversation(self, conversation: Conversation) -> Conversation:
        """Create a new conversation."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversations (
                    user_id, session_id, context_id, title, summary,
                    metadata, total_messages, total_tokens, agents_used, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation.user_id,
                    conversation.session_id,
                    conversation.context_id,
                    conversation.title,
                    conversation.summary,
                    json.dumps(conversation.metadata or {}),
                    conversation.total_messages,
                    conversation.total_tokens,
                    json.dumps(conversation.agents_used),
                    json.dumps(conversation.tags)
                )
            )
            conversation.id = cursor.lastrowid
            
            # Update user statistics
            conn.execute(
                """
                UPDATE user_profiles 
                SET total_conversations = total_conversations + 1,
                    last_active = CURRENT_TIMESTAMP
                WHERE user_id = ?
                """,
                (conversation.user_id,)
            )
            
            return conversation
    
    async def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Get a conversation by ID."""
        async with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            ).fetchone()
            
            if not row:
                return None
            
            return self._row_to_conversation(row)
    
    async def update_conversation(self, conversation: Conversation) -> Conversation:
        """Update an existing conversation."""
        async with self._get_connection() as conn:
            conversation.updated_at = datetime.now()
            conn.execute(
                """
                UPDATE conversations SET
                    title = ?, summary = ?, metadata = ?,
                    total_messages = ?, total_tokens = ?,
                    agents_used = ?, tags = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    conversation.title,
                    conversation.summary,
                    json.dumps(conversation.metadata or {}),
                    conversation.total_messages,
                    conversation.total_tokens,
                    json.dumps(conversation.agents_used),
                    json.dumps(conversation.tags),
                    conversation.updated_at,
                    conversation.id
                )
            )
            return conversation
    
    async def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            return cursor.rowcount > 0
    
    async def list_conversations(
        self,
        user_id: str,
        context_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Conversation]:
        """List conversations for a user."""
        async with self._get_connection() as conn:
            query = "SELECT * FROM conversations WHERE user_id = ?"
            params = [user_id]
            
            if context_id is not None:
                query += " AND context_id = ?"
                params.append(context_id)
            
            query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_conversation(row) for row in rows] 

    # Message methods
    async def add_message(self, message: Message) -> Message:
        """Add a message to a conversation."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, agent_used,
                    metadata, tokens_used, processing_time,
                    confidence_score, referenced_messages
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.conversation_id,
                    message.role.value,
                    message.content,
                    message.agent_used,
                    json.dumps(message.metadata or {}),
                    message.tokens_used,
                    message.processing_time,
                    message.confidence_score,
                    json.dumps(message.referenced_messages or [])
                )
            )
            message.id = cursor.lastrowid
            
            # Update conversation statistics
            conn.execute(
                """
                UPDATE conversations 
                SET total_messages = total_messages + 1,
                    total_tokens = total_tokens + COALESCE(?, 0),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (message.tokens_used or 0, message.conversation_id)
            )
            
            # Update user statistics
            conn.execute(
                """
                UPDATE user_profiles 
                SET total_messages = total_messages + 1,
                    last_active = CURRENT_TIMESTAMP
                WHERE user_id = (
                    SELECT user_id FROM conversations WHERE id = ?
                )
                """,
                (message.conversation_id,)
            )
            
            return message
    
    async def get_messages(
        self,
        conversation_id: int,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """Get messages from a conversation."""
        async with self._get_connection() as conn:
            query = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp"
            params = [conversation_id]
            
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_message(row) for row in rows]
    
    async def update_message(self, message: Message) -> Message:
        """Update a message."""
        async with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE messages SET
                    content = ?, metadata = ?, tokens_used = ?,
                    processing_time = ?, confidence_score = ?,
                    referenced_messages = ?
                WHERE id = ?
                """,
                (
                    message.content,
                    json.dumps(message.metadata or {}),
                    message.tokens_used,
                    message.processing_time,
                    message.confidence_score,
                    json.dumps(message.referenced_messages or []),
                    message.id
                )
            )
            return message
    
    async def delete_message(self, message_id: int) -> bool:
        """Delete a message."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE id = ?",
                (message_id,)
            )
            return cursor.rowcount > 0
    
    # Helper methods to convert database rows to model objects
    def _row_to_conversation(self, row: sqlite3.Row) -> Conversation:
        """Convert a database row to a Conversation object."""
        return Conversation(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            context_id=row["context_id"],
            title=row["title"],
            summary=row["summary"],
            metadata=json.loads(row["metadata"] or "{}"),
            total_messages=row["total_messages"],
            total_tokens=row["total_tokens"],
            agents_used=json.loads(row["agents_used"] or "[]"),
            tags=json.loads(row["tags"] or "[]"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )
    
    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """Convert a database row to a Message object."""
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            agent_used=row["agent_used"],
            metadata=json.loads(row["metadata"] or "{}"),
            tokens_used=row["tokens_used"],
            processing_time=row["processing_time"],
            confidence_score=row["confidence_score"],
            referenced_messages=json.loads(row["referenced_messages"] or "[]"),
            timestamp=datetime.fromisoformat(row["timestamp"])
        )

    # Context methods
    async def create_context(self, context: Context) -> Context:
        """Create a new context."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO contexts (
                    user_id, name, description, status, metadata,
                    document_ids, knowledge_graph_id, default_agents, settings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    context.user_id,
                    context.name,
                    context.description,
                    context.status.value,
                    json.dumps(context.metadata or {}),
                    json.dumps(context.document_ids),
                    context.knowledge_graph_id,
                    json.dumps(context.default_agents),
                    json.dumps(context.settings)
                )
            )
            context.id = cursor.lastrowid
            
            # Update user statistics
            conn.execute(
                """
                UPDATE user_profiles 
                SET total_contexts = total_contexts + 1
                WHERE user_id = ?
                """,
                (context.user_id,)
            )
            
            return context

    async def get_context(self, context_id: int) -> Optional[Context]:
        """Get a context by ID."""
        async with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM contexts WHERE id = ?",
                (context_id,)
            ).fetchone()
            
            if not row:
                return None
            
            return self._row_to_context(row)

    async def update_context(self, context: Context) -> Context:
        """Update a context."""
        async with self._get_connection() as conn:
            context.updated_at = datetime.now()
            conn.execute(
                """
                UPDATE contexts SET
                    name = ?, description = ?, status = ?,
                    metadata = ?, document_ids = ?,
                    knowledge_graph_id = ?, default_agents = ?,
                    settings = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    context.name,
                    context.description,
                    context.status.value,
                    json.dumps(context.metadata or {}),
                    json.dumps(context.document_ids),
                    context.knowledge_graph_id,
                    json.dumps(context.default_agents),
                    json.dumps(context.settings),
                    context.updated_at,
                    context.id
                )
            )
            return context

    async def delete_context(self, context_id: int) -> bool:
        """Delete a context."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM contexts WHERE id = ?",
                (context_id,)
            )
            return cursor.rowcount > 0

    async def list_contexts(
        self,
        user_id: str,
        include_archived: bool = False
    ) -> List[Context]:
        """List contexts for a user."""
        async with self._get_connection() as conn:
            query = "SELECT * FROM contexts WHERE user_id = ?"
            params = [user_id]
            
            if not include_archived:
                query += " AND status != ?"
                params.append(ContextStatus.ARCHIVED.value)
            
            query += " ORDER BY updated_at DESC"
            
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_context(row) for row in rows]

    async def get_active_context(self, user_id: str) -> Optional[Context]:
        """Get the active context for a user."""
        async with self._get_connection() as conn:
            # First try to get the user's default context
            profile = await self.get_user_profile(user_id)
            if profile and profile.default_context_id:
                context = await self.get_context(profile.default_context_id)
                if context and context.status == ContextStatus.ACTIVE:
                    return context
            
            # Otherwise get the most recently updated active context
            row = conn.execute(
                """
                SELECT * FROM contexts 
                WHERE user_id = ? AND status = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id, ContextStatus.ACTIVE.value)
            ).fetchone()
            
            if row:
                return self._row_to_context(row)
            
            return None

    # User profile methods
    async def create_or_update_user_profile(self, profile: UserProfile) -> UserProfile:
        """Create or update a user profile."""
        async with self._get_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM user_profiles WHERE user_id = ?",
                (profile.user_id,)
            ).fetchone()
            
            if existing:
                profile.id = existing["id"]
                profile.updated_at = datetime.now()
                conn.execute(
                    """
                    UPDATE user_profiles SET
                        name = ?, email = ?, preferences = ?,
                        default_context_id = ?, preferred_agents = ?,
                        language = ?, timezone = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (
                        profile.name,
                        profile.email,
                        json.dumps(profile.preferences),
                        profile.default_context_id,
                        json.dumps(profile.preferred_agents),
                        profile.language,
                        profile.timezone,
                        profile.updated_at,
                        profile.user_id
                    )
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO user_profiles (
                        user_id, name, email, preferences,
                        default_context_id, preferred_agents,
                        language, timezone
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.user_id,
                        profile.name,
                        profile.email,
                        json.dumps(profile.preferences),
                        profile.default_context_id,
                        json.dumps(profile.preferred_agents),
                        profile.language,
                        profile.timezone
                    )
                )
                profile.id = cursor.lastrowid
            
            return profile

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user profile."""
        async with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            
            if not row:
                return None
            
            return self._row_to_user_profile(row)

    async def delete_user_profile(self, user_id: str) -> bool:
        """Delete a user profile."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM user_profiles WHERE user_id = ?",
                (user_id,)
            )
            return cursor.rowcount > 0

    # Search methods
    async def search_messages(
        self,
        user_id: str,
        query: str,
        context_id: Optional[int] = None,
        limit: int = 20
    ) -> List[MemorySearchResult]:
        """Search through messages using full-text search."""
        async with self._get_connection() as conn:
            # Build the search query
            search_query = """
                SELECT 
                    m.*, c.*, ctx.name as context_name,
                    messages_fts.rank as relevance_score
                FROM messages_fts
                JOIN messages m ON messages_fts.message_id = m.id
                JOIN conversations c ON m.conversation_id = c.id
                LEFT JOIN contexts ctx ON c.context_id = ctx.id
                WHERE messages_fts MATCH ?
                AND c.user_id = ?
            """
            params = [query, user_id]
            
            if context_id is not None:
                search_query += " AND c.context_id = ?"
                params.append(context_id)
            
            search_query += " ORDER BY relevance_score LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(search_query, params).fetchall()
            
            results = []
            for row in rows:
                message = self._row_to_message(row)
                conversation = self._row_to_conversation(row)
                
                # Get context if available
                context = None
                if row["context_id"]:
                    context = await self.get_context(row["context_id"])
                
                results.append(MemorySearchResult(
                    message=message,
                    conversation=conversation,
                    context=context,
                    relevance_score=abs(row["relevance_score"]),
                    highlights=[]  # Could implement snippet extraction
                ))
            
            return results

    def _row_to_context(self, row: sqlite3.Row) -> Context:
        """Convert a database row to a Context object."""
        return Context(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            description=row["description"],
            status=ContextStatus(row["status"]),
            metadata=json.loads(row["metadata"] or "{}"),
            document_ids=json.loads(row["document_ids"] or "[]"),
            knowledge_graph_id=row["knowledge_graph_id"],
            default_agents=json.loads(row["default_agents"] or "[]"),
            settings=json.loads(row["settings"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )

    def _row_to_user_profile(self, row: sqlite3.Row) -> UserProfile:
        """Convert a database row to a UserProfile object."""
        return UserProfile(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            email=row["email"],
            preferences=json.loads(row["preferences"] or "{}"),
            default_context_id=row["default_context_id"],
            preferred_agents=json.loads(row["preferred_agents"] or "[]"),
            language=row["language"],
            timezone=row["timezone"],
            total_conversations=row["total_conversations"],
            total_messages=row["total_messages"],
            total_contexts=row["total_contexts"],
            last_active=datetime.fromisoformat(row["last_active"]) if row["last_active"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )

    # Placeholder methods - Basic implementations
    async def create_knowledge_snapshot(self, snapshot: KnowledgeSnapshot) -> KnowledgeSnapshot:
        """Create a knowledge graph snapshot."""
        async with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO knowledge_snapshots (
                    context_id, snapshot_data, entity_count,
                    relationship_count, document_count, description,
                    version, compression, size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.context_id,
                    json.dumps(snapshot.snapshot_data),
                    snapshot.entity_count,
                    snapshot.relationship_count,
                    snapshot.document_count,
                    snapshot.description,
                    snapshot.version,
                    snapshot.compression,
                    snapshot.size_bytes
                )
            )
            snapshot.id = cursor.lastrowid
            return snapshot

    async def get_knowledge_snapshot(self, snapshot_id: int) -> Optional[KnowledgeSnapshot]:
        """Get a knowledge snapshot by ID."""
        async with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_snapshots WHERE id = ?",
                (snapshot_id,)
            ).fetchone()
            
            if not row:
                return None
            
            return KnowledgeSnapshot(
                id=row["id"],
                context_id=row["context_id"],
                snapshot_data=json.loads(row["snapshot_data"]),
                entity_count=row["entity_count"],
                relationship_count=row["relationship_count"],
                document_count=row["document_count"],
                description=row["description"],
                version=row["version"],
                compression=row["compression"],
                size_bytes=row["size_bytes"],
                created_at=datetime.fromisoformat(row["created_at"])
            )

    async def list_knowledge_snapshots(self, context_id: int, limit: int = 10) -> List[KnowledgeSnapshot]:
        """List knowledge snapshots for a context."""
        async with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM knowledge_snapshots 
                WHERE context_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
                """,
                (context_id, limit)
            ).fetchall()
            
            return [
                KnowledgeSnapshot(
                    id=row["id"],
                    context_id=row["context_id"],
                    snapshot_data=json.loads(row["snapshot_data"]),
                    entity_count=row["entity_count"],
                    relationship_count=row["relationship_count"],
                    document_count=row["document_count"],
                    description=row["description"],
                    version=row["version"],
                    compression=row["compression"],
                    size_bytes=row["size_bytes"],
                    created_at=datetime.fromisoformat(row["created_at"])
                )
                for row in rows
            ]

    async def get_latest_knowledge_snapshot(self, context_id: int) -> Optional[KnowledgeSnapshot]:
        """Get the latest knowledge snapshot for a context."""
        snapshots = await self.list_knowledge_snapshots(context_id, limit=1)
        return snapshots[0] if snapshots else None

    async def search_by_entities(self, user_id: str, entities: List[str], context_id: Optional[int] = None, limit: int = 20) -> List[MemorySearchResult]:
        """Search messages that mention specific entities."""
        # Simple implementation - search for any of the entities
        entity_query = " OR ".join([f'"{entity}"' for entity in entities])
        return await self.search_messages(user_id, entity_query, context_id, limit)

    async def get_conversation_summary(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Get a summary of a conversation."""
        async with self._get_connection() as conn:
            conv = await self.get_conversation(conversation_id)
            if not conv:
                return None
            
            messages = await self.get_messages(conversation_id)
            
            return {
                "conversation_id": conversation_id,
                "title": conv.title,
                "summary": conv.summary,
                "message_count": len(messages),
                "total_tokens": conv.total_tokens,
                "agents_used": conv.agents_used,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat()
            }

    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get usage statistics for a user."""
        profile = await self.get_user_profile(user_id)
        if not profile:
            return {}
        
        return {
            "user_id": user_id,
            "total_conversations": profile.total_conversations,
            "total_messages": profile.total_messages,
            "total_contexts": profile.total_contexts,
            "last_active": profile.last_active.isoformat() if profile.last_active else None,
            "created_at": profile.created_at.isoformat()
        }

    async def get_context_statistics(self, context_id: int) -> Dict[str, Any]:
        """Get statistics for a context."""
        context = await self.get_context(context_id)
        if not context:
            return {}
        
        conversations = await self.list_conversations(context.user_id, context_id)
        
        return {
            "context_id": context_id,
            "name": context.name,
            "conversation_count": len(conversations),
            "document_count": len(context.document_ids),
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat()
        }

    async def get_agent_usage_stats(self, user_id: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get agent usage statistics."""
        async with self._get_connection() as conn:
            query = """
                SELECT agent_used, COUNT(*) as count
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.user_id = ? AND agent_used IS NOT NULL
            """
            params = [user_id]
            
            if start_date:
                query += " AND m.timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND m.timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " GROUP BY agent_used"
            
            rows = conn.execute(query, params).fetchall()
            
            return {
                "user_id": user_id,
                "agent_usage": {row["agent_used"]: row["count"] for row in rows},
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }

    async def cleanup_old_conversations(self, days_to_keep: int = 90) -> int:
        """Clean up old conversations."""
        async with self._get_connection() as conn:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cursor = conn.execute(
                "DELETE FROM conversations WHERE updated_at < ?",
                (cutoff_date.isoformat(),)
            )
            return cursor.rowcount

    async def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user."""
        profile = await self.get_user_profile(user_id)
        contexts = await self.list_contexts(user_id, include_archived=True)
        conversations = await self.list_conversations(user_id)
        
        # Get all messages for all conversations
        all_messages = []
        for conv in conversations:
            messages = await self.get_messages(conv.id)
            all_messages.extend([msg.model_dump() for msg in messages])
        
        return {
            "user_profile": profile.model_dump() if profile else None,
            "contexts": [ctx.model_dump() for ctx in contexts],
            "conversations": [conv.model_dump() for conv in conversations],
            "messages": all_messages,
            "export_date": datetime.now().isoformat()
        }

    async def import_user_data(self, user_id: str, data: Dict[str, Any]) -> bool:
        """Import user data from an export."""
        # This is a simplified implementation - in production, you'd want more error handling
        try:
            # Import user profile
            if data.get("user_profile"):
                profile = UserProfile(**data["user_profile"])
                await self.create_or_update_user_profile(profile)
            
            # Import contexts
            context_mapping = {}
            for ctx_data in data.get("contexts", []):
                old_id = ctx_data.pop("id", None)
                context = Context(**ctx_data)
                context = await self.create_context(context)
                if old_id:
                    context_mapping[old_id] = context.id
            
            # Import conversations
            conv_mapping = {}
            for conv_data in data.get("conversations", []):
                old_id = conv_data.pop("id", None)
                # Map old context_id to new one
                if conv_data.get("context_id") in context_mapping:
                    conv_data["context_id"] = context_mapping[conv_data["context_id"]]
                
                conversation = Conversation(**conv_data)
                conversation = await self.create_conversation(conversation)
                if old_id:
                    conv_mapping[old_id] = conversation.id
            
            # Import messages
            for msg_data in data.get("messages", []):
                msg_data.pop("id", None)
                # Map old conversation_id to new one
                if msg_data.get("conversation_id") in conv_mapping:
                    msg_data["conversation_id"] = conv_mapping[msg_data["conversation_id"]]
                    message = Message(**msg_data)
                    await self.add_message(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Error importing user data: {e}")
            return False 