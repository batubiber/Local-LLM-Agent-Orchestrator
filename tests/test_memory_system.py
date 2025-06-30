"""
Tests for the Persistent Memory System.
"""
import pytest
import asyncio
from datetime import datetime

from src.memory.memory_manager import MemoryManager
from src.memory.sqlite_memory_store import SQLiteMemoryStore
from src.memory.memory_models import (
    UserProfile,
    Context,
    Conversation,
    Message,
    MessageRole
)


@pytest.fixture
async def memory_manager():
    """Create a memory manager with in-memory database for testing."""
    store = SQLiteMemoryStore(":memory:")  # In-memory SQLite
    manager = MemoryManager(store)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.mark.asyncio
async def test_user_management(memory_manager):
    """Test user profile creation and management."""
    # Set user
    profile = await memory_manager.set_user("test_user")
    assert profile.user_id == "test_user"
    
    # Get current user
    current_user = await memory_manager.get_current_user_id()
    assert current_user == "test_user"


@pytest.mark.asyncio
async def test_context_management(memory_manager):
    """Test context creation and switching."""
    # Set user first
    await memory_manager.set_user("test_user")
    
    # Create context
    context = await memory_manager.create_context(
        name="Test Project",
        description="Test context for unit tests"
    )
    assert context.name == "Test Project"
    assert context.user_id == "test_user"
    
    # Get current context
    current = await memory_manager.get_current_context()
    assert current.id == context.id
    
    # Create another context
    context2 = await memory_manager.create_context(
        name="Another Project",
        set_as_active=False
    )
    
    # List contexts
    contexts = await memory_manager.list_contexts()
    assert len(contexts) == 2
    
    # Switch context
    switched = await memory_manager.switch_context(context2.id)
    assert switched.id == context2.id


@pytest.mark.asyncio
async def test_conversation_management(memory_manager):
    """Test conversation creation and message handling."""
    # Set user
    await memory_manager.set_user("test_user")
    
    # Start conversation
    conversation = await memory_manager.start_conversation(
        title="Test Conversation"
    )
    assert conversation.title == "Test Conversation"
    
    # Add messages
    user_msg = await memory_manager.add_user_message("Hello, how are you?")
    assert user_msg.role == MessageRole.USER
    assert user_msg.content == "Hello, how are you?"
    
    assistant_msg = await memory_manager.add_assistant_message(
        content="I'm doing well, thank you!",
        agent_used="test_agent",
        tokens_used=10
    )
    assert assistant_msg.role == MessageRole.ASSISTANT
    assert assistant_msg.agent_used == "test_agent"
    
    # Get messages
    messages = await memory_manager.get_conversation_messages()
    assert len(messages) == 2
    assert messages[0].content == "Hello, how are you?"
    assert messages[1].content == "I'm doing well, thank you!"


@pytest.mark.asyncio
async def test_memory_search(memory_manager):
    """Test searching through memory."""
    # Set up test data
    await memory_manager.set_user("test_user")
    await memory_manager.start_conversation()
    
    # Add some messages
    await memory_manager.add_user_message("Tell me about machine learning")
    await memory_manager.add_assistant_message(
        "Machine learning is a subset of artificial intelligence...",
        agent_used="test_agent"
    )
    
    await memory_manager.add_user_message("What about deep learning?")
    await memory_manager.add_assistant_message(
        "Deep learning is a subset of machine learning...",
        agent_used="test_agent"
    )
    
    # Search memory
    results = await memory_manager.search_memory("machine learning")
    assert len(results) > 0
    assert any("machine learning" in r.message.content.lower() for r in results)


@pytest.mark.asyncio
async def test_quick_save(memory_manager):
    """Test the quick save functionality."""
    await memory_manager.set_user("test_user")
    
    user_msg, assistant_msg = await memory_manager.quick_save(
        user_message="What is Python?",
        assistant_response="Python is a high-level programming language...",
        agent_used="test_agent"
    )
    
    assert user_msg.content == "What is Python?"
    assert assistant_msg.content == "Python is a high-level programming language..."
    assert assistant_msg.agent_used == "test_agent"


@pytest.mark.asyncio
async def test_statistics(memory_manager):
    """Test getting usage statistics."""
    await memory_manager.set_user("test_user")
    
    # Create some data
    await memory_manager.create_context("Test Context")
    await memory_manager.start_conversation()
    await memory_manager.quick_save(
        "Test question",
        "Test answer",
        agent_used="test_agent"
    )
    
    # Get statistics
    stats = await memory_manager.get_statistics()
    assert stats["total_contexts"] == 1
    assert stats["total_conversations"] == 1
    assert stats["total_messages"] == 2


@pytest.mark.asyncio
async def test_export_import(memory_manager):
    """Test data export functionality."""
    await memory_manager.set_user("test_user")
    
    # Create some data
    context = await memory_manager.create_context("Export Test")
    await memory_manager.start_conversation(title="Export Conversation")
    await memory_manager.quick_save("Question", "Answer", "test_agent")
    
    # Export data
    exported_data = await memory_manager.export_data()
    
    assert "user_profile" in exported_data
    assert "contexts" in exported_data
    assert "conversations" in exported_data
    assert "messages" in exported_data
    assert len(exported_data["contexts"]) == 1
    assert len(exported_data["conversations"]) == 1
    assert len(exported_data["messages"]) == 2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 