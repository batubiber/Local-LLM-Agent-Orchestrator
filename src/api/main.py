"""
FastAPI application for the GraphRAG Agent Orchestrator.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.agent_factory import RAGStrategy
from src.agents.graph_rag_agent import GraphRAGAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.code_rag_agent import CodeRAGAgent
from src.agents.web_scraping_agent import WebScrapingAgent
from src.memory.memory_manager import MemoryManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator: Optional[Orchestrator] = None
memory_manager: Optional[MemoryManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    global orchestrator, memory_manager

    # Startup
    logger.info("Starting GraphRAG API server...")

    try:
        # Initialize memory manager
        memory_manager = MemoryManager()
        await memory_manager.initialize()
        logger.info("Memory manager initialized successfully")

        # Initialize orchestrator
        orchestrator = await create_orchestrator()
        logger.info("Orchestrator initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down GraphRAG API server...")
        if memory_manager:
            await memory_manager.close()


# Create FastAPI app
app = FastAPI(
    title="GraphRAG Agent Orchestrator API",
    description="API for the GraphRAG system with multiple specialized agents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str = Field(..., description="The query to process")
    agent_name: Optional[str] = Field(None, description="Specific agent to use (optional)")


class QueryResponse(BaseModel):
    """Response model for queries."""
    query: str = Field(..., description="The original query")
    agents_used: List[str] = Field(..., description="List of agents that processed the query")
    responses: Dict[str, Any] = Field(..., description="Responses from each agent")
    success: bool = Field(..., description="Whether the query was processed successfully")


class AgentStatus(BaseModel):
    """Model for agent status information."""
    name: str = Field(..., description="Agent name")
    type: str = Field(..., description="Agent class type")
    enabled: bool = Field(..., description="Whether the agent is enabled")
    can_handle_requests: bool = Field(..., description="Whether the agent can handle requests")


class SystemStatus(BaseModel):
    """Model for system status."""
    status: str = Field(..., description="Overall system status")
    agents: List[AgentStatus] = Field(..., description="Status of all agents")
    total_agents: int = Field(..., description="Total number of agents")


class SummaryRequest(BaseModel):
    """Request model for summaries."""
    content: str = Field(..., description="Content to summarize")
    summary_type: Optional[str] = Field("narrative", description="Type of summary (executive, technical, narrative, bullet_points, abstract)")


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis."""
    code: str = Field(..., description="Code to analyze")
    language: Optional[str] = Field(None, description="Programming language")
    analysis_type: Optional[str] = Field("explain", description="Type of analysis (review, explain, debug, optimize, refactor, security, test)")


class WebScrapingRequest(BaseModel):
    """Request model for web scraping."""
    urls: List[str] = Field(..., description="URLs to scrape")
    analysis_request: Optional[str] = Field("", description="Analysis to perform on scraped content")


# Memory-related models
class UserRequest(BaseModel):
    """Request model for user operations."""
    user_id: str = Field(..., description="User identifier")
    name: Optional[str] = Field(None, description="User's display name")
    email: Optional[str] = Field(None, description="User's email address")


class ContextRequest(BaseModel):
    """Request model for context operations."""
    name: str = Field(..., description="Context name")
    description: Optional[str] = Field(None, description="Context description")
    set_as_active: bool = Field(True, description="Set as active context")


class ContextResponse(BaseModel):
    """Response model for context information."""
    id: int = Field(..., description="Context ID")
    name: str = Field(..., description="Context name")
    description: Optional[str] = Field(None, description="Context description")
    status: str = Field(..., description="Context status")
    document_count: int = Field(0, description="Number of documents in context")
    created_at: str = Field(..., description="Creation timestamp")
    is_active: bool = Field(False, description="Whether this is the active context")


class ConversationResponse(BaseModel):
    """Response model for conversation information."""
    id: int = Field(..., description="Conversation ID")
    title: str = Field(..., description="Conversation title")
    summary: Optional[str] = Field(None, description="Conversation summary")
    message_count: int = Field(0, description="Number of messages")
    context_id: Optional[int] = Field(None, description="Associated context ID")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class MemorySearchRequest(BaseModel):
    """Request model for memory search."""
    query: str = Field(..., description="Search query")
    context_id: Optional[int] = Field(None, description="Limit search to specific context")
    limit: int = Field(20, description="Maximum number of results")


class MemorySearchResponse(BaseModel):
    """Response model for memory search results."""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    messages: List[Dict[str, Any]] = Field(..., description="List of messages")
    conversation_id: int = Field(..., description="Conversation ID")
    title: str = Field(..., description="Conversation title")


# Dependency to get orchestrator
async def get_orchestrator() -> Orchestrator:
    """Get the global orchestrator instance."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


# Dependency to get memory manager
async def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    if memory_manager is None:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    return memory_manager


async def create_orchestrator() -> Orchestrator:
    """Create and configure the orchestrator with all agents."""
    # Get Azure OpenAI configuration
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    # Get Milvus configuration
    milvus_uri = os.getenv("MILVUS_URI")
    milvus_token = os.getenv("MILVUS_TOKEN")
    
    if not all([azure_endpoint, azure_api_key, azure_deployment]):
        raise ValueError("Azure OpenAI configuration is incomplete. Please check your .env file.")
    
    if not milvus_uri:
        raise ValueError("Milvus URI is not configured. Please check your .env file.")
    
    # Initialize agents
    agents = []
    strategies = {}
    
    try:
        # GraphRAG Agent
        graph_rag_agent = GraphRAGAgent(
            name="graph_rag",
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_deployment=azure_deployment,
            api_version=api_version,
            temperature=float(os.getenv("TEMPERATURE", 0.0)),
            use_llm_extraction=os.getenv("USE_LLM_EXTRACTION", "true").lower() == "true"
        )
        agents.append(graph_rag_agent)
        strategies["graph_rag"] = RAGStrategy()
        logger.info("GraphRAG agent initialized")
        
    except Exception as e:
        logger.warning(f"Failed to initialize GraphRAG agent: {e}")
    
    try:
        # Summary Agent
        summary_agent = SummaryAgent(
            name="summary",
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_deployment=azure_deployment,
            api_version=api_version,
            temperature=0.3
        )
        agents.append(summary_agent)
        strategies["summary"] = RAGStrategy()
        logger.info("Summary agent initialized")
        
    except Exception as e:
        logger.warning(f"Failed to initialize Summary agent: {e}")
    
    try:
        # Code RAG Agent
        code_agent = CodeRAGAgent(
            name="code_rag",
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_deployment=azure_deployment,
            api_version=api_version,
            temperature=0.1
        )
        agents.append(code_agent)
        strategies["code_rag"] = RAGStrategy()
        logger.info("Code RAG agent initialized")
        
    except Exception as e:
        logger.warning(f"Failed to initialize Code RAG agent: {e}")
    
    try:
        # Web Scraping Agent
        web_agent = WebScrapingAgent(
            name="web_scraping",
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_deployment=azure_deployment,
            api_version=api_version,
            temperature=0.2
        )
        agents.append(web_agent)
        strategies["web_scraping"] = RAGStrategy()
        logger.info("Web scraping agent initialized")
        
    except Exception as e:
        logger.warning(f"Failed to initialize Web scraping agent: {e}")
    
    if not agents:
        raise ValueError("No agents could be initialized. Please check your configuration.")
    
    return Orchestrator(agents=agents, strategies=strategies)


# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "GraphRAG Agent Orchestrator API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "GraphRAG Agent Orchestrator"}


@app.get("/status", response_model=SystemStatus)
async def get_system_status(orchestrator: Orchestrator = Depends(get_orchestrator)):
    """Get system and agent status."""
    agent_status = orchestrator.get_agent_status()
    
    agents = [
        AgentStatus(
            name=name,
            type=info["type"],
            enabled=info["enabled"],
            can_handle_requests=info["can_handle_requests"]
        )
        for name, info in agent_status.items()
    ]
    
    return SystemStatus(
        status="operational",
        agents=agents,
        total_agents=len(agents)
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    memory: MemoryManager = Depends(get_memory_manager),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Process a query using appropriate agents with memory support."""
    try:
        # Set user in memory manager if provided
        if user_id:
            await memory.set_user(user_id)
        
        # Save the query to memory
        if memory._current_user_id:
            await memory.add_user_message(request.query)
        
        if request.agent_name:
            # Use specific agent if requested
            agent = next((a for a in orchestrator._agents if a.name == request.agent_name), None)
            if not agent:
                raise HTTPException(status_code=400, detail=f"Agent '{request.agent_name}' not found")
            
            if not agent.can_handle(request.query):
                raise HTTPException(status_code=400, detail=f"Agent '{request.agent_name}' cannot handle this query")
            
            strategy = orchestrator._strategies.get(agent.name)
            if not strategy:
                raise HTTPException(status_code=500, detail=f"No strategy configured for agent '{request.agent_name}'")
            
            response = strategy.execute(agent, request.query)
            responses = {agent.name: response}
            agents_used = [agent.name]
        else:
            # Use orchestrator to determine appropriate agents
            responses = orchestrator.handle(request.query)
            agents_used = list(responses.keys())
        
        if not responses:
            raise HTTPException(status_code=400, detail="No agents could handle this query")
        
        # Save responses to memory
        if memory._current_user_id:
            for agent_name, response_data in responses.items():
                if isinstance(response_data, dict) and response_data.get("response"):
                    await memory.add_assistant_message(
                        content=response_data["response"],
                        agent_used=agent_name,
                        metadata=response_data.get("metadata", {}),
                        tokens_used=response_data.get("tokens_used")
                    )
        
        return QueryResponse(
            query=request.query,
            agents_used=agents_used,
            responses=responses,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@app.post("/summary", response_model=Dict[str, Any])
async def generate_summary(
    request: SummaryRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """Generate a specific type of summary."""
    try:
        # Find summary agent
        summary_agent = next((a for a in orchestrator._agents if a.name == "summary"), None)
        if not summary_agent:
            raise HTTPException(status_code=503, detail="Summary agent not available")
        
        # Generate summary by type
        if hasattr(summary_agent, 'generate_summary_by_type'):
            result = summary_agent.generate_summary_by_type(request.content, request.summary_type)
        else:
            # Fallback to regular summarization
            query = f"Create a {request.summary_type} summary of: {request.content}"
            result = summary_agent.retrieve_and_generate(query)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/code/analyze", response_model=Dict[str, Any])
async def analyze_code(
    request: CodeAnalysisRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """Analyze code using the Code RAG agent."""
    try:
        # Find code agent
        code_agent = next((a for a in orchestrator._agents if a.name == "code_rag"), None)
        if not code_agent:
            raise HTTPException(status_code=503, detail="Code RAG agent not available")
        
        # Construct query
        language_info = f" (Language: {request.language})" if request.language else ""
        query = f"Please {request.analysis_type} this code{language_info}:\n\n```\n{request.code}\n```"
        
        result = code_agent.retrieve_and_generate(query)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing code: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/web/scrape", response_model=Dict[str, Any])
async def scrape_web_content(
    request: WebScrapingRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """Scrape and analyze web content."""
    try:
        # Find web scraping agent
        web_agent = next((a for a in orchestrator._agents if a.name == "web_scraping"), None)
        if not web_agent:
            raise HTTPException(status_code=503, detail="Web scraping agent not available")
        
        # Construct query
        urls_text = " ".join(request.urls)
        query = f"{request.analysis_request} {urls_text}" if request.analysis_request else f"Extract content from {urls_text}"
        
        result = web_agent.retrieve_and_generate(query)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scraping web content: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/agents", response_model=List[Dict[str, Any]])
async def list_agents(orchestrator: Orchestrator = Depends(get_orchestrator)):
    """List all available agents and their capabilities."""
    agents_info = []
    
    for agent in orchestrator._agents:
        agent_info = {
            "name": agent.name,
            "type": agent.__class__.__name__,
            "description": agent.__class__.__doc__ or "No description available",
            "can_handle_summary": hasattr(agent, 'summarize'),
            "can_handle_rag": hasattr(agent, 'retrieve_and_generate')
        }
        
        # Add specific capabilities based on agent type
        if hasattr(agent, 'supported_languages'):
            agent_info["supported_languages"] = getattr(agent, 'supported_languages', [])
        
        if hasattr(agent, 'summary_templates'):
            agent_info["summary_types"] = list(getattr(agent, 'summary_templates', {}).keys())
        
        if hasattr(agent, 'analysis_templates'):
            agent_info["analysis_types"] = list(getattr(agent, 'analysis_templates', {}).keys())
        
        agents_info.append(agent_info)
    
    return agents_info


# Memory API endpoints

@app.post("/users/set", response_model=Dict[str, Any])
async def set_user(
    request: UserRequest,
    memory: MemoryManager = Depends(get_memory_manager)
):
    """Set or create a user profile."""
    try:
        profile = await memory.set_user(
            user_id=request.user_id,
            create_if_not_exists=True
        )
        
        # Update profile if name or email provided
        if request.name or request.email:
            if request.name:
                profile.name = request.name
            if request.email:
                profile.email = request.email
            profile = await memory.store.create_or_update_user_profile(profile)
        
        return {
            "user_id": profile.user_id,
            "name": profile.name,
            "email": profile.email,
            "created": True
        }
    except Exception as e:
        logger.error(f"Error setting user: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/contexts", response_model=List[ContextResponse])
async def list_contexts(
    memory: MemoryManager = Depends(get_memory_manager),
    user_id: str = Header(..., alias="X-User-ID"),
    include_archived: bool = False
):
    """List all contexts for a user."""
    try:
        await memory.set_user(user_id)
        contexts = await memory.list_contexts(include_archived=include_archived)
        current_context = await memory.get_current_context()
        
        return [
            ContextResponse(
                id=ctx.id,
                name=ctx.name,
                description=ctx.description,
                status=ctx.status.value,
                document_count=len(ctx.document_ids),
                created_at=ctx.created_at.isoformat(),
                is_active=current_context and ctx.id == current_context.id
            )
            for ctx in contexts
        ]
    except Exception as e:
        logger.error(f"Error listing contexts: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/contexts", response_model=ContextResponse)
async def create_context(
    request: ContextRequest,
    memory: MemoryManager = Depends(get_memory_manager),
    user_id: str = Header(..., alias="X-User-ID")
):
    """Create a new context."""
    try:
        await memory.set_user(user_id)
        context = await memory.create_context(
            name=request.name,
            description=request.description,
            set_as_active=request.set_as_active
        )
        
        return ContextResponse(
            id=context.id,
            name=context.name,
            description=context.description,
            status=context.status.value,
            document_count=0,
            created_at=context.created_at.isoformat(),
            is_active=request.set_as_active
        )
    except Exception as e:
        logger.error(f"Error creating context: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.put("/contexts/{context_id}/activate")
async def activate_context(
    context_id: int,
    memory: MemoryManager = Depends(get_memory_manager),
    user_id: str = Header(..., alias="X-User-ID")
):
    """Switch to a different context."""
    try:
        await memory.set_user(user_id)
        context = await memory.switch_context(context_id)
        
        return {
            "message": f"Switched to context '{context.name}'",
            "context_id": context.id
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error switching context: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    memory: MemoryManager = Depends(get_memory_manager),
    user_id: str = Header(..., alias="X-User-ID"),
    context_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0
):
    """List conversations for a user."""
    try:
        await memory.set_user(user_id)
        conversations = await memory.list_conversations(
            context_id=context_id,
            limit=limit,
            offset=offset
        )
        
        return [
            ConversationResponse(
                id=conv.id,
                title=conv.title,
                summary=conv.summary,
                message_count=conv.total_messages,
                context_id=conv.context_id,
                created_at=conv.created_at.isoformat(),
                updated_at=conv.updated_at.isoformat()
            )
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: int,
    memory: MemoryManager = Depends(get_memory_manager),
    user_id: str = Header(..., alias="X-User-ID")
):
    """Get full conversation history."""
    try:
        await memory.set_user(user_id)
        
        # Get conversation details
        conversation = await memory.store.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        if conversation.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get messages
        messages = await memory.store.get_messages(conversation_id)
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            title=conversation.title or f"Conversation {conversation_id}",
            messages=[
                {
                    "id": msg.id,
                    "role": msg.role.value,
                    "content": msg.content,
                    "agent_used": msg.agent_used,
                    "timestamp": msg.timestamp.isoformat(),
                    "tokens_used": msg.tokens_used
                }
                for msg in messages
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/conversations/{conversation_id}/resume")
async def resume_conversation(
    conversation_id: int,
    memory: MemoryManager = Depends(get_memory_manager),
    user_id: str = Header(..., alias="X-User-ID")
):
    """Resume an existing conversation."""
    try:
        await memory.set_user(user_id)
        conversation = await memory.resume_conversation(conversation_id)
        
        return {
            "message": "Conversation resumed",
            "conversation_id": conversation.id,
            "title": conversation.title
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error resuming conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    request: MemorySearchRequest,
    memory: MemoryManager = Depends(get_memory_manager),
    user_id: str = Header(..., alias="X-User-ID")
):
    """Search through conversation history."""
    try:
        await memory.set_user(user_id)
        results = await memory.search_memory(
            query=request.query,
            context_id=request.context_id,
            limit=request.limit
        )
        
        return MemorySearchResponse(
            results=[
                {
                    "content": result.message.content,
                    "role": result.message.role.value,
                    "timestamp": result.message.timestamp.isoformat(),
                    "relevance_score": result.relevance_score,
                    "conversation_id": result.conversation.id,
                    "conversation_title": result.conversation.title,
                    "context_name": result.context.name if result.context else None
                }
                for result in results
            ],
            total_results=len(results)
        )
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/memory/statistics")
async def get_memory_statistics(
    memory: MemoryManager = Depends(get_memory_manager),
    user_id: str = Header(..., alias="X-User-ID")
):
    """Get user's memory usage statistics."""
    try:
        await memory.set_user(user_id)
        stats = await memory.get_statistics()
        
        # Get agent usage stats
        agent_stats = await memory.store.get_agent_usage_stats(user_id)
        
        return {
            **stats,
            **agent_stats
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/memory/export")
async def export_memory(
    memory: MemoryManager = Depends(get_memory_manager),
    user_id: str = Header(..., alias="X-User-ID")
):
    """Export all user data."""
    try:
        await memory.set_user(user_id)
        data = await memory.export_data()
        
        return data
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
