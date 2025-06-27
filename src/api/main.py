"""
FastAPI application for the GraphRAG Agent Orchestrator.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.agent_factory import AgentFactory, RAGStrategy, SummaryStrategy
from src.agents.graph_rag_agent import GraphRAGAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.code_rag_agent import CodeRAGAgent
from src.agents.web_scraping_agent import WebScrapingAgent

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    global orchestrator
    
    # Startup
    logger.info("Starting GraphRAG API server...")
    
    try:
        # Initialize orchestrator
        orchestrator = await create_orchestrator()
        logger.info("Orchestrator initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down GraphRAG API server...")


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


# Dependency to get orchestrator
async def get_orchestrator() -> Orchestrator:
    """Get the global orchestrator instance."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


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
    orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """Process a query using appropriate agents."""
    try:
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
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 