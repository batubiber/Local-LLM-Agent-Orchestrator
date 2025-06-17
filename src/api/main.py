"""
FastAPI application integrating the Orchestrator with Dependency Injection.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.orchestrator.app import create_orchestrator
from src.orchestrator.orchestrator import Orchestrator


app = FastAPI(
    title="Local LLM Agent Orchestrator",
    description="RAG-enabled multi-agent system for local LLM inference",
    version="0.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
_orchestrator: Optional[Orchestrator] = None

def get_orchestrator() -> Orchestrator:
    """Provide a singleton Orchestrator instance via DI."""
    global _orchestrator
    if _orchestrator is None:
        model_path = os.getenv("LLM_MODEL_PATH")
        if not model_path:
            raise HTTPException(
                status_code=500,
                detail="LLM_MODEL_PATH environment variable not set"
            )
        _orchestrator = create_orchestrator(model_path=model_path)
    return _orchestrator


class QueryRequest(BaseModel):
    """Request schema for processing queries."""
    query: str


class QueryResponse(BaseModel):
    """Response schema for query results."""
    response: str
    sources: list[dict] = []


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Local LLM Agent Orchestrator API",
        "version": "0.1.0",
        "description": "RAG-enabled multi-agent system for local LLM inference",
        "endpoints": {
            "/": "This documentation",
            "/query": "Process a query through the agent system",
            "/health": "Check system health"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator)
) -> QueryResponse:
    """
    Process a query through the agent system.
    
    The system will:
    1. Analyze the query
    2. Retrieve relevant context if needed
    3. Generate a response using the appropriate agent
    """
    try:
        result = orchestrator.handle(request.query)
        return QueryResponse(
            response=result.get("response", ""),
            sources=result.get("sources", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check(
    orchestrator: Orchestrator = Depends(get_orchestrator)
) -> dict:
    """Check system health."""
    try:
        # Basic health check
        return {
            "status": "healthy",
            "agents": orchestrator.get_agent_status()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"System unhealthy: {str(e)}") from e
