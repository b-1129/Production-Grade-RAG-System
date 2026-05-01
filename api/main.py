"""
FastAPI Backend for Production-Grade RAG System

This module provides REST API endpoints for the RAG system.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import logging

from agents.graph import create_agent, AgentState
from retriever.retriever import Retriever
from retriever.reranker import Reranker
from llm.llm_wrapper import LLMWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Production-Grade RAG System API",
    description="REST API for LangGraph-powered RAG agent",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User query", min_length=1)
    max_iterations: int = Field(default=3, ge=1, le=5, description="Max refinement iterations")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source documents in response")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    response: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    iterations: int
    context_used: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    components: Dict[str, str]


class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    directory: str = Field(default="data", description="Directory to index")
    force_rebuild: bool = Field(default=False, description="Force rebuild of index")


class IndexResponse(BaseModel):
    """Response model for indexing operation."""
    status: str
    message: str
    documents_indexed: int = 0


# ============== Global State ==============

# Lazy-loaded components
_retriever: Optional[Retriever] = None
_reranker: Optional[Reranker] = None
_llm_wrapper: Optional[LLMWrapper] = None
_agent: Optional[Any] = None


def get_retriever() -> Retriever:
    """Get or create Retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_reranker() -> Reranker:
    """Get or create Reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def get_llm_wrapper() -> LLMWrapper:
    """Get or create LLM wrapper instance."""
    global _llm_wrapper
    if _llm_wrapper is None:
        _llm_wrapper = LLMWrapper()
    return _llm_wrapper


def get_agent() -> Any:
    """Get or create LangGraph agent instance."""
    global _agent
    if _agent is None:
        _agent = create_agent(
            retriever=get_retriever(),
            reranker=get_reranker(),
            llm_wrapper=get_llm_wrapper(),
            max_iterations=3
        )
    return _agent


# ============== API Endpoints ==============

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Production-Grade RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {}
    
    # Check Retriever
    try:
        retriever = get_retriever()
        components["retriever"] = "healthy" if retriever else "unavailable"
    except Exception as e:
        components["retriever"] = f"error: {str(e)}"
    
    # Check Reranker
    try:
        reranker = get_reranker()
        components["reranker"] = "healthy" if reranker else "unavailable"
    except Exception as e:
        components["reranker"] = f"error: {str(e)}"
    
    # Check LLM Wrapper
    try:
        llm = get_llm_wrapper()
        components["llm_wrapper"] = "healthy" if llm else "unavailable"
    except Exception as e:
        components["llm_wrapper"] = f"error: {str(e)}"
    
    # Check Agent
    try:
        agent = get_agent()
        components["agent"] = "healthy" if agent else "unavailable"
    except Exception as e:
        components["agent"] = f"error: {str(e)}"
    
    overall_status = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        components=components
    )


@app.post("/query", tags=["RAG"], response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Processes the user query through the LangGraph agent pipeline:
    retrieval -> reranking -> generation -> evaluation -> (optional) refinement
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Create initial state
        initial_state: AgentState = {
            "query": request.query,
            "retrieved_docs": [],
            "reranked_docs": [],
            "context": "",
            "response": "",
            "feedback": "",
            "iteration": 0,
            "max_iterations": request.max_iterations,
            "error": None
        }
        
        # Invoke the agent
        agent = get_agent()
        result = await asyncio.to_thread(agent.invoke, initial_state)
        
        # Extract sources if requested
        sources = []
        if request.include_sources:
            docs = result.get("reranked_docs", result.get("retrieved_docs", []))
            for i, doc in enumerate(docs[:request.top_k]):
                sources.append({
                    "index": i + 1,
                    "content": doc[:500] + "..." if len(doc) > 500 else doc
                })
        
        return QueryResponse(
            query=result.get("query", request.query),
            response=result.get("response", "No response generated"),
            sources=sources,
            iterations=result.get("iteration", 0),
            context_used=len(result.get("context", "")) > 0,
            metadata={
                "max_iterations": request.max_iterations,
                "top_k": request.top_k,
                "has_error": result.get("error") is not None
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/index", tags=["Indexing"], response_model=IndexResponse)
async def index_documents(
    request: IndexRequest,
    background_tasks: BackgroundTasks
):
    """
    Index documents from a directory.
    
    This is a placeholder for document indexing functionality.
    In production, this would trigger the ingestion pipeline.
    """
    try:
        # Placeholder: In production, this would call the data loading
        # and embedding pipeline
        logger.info(f"Indexing documents from: {request.directory}")
        
        return IndexResponse(
            status="success",
            message=f"Document indexing initiated for directory: {request.directory}",
            documents_indexed=0
        )
        
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error indexing: {str(e)}")


@app.get("/sources", tags=["RAG"])
async def get_sources(top_k: int = 5):
    """
    Get available source documents from the vector store.
    
    Returns a list of indexed documents.
    """
    try:
        retriever = get_retriever()
        faiss_db = retriever.faiss_db
        
        # Get metadata from FaissDB
        metadata = faiss_db.metadata if hasattr(faiss_db, 'metadata') else []
        
        sources = []
        for i, meta in enumerate(metadata[:top_k]):
            content = ""
            if isinstance(meta, dict):
                content = meta.get("text", "")[:300]
            elif hasattr(meta, 'page_content'):
                content = meta.page_content[:300]
            else:
                content = str(meta)[:300]
            
            sources.append({
                "index": i + 1,
                "content": content + "..." if len(content) >= 300 else content
            })
        
        return {
            "total_documents": len(metadata),
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error retrieving sources: {str(e)}")
        # Return empty response if vector store not initialized
        return {
            "total_documents": 0,
            "sources": [],
            "message": "Vector store not initialized. Please build the index first."
        }


# ============== Main Entry Point ==============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )