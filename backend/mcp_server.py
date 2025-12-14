"""
ContextPilot MCP Server
FastMCP server with simplified tools + REST API endpoints via FastAPI.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastmcp import FastMCP

from .config import get_config
from .firestore_db import get_firestore_db as get_db, CrawlStatus
from .embed_manager import get_embed_manager
from .crawl_manager import get_crawl_manager
from .normalizer import get_normalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("contextpilot")

# ==================== MCP Server ====================

mcp = FastMCP("ContextPilot")


@mcp.tool()
def search_documentation(
    query: str,
    limit: int = 10,
    url_filter: str = "",
) -> str:
    """
    Search indexed documentation.
    
    Args:
        query: The search query
        limit: Maximum number of results (default 10)
        url_filter: Optional URL prefix to filter results
    
    Returns:
        JSON with search results including url, title, score, and content.
    """
    try:
        logger.info(f"Search query: {query}")
        embed_manager = get_embed_manager()
        
        # Try normalized docs first
        normalized_results = embed_manager.search_normalized(
            query=query,
            limit=limit,
            url_filter=url_filter,
        )
        
        if normalized_results:
            return json.dumps({
                "results": [
                    {
                        "url": r.url,
                        "title": r.title,
                        "score": r.score,
                        "content": r.content,
                        "type": "normalized",
                    }
                    for r in normalized_results
                ],
                "source": "normalized",
            }, ensure_ascii=False)
        
        # Fall back to raw docs
        results = embed_manager.search(
            query=query,
            limit=limit,
            url_filter=url_filter,
        )
        
        return json.dumps({
            "results": [
                {
                    "url": r.url,
                    "page_url": r.page_url,
                    "title": r.title,
                    "score": r.score,
                    "content": r.content,
                    "type": "raw",
                }
                for r in results
            ],
            "source": "raw",
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.exception("Search failed")
        return json.dumps({"error": str(e)})


@mcp.tool()
def crawl_url(url: str) -> str:
    """
    Crawl a URL and index its content.
    
    Args:
        url: The URL to crawl
    
    Returns:
        JSON with crawl status and number of chunks indexed.
    """
    try:
        logger.info(f"Crawling: {url}")
        crawl_manager = get_crawl_manager()
        
        chunks_indexed = crawl_manager.crawl_and_index(url)
        
        return json.dumps({
            "success": chunks_indexed > 0,
            "url": url,
            "chunks_indexed": chunks_indexed,
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.exception("Crawl failed")
        return json.dumps({"error": str(e)})


@mcp.tool()
def build_normalized_doc(url_prefix: str, title: str) -> str:
    """
    Build a normalized document from all chunks matching a URL prefix.
    
    Args:
        url_prefix: URL prefix to match (e.g., "https://docs.example.com/api")
        title: Title for the normalized document
    
    Returns:
        JSON with normalization result.
    """
    try:
        logger.info(f"Normalizing: {url_prefix} -> {title}")
        normalizer = get_normalizer()
        
        result = normalizer.normalize(url_prefix=url_prefix, title=title)
        
        return json.dumps({
            "success": result.success,
            "url_prefix": result.url_prefix,
            "title": result.title,
            "raw_chunk_count": result.raw_chunk_count,
            "pinecone_id": result.pinecone_id if result.success else None,
            "error": result.error,
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.exception("Normalization failed")
        return json.dumps({"error": str(e)})


@mcp.tool()
def health_status() -> str:
    """
    Get system health status and statistics.
    
    Returns:
        JSON with database stats, index stats, and service status.
    """
    try:
        config = get_config()
        db = get_db()
        embed_manager = get_embed_manager()
        crawl_manager = get_crawl_manager()
        
        db_stats = db.get_stats()
        index_stats = embed_manager.get_index_stats()
        
        return json.dumps({
            "status": "healthy",
            "services": {
                "pinecone": embed_manager.is_ready,
                "firecrawl": crawl_manager.has_firecrawl,
            },
            "database": db_stats,
            "index": index_stats,
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.exception("Health check failed")
        return json.dumps({"status": "error", "error": str(e)})


# ==================== REST API (FastAPI) ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting ContextPilot API server")
    yield
    logger.info("Shutting down ContextPilot API server")


api = FastAPI(
    title="ContextPilot API",
    description="Open-source context augmentation layer for AI agents",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for frontend - allow all origins for cloud deployment
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API Authentication ====================

async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    """Verify API key for protected endpoints."""
    config = get_config()
    
    # Skip auth if not enabled or no key configured
    if not config.has_auth:
        return True
    
    # Check X-API-Key header
    if x_api_key and x_api_key == config.auth.api_key:
        return True
    
    # Check Authorization: Bearer header
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            if parts[1] == config.auth.api_key:
                return True
    
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ==================== Pydantic Models ====================

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    url_filter: str = ""


class CrawlRequest(BaseModel):
    url: str


class NormalizeRequest(BaseModel):
    url_prefix: str
    title: str


class SearchResultItem(BaseModel):
    url: str
    page_url: Optional[str] = None
    title: str
    score: float
    content: str
    type: str


class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    source: str


# ==================== Public Endpoints ====================

@api.get("/")
async def root():
    """API root - returns service info."""
    return {
        "name": "ContextPilot",
        "version": "2.0.0",
        "description": "Open-source context augmentation layer for AI agents",
        "docs": "/docs",
    }


@api.get("/health")
async def health():
    """Health check endpoint."""
    try:
        embed_manager = get_embed_manager()
        return {
            "status": "healthy",
            "pinecone": embed_manager.is_ready,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@api.get("/api/stats")
async def get_stats():
    """Get dashboard statistics."""
    try:
        db = get_db()
        embed_manager = get_embed_manager()
        
        db_stats = db.get_stats()
        index_stats = embed_manager.get_index_stats()
        
        return {
            "database": db_stats,
            "index": index_stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/sources")
async def list_sources():
    """List all indexed sources."""
    try:
        db = get_db()
        return {"sources": db.list_indexed_sources()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Protected Endpoints ====================

@api.post("/api/crawl")
async def start_crawl(
    request: CrawlRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key),
):
    """Start a crawl job (requires API key)."""
    try:
        db = get_db()
        job = db.create_crawl_job(request.url)
        
        # Run crawl in background
        def do_crawl():
            crawl_manager = get_crawl_manager()
            crawl_manager.crawl_and_index(request.url)
        
        background_tasks.add_task(do_crawl)
        
        return {
            "job_id": job.id,
            "url": request.url,
            "status": "started",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/crawls")
async def list_crawls(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List crawl jobs."""
    try:
        db = get_db()
        status_filter = CrawlStatus(status) if status else None
        jobs = db.list_crawl_jobs(status=status_filter, limit=limit, offset=offset)
        
        return {
            "jobs": [
                {
                    "id": job.id,
                    "url": job.url,
                    "status": job.status.value,
                    "method": job.method.value if job.method else None,
                    "chunks_count": job.chunks_count,
                    "error_message": job.error_message,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at,
                    "created_at": job.created_at,
                }
                for job in jobs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/crawls/{job_id}")
async def get_crawl(job_id: int):
    """Get a specific crawl job."""
    try:
        db = get_db()
        job = db.get_crawl_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "id": job.id,
            "url": job.url,
            "status": job.status.value,
            "method": job.method.value if job.method else None,
            "chunks_count": job.chunks_count,
            "error_message": job.error_message,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "created_at": job.created_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/search")
async def search_docs(request: SearchRequest):
    """Search documentation."""
    try:
        embed_manager = get_embed_manager()
        
        # Try normalized first
        results = embed_manager.search_normalized(
            query=request.query,
            limit=request.limit,
            url_filter=request.url_filter,
        )
        source = "normalized"
        
        if not results:
            results = embed_manager.search(
                query=request.query,
                limit=request.limit,
                url_filter=request.url_filter,
            )
            source = "raw"
        
        return {
            "results": [
                {
                    "url": r.url,
                    "page_url": r.page_url,
                    "title": r.title,
                    "score": r.score,
                    "content": r.content,
                    "type": source,
                }
                for r in results
            ],
            "source": source,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/normalize")
async def normalize_docs(
    request: NormalizeRequest,
    _: bool = Depends(verify_api_key),
):
    """Build a normalized document (requires API key)."""
    try:
        normalizer = get_normalizer()
        result = normalizer.normalize(
            url_prefix=request.url_prefix,
            title=request.title,
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "success": True,
            "url_prefix": result.url_prefix,
            "title": result.title,
            "raw_chunk_count": result.raw_chunk_count,
            "pinecone_id": result.pinecone_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/normalized")
async def list_normalized():
    """List normalized documents."""
    try:
        normalizer = get_normalizer()
        return {"documents": normalizer.list_normalized_docs()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.delete("/api/sources/{source_url:path}")
async def delete_source(
    source_url: str,
    _: bool = Depends(verify_api_key),
):
    """Delete all indexed docs for a source (requires API key)."""
    try:
        embed_manager = get_embed_manager()
        deleted = embed_manager.delete_by_source(source_url)
        return {"deleted": deleted, "source_url": source_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MCP HTTP Endpoint ====================

@api.post("/mcp/tools/{tool_name}")
async def mcp_tool_call(
    tool_name: str,
    request: dict,
    _: bool = Depends(verify_api_key),
):
    """
    Execute MCP tool via HTTP.
    This allows remote MCP clients to connect via HTTP instead of STDIO.
    """
    tools = {
        "search_documentation": search_documentation,
        "crawl_url": crawl_url,
        "build_normalized_doc": build_normalized_doc,
        "health_status": health_status,
    }
    
    if tool_name not in tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    try:
        tool_fn = tools[tool_name]
        # Get the underlying function from FastMCP tool wrapper
        if hasattr(tool_fn, 'fn'):
            result = tool_fn.fn(**request)
        else:
            result = tool_fn(**request)
        return {"result": json.loads(result) if isinstance(result, str) else result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Entry Points ====================

def run_mcp():
    """Run the MCP server (STDIO transport)."""
    mcp.run()


def run_api():
    """Run the REST API server."""
    import uvicorn
    config = get_config()
    uvicorn.run(
        api,
        host=config.server.host,
        port=config.server.port,
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        run_api()
    else:
        run_mcp()
