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
from pydantic import BaseModel, Field, ConfigDict
from fastmcp import FastMCP

from .config import get_config
from .firestore_db import get_firestore_db as get_db
from .models import CrawlStatus
from .embed_manager import get_embed_manager
from .crawl_manager import get_crawl_manager
from .normalizer import get_normalizer
from .tenant_context import set_tenant_id, reset_tenant_id
from .api_keys import ApiKeyStore
from .billing import is_tenant_active
from .source_registry import get_source_registry
from .scheduler import get_scheduler, start_scheduler, stop_scheduler
from .job_queue import get_job_queue, JobType, JobPriority, JobStatus as QueueJobStatus
from .search_agent import get_search_agent, SearchMode
from .discovery_agent import get_discovery_agent
from .crawl_planner import (
    CrawlPlanner,
    apply_plan_rules,
    build_branch_summaries,
    UniverseUrl,
    QueryParamPolicy,
    CrawlPlanRules,
    canonicalize_url,
)
from .evaluation import EvaluationHarness, RegressionTracker
from .entitlements import get_entitlement_manager, PlanTier

# Optional Firebase auth (for hosted mode)
try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth
    HAS_FIREBASE_ADMIN = True
except Exception:
    firebase_admin = None
    firebase_auth = None
    HAS_FIREBASE_ADMIN = False

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
    mode: str = "quick",
    limit: int = 10,
    url_filter: str = "",
) -> str:
    """
    Search indexed documentation with agentic capabilities.

    Args:
        query: The search query
        mode: Search mode - "quick" (fast with reranking) or "deep" (synthesized answer)
        limit: Maximum number of results (default 10)
        url_filter: Optional URL prefix to filter results

    Returns:
        JSON with search results. Quick mode returns ranked results.
        Deep mode returns a synthesized answer with citations.
    """
    import asyncio

    try:
        logger.info(f"Search query: {query} (mode={mode})")

        # Get search agent
        search_agent = get_search_agent()

        # Determine search mode
        search_mode = SearchMode.DEEP if mode.lower() == "deep" else SearchMode.QUICK

        # Run async search in sync context
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                search_agent.search(
                    query=query,
                    mode=search_mode,
                    limit=limit,
                    url_filter=url_filter,
                )
            )
        finally:
            loop.close()

        return json.dumps(result.to_dict(), ensure_ascii=False)

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


@mcp.tool()
def discover_documentation(topic: str, max_results: int = 10) -> str:
    """
    Discover documentation sources for a topic using AI.

    Uses LLM knowledge to find relevant official documentation sources,
    validates URLs, and returns them for import.

    Args:
        topic: The topic to find documentation for (e.g., "React hooks", "Kubernetes deployment")
        max_results: Maximum number of sources to discover (default 10)

    Returns:
        JSON with discovered sources including names, URLs, descriptions, and validation status.
    """
    import asyncio

    try:
        logger.info(f"Discovering documentation for: {topic}")

        discovery_agent = get_discovery_agent()

        if not discovery_agent.is_ready:
            return json.dumps({
                "error": "Discovery agent not available (Gemini not configured)"
            })

        # Run async discovery in sync context
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                discovery_agent.discover_for_topic(
                    topic=topic,
                    max_results=max_results,
                    validate=True,
                )
            )
        finally:
            loop.close()

        return json.dumps(result.to_dict(), ensure_ascii=False)

    except Exception as e:
        logger.exception("Discovery failed")
        return json.dumps({"error": str(e)})


# ==================== REST API (FastAPI) ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting ContextPilot API server")

    # Optionally start the scheduler on server startup
    config = get_config()
    if getattr(config, 'auto_start_scheduler', False):
        try:
            start_scheduler()
            logger.info("Crawl scheduler started automatically")
        except Exception as e:
            logger.warning(f"Failed to start scheduler: {e}")

    yield

    # Clean shutdown
    logger.info("Shutting down ContextPilot API server")
    try:
        stop_scheduler()
    except Exception:
        pass


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
    """Verify auth for protected endpoints. Returns tenant_id."""
    config = get_config()
    
    # Skip auth if not enabled or no key configured
    if not config.has_auth:
        token = set_tenant_id("default")
        try:
            yield "default"
        finally:
            reset_tenant_id(token)
        return

    mode = (config.auth.mode or "api_key").lower()

    def _extract_bearer() -> Optional[str]:
        if not authorization:
            return None
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
        return None

    bearer = _extract_bearer()

    if mode in ("firebase", "api_key_or_firebase") and bearer:
        if not HAS_FIREBASE_ADMIN:
            raise HTTPException(status_code=500, detail="firebase-admin not installed (AUTH_MODE=firebase)")
        try:
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
            decoded = firebase_auth.verify_id_token(bearer)
            tenant_id = decoded.get("uid") or decoded.get("user_id")
            if not tenant_id:
                raise HTTPException(status_code=401, detail="Invalid token (missing uid)")
            token = set_tenant_id(tenant_id)
            try:
                yield tenant_id
            finally:
                reset_tenant_id(token)
            return
        except HTTPException:
            raise
        except Exception as e:
            # In api_key_or_firebase mode, allow falling back to API key if bearer isn't a Firebase token.
            if mode == "firebase":
                raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    
    # API key modes (admin key or per-tenant keys)
    if mode in ("api_key", "api_key_or_firebase"):
        api_key_value = x_api_key or bearer

        # Check admin/global API key
        if api_key_value and api_key_value == config.auth.api_key:
            token = set_tenant_id("default")
            try:
                yield "default"
            finally:
                reset_tenant_id(token)
            return

        # Check per-tenant API key store (hosted)
        if api_key_value and config.multi_tenant.enabled and config.has_firestore:
            try:
                tenant_id = ApiKeyStore().lookup_tenant(api_key_value)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"API key lookup failed: {e}")
            if tenant_id:
                token = set_tenant_id(tenant_id)
                try:
                    yield tenant_id
                finally:
                    reset_tenant_id(token)
                return

    # Back-compat: explicit X-API-Key header matches configured key
    if x_api_key and x_api_key == config.auth.api_key:
        token = set_tenant_id("default")
        try:
            yield "default"
        finally:
            reset_tenant_id(token)
        return
    if bearer and bearer == config.auth.api_key:
        token = set_tenant_id("default")
        try:
            yield "default"
        finally:
            reset_tenant_id(token)
        return
    
    raise HTTPException(status_code=401, detail="Unauthorized")


async def require_firebase_user(
    authorization: Optional[str] = Header(None),
):
    """Require a Firebase user (for account management endpoints). Returns tenant_id."""
    config = get_config()
    if (config.auth.mode or "").lower() not in ("firebase", "api_key_or_firebase"):
        raise HTTPException(status_code=400, detail="Firebase auth is not enabled")
    if not HAS_FIREBASE_ADMIN:
        raise HTTPException(status_code=500, detail="firebase-admin not installed")
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization bearer token")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    try:
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        decoded = firebase_auth.verify_id_token(parts[1])
        tenant_id = decoded.get("uid") or decoded.get("user_id")
        if not tenant_id:
            raise HTTPException(status_code=401, detail="Invalid token (missing uid)")
        token = set_tenant_id(tenant_id)
        try:
            yield tenant_id
        finally:
            reset_tenant_id(token)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


async def require_paid_tenant(
    tenant_id: str = Depends(verify_api_key),
):
    """Require an active subscription when BILLING_REQUIRED=true."""
    if not is_tenant_active(tenant_id):
        raise HTTPException(status_code=402, detail="Subscription required")
    return tenant_id


# ==================== Pydantic Models ====================

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    url_filter: str = ""


class AgenticSearchRequest(BaseModel):
    query: str
    mode: str = "quick"  # "quick" or "deep"
    limit: int = 10
    url_filter: str = ""


class CrawlRequest(BaseModel):
    url: str


class NormalizeRequest(BaseModel):
    url_prefix: str
    title: str


class CreateApiKeyRequest(BaseModel):
    label: str = "Default"


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


class CreateSourceRequest(BaseModel):
    name: str
    base_url: str
    sitemap_url: Optional[str] = None
    priority_paths: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    crawl_frequency: str = "weekly"
    max_pages: int = 500
    tags: Optional[List[str]] = None
    description: Optional[str] = None


class UpdateSourceRequest(BaseModel):
    name: Optional[str] = None
    sitemap_url: Optional[str] = None
    priority_paths: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    crawl_frequency: Optional[str] = None
    max_pages: Optional[int] = None
    is_enabled: Optional[bool] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None


class SyncCuratedRequest(BaseModel):
    overwrite: bool = False


class DiscoveryRequest(BaseModel):
    topic: str
    max_results: int = 10
    should_validate: bool = Field(default=True, alias="validate")
    existing_sources: Optional[List[str]] = None
    model_config = ConfigDict(populate_by_name=True)


class ImportDiscoveredRequest(BaseModel):
    sources: List[dict]  # List of DiscoveredSource.to_dict() objects


class PlanCrawlRequest(BaseModel):
    max_urls: int = 5000


class ApprovePlanRequest(BaseModel):
    plan_id: str | int


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
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Start a crawl job (requires API key)."""
    try:
        db = get_db()
        job = db.create_crawl_job(request.url)
        
        # Run crawl in background
        def do_crawl(tenant_id: str):
            crawl_manager = get_crawl_manager(tenant_id)
            crawl_manager.crawl_and_index(request.url)
        
        background_tasks.add_task(do_crawl, _tenant_id)
        
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
    """Search documentation (legacy endpoint)."""
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


@api.post("/api/search/agentic")
async def agentic_search(
    request: AgenticSearchRequest,
    _tenant_id: str = Depends(verify_api_key),
):
    """
    Agentic documentation search with quick and deep modes.

    Quick mode: Fast retrieval with query rewriting and reranking
    Deep mode: Full synthesis with citations and follow-up questions
    """
    try:
        search_agent = get_search_agent(_tenant_id)

        # Determine search mode
        mode = SearchMode.DEEP if request.mode.lower() == "deep" else SearchMode.QUICK

        result = await search_agent.search(
            query=request.query,
            mode=mode,
            limit=request.limit,
            url_filter=request.url_filter,
        )

        return result.to_dict()
    except Exception as e:
        logger.exception("Agentic search failed")
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/search/quick")
async def quick_search(
    request: SearchRequest,
    _tenant_id: str = Depends(verify_api_key),
):
    """
    Quick search with query enhancement and reranking.
    Faster than deep search, returns ranked results without synthesis.
    """
    try:
        search_agent = get_search_agent(_tenant_id)
        result = await search_agent.quick_search(
            query=request.query,
            limit=request.limit,
            url_filter=request.url_filter,
        )
        return result.to_dict()
    except Exception as e:
        logger.exception("Quick search failed")
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/search/deep")
async def deep_search(
    request: SearchRequest,
    _tenant_id: str = Depends(verify_api_key),
):
    """
    Deep search with full synthesis.
    Returns a synthesized answer with citations and follow-up questions.
    """
    try:
        search_agent = get_search_agent(_tenant_id)
        result = await search_agent.deep_search(
            query=request.query,
            max_sources=request.limit,
            url_filter=request.url_filter,
        )
        return result.to_dict()
    except Exception as e:
        logger.exception("Deep search failed")
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/normalize")
async def normalize_docs(
    request: NormalizeRequest,
    _tenant_id: str = Depends(require_paid_tenant),
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
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Delete all indexed docs for a source (requires API key)."""
    try:
        embed_manager = get_embed_manager()
        deleted = embed_manager.delete_by_source(source_url)
        return {"deleted": deleted, "source_url": source_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Source Registry ====================


def _source_to_dict(source) -> dict:
    """Convert Source dataclass to API response dict."""
    return {
        "id": source.id,
        "name": source.name,
        "base_url": source.base_url,
        "sitemap_url": source.sitemap_url,
        "priority_paths": source.priority_paths,
        "exclude_paths": source.exclude_paths,
        "crawl_frequency": source.crawl_frequency.value,
        "max_pages": source.max_pages,
        "is_enabled": source.is_enabled,
        "is_curated": source.is_curated,
        "created_by": source.created_by.value,
        "health_status": source.health_status.value,
        "last_crawled_at": source.last_crawled_at,
        "next_crawl_at": source.next_crawl_at,
        "chunks_count": source.chunks_count,
        "error_message": source.error_message,
        "tags": source.tags,
        "description": source.description,
        "created_at": source.created_at,
        "updated_at": source.updated_at,
    }


@api.get("/api/sources/registry")
async def list_registry_sources(
    is_enabled: Optional[bool] = None,
    created_by: Optional[str] = None,
    health_status: Optional[str] = None,
    limit: int = 100,
    _tenant_id: str = Depends(verify_api_key),
):
    """List all sources in the registry."""
    try:
        registry = get_source_registry(_tenant_id)
        sources = registry.list_sources(
            is_enabled=is_enabled,
            created_by=created_by,
            health_status=health_status,
            limit=limit,
        )
        return {
            "sources": [_source_to_dict(s) for s in sources],
            "stats": registry.get_stats(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/sources/registry/{source_id}")
async def get_registry_source(
    source_id: str,
    _tenant_id: str = Depends(verify_api_key),
):
    """Get a specific source from the registry."""
    try:
        registry = get_source_registry(_tenant_id)
        source = registry.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        return _source_to_dict(source)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/sources/registry")
async def create_registry_source(
    request: CreateSourceRequest,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Add a new source to the registry."""
    try:
        registry = get_source_registry(_tenant_id)
        source = registry.add_user_source(
            name=request.name,
            base_url=request.base_url,
            sitemap_url=request.sitemap_url,
            priority_paths=request.priority_paths,
            exclude_paths=request.exclude_paths,
            crawl_frequency=request.crawl_frequency,
            max_pages=request.max_pages,
            tags=request.tags,
            description=request.description,
        )
        return _source_to_dict(source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.put("/api/sources/registry/{source_id}")
async def update_registry_source(
    source_id: str,
    request: UpdateSourceRequest,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Update a source in the registry."""
    try:
        registry = get_source_registry(_tenant_id)

        # Build updates dict from non-None fields
        updates = {}
        if request.name is not None:
            updates["name"] = request.name
        if request.sitemap_url is not None:
            updates["sitemap_url"] = request.sitemap_url
        if request.priority_paths is not None:
            updates["priority_paths"] = request.priority_paths
        if request.exclude_paths is not None:
            updates["exclude_paths"] = request.exclude_paths
        if request.crawl_frequency is not None:
            updates["crawl_frequency"] = request.crawl_frequency
        if request.max_pages is not None:
            updates["max_pages"] = request.max_pages
        if request.is_enabled is not None:
            updates["is_enabled"] = request.is_enabled
        if request.tags is not None:
            updates["tags"] = request.tags
        if request.description is not None:
            updates["description"] = request.description

        source = registry.update_source(source_id, updates)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        return _source_to_dict(source)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.delete("/api/sources/registry/{source_id}")
async def delete_registry_source(
    source_id: str,
    cascade: bool = True,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Delete a source from the registry. With cascade=true (default), deletes ALL related data."""
    try:
        from .data_manager import get_data_manager

        registry = get_source_registry(_tenant_id)
        source = registry.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        if cascade:
            # Use the new cascade delete
            manager = get_data_manager(_tenant_id)
            result = manager.delete_source_cascade(source_id)
            return {
                "deleted": True,
                "source_id": source_id,
                "cascade": True,
                "details": {
                    "vectors": result.deleted_vectors,
                    "chunks": result.deleted_raw_chunks,
                    "indexed_docs": result.deleted_indexed_docs,
                    "crawl_jobs": result.deleted_crawl_jobs,
                    "crawl_plans": result.deleted_crawl_plans,
                    "url_universe": result.deleted_url_universe,
                },
                "total_deleted": result.total_deleted,
                "errors": result.errors if result.errors else None,
            }
        else:
            # Legacy behavior - just delete source record
            registry.delete_source(source_id)
            return {"deleted": True, "source_id": source_id, "cascade": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/sources/registry/sync-curated")
async def sync_curated_sources(
    request: SyncCuratedRequest,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Sync curated sources to the tenant's registry."""
    try:
        registry = get_source_registry(_tenant_id)
        results = registry.sync_curated_to_tenant(overwrite=request.overwrite)
        return {
            "added": results["added"],
            "updated": results["updated"],
            "skipped": results["skipped"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/sources/registry/curated")
async def list_curated_sources():
    """List available curated sources (no auth required)."""
    try:
        registry = get_source_registry()
        curated = registry.load_curated_sources()
        return {
            "sources": [
                {
                    "name": s.get("name", ""),
                    "base_url": s.get("base_url", ""),
                    "sitemap_url": s.get("sitemap_url"),
                    "description": s.get("description"),
                    "tags": s.get("tags", []),
                    "crawl_frequency": s.get("crawl_frequency", "weekly"),
                }
                for s in curated
            ],
            "count": len(curated),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/sources/registry/{source_id}/crawl")
async def trigger_source_crawl(
    source_id: str,
    background_tasks: BackgroundTasks,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Trigger a crawl for a source using the approved plan's URLs.

    The plan is the single source of truth - Gemini curates and classifies URLs,
    and only the approved 'keep' URLs are crawled.
    """
    try:
        registry = get_source_registry(_tenant_id)
        source = registry.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        db = get_db(_tenant_id)
        plan = db.get_latest_crawl_plan(source_id)

        # Require an approved plan - the plan is the single source of truth
        if not plan:
            raise HTTPException(
                status_code=400,
                detail="No crawl plan found. Generate and approve a plan first."
            )
        if str(plan.get("status") or "") != "approved":
            raise HTTPException(
                status_code=400,
                detail="Crawl plan is not approved. Review and approve the plan first."
            )

        # Get URLs from the plan (the single source of truth)
        rules_data = plan.get("rules") or {}
        q = rules_data.get("query_param_policy") or {}
        rules = CrawlPlanRules(
            allow_prefixes=list(rules_data.get("allow_prefixes") or []),
            deny_prefixes=list(rules_data.get("deny_prefixes") or []),
            maybe_prefixes=list(rules_data.get("maybe_prefixes") or []),
            query_param_policy=QueryParamPolicy(
                keep_keys=list(q.get("keep_keys") or []),
                drop_keys=list(q.get("drop_keys") or []),
                default=str(q.get("default") or "drop_unknown_keys"),
            ),
        )

        # Get URL universe and apply plan rules
        universe_rows = db.list_url_universe(source_id)
        universe: List[UniverseUrl] = []
        for row in universe_rows:
            canonical = canonicalize_url(row.get("canonical_url") or row.get("url") or "", rules.query_param_policy)
            if not canonical:
                continue
            universe.append(UniverseUrl(
                url_id=row.get("url_id") or "",
                url=row.get("url") or canonical,
                canonical_url=canonical,
                path=row.get("path") or "/",
                depth=int(row.get("depth") or 0),
                query_keys=list(row.get("query_keys") or []),
            ))

        if not universe:
            raise HTTPException(
                status_code=400,
                detail="No URLs in plan. Re-generate the crawl plan."
            )

        classified = apply_plan_rules(universe, rules)
        urls = [u.canonical_url for u in classified["keep"]]

        if not urls:
            raise HTTPException(
                status_code=400,
                detail="No URLs to crawl after applying plan rules."
            )

        # Run crawl in background using plan URLs
        async def do_crawl(tenant_id: str, src_id: str, src: Source, urls_to_crawl: List[str]):
            crawl_manager = get_crawl_manager(tenant_id)
            result = await crawl_manager.crawl_urls_batch(
                urls=urls_to_crawl,
                source=src,
                incremental=True,
                max_concurrent=3,
            )
            reg = get_source_registry(tenant_id)
            ok = (result.get("pages_crawled", 0) > 0)
            reg.mark_crawl_complete(
                src_id,
                success=ok,
                chunks_count=result.get("chunks_indexed", 0),
                error_message="; ".join(result.get("errors", [])[:3]) if not ok else None,
            )

        background_tasks.add_task(do_crawl, _tenant_id, source_id, source, urls)

        return {
            "source_id": source_id,
            "url": source.base_url,
            "status": "started",
            "planned_urls": len(urls),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/sources/registry/{source_id}/plan")
async def generate_crawl_plan(
    source_id: str,
    request: PlanCrawlRequest,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Generate an LLM-curated crawl plan for a source (returns a JSON report)."""
    try:
        registry = get_source_registry(_tenant_id)
        source = registry.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        planner = CrawlPlanner(tenant_id=_tenant_id)
        universe = await planner.build_url_universe(source, max_urls=request.max_urls)

        branches = build_branch_summaries(universe, base_url=source.base_url)
        rules = planner.propose_rules(source.base_url, branches)
        rules = planner.validate_rules(rules, branches)

        classified = apply_plan_rules(universe, rules)
        counts = {
            "total_urls_seen": len(universe),
            "kept_urls": len(classified["keep"]),
            "dropped_urls": len(classified["drop"]),
            "maybe_urls": len(classified["maybe"]),
        }
        samples = {
            "keep": [u.url_id for u in classified["keep"][:50]],
            "drop": [u.url_id for u in classified["drop"][:50]],
            "maybe": [u.url_id for u in classified["maybe"][:50]],
        }
        id_to_url = {u.url_id: u.canonical_url for u in universe}
        sample_urls = {
            "keep": [{"id": uid, "url": id_to_url.get(uid, "")} for uid in samples["keep"]],
            "drop": [{"id": uid, "url": id_to_url.get(uid, "")} for uid in samples["drop"]],
            "maybe": [{"id": uid, "url": id_to_url.get(uid, "")} for uid in samples["maybe"]],
        }

        report = planner.generate_report(
            base_url=source.base_url,
            branches=branches,
            rules=rules,
            counts=counts,
            samples=samples,
        )

        plan_id = planner.store_plan(
            source_id=source_id,
            rules=rules,
            report=report,
            total_urls_seen=counts["total_urls_seen"],
            kept_urls=counts["kept_urls"],
            dropped_urls=counts["dropped_urls"],
            maybe_urls=counts["maybe_urls"],
        )

        return {
            "plan_id": plan_id,
            "source_id": source_id,
            "counts": counts,
            "rules": {
                "allow_prefixes": rules.allow_prefixes,
                "deny_prefixes": rules.deny_prefixes,
                "maybe_prefixes": rules.maybe_prefixes,
                "query_param_policy": {
                    "keep_keys": rules.query_param_policy.keep_keys,
                    "drop_keys": rules.query_param_policy.drop_keys,
                    "default": rules.query_param_policy.default,
                },
            },
            "report": report,
            "samples": sample_urls,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate crawl plan")
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/sources/registry/{source_id}/plan")
async def get_crawl_plan(
    source_id: str,
    _tenant_id: str = Depends(verify_api_key),
):
    """Get the latest crawl plan for a source."""
    try:
        db = get_db(_tenant_id)
        plan = db.get_latest_crawl_plan(source_id)
        if not plan:
            return {"plan": None}
        return {"plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/sources/registry/{source_id}/plan/approve")
async def approve_crawl_plan(
    source_id: str,
    request: ApprovePlanRequest,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Approve a generated crawl plan (explicit user confirmation)."""
    logger.info(f"[approve_crawl_plan] source_id={source_id}, plan_id={request.plan_id}")
    try:
        db = get_db(_tenant_id)
        latest = db.get_latest_crawl_plan(source_id)
        if not latest:
            logger.warning(f"[approve_crawl_plan] No crawl plan found for source {source_id}")
            raise HTTPException(status_code=404, detail="No crawl plan found")

        plan_id = request.plan_id
        latest_id = str(latest.get("id"))
        logger.info(f"[approve_crawl_plan] latest_id={latest_id}, request_plan_id={plan_id}")
        # Basic safety: ensure the plan being approved is the latest for the source.
        if latest_id != str(plan_id):
            logger.warning(f"[approve_crawl_plan] Plan id mismatch: {plan_id} != {latest_id}")
            raise HTTPException(status_code=400, detail="Plan id is not the latest for this source")

        ok = db.update_crawl_plan_status(plan_id, "approved")
        logger.info(f"[approve_crawl_plan] Updated status to approved: {ok}")
        return {"approved": bool(ok), "plan_id": plan_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/sources/registry/{source_id}/crawl-from-plan")
async def crawl_from_plan(
    source_id: str,
    background_tasks: BackgroundTasks,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Crawl a source using the latest stored plan rules (allow/deny prefixes)."""
    logger.info(f"[crawl_from_plan] Starting crawl for source_id={source_id}")
    try:
        registry = get_source_registry(_tenant_id)
        source = registry.get_source(source_id)
        if not source:
            logger.warning(f"[crawl_from_plan] Source not found: {source_id}")
            raise HTTPException(status_code=404, detail="Source not found")

        db = get_db(_tenant_id)
        plan = db.get_latest_crawl_plan(source_id)
        if not plan:
            logger.warning(f"[crawl_from_plan] No crawl plan found for source {source_id}")
            raise HTTPException(status_code=404, detail="No crawl plan found")
        logger.info(f"[crawl_from_plan] Plan status: {plan.get('status')}")
        if str(plan.get("status") or "") != "approved":
            raise HTTPException(status_code=400, detail="Crawl plan is not approved yet")

        rules_data = plan.get("rules") or {}
        q = rules_data.get("query_param_policy") or {}
        rules = CrawlPlanRules(
            allow_prefixes=list(rules_data.get("allow_prefixes") or []),
            deny_prefixes=list(rules_data.get("deny_prefixes") or []),
            maybe_prefixes=list(rules_data.get("maybe_prefixes") or []),
            query_param_policy=QueryParamPolicy(
                keep_keys=list(q.get("keep_keys") or []),
                drop_keys=list(q.get("drop_keys") or []),
                default=str(q.get("default") or "drop_unknown_keys"),
            ),
        )

        universe_rows = db.list_url_universe(source_id)
        universe: List[UniverseUrl] = []
        for row in universe_rows:
            canonical = canonicalize_url(row.get("canonical_url") or row.get("url") or "", rules.query_param_policy)
            if not canonical:
                continue
            universe.append(UniverseUrl(
                url_id=row.get("url_id") or "",
                url=row.get("url") or canonical,
                canonical_url=canonical,
                path=row.get("path") or "/",
                depth=int(row.get("depth") or 0),
                query_keys=list(row.get("query_keys") or []),
            ))

        if not universe:
            planner = CrawlPlanner(tenant_id=_tenant_id)
            universe = await planner.build_url_universe(source, max_urls=source.max_pages or 2000)

        classified = apply_plan_rules(universe, rules)
        urls = [u.canonical_url for u in classified["keep"]]

        # Run crawl in background (potentially long-running)
        async def do_crawl(tenant_id: str, src_id: str, src, urls_to_crawl: List[str]):
            crawl_manager = get_crawl_manager(tenant_id)
            result = await crawl_manager.crawl_urls_batch(
                urls=urls_to_crawl,
                source=src,
                incremental=True,
                max_concurrent=3,
            )
            reg = get_source_registry(tenant_id)
            ok = (result.get("pages_crawled", 0) > 0)
            reg.mark_crawl_complete(
                src_id,
                success=ok,
                chunks_count=result.get("chunks_indexed", 0),
                error_message="; ".join(result.get("errors", [])[:3]) if not ok else None,
            )

        background_tasks.add_task(do_crawl, _tenant_id, source_id, source, urls)

        return {
            "source_id": source_id,
            "status": "started",
            "planned_urls": len(urls),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to crawl from plan")
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/sources/registry/{source_id}/toggle")
async def toggle_source_enabled(
    source_id: str,
    enabled: bool = True,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Enable or disable a source."""
    try:
        registry = get_source_registry(_tenant_id)
        source = registry.toggle_source(source_id, enabled)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        return _source_to_dict(source)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/sources/registry/health")
async def check_sources_health(
    _tenant_id: str = Depends(verify_api_key),
):
    """Check health of all sources and identify issues."""
    try:
        registry = get_source_registry(_tenant_id)
        issues = registry.check_source_health()
        stats = registry.get_stats()
        return {
            "issues": issues,
            "stats": stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Scheduler ====================


@api.get("/api/scheduler/status")
async def get_scheduler_status(
    _tenant_id: str = Depends(verify_api_key),
):
    """Get the scheduler status and statistics."""
    try:
        scheduler = get_scheduler()
        status = scheduler.get_scheduler_status()
        running_jobs = scheduler.get_running_jobs()
        recent_history = scheduler.get_job_history(limit=10)

        return {
            "status": status,
            "running_jobs": [
                {
                    "id": j.id,
                    "source_id": j.source_id,
                    "tenant_id": j.tenant_id,
                    "status": j.status.value,
                    "started_at": j.started_at,
                    "chunks_indexed": j.chunks_indexed,
                }
                for j in running_jobs
            ],
            "recent_history": [
                {
                    "id": j.id,
                    "source_id": j.source_id,
                    "tenant_id": j.tenant_id,
                    "status": j.status.value,
                    "started_at": j.started_at,
                    "completed_at": j.completed_at,
                    "chunks_indexed": j.chunks_indexed,
                    "error_message": j.error_message,
                }
                for j in recent_history
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/scheduler/start")
async def start_scheduler_endpoint(
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Start the crawl scheduler."""
    try:
        start_scheduler()
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/scheduler/stop")
async def stop_scheduler_endpoint(
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Stop the crawl scheduler."""
    try:
        stop_scheduler()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/scheduler/trigger/{source_id}")
async def trigger_scheduled_crawl(
    source_id: str,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Manually trigger a scheduled crawl for a source."""
    try:
        scheduler = get_scheduler()
        job = await scheduler.trigger_source_crawl(_tenant_id, source_id)
        return {
            "job_id": job.id,
            "source_id": source_id,
            "status": job.status.value,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Job Queue ====================


class EnqueueJobRequest(BaseModel):
    job_type: str
    payload: dict
    priority: str = "normal"


@api.get("/api/jobs")
async def list_jobs(
    limit: int = 50,
    _tenant_id: str = Depends(verify_api_key),
):
    """List background jobs."""
    try:
        queue = get_job_queue()
        running = queue.get_running_jobs()
        stats = queue.get_queue_stats()

        return {
            "stats": stats,
            "running_jobs": [
                {
                    "id": j.id,
                    "type": j.type.value,
                    "tenant_id": j.tenant_id,
                    "status": j.status.value,
                    "priority": j.priority.value,
                    "started_at": j.started_at,
                    "retries": j.retries,
                }
                for j in running
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/jobs/{job_id}")
async def get_job(
    job_id: str,
    _tenant_id: str = Depends(verify_api_key),
):
    """Get a specific job by ID."""
    try:
        queue = get_job_queue()
        job = await queue.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "id": job.id,
            "type": job.type.value,
            "tenant_id": job.tenant_id,
            "payload": job.payload,
            "status": job.status.value,
            "priority": job.priority.value,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error_message": job.error_message,
            "result": job.result,
            "retries": job.retries,
            "max_retries": job.max_retries,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/jobs")
async def enqueue_job(
    request: EnqueueJobRequest,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Enqueue a new background job."""
    try:
        queue = get_job_queue()

        # Map string to enum
        try:
            job_type = JobType(request.job_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid job_type. Valid types: {[t.value for t in JobType]}"
            )

        priority_map = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "urgent": JobPriority.URGENT,
        }
        priority = priority_map.get(request.priority.lower(), JobPriority.NORMAL)

        job = await queue.enqueue(
            job_type=job_type,
            tenant_id=_tenant_id,
            payload=request.payload,
            priority=priority,
        )

        return {
            "job_id": job.id,
            "type": job.type.value,
            "status": job.status.value,
            "priority": job.priority.value,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Cancel a running or pending job."""
    try:
        queue = get_job_queue()
        cancelled = await queue.cancel_job(job_id)

        if not cancelled:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")

        return {"cancelled": True, "job_id": job_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/jobs/queue/start")
async def start_job_queue(
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Start the job queue workers."""
    try:
        queue = get_job_queue()
        await queue.start()
        return {"status": "started", "stats": queue.get_queue_stats()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/jobs/queue/stop")
async def stop_job_queue(
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Stop the job queue workers."""
    try:
        queue = get_job_queue()
        await queue.stop()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== API Keys (Hosted) ====================

@api.get("/api/api-keys")
async def list_api_keys(
    tenant_id: str = Depends(require_firebase_user),
):
    try:
        store = ApiKeyStore()
        keys = store.list_for_tenant(tenant_id)
        return {
            "api_keys": [
                {
                    "digest": k.digest,
                    "label": k.label,
                    "created_at": k.created_at,
                    "last_used_at": k.last_used_at,
                    "revoked_at": k.revoked_at,
                }
                for k in keys
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/api-keys")
async def create_api_key(
    request: CreateApiKeyRequest,
    tenant_id: str = Depends(require_firebase_user),
):
    try:
        if not is_tenant_active(tenant_id):
            raise HTTPException(status_code=402, detail="Subscription required")
        store = ApiKeyStore()
        created = store.create(tenant_id=tenant_id, label=request.label)
        # Only return plaintext once.
        return {"api_key": created["api_key"], "digest": created["digest"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.delete("/api/api-keys/{digest}")
async def revoke_api_key(
    digest: str,
    tenant_id: str = Depends(require_firebase_user),
):
    try:
        if not is_tenant_active(tenant_id):
            raise HTTPException(status_code=402, detail="Subscription required")
        ok = ApiKeyStore().revoke(tenant_id=tenant_id, digest=digest)
        if not ok:
            raise HTTPException(status_code=404, detail="API key not found")
        return {"revoked": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Discovery Endpoints ====================

@api.post("/api/discovery/search")
async def discover_sources(
    request: DiscoveryRequest,
    _tenant_id: str = Depends(verify_api_key),
):
    """
    Discover documentation sources for a topic using AI.

    Uses LLM knowledge to find relevant documentation sources,
    validates URLs, and attempts to find sitemaps.
    """
    try:
        discovery_agent = get_discovery_agent()

        if not discovery_agent.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Discovery agent not available (Gemini not configured)"
            )

        # Get existing source URLs to exclude
        registry = get_source_registry()
        existing = request.existing_sources or []
        if not existing:
            sources = registry.list_sources()
            existing = [s.base_url for s in sources]

        result = await discovery_agent.discover_for_topic(
            topic=request.topic,
            existing_sources=existing,
            max_results=request.max_results,
            validate=request.should_validate,
        )

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Discovery failed")
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/discovery/import")
async def import_discovered_sources(
    request: ImportDiscoveredRequest,
    _tenant_id: str = Depends(verify_api_key),
):
    """
    Import discovered sources into the registry.

    Takes validated discovered sources and adds them to the source registry
    for scheduled crawling.
    """
    try:
        from .models import SourceCreatedBy, CrawlFrequency

        registry = get_source_registry()
        imported = []
        errors = []

        for source_data in request.sources:
            try:
                # Create source from discovered data
                source = registry.create_source(
                    name=source_data.get("name", "Unknown"),
                    base_url=source_data["base_url"],
                    sitemap_url=source_data.get("sitemap_url"),
                    priority_paths=source_data.get("suggested_paths", []),
                    crawl_frequency=CrawlFrequency.WEEKLY,
                    created_by=SourceCreatedBy.DISCOVERY,
                    description=source_data.get("description", ""),
                    tags=[source_data.get("category", "discovered")],
                )
                imported.append({
                    "id": source.id,
                    "name": source.name,
                    "base_url": source.base_url,
                })
            except Exception as e:
                errors.append({
                    "base_url": source_data.get("base_url", "unknown"),
                    "error": str(e),
                })

        return {
            "imported": imported,
            "imported_count": len(imported),
            "errors": errors,
            "error_count": len(errors),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Import failed")
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/discovery/related")
async def discover_related_docs(
    base_url: str,
    max_links: int = 20,
    _tenant_id: str = Depends(verify_api_key),
):
    """
    Discover related documentation pages from a base URL.

    Crawls links from the base URL and returns documentation-like URLs.
    """
    try:
        discovery_agent = get_discovery_agent()

        links = await discovery_agent.discover_related_docs(
            base_url=base_url,
            max_links=max_links,
        )

        return {
            "base_url": base_url,
            "discovered_links": links,
            "count": len(links),
        }

    except Exception as e:
        logger.exception("Related docs discovery failed")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== WebSocket ====================

from fastapi import WebSocket, WebSocketDisconnect
from .websocket_manager import (
    get_connection_manager,
    Event,
    EventType,
    broadcast_stats_update,
)


@api.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Clients connect here to receive:
    - Crawl progress events
    - Indexing updates
    - Stats changes
    - Health status
    """
    manager = get_connection_manager()
    tenant_id = "default"  # TODO: Extract from query params or auth

    await manager.connect(websocket, tenant_id)

    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                msg_type = message.get("type", "")

                # Handle ping/pong for keepalive
                if msg_type == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": message.get("timestamp", ""),
                    }))

                # Handle history request
                elif msg_type == "get_history":
                    count = message.get("count", 50)
                    events = manager.get_recent_events(count)
                    await websocket.send_text(json.dumps({
                        "type": "history",
                        "events": [
                            {
                                "type": e.type.value,
                                "data": e.data,
                                "timestamp": e.timestamp,
                            }
                            for e in events
                        ],
                    }))

                # Handle subscription changes (future use)
                elif msg_type == "subscribe":
                    pass  # Could implement topic-based subscriptions

            except json.JSONDecodeError:
                pass  # Ignore invalid JSON

    except WebSocketDisconnect:
        await manager.disconnect(websocket, tenant_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket, tenant_id)


# ==================== Vector Visualization ====================


class VectorSampleRequest(BaseModel):
    limit: int = 500
    namespace: str = "docs"
    url_filter: str = ""


class VectorQueryRequest(BaseModel):
    query: str
    limit: int = 20
    namespace: str = "docs"
    include_vectors: bool = True


@api.post("/api/vectors/sample")
async def get_vector_samples(
    request: VectorSampleRequest,
    _tenant_id: str = Depends(verify_api_key),
):
    """
    Get a sample of vectors for visualization.

    Returns vectors with their metadata for 2D/3D visualization.
    Uses random sampling for large indexes.
    """
    try:
        embed_manager = get_embed_manager(_tenant_id)

        if not embed_manager._index:
            raise HTTPException(status_code=503, detail="Pinecone not initialized")

        # Query for random samples using a zero vector
        # This gives us a diverse sample across the embedding space
        import random
        dimension = embed_manager.config.pinecone.dimension

        # Generate a random unit vector for sampling
        random_vector = [random.gauss(0, 1) for _ in range(dimension)]
        magnitude = sum(x * x for x in random_vector) ** 0.5
        random_vector = [x / magnitude for x in random_vector]

        # Build filter if needed
        filter_dict = None
        if request.url_filter:
            filter_dict = {
                "url": {"$gte": request.url_filter, "$lt": request.url_filter + "\uffff"}
            }

        # Query Pinecone with include_values to get vectors
        ns = embed_manager._ns(request.namespace)
        results = embed_manager._index.query(
            vector=random_vector,
            top_k=request.limit,
            filter=filter_dict,
            include_metadata=True,
            include_values=True,
            namespace=ns,
        )

        # Format response
        vectors = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            vectors.append({
                "id": match["id"],
                "score": match.get("score", 0),
                "values": match.get("values", []),  # The actual embedding
                "url": metadata.get("url", ""),
                "page_url": metadata.get("page_url", ""),
                "title": metadata.get("title", ""),
                "content_preview": metadata.get("content", "")[:200],
            })

        return {
            "vectors": vectors,
            "count": len(vectors),
            "dimension": dimension,
            "namespace": ns,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get vector samples")
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/vectors/query")
async def query_vectors_with_embedding(
    request: VectorQueryRequest,
    _tenant_id: str = Depends(verify_api_key),
):
    """
    Query vectors and return with embeddings for visualization.

    Useful for showing query results in vector space.
    """
    try:
        embed_manager = get_embed_manager(_tenant_id)

        if not embed_manager._index:
            raise HTTPException(status_code=503, detail="Pinecone not initialized")

        # Get query embedding
        query_embedding = embed_manager.get_embedding(request.query)

        # Query Pinecone
        ns = embed_manager._ns(request.namespace)
        results = embed_manager._index.query(
            vector=query_embedding,
            top_k=request.limit,
            include_metadata=True,
            include_values=request.include_vectors,
            namespace=ns,
        )

        # Format response
        vectors = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            item = {
                "id": match["id"],
                "score": match.get("score", 0),
                "url": metadata.get("url", ""),
                "page_url": metadata.get("page_url", ""),
                "title": metadata.get("title", ""),
                "content": metadata.get("content", ""),
            }
            if request.include_vectors:
                item["values"] = match.get("values", [])
            vectors.append(item)

        return {
            "query": request.query,
            "query_vector": query_embedding if request.include_vectors else None,
            "results": vectors,
            "count": len(vectors),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to query vectors")
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/vectors/stats")
async def get_vector_stats(
    _tenant_id: str = Depends(verify_api_key),
):
    """Get detailed vector index statistics."""
    try:
        embed_manager = get_embed_manager(_tenant_id)
        stats = embed_manager.get_index_stats()

        # Get source breakdown from database
        db = get_db(_tenant_id)
        sources = db.list_indexed_sources()

        return {
            "index": stats,
            "sources": [
                {
                    "url": s.get("source_url", ""),
                    "title": s.get("title", ""),
                    "chunks": s.get("chunks", 0),
                }
                for s in sources
            ],
        }

    except Exception as e:
        logger.exception("Failed to get vector stats")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MCP Playground ====================


@api.get("/api/mcp/tools")
async def list_mcp_tools():
    """
    List all available MCP tools with their schemas.

    Used by the MCP Playground to display tool forms.
    """
    tools_info = [
        {
            "name": "search_documentation",
            "description": "Search indexed documentation with agentic capabilities.",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                    "required": True,
                },
                "mode": {
                    "type": "string",
                    "description": "Search mode - 'quick' (fast with reranking) or 'deep' (synthesized answer)",
                    "default": "quick",
                    "enum": ["quick", "deep"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10,
                },
                "url_filter": {
                    "type": "string",
                    "description": "Optional URL prefix to filter results",
                    "default": "",
                },
            },
        },
        {
            "name": "crawl_url",
            "description": "Crawl a URL and index its content.",
            "parameters": {
                "url": {
                    "type": "string",
                    "description": "The URL to crawl",
                    "required": True,
                },
            },
        },
        {
            "name": "build_normalized_doc",
            "description": "Build a normalized document from all chunks matching a URL prefix.",
            "parameters": {
                "url_prefix": {
                    "type": "string",
                    "description": "URL prefix to match (e.g., 'https://docs.example.com/api')",
                    "required": True,
                },
                "title": {
                    "type": "string",
                    "description": "Title for the normalized document",
                    "required": True,
                },
            },
        },
        {
            "name": "health_status",
            "description": "Get system health status and statistics.",
            "parameters": {},
        },
        {
            "name": "discover_documentation",
            "description": "Discover documentation sources for a topic using AI.",
            "parameters": {
                "topic": {
                    "type": "string",
                    "description": "The topic to find documentation for",
                    "required": True,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of sources to discover",
                    "default": 10,
                },
            },
        },
    ]

    return {"tools": tools_info}


@api.post("/api/mcp/execute")
async def execute_mcp_tool(
    tool_name: str,
    parameters: dict,
    _tenant_id: str = Depends(verify_api_key),
):
    """
    Execute an MCP tool and return the result.

    This is the main endpoint for the MCP Playground.
    """
    tools = {
        "search_documentation": search_documentation,
        "crawl_url": crawl_url,
        "build_normalized_doc": build_normalized_doc,
        "health_status": health_status,
        "discover_documentation": discover_documentation,
    }

    if tool_name not in tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    try:
        tool_fn = tools[tool_name]

        # Get the underlying function from FastMCP tool wrapper
        if hasattr(tool_fn, "fn"):
            result = tool_fn.fn(**parameters)
        else:
            result = tool_fn(**parameters)

        # Parse JSON result
        parsed = json.loads(result) if isinstance(result, str) else result

        return {
            "tool": tool_name,
            "parameters": parameters,
            "result": parsed,
            "raw": result if isinstance(result, str) else json.dumps(parsed, indent=2),
        }

    except Exception as e:
        logger.exception(f"MCP tool execution failed: {tool_name}")
        return {
            "tool": tool_name,
            "parameters": parameters,
            "error": str(e),
            "raw": None,
        }


# ==================== LLM Chat ====================


class ChatRequest(BaseModel):
    message: str
    mode: str = "quick"  # quick or deep
    history: list = []


@api.post("/api/chat")
async def chat_with_docs(
    request: ChatRequest,
    _tenant_id: str = Depends(verify_api_key),
):
    """
    Chat interface that queries documentation and generates responses.

    Uses the search agent to find relevant docs, then synthesizes an answer.
    """
    try:
        search_agent = get_search_agent(_tenant_id)
        mode = SearchMode.DEEP if request.mode == "deep" else SearchMode.QUICK

        # Search for relevant documentation
        result = await search_agent.search(
            query=request.message,
            mode=mode,
            limit=10,
        )

        result_dict = result.to_dict()

        # For quick mode, format a simple response
        if mode == SearchMode.QUICK:
            sources = result_dict.get("results", [])
            return {
                "message": request.message,
                "mode": "quick",
                "answer": None,  # Quick mode returns sources, not synthesized answer
                "sources": [
                    {
                        "title": s.get("title", ""),
                        "url": s.get("url", "") or s.get("page_url", ""),
                        "score": s.get("score", 0),
                        "content_preview": s.get("content", "")[:300],
                    }
                    for s in sources[:5]
                ],
                "total_results": len(sources),
            }

        # Deep mode returns synthesized answer
        return {
            "message": request.message,
            "mode": "deep",
            "answer": result_dict.get("answer", ""),
            "sources": result_dict.get("sources", []),
            "follow_up_questions": result_dict.get("follow_up_questions", []),
        }

    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Evaluation Endpoints ====================

class GoldenQueryRequest(BaseModel):
    query: str
    source_id: str
    expected_urls: List[str] = []
    expected_keywords: List[str] = []
    min_relevance_score: float = 0.7


class RunEvaluationRequest(BaseModel):
    source_id: str
    top_k: int = 10


@api.post("/api/evaluation/golden-queries")
async def add_golden_query(
    request: GoldenQueryRequest,
    _tenant_id: str = Depends(verify_api_key),
):
    """Add a golden query for evaluation."""
    try:
        db = get_db()
        embed_manager = get_embed_manager()
        harness = EvaluationHarness(db, embed_manager)

        golden = harness.add_golden_query(
            query=request.query,
            source_id=request.source_id,
            expected_urls=request.expected_urls,
            expected_keywords=request.expected_keywords,
            min_relevance_score=request.min_relevance_score,
        )

        return {"success": True, "golden_query": golden.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/evaluation/golden-queries/{source_id}")
async def list_golden_queries(
    source_id: str,
    _tenant_id: str = Depends(verify_api_key),
):
    """List golden queries for a source."""
    try:
        db = get_db()
        embed_manager = get_embed_manager()
        harness = EvaluationHarness(db, embed_manager)

        queries = harness.get_golden_queries(source_id)
        return {"source_id": source_id, "queries": [q.to_dict() for q in queries]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/evaluation/run")
async def run_evaluation(
    request: RunEvaluationRequest,
    _tenant_id: str = Depends(verify_api_key),
):
    """Run evaluation for a source."""
    try:
        db = get_db()
        embed_manager = get_embed_manager()
        harness = EvaluationHarness(db, embed_manager)
        tracker = RegressionTracker(db)

        summary = await harness.run_evaluation(
            source_id=request.source_id,
            top_k=request.top_k,
        )

        # Record snapshot for regression tracking
        tracker.record_snapshot(summary)

        # Check for regressions
        regression = tracker.detect_regression(request.source_id)

        return {
            "success": True,
            "summary": summary.to_dict(),
            "regression_detected": regression is not None,
            "regression_details": regression,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/evaluation/results/{source_id}")
async def get_evaluation_results(
    source_id: str,
    _tenant_id: str = Depends(verify_api_key),
):
    """Get evaluation results for a source."""
    try:
        db = get_db()
        tracker = RegressionTracker(db)

        history = tracker.get_history(source_id, limit=10)
        trend = tracker.get_trend(source_id, metric="quality_score", limit=10)

        return {
            "source_id": source_id,
            "history": [s.to_dict() for s in history],
            "trend": trend,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/evaluation/regression/{source_id}")
async def check_regression(
    source_id: str,
    threshold: float = 0.10,
    _tenant_id: str = Depends(verify_api_key),
):
    """Check for quality regressions."""
    try:
        db = get_db()
        tracker = RegressionTracker(db)

        regression = tracker.detect_regression(source_id, threshold)

        return {
            "source_id": source_id,
            "regression_detected": regression is not None,
            "details": regression,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Entitlements Endpoints ====================

@api.get("/api/entitlements")
async def get_entitlements(
    _tenant_id: str = Depends(verify_api_key),
):
    """Get current tenant entitlements and usage."""
    try:
        manager = get_entitlement_manager()
        summary = manager.get_usage_summary(_tenant_id)
        return {"tenant_id": _tenant_id, **summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/entitlements/check/{quota_type}")
async def check_quota(
    quota_type: str,
    increment: int = 1,
    _tenant_id: str = Depends(verify_api_key),
):
    """Check if an operation is allowed within quotas."""
    try:
        manager = get_entitlement_manager()
        result = manager.check_quota(_tenant_id, quota_type, increment)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/entitlements/feature/{feature}")
async def check_feature(
    feature: str,
    _tenant_id: str = Depends(verify_api_key),
):
    """Check if tenant has access to a feature."""
    try:
        manager = get_entitlement_manager()
        allowed = manager.check_feature(_tenant_id, feature)
        return {"feature": feature, "allowed": allowed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MCP HTTP Endpoint ====================

@api.post("/mcp/tools/{tool_name}")
async def mcp_tool_call(
    tool_name: str,
    request: dict,
    _tenant_id: str = Depends(verify_api_key),
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
        "discover_documentation": discover_documentation,
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


# ==================== Data Management ====================

@api.get("/api/data/sources")
async def list_sources_with_stats(
    _tenant_id: str = Depends(verify_api_key),
):
    """List all sources with comprehensive data stats."""
    try:
        from .data_manager import get_data_manager
        manager = get_data_manager(_tenant_id)
        stats = manager.get_all_sources_stats()
        return {
            "sources": [
                {
                    "source_id": s.source_id,
                    "source_name": s.source_name,
                    "base_url": s.base_url,
                    "indexed_docs": s.indexed_docs,
                    "raw_chunks": s.raw_chunks,
                    "vectors": s.vectors,
                    "crawl_jobs": s.crawl_jobs,
                    "crawl_plans": s.crawl_plans,
                    "url_universe_urls": s.url_universe_urls,
                    "chunk_hashes": s.chunk_hashes,
                }
                for s in stats
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/data/sources/{source_id}/stats")
async def get_source_data_stats(
    source_id: str,
    _tenant_id: str = Depends(verify_api_key),
):
    """Get comprehensive data stats for a single source."""
    try:
        from .data_manager import get_data_manager
        manager = get_data_manager(_tenant_id)
        stats = manager.get_source_stats(source_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Source not found")
        return {
            "source_id": stats.source_id,
            "source_name": stats.source_name,
            "base_url": stats.base_url,
            "indexed_docs": stats.indexed_docs,
            "raw_chunks": stats.raw_chunks,
            "vectors": stats.vectors,
            "crawl_jobs": stats.crawl_jobs,
            "crawl_plans": stats.crawl_plans,
            "url_universe_urls": stats.url_universe_urls,
            "chunk_hashes": stats.chunk_hashes,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/data/sources/{source_id}/chunks")
async def get_source_chunks(
    source_id: str,
    limit: int = 50,
    offset: int = 0,
    _tenant_id: str = Depends(verify_api_key),
):
    """Get raw chunks for a source with pagination."""
    try:
        from .data_manager import get_data_manager
        manager = get_data_manager(_tenant_id)
        return manager.get_chunks_by_source(source_id, limit, offset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/data/sources/{source_id}/history")
async def get_source_crawl_history(
    source_id: str,
    limit: int = 50,
    _tenant_id: str = Depends(verify_api_key),
):
    """Get crawl history for a source."""
    try:
        from .data_manager import get_data_manager
        manager = get_data_manager(_tenant_id)
        return manager.get_crawl_history_by_source(source_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.delete("/api/data/sources/{source_id}/history")
async def delete_source_crawl_history(
    source_id: str,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Delete all crawl history for a source."""
    try:
        from .data_manager import get_data_manager
        manager = get_data_manager(_tenant_id)
        deleted = manager.delete_crawl_history_by_source(source_id)
        return {"deleted": deleted, "source_id": source_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.delete("/api/data/sources/{source_id}/cascade")
async def delete_source_cascade_endpoint(
    source_id: str,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Delete a source and ALL related data (complete cascade delete)."""
    try:
        from .data_manager import get_data_manager
        manager = get_data_manager(_tenant_id)
        result = manager.delete_source_cascade(source_id)
        return {
            "source_id": result.source_id,
            "deleted": {
                "indexed_docs": result.deleted_indexed_docs,
                "raw_chunks": result.deleted_raw_chunks,
                "vectors": result.deleted_vectors,
                "crawl_jobs": result.deleted_crawl_jobs,
                "crawl_plans": result.deleted_crawl_plans,
                "url_universe": result.deleted_url_universe,
                "chunk_hashes": result.deleted_chunk_hashes,
                "content_hashes": result.deleted_content_hashes,
                "source": result.deleted_source,
            },
            "total_deleted": result.total_deleted,
            "errors": result.errors,
            "success": len(result.errors) == 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.delete("/api/data/all")
async def delete_all_data(
    confirm: bool = False,
    _tenant_id: str = Depends(require_paid_tenant),
):
    """Nuclear option - delete ALL data for tenant. Requires confirm=true."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must pass confirm=true to delete all data"
        )
    try:
        from .data_manager import get_data_manager
        manager = get_data_manager(_tenant_id)
        return manager.delete_all_data()
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
