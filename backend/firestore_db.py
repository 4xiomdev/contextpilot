"""
ContextPilot Firestore Database Layer
Cloud Firestore-based metadata storage for crawl jobs, indexed docs, and normalized docs.
Replaces SQLite for cloud deployment.
"""

import time
from typing import List, Optional, Dict, Any
import os

# Use google-cloud-firestore
try:
    from google.cloud import firestore
    from google.cloud.firestore_v1 import FieldFilter
    HAS_FIRESTORE = True
except ImportError:
    HAS_FIRESTORE = False
    firestore = None
    FieldFilter = None

from .config import get_config
from .models import (
    CrawlStatus,
    CrawlMethod,
    CrawlFrequency,
    SourceHealthStatus,
    SourceCreatedBy,
    CrawlJob,
    IndexedDoc,
    NormalizedDoc,
    RawChunk,
    Source,
)
from .tenant_context import get_tenant_id
import logging

logger = logging.getLogger("contextpilot.firestore")


 


class FirestoreDatabase:
    """
    Firestore database manager for ContextPilot metadata.
    
    Collections:
    - crawl_jobs: Crawl job tracking
    - indexed_docs: Document chunk metadata
    - normalized_docs: Normalized document metadata
    """
    
    def __init__(self, tenant_id: Optional[str] = None):
        self.config = get_config()
        self.tenant_id = tenant_id or get_tenant_id()
        self._client = None
        
        if HAS_FIRESTORE and self.config.has_firestore:
            self._init_firestore()

    def _col(self, name: str):
        """Get a collection reference (tenant-scoped when multi-tenant is enabled)."""
        if not self._client:
            raise RuntimeError("Firestore client not initialized")
        if self.config.multi_tenant.enabled:
            return (
                self._client.collection(self.config.multi_tenant.tenants_collection)
                .document(self.tenant_id)
                .collection(name)
            )
        return self._client.collection(name)
    
    def _init_firestore(self) -> None:
        """Initialize Firestore client."""
        try:
            # Use project ID from config
            if self.config.firestore.project_id:
                self._client = firestore.Client(
                    project=self.config.firestore.project_id
                )
            else:
                # Use default credentials (for Cloud Run)
                self._client = firestore.Client()
            
            logger.info(f"Connected to Firestore: {self.config.firestore.project_id or 'default'}")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore: {e}")
            self._client = None
    
    @property
    def is_ready(self) -> bool:
        """Check if Firestore is ready."""
        return self._client is not None
    
    # ==================== Crawl Jobs ====================
    
    def create_crawl_job(self, url: str) -> CrawlJob:
        """Create a new crawl job."""
        now = time.time()
        data = {
            "url": url,
            "status": CrawlStatus.PENDING.value,
            "method": None,
            "started_at": None,
            "completed_at": None,
            "chunks_count": 0,
            "error_message": None,
            "created_at": now,
        }
        
        doc_ref = self._col("crawl_jobs").add(data)
        doc_id = doc_ref[1].id
        
        return CrawlJob(
            id=doc_id,
            url=url,
            status=CrawlStatus.PENDING,
            created_at=now,
        )
    
    def update_crawl_job(
        self,
        job_id: str,
        status: Optional[CrawlStatus] = None,
        method: Optional[CrawlMethod] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        chunks_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update a crawl job."""
        updates = {}
        
        if status is not None:
            updates["status"] = status.value
        if method is not None:
            updates["method"] = method.value
        if started_at is not None:
            updates["started_at"] = started_at
        if completed_at is not None:
            updates["completed_at"] = completed_at
        if chunks_count is not None:
            updates["chunks_count"] = chunks_count
        if error_message is not None:
            updates["error_message"] = error_message
        
        if updates:
            self._col("crawl_jobs").document(job_id).update(updates)
    
    def get_crawl_job(self, job_id: str) -> Optional[CrawlJob]:
        """Get a crawl job by ID."""
        doc = self._col("crawl_jobs").document(job_id).get()
        
        if doc.exists:
            data = doc.to_dict()
            return CrawlJob(
                id=doc.id,
                url=data.get("url", ""),
                status=CrawlStatus(data.get("status", "pending")),
                method=CrawlMethod(data["method"]) if data.get("method") else None,
                started_at=data.get("started_at"),
                completed_at=data.get("completed_at"),
                chunks_count=data.get("chunks_count", 0),
                error_message=data.get("error_message"),
                created_at=data.get("created_at", 0),
            )
        return None
    
    def list_crawl_jobs(
        self,
        status: Optional[CrawlStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CrawlJob]:
        """List crawl jobs with optional filtering."""
        query = self._col("crawl_jobs")
        
        if status:
            query = query.where(filter=FieldFilter("status", "==", status.value))
        
        query = query.order_by("created_at", direction=firestore.Query.DESCENDING)
        query = query.limit(limit).offset(offset)
        
        jobs = []
        for doc in query.stream():
            data = doc.to_dict()
            jobs.append(CrawlJob(
                id=doc.id,
                url=data.get("url", ""),
                status=CrawlStatus(data.get("status", "pending")),
                method=CrawlMethod(data["method"]) if data.get("method") else None,
                started_at=data.get("started_at"),
                completed_at=data.get("completed_at"),
                chunks_count=data.get("chunks_count", 0),
                error_message=data.get("error_message"),
                created_at=data.get("created_at", 0),
            ))
        
        return jobs
    
    # ==================== Indexed Docs ====================
    
    def upsert_indexed_doc(
        self,
        url: str,
        page_url: str,
        source_url: str,
        title: str,
        content_hash: str,
        pinecone_id: str,
        content_preview: str = "",
    ) -> IndexedDoc:
        """Insert or update an indexed document."""
        now = time.time()
        
        # Check for existing by content_hash
        existing = self.get_indexed_doc_by_hash(content_hash)
        
        data = {
            "url": url,
            "page_url": page_url,
            "source_url": source_url,
            "title": title,
            "content_hash": content_hash,
            "pinecone_id": pinecone_id,
            "content_preview": content_preview,
            "updated_at": now,
        }
        
        if existing:
            # Update existing
            self._col("indexed_docs").document(existing.id).update(data)
            doc_id = existing.id
        else:
            # Insert new
            data["created_at"] = now
            doc_ref = self._col("indexed_docs").add(data)
            doc_id = doc_ref[1].id
        
        return IndexedDoc(
            id=doc_id,
            url=url,
            page_url=page_url,
            source_url=source_url,
            title=title,
            content_hash=content_hash,
            pinecone_id=pinecone_id,
            content_preview=content_preview,
            created_at=now if not existing else existing.created_at,
            updated_at=now,
        )
    
    def get_indexed_doc_by_hash(self, content_hash: str) -> Optional[IndexedDoc]:
        """Check if a document with this hash already exists."""
        docs = self._col("indexed_docs").where(
            filter=FieldFilter("content_hash", "==", content_hash)
        ).limit(1).stream()
        
        for doc in docs:
            data = doc.to_dict()
            return IndexedDoc(
                id=doc.id,
                url=data.get("url", ""),
                page_url=data.get("page_url", ""),
                source_url=data.get("source_url", ""),
                title=data.get("title", ""),
                content_hash=data.get("content_hash", ""),
                pinecone_id=data.get("pinecone_id", ""),
                content_preview=data.get("content_preview", ""),
                created_at=data.get("created_at", 0),
                updated_at=data.get("updated_at", 0),
            )
        
        return None
    
    def count_indexed_docs(self, source_url: Optional[str] = None) -> int:
        """Count indexed documents."""
        query = self._col("indexed_docs")
        
        if source_url:
            query = query.where(filter=FieldFilter("source_url", "==", source_url))
        
        # Firestore doesn't have a direct count, so we need to stream
        return sum(1 for _ in query.stream())
    
    def list_indexed_sources(self) -> List[Dict[str, Any]]:
        """List all indexed sources with stats."""
        # Group by source_url and aggregate
        sources_map = {}
        
        for doc in self._col("indexed_docs").stream():
            data = doc.to_dict()
            source = data.get("source_url", "")
            
            if source not in sources_map:
                sources_map[source] = {
                    "source_url": source,
                    "title": data.get("title", ""),
                    "chunks": 0,
                    "last_indexed": 0,
                }
            
            sources_map[source]["chunks"] += 1
            created = data.get("created_at", 0)
            if created > sources_map[source]["last_indexed"]:
                sources_map[source]["last_indexed"] = created
        
        # Sort by last_indexed descending
        return sorted(
            sources_map.values(),
            key=lambda x: x["last_indexed"],
            reverse=True,
        )
    
    def delete_indexed_docs_by_source(self, source_url: str) -> List[str]:
        """Delete all indexed docs for a source. Returns Pinecone IDs to delete."""
        pinecone_ids = []
        
        docs = self._col("indexed_docs").where(
            filter=FieldFilter("source_url", "==", source_url)
        ).stream()
        
        batch = self._client.batch()
        batch_count = 0
        
        for doc in docs:
            data = doc.to_dict()
            pinecone_ids.append(data.get("pinecone_id", ""))
            batch.delete(doc.reference)
            batch_count += 1
            
            # Firestore batch limit is 500
            if batch_count >= 450:
                batch.commit()
                batch = self._client.batch()
                batch_count = 0
        
        if batch_count > 0:
            batch.commit()
        
        return pinecone_ids

    # ==================== Raw Chunks ====================

    def upsert_raw_chunk(
        self,
        source_url: str,
        page_url: str,
        title: str,
        content: str,
        content_hash: str,
    ) -> RawChunk:
        """Insert or update a raw chunk used for normalization."""
        now = time.time()
        data = {
            "source_url": source_url,
            "page_url": page_url,
            "title": title,
            "content": content,
            "content_hash": content_hash,
            "created_at": now,
        }
        self._col("raw_chunks").document(content_hash).set(data, merge=True)
        return RawChunk(
            id=content_hash,
            source_url=source_url,
            page_url=page_url,
            title=title,
            content=content,
            content_hash=content_hash,
            created_at=now,
        )

    def list_raw_chunks_by_prefix(
        self,
        url_prefix: str,
        limit: int = 2000,
    ) -> List[RawChunk]:
        """List raw chunks by page URL prefix."""
        chunks: List[RawChunk] = []

        try:
            query = self._col("raw_chunks").where(
                filter=FieldFilter("page_url", ">=", url_prefix)
            ).where(
                filter=FieldFilter("page_url", "<=", f"{url_prefix}\uf8ff")
            ).limit(limit)
            docs = query.stream()
        except Exception:
            docs = self._col("raw_chunks").where(
                filter=FieldFilter("source_url", "==", url_prefix)
            ).limit(limit).stream()

        for doc in docs:
            data = doc.to_dict()
            page_url = data.get("page_url", "")
            if not page_url.startswith(url_prefix):
                continue
            chunks.append(RawChunk(
                id=doc.id,
                source_url=data.get("source_url", ""),
                page_url=page_url,
                title=data.get("title", ""),
                content=data.get("content", ""),
                content_hash=data.get("content_hash", ""),
                created_at=data.get("created_at", 0),
            ))

        return chunks

    def delete_raw_chunks_by_source(self, source_url: str) -> int:
        """Delete raw chunks for a given source URL."""
        docs = self._col("raw_chunks").where(
            filter=FieldFilter("source_url", "==", source_url)
        ).stream()

        batch = self._client.batch()
        batch_count = 0
        deleted = 0

        for doc in docs:
            batch.delete(doc.reference)
            batch_count += 1
            deleted += 1
            if batch_count >= 450:
                batch.commit()
                batch = self._client.batch()
                batch_count = 0

        if batch_count > 0:
            batch.commit()

        return deleted
    
    # ==================== Normalized Docs ====================
    
    def upsert_normalized_doc(
        self,
        url_prefix: str,
        title: str,
        doc_hash: str,
        pinecone_id: str,
        raw_chunk_count: int = 0,
        content_preview: str = "",
    ) -> NormalizedDoc:
        """Insert or update a normalized document."""
        now = time.time()
        
        # Check for existing by url_prefix
        existing = self.get_normalized_doc_by_prefix(url_prefix)
        
        data = {
            "url_prefix": url_prefix,
            "title": title,
            "doc_hash": doc_hash,
            "pinecone_id": pinecone_id,
            "raw_chunk_count": raw_chunk_count,
            "content_preview": content_preview,
            "created_at": now,
        }
        
        if existing:
            self._col("normalized_docs").document(existing.id).update(data)
            doc_id = existing.id
        else:
            doc_ref = self._col("normalized_docs").add(data)
            doc_id = doc_ref[1].id
        
        return NormalizedDoc(
            id=doc_id,
            url_prefix=url_prefix,
            title=title,
            doc_hash=doc_hash,
            pinecone_id=pinecone_id,
            raw_chunk_count=raw_chunk_count,
            content_preview=content_preview,
            created_at=now,
        )
    
    def list_normalized_docs(self) -> List[NormalizedDoc]:
        """List all normalized documents."""
        docs = []
        
        for doc in self._col("normalized_docs").order_by(
            "created_at", direction=firestore.Query.DESCENDING
        ).stream():
            data = doc.to_dict()
            docs.append(NormalizedDoc(
                id=doc.id,
                url_prefix=data.get("url_prefix", ""),
                title=data.get("title", ""),
                doc_hash=data.get("doc_hash", ""),
                pinecone_id=data.get("pinecone_id", ""),
                raw_chunk_count=data.get("raw_chunk_count", 0),
                content_preview=data.get("content_preview", ""),
                created_at=data.get("created_at", 0),
            ))
        
        return docs
    
    def get_normalized_doc_by_prefix(self, url_prefix: str) -> Optional[NormalizedDoc]:
        """Get a normalized doc by URL prefix."""
        docs = self._col("normalized_docs").where(
            filter=FieldFilter("url_prefix", "==", url_prefix)
        ).limit(1).stream()

        for doc in docs:
            data = doc.to_dict()
            return NormalizedDoc(
                id=doc.id,
                url_prefix=data.get("url_prefix", ""),
                title=data.get("title", ""),
                doc_hash=data.get("doc_hash", ""),
                pinecone_id=data.get("pinecone_id", ""),
                raw_chunk_count=data.get("raw_chunk_count", 0),
                content_preview=data.get("content_preview", ""),
                created_at=data.get("created_at", 0),
            )

        return None

    # ==================== Sources ====================

    def _source_from_dict(self, doc_id: str, data: Dict[str, Any]) -> Source:
        """Convert Firestore document to Source dataclass."""
        return Source(
            id=doc_id,
            name=data.get("name", ""),
            base_url=data.get("base_url", ""),
            sitemap_url=data.get("sitemap_url"),
            priority_paths=data.get("priority_paths", []),
            exclude_paths=data.get("exclude_paths", []),
            crawl_frequency=CrawlFrequency(data.get("crawl_frequency", "weekly")),
            max_pages=data.get("max_pages", 500),
            is_enabled=data.get("is_enabled", True),
            is_curated=data.get("is_curated", False),
            created_by=SourceCreatedBy(data.get("created_by", "user")),
            health_status=SourceHealthStatus(data.get("health_status", "unknown")),
            last_crawled_at=data.get("last_crawled_at"),
            last_content_hash=data.get("last_content_hash"),
            next_crawl_at=data.get("next_crawl_at"),
            chunks_count=data.get("chunks_count", 0),
            error_message=data.get("error_message"),
            tags=data.get("tags", []),
            description=data.get("description"),
            created_at=data.get("created_at", 0),
            updated_at=data.get("updated_at", 0),
        )

    def _source_to_dict(self, source: Source) -> Dict[str, Any]:
        """Convert Source dataclass to Firestore document."""
        return {
            "name": source.name,
            "base_url": source.base_url,
            "sitemap_url": source.sitemap_url,
            "priority_paths": source.priority_paths or [],
            "exclude_paths": source.exclude_paths or [],
            "crawl_frequency": source.crawl_frequency.value,
            "max_pages": source.max_pages,
            "is_enabled": source.is_enabled,
            "is_curated": source.is_curated,
            "created_by": source.created_by.value,
            "health_status": source.health_status.value,
            "last_crawled_at": source.last_crawled_at,
            "last_content_hash": source.last_content_hash,
            "next_crawl_at": source.next_crawl_at,
            "chunks_count": source.chunks_count,
            "error_message": source.error_message,
            "tags": source.tags or [],
            "description": source.description,
            "created_at": source.created_at,
            "updated_at": source.updated_at,
        }

    def create_source(self, source: Source) -> Source:
        """Create a new source."""
        now = time.time()
        source.created_at = now
        source.updated_at = now

        data = self._source_to_dict(source)
        doc_ref = self._col("sources").add(data)
        source.id = doc_ref[1].id

        return source

    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        doc = self._col("sources").document(source_id).get()

        if doc.exists:
            return self._source_from_dict(doc.id, doc.to_dict())
        return None

    def get_source_by_url(self, base_url: str) -> Optional[Source]:
        """Get a source by base URL."""
        docs = self._col("sources").where(
            filter=FieldFilter("base_url", "==", base_url)
        ).limit(1).stream()

        for doc in docs:
            return self._source_from_dict(doc.id, doc.to_dict())
        return None

    def update_source(self, source_id: str, updates: Dict[str, Any]) -> Optional[Source]:
        """Update a source with partial data."""
        updates["updated_at"] = time.time()

        # Convert enum values to strings
        if "crawl_frequency" in updates and isinstance(updates["crawl_frequency"], CrawlFrequency):
            updates["crawl_frequency"] = updates["crawl_frequency"].value
        if "health_status" in updates and isinstance(updates["health_status"], SourceHealthStatus):
            updates["health_status"] = updates["health_status"].value
        if "created_by" in updates and isinstance(updates["created_by"], SourceCreatedBy):
            updates["created_by"] = updates["created_by"].value

        self._col("sources").document(source_id).update(updates)
        return self.get_source(source_id)

    def delete_source(self, source_id: str) -> bool:
        """Delete a source."""
        self._col("sources").document(source_id).delete()
        return True

    def list_sources(
        self,
        is_enabled: Optional[bool] = None,
        created_by: Optional[SourceCreatedBy] = None,
        health_status: Optional[SourceHealthStatus] = None,
        limit: int = 100,
    ) -> List[Source]:
        """List sources with optional filtering."""
        query = self._col("sources")

        if is_enabled is not None:
            query = query.where(filter=FieldFilter("is_enabled", "==", is_enabled))
        if created_by is not None:
            query = query.where(filter=FieldFilter("created_by", "==", created_by.value))
        if health_status is not None:
            query = query.where(filter=FieldFilter("health_status", "==", health_status.value))

        query = query.order_by("name").limit(limit)

        sources = []
        for doc in query.stream():
            sources.append(self._source_from_dict(doc.id, doc.to_dict()))

        return sources

    def list_sources_due_for_crawl(self) -> List[Source]:
        """List sources that are due for scheduled crawling."""
        now = time.time()
        sources = []

        # Get enabled sources with next_crawl_at in the past
        query = self._col("sources").where(
            filter=FieldFilter("is_enabled", "==", True)
        )

        for doc in query.stream():
            source = self._source_from_dict(doc.id, doc.to_dict())
            if source.is_due_for_crawl():
                sources.append(source)

        return sources

    def update_source_after_crawl(
        self,
        source_id: str,
        success: bool,
        chunks_count: int = 0,
        content_hash: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Source]:
        """Update source after a crawl completes."""
        now = time.time()
        source = self.get_source(source_id)

        if not source:
            return None

        updates = {
            "last_crawled_at": now,
            "updated_at": now,
        }

        if success:
            updates["health_status"] = SourceHealthStatus.HEALTHY.value
            updates["chunks_count"] = chunks_count
            updates["error_message"] = None
            updates["next_crawl_at"] = source.compute_next_crawl_time()
            if content_hash:
                updates["last_content_hash"] = content_hash
        else:
            updates["health_status"] = SourceHealthStatus.ERROR.value
            updates["error_message"] = error_message
            # Still schedule next crawl even on failure
            updates["next_crawl_at"] = source.compute_next_crawl_time()

        self._col("sources").document(source_id).update(updates)
        return self.get_source(source_id)

    def mark_source_stale(self, source_id: str) -> Optional[Source]:
        """Mark a source as stale (needs re-crawling)."""
        return self.update_source(source_id, {
            "health_status": SourceHealthStatus.STALE.value,
            "next_crawl_at": time.time(),  # Due now
        })

    def count_sources(self) -> Dict[str, int]:
        """Count sources by status."""
        counts = {
            "total": 0,
            "healthy": 0,
            "stale": 0,
            "error": 0,
            "enabled": 0,
            "curated": 0,
        }

        for doc in self._col("sources").stream():
            data = doc.to_dict()
            counts["total"] += 1

            status = data.get("health_status", "unknown")
            if status in counts:
                counts[status] += 1

            if data.get("is_enabled", False):
                counts["enabled"] += 1
            if data.get("is_curated", False):
                counts["curated"] += 1

        return counts

    # ==================== Content Hashes (for freshness detection) ====================

    def get_content_hashes(self, source_id: str) -> Dict[str, str]:
        """
        Get stored content hashes for a source.

        Args:
            source_id: Source ID

        Returns:
            Dict mapping page URLs to their content hashes
        """
        doc = self._col("content_hashes").document(source_id).get()
        if doc.exists:
            return doc.to_dict().get("hashes", {})
        return {}

    def store_content_hashes(self, source_id: str, hashes: Dict[str, str]) -> None:
        """
        Store content hashes for a source.

        Args:
            source_id: Source ID
            hashes: Dict mapping page URLs to their content hashes
        """
        self._col("content_hashes").document(source_id).set({
            "hashes": hashes,
            "updated_at": time.time(),
        })

    def clear_content_hashes(self, source_id: str) -> None:
        """Clear stored content hashes for a source."""
        self._col("content_hashes").document(source_id).delete()

    # ==================== URL Universe + Crawl Plans ====================

    def replace_url_universe(self, source_id: str, urls: List[Dict[str, Any]]) -> int:
        """
        Store a URL universe for a source.

        Firestore implementation stores only a summarized snapshot to avoid large writes.
        """
        try:
            now = time.time()
            snapshot = {
                "source_id": source_id,
                "total_urls": len(urls),
                "created_at": now,
                "sample": urls[:200],
            }
            self._col("url_universes").document(str(source_id)).set(snapshot)
            return len(urls)
        except Exception:
            return 0

    def list_url_universe(self, source_id: str, limit: int = 200_000) -> List[Dict[str, Any]]:
        """Return the stored universe sample (Firestore stores a capped sample)."""
        doc = self._col("url_universes").document(str(source_id)).get()
        if doc.exists:
            data = doc.to_dict() or {}
            return list(data.get("sample", []))[:limit]
        return []

    def get_universe_meta(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get URL universe metadata for a source."""
        doc = self._col("url_universes").document(str(source_id)).get()
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        return {
            "source_id": source_id,
            "url_count": data.get("total_urls", 0),
            "ttl_hours": data.get("ttl_hours", 24),
            "cached_at": data.get("cached_at", data.get("created_at", 0)),
            "discovery_method": data.get("discovery_method"),
            "created_at": data.get("created_at", 0),
            "updated_at": data.get("updated_at", data.get("created_at", 0)),
        }

    def set_universe_meta(
        self,
        source_id: str,
        url_count: int,
        ttl_hours: int = 24,
        discovery_method: Optional[str] = None,
    ) -> None:
        """Set or update URL universe metadata for a source."""
        now = time.time()
        doc_ref = self._col("url_universes").document(str(source_id))
        doc = doc_ref.get()

        if doc.exists:
            # Update existing document with metadata
            doc_ref.update({
                "total_urls": url_count,
                "ttl_hours": ttl_hours,
                "cached_at": now,
                "discovery_method": discovery_method,
                "updated_at": now,
            })
        else:
            # Create metadata-only document
            doc_ref.set({
                "source_id": source_id,
                "total_urls": url_count,
                "ttl_hours": ttl_hours,
                "cached_at": now,
                "discovery_method": discovery_method,
                "created_at": now,
                "updated_at": now,
                "sample": [],
            })

    def is_universe_stale(self, source_id: str, default_ttl_hours: int = 24) -> bool:
        """
        Check if the URL universe for a source is stale.

        Returns True if:
        - No universe exists for the source
        - The cached universe has exceeded its TTL
        """
        meta = self.get_universe_meta(source_id)
        if not meta:
            return True

        ttl_hours = meta.get("ttl_hours", default_ttl_hours)
        cached_at = meta.get("cached_at", 0)
        ttl_seconds = ttl_hours * 3600

        return (time.time() - cached_at) > ttl_seconds

    def delete_universe_meta(self, source_id: str) -> bool:
        """Delete URL universe and metadata for a source."""
        self._col("url_universes").document(str(source_id)).delete()
        return True

    def upsert_crawl_plan(
        self,
        source_id: str,
        status: str,
        rules: Dict[str, Any],
        report: Dict[str, Any],
        total_urls_seen: int,
        kept_urls: int,
        dropped_urls: int,
        maybe_urls: int,
    ) -> str:
        """Insert a crawl plan report."""
        now = time.time()
        doc_ref = self._col("crawl_plans").add({
            "source_id": str(source_id),
            "status": status,
            "rules": rules,
            "report": report,
            "total_urls_seen": int(total_urls_seen),
            "kept_urls": int(kept_urls),
            "dropped_urls": int(dropped_urls),
            "maybe_urls": int(maybe_urls),
            "created_at": now,
            "updated_at": now,
        })
        return doc_ref[1].id

    def get_latest_crawl_plan(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest crawl plan for a source."""
        docs = (
            self._col("crawl_plans")
            .where(filter=FieldFilter("source_id", "==", str(source_id)))
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        for doc in docs:
            data = doc.to_dict() or {}
            return {
                "id": doc.id,
                "source_id": data.get("source_id"),
                "status": data.get("status"),
                "rules": data.get("rules") or {},
                "report": data.get("report") or {},
                "total_urls_seen": data.get("total_urls_seen", 0),
                "kept_urls": data.get("kept_urls", 0),
                "dropped_urls": data.get("dropped_urls", 0),
                "maybe_urls": data.get("maybe_urls", 0),
                "created_at": data.get("created_at", 0),
                "updated_at": data.get("updated_at", 0),
            }
        return None

    def update_crawl_plan_status(self, plan_id: str, status: str) -> bool:
        """Update crawl plan status."""
        self._col("crawl_plans").document(str(plan_id)).update({
            "status": status,
            "updated_at": time.time(),
        })
        return True

    # ==================== Normalization Lineage ====================

    def store_normalization_lineage(self, lineage: Dict[str, Any]) -> str:
        """
        Store normalization lineage data.

        Args:
            lineage: Dict with keys: normalized_doc_id, url_prefix, source_chunk_count,
                     used_chunk_count, sections, created_at

        Returns:
            Document ID of inserted lineage record.
        """
        now = lineage.get("created_at", time.time())
        doc_ref = self._col("normalization_lineage").add({
            "normalized_doc_id": lineage.get("normalized_doc_id", ""),
            "url_prefix": lineage.get("url_prefix", ""),
            "source_chunk_count": lineage.get("source_chunk_count", 0),
            "used_chunk_count": lineage.get("used_chunk_count", 0),
            "sections": lineage.get("sections", []),
            "created_at": now,
        })
        return doc_ref[1].id

    def get_normalization_lineage(self, normalized_doc_id: str) -> Optional[Dict[str, Any]]:
        """Get lineage for a normalized document."""
        docs = (
            self._col("normalization_lineage")
            .where(filter=FieldFilter("normalized_doc_id", "==", normalized_doc_id))
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        for doc in docs:
            data = doc.to_dict() or {}
            return {
                "id": doc.id,
                "normalized_doc_id": data.get("normalized_doc_id", ""),
                "url_prefix": data.get("url_prefix", ""),
                "source_chunk_count": data.get("source_chunk_count", 0),
                "used_chunk_count": data.get("used_chunk_count", 0),
                "sections": data.get("sections", []),
                "created_at": data.get("created_at", 0),
            }
        return None

    def get_lineage_by_prefix(self, url_prefix: str) -> Optional[Dict[str, Any]]:
        """Get lineage for a URL prefix."""
        docs = (
            self._col("normalization_lineage")
            .where(filter=FieldFilter("url_prefix", "==", url_prefix))
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        for doc in docs:
            data = doc.to_dict() or {}
            return {
                "id": doc.id,
                "normalized_doc_id": data.get("normalized_doc_id", ""),
                "url_prefix": data.get("url_prefix", ""),
                "source_chunk_count": data.get("source_chunk_count", 0),
                "used_chunk_count": data.get("used_chunk_count", 0),
                "sections": data.get("sections", []),
                "created_at": data.get("created_at", 0),
            }
        return None

    def list_lineage(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all normalization lineage records."""
        docs = (
            self._col("normalization_lineage")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [
            {
                "id": doc.id,
                "normalized_doc_id": (doc.to_dict() or {}).get("normalized_doc_id", ""),
                "url_prefix": (doc.to_dict() or {}).get("url_prefix", ""),
                "source_chunk_count": (doc.to_dict() or {}).get("source_chunk_count", 0),
                "used_chunk_count": (doc.to_dict() or {}).get("used_chunk_count", 0),
                "sections": (doc.to_dict() or {}).get("sections", []),
                "created_at": (doc.to_dict() or {}).get("created_at", 0),
            }
            for doc in docs
        ]

    # ==================== Stats ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        # Crawl jobs stats
        crawl_jobs = list(self._col("crawl_jobs").stream())
        crawl_stats = {
            "total": len(crawl_jobs),
            "completed": sum(1 for j in crawl_jobs if j.to_dict().get("status") == "completed"),
            "failed": sum(1 for j in crawl_jobs if j.to_dict().get("status") == "failed"),
            "running": sum(1 for j in crawl_jobs if j.to_dict().get("status") == "running"),
        }
        
        # Indexed docs stats
        indexed_docs = list(self._col("indexed_docs").stream())
        sources = set(d.to_dict().get("source_url", "") for d in indexed_docs)
        indexed_stats = {
            "total": len(indexed_docs),
            "sources": len(sources),
        }
        
        # Normalized docs stats
        normalized_docs = list(self._col("normalized_docs").stream())
        normalized_stats = {
            "total": len(normalized_docs),
        }

        # Source registry stats
        source_stats = self.count_sources()

        return {
            "crawl_jobs": crawl_stats,
            "indexed_docs": indexed_stats,
            "normalized_docs": normalized_stats,
            "sources": source_stats,
        }

    # ==================== Data Management (Cascade Delete + Stats) ====================

    def delete_crawl_jobs_by_url(self, base_url: str) -> int:
        """Delete all crawl jobs matching base URL prefix."""
        count = 0
        docs = self._col("crawl_jobs").stream()
        for doc in docs:
            data = doc.to_dict() or {}
            if data.get("url", "").startswith(base_url):
                doc.reference.delete()
                count += 1
        return count

    def delete_crawl_plans_by_source(self, source_id: str) -> int:
        """Delete all crawl plans for a source."""
        count = 0
        docs = (
            self._col("crawl_plans")
            .where(filter=FieldFilter("source_id", "==", str(source_id)))
            .stream()
        )
        for doc in docs:
            doc.reference.delete()
            count += 1
        return count

    def delete_url_universe_by_source(self, source_id: str) -> int:
        """Delete URL universe for a source (stored as single doc)."""
        self._col("url_universes").document(str(source_id)).delete()
        return 1

    def count_indexed_docs_by_source(self, source_url: str) -> int:
        """Count indexed docs for a source URL."""
        docs = (
            self._col("indexed_docs")
            .where(filter=FieldFilter("source_url", "==", source_url))
            .stream()
        )
        return sum(1 for _ in docs)

    def count_raw_chunks_by_source(self, source_url: str) -> int:
        """Count raw chunks for a source URL."""
        docs = (
            self._col("raw_chunks")
            .where(filter=FieldFilter("source_url", "==", source_url))
            .stream()
        )
        return sum(1 for _ in docs)

    def count_crawl_jobs_by_url(self, base_url: str) -> int:
        """Count crawl jobs matching base URL prefix."""
        count = 0
        docs = self._col("crawl_jobs").stream()
        for doc in docs:
            data = doc.to_dict() or {}
            if data.get("url", "").startswith(base_url):
                count += 1
        return count

    def count_crawl_plans_by_source(self, source_id: str) -> int:
        """Count crawl plans for a source."""
        docs = (
            self._col("crawl_plans")
            .where(filter=FieldFilter("source_id", "==", str(source_id)))
            .stream()
        )
        return sum(1 for _ in docs)

    def count_url_universe_by_source(self, source_id: str) -> int:
        """Count URL universe entries (single doc stores all URLs as array)."""
        doc = self._col("url_universes").document(str(source_id)).get()
        if doc.exists:
            data = doc.to_dict() or {}
            return len(data.get("urls", []))
        return 0

    def count_chunk_hashes_by_source(self, source_id: str) -> int:
        """Count chunk hashes for a source."""
        docs = (
            self._col("chunk_hashes")
            .where(filter=FieldFilter("source_id", "==", str(source_id)))
            .stream()
        )
        return sum(1 for _ in docs)

    def delete_all_chunk_hashes_for_source(self, source_id: str) -> int:
        """Delete all chunk hashes for a source."""
        count = 0
        docs = (
            self._col("chunk_hashes")
            .where(filter=FieldFilter("source_id", "==", str(source_id)))
            .stream()
        )
        for doc in docs:
            doc.reference.delete()
            count += 1
        return count

    def list_crawl_jobs_by_url(self, base_url: str, limit: int = 100) -> List[CrawlJob]:
        """List crawl jobs matching base URL prefix."""
        jobs = []
        docs = self._col("crawl_jobs").stream()
        for doc in docs:
            data = doc.to_dict() or {}
            if data.get("url", "").startswith(base_url):
                jobs.append(CrawlJob(
                    id=doc.id,
                    url=data.get("url", ""),
                    status=CrawlStatus(data.get("status", "running")),
                    method=CrawlMethod(data.get("method")) if data.get("method") else None,
                    started_at=data.get("started_at"),
                    completed_at=data.get("completed_at"),
                    chunks_count=data.get("chunks_count", 0),
                    error_message=data.get("error_message"),
                    created_at=data.get("created_at", 0),
                ))
                if len(jobs) >= limit:
                    break
        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at or 0, reverse=True)
        return jobs[:limit]

    def list_raw_chunks_by_source(self, source_url: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List raw chunks for a source URL with pagination."""
        chunks = []
        docs = (
            self._col("raw_chunks")
            .where(filter=FieldFilter("source_url", "==", source_url))
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit + offset)
            .stream()
        )
        for i, doc in enumerate(docs):
            if i < offset:
                continue
            data = doc.to_dict() or {}
            chunks.append({
                "id": doc.id,
                "source_url": data.get("source_url", ""),
                "page_url": data.get("page_url", ""),
                "title": data.get("title", ""),
                "content": data.get("content", ""),
                "content_hash": data.get("content_hash", ""),
                "created_at": data.get("created_at", 0),
            })
        return chunks

    # ========== Nuclear Delete Methods ==========

    def delete_all_indexed_docs(self) -> int:
        """Delete ALL indexed docs (nuclear option)."""
        count = 0
        batch_size = 500
        while True:
            docs = list(self._col("indexed_docs").limit(batch_size).stream())
            if not docs:
                break
            batch = self._db.batch()
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
            batch.commit()
        return count

    def delete_all_raw_chunks(self) -> int:
        """Delete ALL raw chunks (nuclear option)."""
        count = 0
        batch_size = 500
        while True:
            docs = list(self._col("raw_chunks").limit(batch_size).stream())
            if not docs:
                break
            batch = self._db.batch()
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
            batch.commit()
        return count

    def delete_all_crawl_jobs(self) -> int:
        """Delete ALL crawl jobs (nuclear option)."""
        count = 0
        batch_size = 500
        while True:
            docs = list(self._col("crawl_jobs").limit(batch_size).stream())
            if not docs:
                break
            batch = self._db.batch()
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
            batch.commit()
        return count


# ==================== Unified Database Interface ====================

class Database:
    """
    Unified database interface that uses Firestore when available,
    falls back to SQLite for local development.
    """
    
    def __init__(self, tenant_id: Optional[str] = None):
        config = get_config()
        self._use_firestore = HAS_FIRESTORE and config.has_firestore
        
        if self._use_firestore:
            self._impl = FirestoreDatabase(tenant_id=tenant_id)
            logger.info("Using Firestore database")
        else:
            # Fall back to SQLite
            from .db import Database as SQLiteDatabase
            self._impl = SQLiteDatabase()
            logger.info("Using SQLite database (Firestore not configured)")
    
    def __getattr__(self, name):
        """Delegate all method calls to the implementation."""
        return getattr(self._impl, name)

    def list_sources(
        self,
        is_enabled: Optional[bool] = None,
        created_by: Optional[SourceCreatedBy] = None,
        health_status: Optional[SourceHealthStatus] = None,
        limit: int = 100,
    ) -> List[Source]:
        """List sources if supported by the backend, otherwise return empty."""
        if hasattr(self._impl, "list_sources"):
            return self._impl.list_sources(
                is_enabled=is_enabled,
                created_by=created_by,
                health_status=health_status,
                limit=limit,
            )
        return []

    def count_sources(self) -> Dict[str, int]:
        """Count sources if supported by the backend, otherwise return empty counts."""
        if hasattr(self._impl, "count_sources"):
            return self._impl.count_sources()
        return {"total": 0, "healthy": 0, "stale": 0, "error": 0, "enabled": 0, "curated": 0}

    def replace_url_universe(self, source_id: str, urls: List[Dict[str, Any]]) -> int:
        """Replace the stored URL universe if supported."""
        if hasattr(self._impl, "replace_url_universe"):
            return self._impl.replace_url_universe(source_id, urls)
        return 0

    def list_url_universe(self, source_id: str, limit: int = 200_000) -> List[Dict[str, Any]]:
        """List URL universe entries if supported."""
        if hasattr(self._impl, "list_url_universe"):
            return self._impl.list_url_universe(source_id, limit=limit)
        return []

    def get_universe_meta(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get URL universe metadata if supported."""
        if hasattr(self._impl, "get_universe_meta"):
            return self._impl.get_universe_meta(source_id)
        return None

    def set_universe_meta(
        self,
        source_id: str,
        url_count: int,
        ttl_hours: int = 24,
        discovery_method: Optional[str] = None,
    ) -> None:
        """Set URL universe metadata if supported."""
        if hasattr(self._impl, "set_universe_meta"):
            self._impl.set_universe_meta(source_id, url_count, ttl_hours, discovery_method)

    def is_universe_stale(self, source_id: str, default_ttl_hours: int = 24) -> bool:
        """Check if URL universe is stale if supported."""
        if hasattr(self._impl, "is_universe_stale"):
            return self._impl.is_universe_stale(source_id, default_ttl_hours)
        return True  # Default to stale if not supported

    def delete_universe_meta(self, source_id: str) -> bool:
        """Delete URL universe metadata if supported."""
        if hasattr(self._impl, "delete_universe_meta"):
            return self._impl.delete_universe_meta(source_id)
        return False

    def store_normalization_lineage(self, lineage: Dict[str, Any]) -> Any:
        """Store normalization lineage if supported."""
        if hasattr(self._impl, "store_normalization_lineage"):
            return self._impl.store_normalization_lineage(lineage)
        return None

    def get_normalization_lineage(self, normalized_doc_id: str) -> Optional[Dict[str, Any]]:
        """Get lineage for a normalized document if supported."""
        if hasattr(self._impl, "get_normalization_lineage"):
            return self._impl.get_normalization_lineage(normalized_doc_id)
        return None

    def get_lineage_by_prefix(self, url_prefix: str) -> Optional[Dict[str, Any]]:
        """Get lineage for a URL prefix if supported."""
        if hasattr(self._impl, "get_lineage_by_prefix"):
            return self._impl.get_lineage_by_prefix(url_prefix)
        return None

    def list_lineage(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List normalization lineage records if supported."""
        if hasattr(self._impl, "list_lineage"):
            return self._impl.list_lineage(limit=limit)
        return []

    def upsert_crawl_plan(
        self,
        source_id: str,
        status: str,
        rules: Dict[str, Any],
        report: Dict[str, Any],
        total_urls_seen: int,
        kept_urls: int,
        dropped_urls: int,
        maybe_urls: int,
    ) -> Any:
        """Store a crawl plan if supported."""
        if hasattr(self._impl, "upsert_crawl_plan"):
            return self._impl.upsert_crawl_plan(
                source_id=source_id,
                status=status,
                rules=rules,
                report=report,
                total_urls_seen=total_urls_seen,
                kept_urls=kept_urls,
                dropped_urls=dropped_urls,
                maybe_urls=maybe_urls,
            )
        return None

    def get_latest_crawl_plan(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest crawl plan if supported."""
        if hasattr(self._impl, "get_latest_crawl_plan"):
            return self._impl.get_latest_crawl_plan(source_id)
        return None

    def update_crawl_plan_status(self, plan_id: Any, status: str) -> bool:
        """Update crawl plan status if supported."""
        if hasattr(self._impl, "update_crawl_plan_status"):
            return self._impl.update_crawl_plan_status(plan_id, status)
        return False


# Singleton instances (keyed by tenant id for multi-tenant Firestore).
_firestore_dbs: Dict[str, Database] = {}


def get_firestore_db(tenant_id: Optional[str] = None) -> Database:
    """Get the database instance (Firestore or SQLite fallback)."""
    # SQLite fallback is single-tenant; reuse one instance.
    config = get_config()
    if not (HAS_FIRESTORE and config.has_firestore):
        if "sqlite" not in _firestore_dbs:
            _firestore_dbs["sqlite"] = Database()
        return _firestore_dbs["sqlite"]

    key = tenant_id or get_tenant_id()
    if key not in _firestore_dbs:
        _firestore_dbs[key] = Database(tenant_id=key)
    return _firestore_dbs[key]
