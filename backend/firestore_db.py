"""
ContextPilot Firestore Database Layer
Cloud Firestore-based metadata storage for crawl jobs, indexed docs, and normalized docs.
Replaces SQLite for cloud deployment.
"""

import time
from dataclasses import dataclass, asdict
from enum import Enum
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
import logging

logger = logging.getLogger("contextpilot.firestore")


class CrawlStatus(str, Enum):
    """Status of a crawl job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CrawlMethod(str, Enum):
    """Method used for crawling."""
    FIRECRAWL = "firecrawl"
    LOCAL = "local"


@dataclass
class CrawlJob:
    """Represents a crawl job record."""
    id: Optional[str] = None
    url: str = ""
    status: CrawlStatus = CrawlStatus.PENDING
    method: Optional[CrawlMethod] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    chunks_count: int = 0
    error_message: Optional[str] = None
    created_at: float = 0.0


@dataclass
class IndexedDoc:
    """Represents an indexed document chunk."""
    id: Optional[str] = None
    url: str = ""
    page_url: str = ""
    source_url: str = ""
    title: str = ""
    content_hash: str = ""
    pinecone_id: str = ""
    content_preview: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class NormalizedDoc:
    """Represents a normalized document."""
    id: Optional[str] = None
    url_prefix: str = ""
    title: str = ""
    doc_hash: str = ""
    pinecone_id: str = ""
    raw_chunk_count: int = 0
    content_preview: str = ""
    created_at: float = 0.0


class FirestoreDatabase:
    """
    Firestore database manager for ContextPilot metadata.
    
    Collections:
    - crawl_jobs: Crawl job tracking
    - indexed_docs: Document chunk metadata
    - normalized_docs: Normalized document metadata
    """
    
    def __init__(self):
        self.config = get_config()
        self._client = None
        
        if HAS_FIRESTORE and self.config.has_firestore:
            self._init_firestore()
    
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
        
        doc_ref = self._client.collection("crawl_jobs").add(data)
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
            self._client.collection("crawl_jobs").document(job_id).update(updates)
    
    def get_crawl_job(self, job_id: str) -> Optional[CrawlJob]:
        """Get a crawl job by ID."""
        doc = self._client.collection("crawl_jobs").document(job_id).get()
        
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
        query = self._client.collection("crawl_jobs")
        
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
            self._client.collection("indexed_docs").document(existing.id).update(data)
            doc_id = existing.id
        else:
            # Insert new
            data["created_at"] = now
            doc_ref = self._client.collection("indexed_docs").add(data)
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
        docs = self._client.collection("indexed_docs").where(
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
        query = self._client.collection("indexed_docs")
        
        if source_url:
            query = query.where(filter=FieldFilter("source_url", "==", source_url))
        
        # Firestore doesn't have a direct count, so we need to stream
        return sum(1 for _ in query.stream())
    
    def list_indexed_sources(self) -> List[Dict[str, Any]]:
        """List all indexed sources with stats."""
        # Group by source_url and aggregate
        sources_map = {}
        
        for doc in self._client.collection("indexed_docs").stream():
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
        
        docs = self._client.collection("indexed_docs").where(
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
            self._client.collection("normalized_docs").document(existing.id).update(data)
            doc_id = existing.id
        else:
            doc_ref = self._client.collection("normalized_docs").add(data)
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
        
        for doc in self._client.collection("normalized_docs").order_by(
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
        docs = self._client.collection("normalized_docs").where(
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
    
    # ==================== Stats ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        # Crawl jobs stats
        crawl_jobs = list(self._client.collection("crawl_jobs").stream())
        crawl_stats = {
            "total": len(crawl_jobs),
            "completed": sum(1 for j in crawl_jobs if j.to_dict().get("status") == "completed"),
            "failed": sum(1 for j in crawl_jobs if j.to_dict().get("status") == "failed"),
            "running": sum(1 for j in crawl_jobs if j.to_dict().get("status") == "running"),
        }
        
        # Indexed docs stats
        indexed_docs = list(self._client.collection("indexed_docs").stream())
        sources = set(d.to_dict().get("source_url", "") for d in indexed_docs)
        indexed_stats = {
            "total": len(indexed_docs),
            "sources": len(sources),
        }
        
        # Normalized docs stats
        normalized_docs = list(self._client.collection("normalized_docs").stream())
        normalized_stats = {
            "total": len(normalized_docs),
        }
        
        return {
            "crawl_jobs": crawl_stats,
            "indexed_docs": indexed_stats,
            "normalized_docs": normalized_stats,
        }


# ==================== Unified Database Interface ====================

class Database:
    """
    Unified database interface that uses Firestore when available,
    falls back to SQLite for local development.
    """
    
    def __init__(self):
        config = get_config()
        self._use_firestore = HAS_FIRESTORE and config.has_firestore
        
        if self._use_firestore:
            self._impl = FirestoreDatabase()
            logger.info("Using Firestore database")
        else:
            # Fall back to SQLite
            from .db import Database as SQLiteDatabase
            self._impl = SQLiteDatabase()
            logger.info("Using SQLite database (Firestore not configured)")
    
    def __getattr__(self, name):
        """Delegate all method calls to the implementation."""
        return getattr(self._impl, name)


# Singleton instance
_firestore_db: Optional[Database] = None


def get_firestore_db() -> Database:
    """Get the singleton database instance (Firestore or SQLite fallback)."""
    global _firestore_db
    if _firestore_db is None:
        _firestore_db = Database()
    return _firestore_db


