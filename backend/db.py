"""
ContextPilot Database Layer
SQLite-based metadata storage for crawl jobs, indexed docs, and normalized docs.
"""

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator
import json

from .config import get_config


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
    id: Optional[int] = None
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
    id: Optional[int] = None
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
    id: Optional[int] = None
    url_prefix: str = ""
    title: str = ""
    doc_hash: str = ""
    pinecone_id: str = ""
    raw_chunk_count: int = 0
    content_preview: str = ""
    created_at: float = 0.0


# SQL Schema
SCHEMA = """
-- Crawl jobs tracking
CREATE TABLE IF NOT EXISTS crawl_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    method TEXT,
    started_at REAL,
    completed_at REAL,
    chunks_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_crawl_jobs_url ON crawl_jobs(url);
CREATE INDEX IF NOT EXISTS idx_crawl_jobs_status ON crawl_jobs(status);
CREATE INDEX IF NOT EXISTS idx_crawl_jobs_created ON crawl_jobs(created_at);

-- Indexed document chunks
CREATE TABLE IF NOT EXISTS indexed_docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    page_url TEXT NOT NULL,
    source_url TEXT NOT NULL,
    title TEXT,
    content_hash TEXT NOT NULL UNIQUE,
    pinecone_id TEXT NOT NULL,
    content_preview TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_indexed_docs_url ON indexed_docs(url);
CREATE INDEX IF NOT EXISTS idx_indexed_docs_source ON indexed_docs(source_url);
CREATE INDEX IF NOT EXISTS idx_indexed_docs_hash ON indexed_docs(content_hash);
CREATE INDEX IF NOT EXISTS idx_indexed_docs_pinecone ON indexed_docs(pinecone_id);

-- Normalized documents
CREATE TABLE IF NOT EXISTS normalized_docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url_prefix TEXT NOT NULL,
    title TEXT NOT NULL,
    doc_hash TEXT NOT NULL,
    pinecone_id TEXT NOT NULL,
    raw_chunk_count INTEGER DEFAULT 0,
    content_preview TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_normalized_docs_prefix ON normalized_docs(url_prefix);
CREATE INDEX IF NOT EXISTS idx_normalized_docs_hash ON normalized_docs(doc_hash);
"""


class Database:
    """SQLite database manager for ContextPilot metadata."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_config().database.path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript(SCHEMA)
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    # ==================== Crawl Jobs ====================
    
    def create_crawl_job(self, url: str) -> CrawlJob:
        """Create a new crawl job."""
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO crawl_jobs (url, status, created_at)
                VALUES (?, ?, ?)
                """,
                (url, CrawlStatus.PENDING.value, now)
            )
            return CrawlJob(
                id=cursor.lastrowid,
                url=url,
                status=CrawlStatus.PENDING,
                created_at=now
            )
    
    def update_crawl_job(
        self,
        job_id: int,
        status: Optional[CrawlStatus] = None,
        method: Optional[CrawlMethod] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        chunks_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update a crawl job."""
        updates = []
        values = []
        
        if status is not None:
            updates.append("status = ?")
            values.append(status.value)
        if method is not None:
            updates.append("method = ?")
            values.append(method.value)
        if started_at is not None:
            updates.append("started_at = ?")
            values.append(started_at)
        if completed_at is not None:
            updates.append("completed_at = ?")
            values.append(completed_at)
        if chunks_count is not None:
            updates.append("chunks_count = ?")
            values.append(chunks_count)
        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)
        
        if not updates:
            return
        
        values.append(job_id)
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE crawl_jobs SET {', '.join(updates)} WHERE id = ?",
                values
            )
    
    def get_crawl_job(self, job_id: int) -> Optional[CrawlJob]:
        """Get a crawl job by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM crawl_jobs WHERE id = ?",
                (job_id,)
            ).fetchone()
            
            if row:
                return CrawlJob(
                    id=row["id"],
                    url=row["url"],
                    status=CrawlStatus(row["status"]),
                    method=CrawlMethod(row["method"]) if row["method"] else None,
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    chunks_count=row["chunks_count"],
                    error_message=row["error_message"],
                    created_at=row["created_at"],
                )
            return None
    
    def list_crawl_jobs(
        self,
        status: Optional[CrawlStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CrawlJob]:
        """List crawl jobs with optional filtering."""
        query = "SELECT * FROM crawl_jobs"
        params: List[Any] = []
        
        if status:
            query += " WHERE status = ?"
            params.append(status.value)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                CrawlJob(
                    id=row["id"],
                    url=row["url"],
                    status=CrawlStatus(row["status"]),
                    method=CrawlMethod(row["method"]) if row["method"] else None,
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    chunks_count=row["chunks_count"],
                    error_message=row["error_message"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]
    
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
        with self._get_connection() as conn:
            # Try to find existing by content_hash
            existing = conn.execute(
                "SELECT id FROM indexed_docs WHERE content_hash = ?",
                (content_hash,)
            ).fetchone()
            
            if existing:
                # Update existing
                conn.execute(
                    """
                    UPDATE indexed_docs 
                    SET url = ?, page_url = ?, source_url = ?, title = ?, 
                        pinecone_id = ?, content_preview = ?, updated_at = ?
                    WHERE content_hash = ?
                    """,
                    (url, page_url, source_url, title, pinecone_id, 
                     content_preview, now, content_hash)
                )
                doc_id = existing["id"]
            else:
                # Insert new
                cursor = conn.execute(
                    """
                    INSERT INTO indexed_docs 
                    (url, page_url, source_url, title, content_hash, pinecone_id, 
                     content_preview, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (url, page_url, source_url, title, content_hash, pinecone_id,
                     content_preview, now, now)
                )
                doc_id = cursor.lastrowid
            
            return IndexedDoc(
                id=doc_id,
                url=url,
                page_url=page_url,
                source_url=source_url,
                title=title,
                content_hash=content_hash,
                pinecone_id=pinecone_id,
                content_preview=content_preview,
                created_at=now,
                updated_at=now,
            )
    
    def get_indexed_doc_by_hash(self, content_hash: str) -> Optional[IndexedDoc]:
        """Check if a document with this hash already exists."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM indexed_docs WHERE content_hash = ?",
                (content_hash,)
            ).fetchone()
            
            if row:
                return IndexedDoc(
                    id=row["id"],
                    url=row["url"],
                    page_url=row["page_url"],
                    source_url=row["source_url"],
                    title=row["title"],
                    content_hash=row["content_hash"],
                    pinecone_id=row["pinecone_id"],
                    content_preview=row["content_preview"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            return None
    
    def count_indexed_docs(self, source_url: Optional[str] = None) -> int:
        """Count indexed documents."""
        with self._get_connection() as conn:
            if source_url:
                row = conn.execute(
                    "SELECT COUNT(*) as count FROM indexed_docs WHERE source_url = ?",
                    (source_url,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) as count FROM indexed_docs"
                ).fetchone()
            return row["count"] if row else 0
    
    def list_indexed_sources(self) -> List[Dict[str, Any]]:
        """List all indexed sources with stats."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT source_url, title, COUNT(*) as chunks, 
                       MAX(created_at) as last_indexed
                FROM indexed_docs
                GROUP BY source_url
                ORDER BY last_indexed DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]
    
    def delete_indexed_docs_by_source(self, source_url: str) -> List[str]:
        """Delete all indexed docs for a source. Returns Pinecone IDs to delete."""
        with self._get_connection() as conn:
            # Get Pinecone IDs first
            rows = conn.execute(
                "SELECT pinecone_id FROM indexed_docs WHERE source_url = ?",
                (source_url,)
            ).fetchall()
            pinecone_ids = [row["pinecone_id"] for row in rows]
            
            # Delete from SQLite
            conn.execute(
                "DELETE FROM indexed_docs WHERE source_url = ?",
                (source_url,)
            )
            
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
        with self._get_connection() as conn:
            # Check if exists by url_prefix
            existing = conn.execute(
                "SELECT id FROM normalized_docs WHERE url_prefix = ?",
                (url_prefix,)
            ).fetchone()
            
            if existing:
                conn.execute(
                    """
                    UPDATE normalized_docs 
                    SET title = ?, doc_hash = ?, pinecone_id = ?, 
                        raw_chunk_count = ?, content_preview = ?, created_at = ?
                    WHERE url_prefix = ?
                    """,
                    (title, doc_hash, pinecone_id, raw_chunk_count, 
                     content_preview, now, url_prefix)
                )
                doc_id = existing["id"]
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO normalized_docs 
                    (url_prefix, title, doc_hash, pinecone_id, raw_chunk_count, 
                     content_preview, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (url_prefix, title, doc_hash, pinecone_id, raw_chunk_count,
                     content_preview, now)
                )
                doc_id = cursor.lastrowid
            
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
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM normalized_docs ORDER BY created_at DESC"
            ).fetchall()
            return [
                NormalizedDoc(
                    id=row["id"],
                    url_prefix=row["url_prefix"],
                    title=row["title"],
                    doc_hash=row["doc_hash"],
                    pinecone_id=row["pinecone_id"],
                    raw_chunk_count=row["raw_chunk_count"],
                    content_preview=row["content_preview"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]
    
    def get_normalized_doc_by_prefix(self, url_prefix: str) -> Optional[NormalizedDoc]:
        """Get a normalized doc by URL prefix."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM normalized_docs WHERE url_prefix = ?",
                (url_prefix,)
            ).fetchone()
            
            if row:
                return NormalizedDoc(
                    id=row["id"],
                    url_prefix=row["url_prefix"],
                    title=row["title"],
                    doc_hash=row["doc_hash"],
                    pinecone_id=row["pinecone_id"],
                    raw_chunk_count=row["raw_chunk_count"],
                    content_preview=row["content_preview"],
                    created_at=row["created_at"],
                )
            return None
    
    # ==================== Stats ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        with self._get_connection() as conn:
            crawl_stats = conn.execute(
                """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running
                FROM crawl_jobs
                """
            ).fetchone()
            
            indexed_stats = conn.execute(
                "SELECT COUNT(*) as total, COUNT(DISTINCT source_url) as sources FROM indexed_docs"
            ).fetchone()
            
            normalized_stats = conn.execute(
                "SELECT COUNT(*) as total FROM normalized_docs"
            ).fetchone()
            
            return {
                "crawl_jobs": {
                    "total": crawl_stats["total"],
                    "completed": crawl_stats["completed"],
                    "failed": crawl_stats["failed"],
                    "running": crawl_stats["running"],
                },
                "indexed_docs": {
                    "total": indexed_stats["total"],
                    "sources": indexed_stats["sources"],
                },
                "normalized_docs": {
                    "total": normalized_stats["total"],
                },
            }


# Singleton instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get the singleton database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


