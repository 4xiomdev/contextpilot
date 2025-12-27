"""
ContextPilot Database Layer
SQLite-based metadata storage for crawl jobs, indexed docs, and normalized docs.
"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator
import json

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

-- Raw chunks (for normalization)
CREATE TABLE IF NOT EXISTS raw_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url TEXT NOT NULL,
    page_url TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_chunks_source ON raw_chunks(source_url);
CREATE INDEX IF NOT EXISTS idx_raw_chunks_page ON raw_chunks(page_url);
CREATE INDEX IF NOT EXISTS idx_raw_chunks_hash ON raw_chunks(content_hash);

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

-- Sources registry (local fallback)
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    base_url TEXT NOT NULL UNIQUE,
    sitemap_url TEXT,
    priority_paths TEXT,
    exclude_paths TEXT,
    crawl_frequency TEXT,
    max_pages INTEGER,
    is_enabled INTEGER,
    is_curated INTEGER,
    created_by TEXT,
    health_status TEXT,
    last_crawled_at REAL,
    last_content_hash TEXT,
    next_crawl_at REAL,
    chunks_count INTEGER,
    error_message TEXT,
    tags TEXT,
    description TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_base_url ON sources(base_url);
CREATE INDEX IF NOT EXISTS idx_sources_enabled ON sources(is_enabled);
CREATE INDEX IF NOT EXISTS idx_sources_health ON sources(health_status);

-- Content hashes (freshness detection)
CREATE TABLE IF NOT EXISTS content_hashes (
    source_id TEXT PRIMARY KEY,
    hashes_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- URL universe (discovered URLs for planning)
CREATE TABLE IF NOT EXISTS url_universe_urls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    url_id TEXT NOT NULL,
    url TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    path TEXT NOT NULL,
    depth INTEGER NOT NULL,
    query_keys TEXT,
    created_at REAL NOT NULL,
    UNIQUE(source_id, url_id)
);

CREATE INDEX IF NOT EXISTS idx_universe_source ON url_universe_urls(source_id);
CREATE INDEX IF NOT EXISTS idx_universe_path ON url_universe_urls(path);
CREATE INDEX IF NOT EXISTS idx_universe_url ON url_universe_urls(url);

-- Crawl plans (LLM-curated crawl strategy)
CREATE TABLE IF NOT EXISTS crawl_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    status TEXT NOT NULL,
    rules_json TEXT NOT NULL,
    report_json TEXT NOT NULL,
    total_urls_seen INTEGER NOT NULL DEFAULT 0,
    kept_urls INTEGER NOT NULL DEFAULT 0,
    dropped_urls INTEGER NOT NULL DEFAULT 0,
    maybe_urls INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_crawl_plans_source ON crawl_plans(source_id);
CREATE INDEX IF NOT EXISTS idx_crawl_plans_created ON crawl_plans(created_at);

-- Chunk-level content hashes (for incremental re-indexing)
CREATE TABLE IF NOT EXISTS chunk_hashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    page_url TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    pinecone_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(source_id, page_url, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunk_hashes_source ON chunk_hashes(source_id);
CREATE INDEX IF NOT EXISTS idx_chunk_hashes_page ON chunk_hashes(page_url);
CREATE INDEX IF NOT EXISTS idx_chunk_hashes_hash ON chunk_hashes(content_hash);

-- URL universe metadata (TTL caching)
CREATE TABLE IF NOT EXISTS url_universe_meta (
    source_id TEXT PRIMARY KEY,
    url_count INTEGER NOT NULL DEFAULT 0,
    ttl_hours INTEGER NOT NULL DEFAULT 24,
    cached_at REAL NOT NULL,
    discovery_method TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Normalization lineage (tracking source chunks -> normalized docs)
CREATE TABLE IF NOT EXISTS normalization_lineage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    normalized_doc_id TEXT NOT NULL,
    url_prefix TEXT NOT NULL,
    source_chunk_count INTEGER NOT NULL DEFAULT 0,
    used_chunk_count INTEGER NOT NULL DEFAULT 0,
    sections_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lineage_doc ON normalization_lineage(normalized_doc_id);
CREATE INDEX IF NOT EXISTS idx_lineage_prefix ON normalization_lineage(url_prefix);
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

    # ==================== Raw Chunks ====================

    def upsert_raw_chunk(
        self,
        source_url: str,
        page_url: str,
        title: str,
        content: str,
        content_hash: str,
    ) -> RawChunk:
        """Insert or update a raw chunk for normalization."""
        now = time.time()
        with self._get_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM raw_chunks WHERE content_hash = ?",
                (content_hash,)
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE raw_chunks
                    SET source_url = ?, page_url = ?, title = ?, content = ?
                    WHERE content_hash = ?
                    """,
                    (source_url, page_url, title, content, content_hash)
                )
                doc_id = existing["id"]
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO raw_chunks
                    (source_url, page_url, title, content, content_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (source_url, page_url, title, content, content_hash, now)
                )
                doc_id = cursor.lastrowid

            return RawChunk(
                id=doc_id,
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
        """List raw chunks matching a URL prefix."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM raw_chunks
                WHERE page_url LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f"{url_prefix}%", limit)
            ).fetchall()

            return [
                RawChunk(
                    id=row["id"],
                    source_url=row["source_url"],
                    page_url=row["page_url"],
                    title=row["title"],
                    content=row["content"],
                    content_hash=row["content_hash"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def delete_raw_chunks_by_source(self, source_url: str) -> int:
        """Delete raw chunks for a source."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM raw_chunks WHERE source_url = ?",
                (source_url,)
            )
            return cursor.rowcount or 0
    
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

    # ==================== Sources ====================

    def _row_to_source(self, row: sqlite3.Row) -> Source:
        """Convert a DB row to a Source dataclass."""
        return Source(
            id=str(row["id"]),
            name=row["name"] or "",
            base_url=row["base_url"] or "",
            sitemap_url=row["sitemap_url"],
            priority_paths=json.loads(row["priority_paths"] or "[]"),
            exclude_paths=json.loads(row["exclude_paths"] or "[]"),
            crawl_frequency=CrawlFrequency(row["crawl_frequency"] or "weekly"),
            max_pages=row["max_pages"] or 500,
            is_enabled=bool(row["is_enabled"]),
            is_curated=bool(row["is_curated"]),
            created_by=SourceCreatedBy(row["created_by"] or "user"),
            health_status=SourceHealthStatus(row["health_status"] or "unknown"),
            last_crawled_at=row["last_crawled_at"],
            last_content_hash=row["last_content_hash"],
            next_crawl_at=row["next_crawl_at"],
            chunks_count=row["chunks_count"] or 0,
            error_message=row["error_message"],
            tags=json.loads(row["tags"] or "[]"),
            description=row["description"],
            created_at=row["created_at"] or 0.0,
            updated_at=row["updated_at"] or 0.0,
        )

    def create_source(self, source: Source) -> Source:
        """Create a new source."""
        now = time.time()
        source.created_at = now
        source.updated_at = now
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sources
                (name, base_url, sitemap_url, priority_paths, exclude_paths,
                 crawl_frequency, max_pages, is_enabled, is_curated, created_by,
                 health_status, last_crawled_at, last_content_hash, next_crawl_at,
                 chunks_count, error_message, tags, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source.name,
                    source.base_url,
                    source.sitemap_url,
                    json.dumps(source.priority_paths or []),
                    json.dumps(source.exclude_paths or []),
                    source.crawl_frequency.value,
                    source.max_pages,
                    int(source.is_enabled),
                    int(source.is_curated),
                    source.created_by.value,
                    source.health_status.value,
                    source.last_crawled_at,
                    source.last_content_hash,
                    source.next_crawl_at,
                    source.chunks_count,
                    source.error_message,
                    json.dumps(source.tags or []),
                    source.description,
                    source.created_at,
                    source.updated_at,
                )
            )
            source.id = str(cursor.lastrowid)
        return source

    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE id = ?",
                (source_id,)
            ).fetchone()
            return self._row_to_source(row) if row else None

    def get_source_by_url(self, base_url: str) -> Optional[Source]:
        """Get a source by base URL."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE base_url = ?",
                (base_url,)
            ).fetchone()
            return self._row_to_source(row) if row else None

    def update_source(self, source_id: str, updates: Dict[str, Any]) -> Optional[Source]:
        """Update a source with partial data."""
        if not updates:
            return self.get_source(source_id)

        updates = dict(updates)
        updates["updated_at"] = time.time()

        if "crawl_frequency" in updates and isinstance(updates["crawl_frequency"], CrawlFrequency):
            updates["crawl_frequency"] = updates["crawl_frequency"].value
        if "health_status" in updates and isinstance(updates["health_status"], SourceHealthStatus):
            updates["health_status"] = updates["health_status"].value
        if "created_by" in updates and isinstance(updates["created_by"], SourceCreatedBy):
            updates["created_by"] = updates["created_by"].value

        json_fields = {"priority_paths", "exclude_paths", "tags"}
        for key in json_fields:
            if key in updates:
                updates[key] = json.dumps(updates[key] or [])

        update_sql = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [source_id]

        with self._get_connection() as conn:
            conn.execute(f"UPDATE sources SET {update_sql} WHERE id = ?", values)
        return self.get_source(source_id)

    def delete_source(self, source_id: str) -> bool:
        """Delete a source."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM sources WHERE id = ?",
                (source_id,)
            )
        return True

    def list_sources(
        self,
        is_enabled: Optional[bool] = None,
        created_by: Optional[SourceCreatedBy] = None,
        health_status: Optional[SourceHealthStatus] = None,
        limit: int = 100,
    ) -> List[Source]:
        """List sources with optional filtering."""
        query = "SELECT * FROM sources"
        clauses = []
        params: List[Any] = []

        if is_enabled is not None:
            clauses.append("is_enabled = ?")
            params.append(int(is_enabled))
        if created_by is not None:
            clauses.append("created_by = ?")
            params.append(created_by.value)
        if health_status is not None:
            clauses.append("health_status = ?")
            params.append(health_status.value)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY name LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_source(row) for row in rows]

    def list_sources_due_for_crawl(self) -> List[Source]:
        """List sources that are due for scheduled crawling."""
        sources = self.list_sources(is_enabled=True, limit=1000)
        return [s for s in sources if s.is_due_for_crawl()]

    def update_source_after_crawl(
        self,
        source_id: str,
        success: bool,
        chunks_count: int = 0,
        content_hash: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Source]:
        """Update source after a crawl completes."""
        source = self.get_source(source_id)
        if not source:
            return None

        updates: Dict[str, Any] = {
            "last_crawled_at": time.time(),
        }

        if success:
            updates["health_status"] = SourceHealthStatus.HEALTHY
            updates["chunks_count"] = chunks_count
            updates["error_message"] = None
            updates["next_crawl_at"] = source.compute_next_crawl_time()
            if content_hash:
                updates["last_content_hash"] = content_hash
        else:
            updates["health_status"] = SourceHealthStatus.ERROR
            updates["error_message"] = error_message
            updates["next_crawl_at"] = source.compute_next_crawl_time()

        return self.update_source(source_id, updates)

    def mark_source_stale(self, source_id: str) -> Optional[Source]:
        """Mark a source as stale (needs re-crawling)."""
        return self.update_source(source_id, {
            "health_status": SourceHealthStatus.STALE,
            "next_crawl_at": time.time(),
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
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM sources").fetchall()
            for row in rows:
                counts["total"] += 1
                status = row["health_status"] or "unknown"
                if status in counts:
                    counts[status] += 1
                if row["is_enabled"]:
                    counts["enabled"] += 1
                if row["is_curated"]:
                    counts["curated"] += 1
        return counts

    # ==================== Content Hashes ====================

    def get_content_hashes(self, source_id: str) -> Dict[str, str]:
        """Get stored content hashes for a source."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT hashes_json FROM content_hashes WHERE source_id = ?",
                (str(source_id),)
            ).fetchone()
            if row and row["hashes_json"]:
                return json.loads(row["hashes_json"])
            return {}

    def store_content_hashes(self, source_id: str, hashes: Dict[str, str]) -> None:
        """Store content hashes for a source."""
        now = time.time()
        with self._get_connection() as conn:
            existing = conn.execute(
                "SELECT source_id FROM content_hashes WHERE source_id = ?",
                (str(source_id),)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE content_hashes SET hashes_json = ?, updated_at = ? WHERE source_id = ?",
                    (json.dumps(hashes), now, str(source_id))
                )
            else:
                conn.execute(
                    "INSERT INTO content_hashes (source_id, hashes_json, updated_at) VALUES (?, ?, ?)",
                    (str(source_id), json.dumps(hashes), now)
                )

    def clear_content_hashes(self, source_id: str) -> None:
        """Clear stored content hashes for a source."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM content_hashes WHERE source_id = ?",
                (str(source_id),)
            )

    # ==================== URL Universe + Crawl Plans ====================

    def replace_url_universe(self, source_id: str, urls: List[Dict[str, Any]]) -> int:
        """
        Replace the stored URL universe for a source.

        Args:
            source_id: Registry source ID
            urls: List of dicts with keys: url_id, url, canonical_url, path, depth, query_keys (list[str])

        Returns:
            Number of rows inserted.
        """
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM url_universe_urls WHERE source_id = ?",
                (str(source_id),)
            )

            rows = [
                (
                    str(source_id),
                    u["url_id"],
                    u["url"],
                    u["canonical_url"],
                    u["path"],
                    int(u["depth"]),
                    json.dumps(u.get("query_keys") or []),
                    now,
                )
                for u in urls
            ]

            if rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO url_universe_urls
                    (source_id, url_id, url, canonical_url, path, depth, query_keys, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

        return len(rows)

    def list_url_universe(self, source_id: str, limit: int = 200_000) -> List[Dict[str, Any]]:
        """List stored URL universe rows for a source."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT url_id, url, canonical_url, path, depth, query_keys
                FROM url_universe_urls
                WHERE source_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (str(source_id), limit),
            ).fetchall()
            result = []
            for row in rows:
                result.append({
                    "url_id": row["url_id"],
                    "url": row["url"],
                    "canonical_url": row["canonical_url"],
                    "path": row["path"],
                    "depth": row["depth"],
                    "query_keys": json.loads(row["query_keys"] or "[]"),
                })
            return result

    def get_universe_meta(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get URL universe metadata for a source."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT source_id, url_count, ttl_hours, cached_at, discovery_method, created_at, updated_at
                FROM url_universe_meta
                WHERE source_id = ?
                """,
                (str(source_id),),
            ).fetchone()
            if not row:
                return None
            return {
                "source_id": row["source_id"],
                "url_count": row["url_count"],
                "ttl_hours": row["ttl_hours"],
                "cached_at": row["cached_at"],
                "discovery_method": row["discovery_method"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
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
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO url_universe_meta
                (source_id, url_count, ttl_hours, cached_at, discovery_method, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    url_count = excluded.url_count,
                    ttl_hours = excluded.ttl_hours,
                    cached_at = excluded.cached_at,
                    discovery_method = excluded.discovery_method,
                    updated_at = excluded.updated_at
                """,
                (str(source_id), url_count, ttl_hours, now, discovery_method, now, now),
            )

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
        """Delete URL universe metadata for a source."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM url_universe_meta WHERE source_id = ?",
                (str(source_id),),
            )
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
    ) -> int:
        """Insert a crawl plan report."""
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO crawl_plans
                (source_id, status, rules_json, report_json, total_urls_seen, kept_urls, dropped_urls, maybe_urls, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(source_id),
                    status,
                    json.dumps(rules, ensure_ascii=False),
                    json.dumps(report, ensure_ascii=False),
                    int(total_urls_seen),
                    int(kept_urls),
                    int(dropped_urls),
                    int(maybe_urls),
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def get_latest_crawl_plan(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest crawl plan for a source."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM crawl_plans
                WHERE source_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (str(source_id),),
            ).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "source_id": row["source_id"],
                "status": row["status"],
                "rules": json.loads(row["rules_json"] or "{}"),
                "report": json.loads(row["report_json"] or "{}"),
                "total_urls_seen": row["total_urls_seen"],
                "kept_urls": row["kept_urls"],
                "dropped_urls": row["dropped_urls"],
                "maybe_urls": row["maybe_urls"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def update_crawl_plan_status(self, plan_id: int, status: str) -> bool:
        """Update crawl plan status."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE crawl_plans SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), int(plan_id)),
            )
        return True

    # ==================== Chunk Hashes (Incremental Re-indexing) ====================

    def get_chunk_hashes(self, source_id: str, page_url: str) -> Dict[int, Dict[str, str]]:
        """
        Get stored chunk hashes for a page.

        Returns:
            Dict mapping chunk_index -> {"hash": content_hash, "pinecone_id": pinecone_id}
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT chunk_index, content_hash, pinecone_id
                FROM chunk_hashes
                WHERE source_id = ? AND page_url = ?
                """,
                (str(source_id), page_url)
            ).fetchall()
            return {
                row["chunk_index"]: {
                    "hash": row["content_hash"],
                    "pinecone_id": row["pinecone_id"],
                }
                for row in rows
            }

    def get_all_chunk_hashes_for_source(self, source_id: str) -> Dict[str, Dict[int, Dict[str, str]]]:
        """
        Get all chunk hashes for a source.

        Returns:
            Dict mapping page_url -> {chunk_index -> {"hash": ..., "pinecone_id": ...}}
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT page_url, chunk_index, content_hash, pinecone_id
                FROM chunk_hashes
                WHERE source_id = ?
                """,
                (str(source_id),)
            ).fetchall()

            result: Dict[str, Dict[int, Dict[str, str]]] = {}
            for row in rows:
                page_url = row["page_url"]
                if page_url not in result:
                    result[page_url] = {}
                result[page_url][row["chunk_index"]] = {
                    "hash": row["content_hash"],
                    "pinecone_id": row["pinecone_id"],
                }
            return result

    def upsert_chunk_hash(
        self,
        source_id: str,
        page_url: str,
        chunk_index: int,
        content_hash: str,
        pinecone_id: Optional[str] = None,
    ) -> None:
        """Insert or update a chunk hash."""
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chunk_hashes
                (source_id, page_url, chunk_index, content_hash, pinecone_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, page_url, chunk_index) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    pinecone_id = excluded.pinecone_id,
                    updated_at = excluded.updated_at
                """,
                (str(source_id), page_url, chunk_index, content_hash, pinecone_id, now, now)
            )

    def upsert_chunk_hashes_batch(
        self,
        source_id: str,
        page_url: str,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """
        Batch upsert chunk hashes for a page.

        Args:
            source_id: Source ID
            page_url: Page URL
            chunks: List of dicts with keys: index, hash, pinecone_id

        Returns:
            Number of rows upserted.
        """
        now = time.time()
        rows = [
            (
                str(source_id),
                page_url,
                int(c["index"]),
                c["hash"],
                c.get("pinecone_id"),
                now,
                now,
            )
            for c in chunks
        ]

        if not rows:
            return 0

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO chunk_hashes
                (source_id, page_url, chunk_index, content_hash, pinecone_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, page_url, chunk_index) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    pinecone_id = excluded.pinecone_id,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def delete_chunk_hashes(
        self,
        source_id: str,
        page_url: str,
        chunk_indices: Optional[List[int]] = None,
    ) -> int:
        """
        Delete chunk hashes for a page.

        Args:
            source_id: Source ID
            page_url: Page URL
            chunk_indices: Specific indices to delete, or None to delete all for the page

        Returns:
            Number of rows deleted.
        """
        with self._get_connection() as conn:
            if chunk_indices is None:
                cursor = conn.execute(
                    "DELETE FROM chunk_hashes WHERE source_id = ? AND page_url = ?",
                    (str(source_id), page_url)
                )
            else:
                placeholders = ",".join("?" * len(chunk_indices))
                cursor = conn.execute(
                    f"DELETE FROM chunk_hashes WHERE source_id = ? AND page_url = ? AND chunk_index IN ({placeholders})",
                    [str(source_id), page_url] + list(chunk_indices)
                )
            return cursor.rowcount or 0

    def delete_all_chunk_hashes_for_source(self, source_id: str) -> int:
        """Delete all chunk hashes for a source."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM chunk_hashes WHERE source_id = ?",
                (str(source_id),)
            )
            return cursor.rowcount or 0

    def get_stale_chunks(
        self,
        source_id: str,
        page_url: str,
        current_hashes: Dict[int, str],
    ) -> Dict[str, List[int]]:
        """
        Compare current chunk hashes with stored ones to find changes.

        Args:
            source_id: Source ID
            page_url: Page URL
            current_hashes: Dict mapping chunk_index -> content_hash

        Returns:
            Dict with keys:
                - "added": indices of new chunks
                - "modified": indices of changed chunks
                - "deleted": indices of removed chunks
                - "unchanged": indices of unchanged chunks
        """
        stored = self.get_chunk_hashes(source_id, page_url)

        stored_indices = set(stored.keys())
        current_indices = set(current_hashes.keys())

        added = list(current_indices - stored_indices)
        deleted = list(stored_indices - current_indices)
        unchanged = []
        modified = []

        for idx in stored_indices & current_indices:
            if stored[idx]["hash"] == current_hashes[idx]:
                unchanged.append(idx)
            else:
                modified.append(idx)

        return {
            "added": sorted(added),
            "modified": sorted(modified),
            "deleted": sorted(deleted),
            "unchanged": sorted(unchanged),
        }

    # ==================== Normalization Lineage ====================

    def store_normalization_lineage(self, lineage: Dict[str, Any]) -> int:
        """
        Store normalization lineage data.

        Args:
            lineage: Dict with keys: normalized_doc_id, url_prefix, source_chunk_count,
                     used_chunk_count, sections, created_at

        Returns:
            Row ID of inserted lineage record.
        """
        now = lineage.get("created_at", time.time())
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO normalization_lineage
                (normalized_doc_id, url_prefix, source_chunk_count, used_chunk_count, sections_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    lineage.get("normalized_doc_id", ""),
                    lineage.get("url_prefix", ""),
                    lineage.get("source_chunk_count", 0),
                    lineage.get("used_chunk_count", 0),
                    json.dumps(lineage.get("sections", [])),
                    now,
                ),
            )
            return cursor.lastrowid or 0

    def get_normalization_lineage(self, normalized_doc_id: str) -> Optional[Dict[str, Any]]:
        """Get lineage for a normalized document."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM normalization_lineage
                WHERE normalized_doc_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (normalized_doc_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "normalized_doc_id": row["normalized_doc_id"],
                "url_prefix": row["url_prefix"],
                "source_chunk_count": row["source_chunk_count"],
                "used_chunk_count": row["used_chunk_count"],
                "sections": json.loads(row["sections_json"] or "[]"),
                "created_at": row["created_at"],
            }

    def get_lineage_by_prefix(self, url_prefix: str) -> Optional[Dict[str, Any]]:
        """Get lineage for a URL prefix."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM normalization_lineage
                WHERE url_prefix = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (url_prefix,),
            ).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "normalized_doc_id": row["normalized_doc_id"],
                "url_prefix": row["url_prefix"],
                "source_chunk_count": row["source_chunk_count"],
                "used_chunk_count": row["used_chunk_count"],
                "sections": json.loads(row["sections_json"] or "[]"),
                "created_at": row["created_at"],
            }

    def list_lineage(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all normalization lineage records."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM normalization_lineage
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "normalized_doc_id": row["normalized_doc_id"],
                    "url_prefix": row["url_prefix"],
                    "source_chunk_count": row["source_chunk_count"],
                    "used_chunk_count": row["used_chunk_count"],
                    "sections": json.loads(row["sections_json"] or "[]"),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

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

            source_stats = self.count_sources()
            
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
                "sources": source_stats,
            }

    # ==================== Data Management (Cascade Delete + Stats) ====================

    def delete_crawl_jobs_by_url(self, base_url: str) -> int:
        """Delete all crawl jobs matching base URL prefix."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM crawl_jobs WHERE url LIKE ?",
                (f"{base_url}%",)
            )
            return cursor.rowcount or 0

    def delete_crawl_plans_by_source(self, source_id: str) -> int:
        """Delete all crawl plans for a source."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM crawl_plans WHERE source_id = ?",
                (str(source_id),)
            )
            return cursor.rowcount or 0

    def delete_url_universe_by_source(self, source_id: str) -> int:
        """Delete URL universe entries for a source."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM url_universe_urls WHERE source_id = ?",
                (str(source_id),)
            )
            return cursor.rowcount or 0

    def count_indexed_docs_by_source(self, source_url: str) -> int:
        """Count indexed docs for a source URL."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM indexed_docs WHERE source_url = ?",
                (source_url,)
            ).fetchone()
            return row["cnt"] if row else 0

    def count_raw_chunks_by_source(self, source_url: str) -> int:
        """Count raw chunks for a source URL."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM raw_chunks WHERE source_url = ?",
                (source_url,)
            ).fetchone()
            return row["cnt"] if row else 0

    def count_crawl_jobs_by_url(self, base_url: str) -> int:
        """Count crawl jobs matching base URL prefix."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM crawl_jobs WHERE url LIKE ?",
                (f"{base_url}%",)
            ).fetchone()
            return row["cnt"] if row else 0

    def count_crawl_plans_by_source(self, source_id: str) -> int:
        """Count crawl plans for a source."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM crawl_plans WHERE source_id = ?",
                (str(source_id),)
            ).fetchone()
            return row["cnt"] if row else 0

    def count_url_universe_by_source(self, source_id: str) -> int:
        """Count URL universe entries for a source."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM url_universe_urls WHERE source_id = ?",
                (str(source_id),)
            ).fetchone()
            return row["cnt"] if row else 0

    def count_chunk_hashes_by_source(self, source_id: str) -> int:
        """Count chunk hashes for a source."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM chunk_hashes WHERE source_id = ?",
                (str(source_id),)
            ).fetchone()
            return row["cnt"] if row else 0

    def list_crawl_jobs_by_url(self, base_url: str, limit: int = 100) -> List[CrawlJob]:
        """List crawl jobs matching base URL prefix."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM crawl_jobs
                WHERE url LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f"{base_url}%", limit)
            ).fetchall()
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

    def list_raw_chunks_by_source(self, source_url: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List raw chunks for a source URL with pagination."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, source_url, page_url, title, content, content_hash, created_at
                FROM raw_chunks
                WHERE source_url = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (source_url, limit, offset)
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "source_url": row["source_url"],
                    "page_url": row["page_url"],
                    "title": row["title"],
                    "content": row["content"],
                    "content_hash": row["content_hash"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    # ========== Nuclear Delete Methods ==========

    def delete_all_indexed_docs(self) -> int:
        """Delete ALL indexed docs (nuclear option)."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM indexed_docs")
            return cursor.rowcount or 0

    def delete_all_raw_chunks(self) -> int:
        """Delete ALL raw chunks (nuclear option)."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM raw_chunks")
            return cursor.rowcount or 0

    def delete_all_crawl_jobs(self) -> int:
        """Delete ALL crawl jobs (nuclear option)."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM crawl_jobs")
            return cursor.rowcount or 0


# Singleton instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get the singleton database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db
