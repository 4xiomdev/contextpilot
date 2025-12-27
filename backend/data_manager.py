"""
ContextPilot Data Manager

Handles complete source deletion with cascade and data inspection.
Provides full visibility into data associated with each source.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger("contextpilot.data_manager")


@dataclass
class SourceDataStats:
    """Statistics for data associated with a source."""
    source_id: str
    source_name: str
    base_url: str
    indexed_docs: int = 0
    raw_chunks: int = 0
    vectors: int = 0
    crawl_jobs: int = 0
    crawl_plans: int = 0
    url_universe_urls: int = 0
    chunk_hashes: int = 0
    content_hashes: int = 0


@dataclass
class DeleteResult:
    """Result of a cascade delete operation."""
    source_id: str
    deleted_indexed_docs: int = 0
    deleted_raw_chunks: int = 0
    deleted_vectors: int = 0
    deleted_crawl_jobs: int = 0
    deleted_crawl_plans: int = 0
    deleted_url_universe: int = 0
    deleted_chunk_hashes: int = 0
    deleted_content_hashes: int = 0
    deleted_source: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def total_deleted(self) -> int:
        return (
            self.deleted_indexed_docs +
            self.deleted_raw_chunks +
            self.deleted_crawl_jobs +
            self.deleted_crawl_plans +
            self.deleted_url_universe +
            self.deleted_chunk_hashes +
            self.deleted_content_hashes +
            (1 if self.deleted_source else 0)
        )


class DataManager:
    """Manages data inspection and complete cascade deletions."""

    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id
        self._db = None
        self._embed_manager = None
        self._registry = None

    @property
    def db(self):
        """Lazy load database."""
        if self._db is None:
            from .firestore_db import get_firestore_db
            self._db = get_firestore_db(self.tenant_id)
        return self._db

    @property
    def embed_manager(self):
        """Lazy load embed manager."""
        if self._embed_manager is None:
            from .embed_manager import get_embed_manager
            self._embed_manager = get_embed_manager(self.tenant_id)
        return self._embed_manager

    @property
    def registry(self):
        """Lazy load source registry."""
        if self._registry is None:
            from .source_registry import get_source_registry
            self._registry = get_source_registry(self.tenant_id)
        return self._registry

    def get_source_stats(self, source_id: str) -> Optional[SourceDataStats]:
        """Get comprehensive data stats for a source."""
        source = self.registry.get_source(source_id)
        if not source:
            return None

        base_url = source.base_url

        # Count each type of data
        try:
            indexed_count = self.db.count_indexed_docs_by_source(base_url)
        except Exception:
            indexed_count = 0

        try:
            raw_count = self.db.count_raw_chunks_by_source(base_url)
        except Exception:
            raw_count = 0

        try:
            crawl_count = self.db.count_crawl_jobs_by_url(base_url)
        except Exception:
            crawl_count = 0

        try:
            plan_count = self.db.count_crawl_plans_by_source(source_id)
        except Exception:
            plan_count = 0

        try:
            universe_count = self.db.count_url_universe_by_source(source_id)
        except Exception:
            universe_count = 0

        try:
            chunk_hash_count = self.db.count_chunk_hashes_by_source(source_id)
        except Exception:
            chunk_hash_count = 0

        # Vector count from Pinecone (estimated from indexed_docs)
        vector_count = indexed_count  # Each indexed doc = 1 vector

        return SourceDataStats(
            source_id=source_id,
            source_name=source.name or "",
            base_url=base_url,
            indexed_docs=indexed_count,
            raw_chunks=raw_count,
            vectors=vector_count,
            crawl_jobs=crawl_count,
            crawl_plans=plan_count,
            url_universe_urls=universe_count,
            chunk_hashes=chunk_hash_count,
            content_hashes=1,  # One hash document per source
        )

    def delete_source_cascade(self, source_id: str) -> DeleteResult:
        """
        Delete a source and ALL related data.

        Order matters for data integrity:
        1. Get source info first (need base_url)
        2. Delete Pinecone vectors
        3. Delete indexed_docs
        4. Delete raw_chunks
        5. Delete chunk_hashes
        6. Delete content_hashes
        7. Delete crawl_jobs
        8. Delete crawl_plans
        9. Delete url_universe_urls
        10. Delete url_universe_meta
        11. Delete source record
        """
        result = DeleteResult(source_id=source_id)
        source = self.registry.get_source(source_id)

        if not source:
            result.errors.append("Source not found")
            return result

        base_url = source.base_url
        logger.info(f"Starting cascade delete for source {source_id} ({base_url})")

        # 1. Delete vectors from Pinecone
        try:
            result.deleted_vectors = self.embed_manager.delete_by_source(base_url)
            logger.info(f"Deleted {result.deleted_vectors} vectors from Pinecone")
        except Exception as e:
            result.errors.append(f"Failed to delete vectors: {e}")
            logger.error(f"Failed to delete vectors: {e}")

        # 2. Delete indexed_docs (if not already done by embed_manager)
        try:
            self.db.delete_indexed_docs_by_source(base_url)
            result.deleted_indexed_docs = 1  # Mark as done
            logger.info("Deleted indexed_docs")
        except Exception as e:
            result.errors.append(f"Failed to delete indexed_docs: {e}")
            logger.error(f"Failed to delete indexed_docs: {e}")

        # 3. Delete raw_chunks
        try:
            deleted = self.db.delete_raw_chunks_by_source(base_url)
            result.deleted_raw_chunks = deleted if isinstance(deleted, int) else 1
            logger.info(f"Deleted {result.deleted_raw_chunks} raw_chunks")
        except Exception as e:
            result.errors.append(f"Failed to delete raw_chunks: {e}")
            logger.error(f"Failed to delete raw_chunks: {e}")

        # 4. Delete chunk_hashes
        try:
            result.deleted_chunk_hashes = self.db.delete_all_chunk_hashes_for_source(source_id)
            logger.info(f"Deleted {result.deleted_chunk_hashes} chunk_hashes")
        except Exception as e:
            result.errors.append(f"Failed to delete chunk_hashes: {e}")
            logger.error(f"Failed to delete chunk_hashes: {e}")

        # 5. Clear content_hashes
        try:
            self.db.clear_content_hashes(source_id)
            result.deleted_content_hashes = 1
            logger.info("Cleared content_hashes")
        except Exception as e:
            result.errors.append(f"Failed to clear content_hashes: {e}")
            logger.error(f"Failed to clear content_hashes: {e}")

        # 6. Delete crawl_jobs
        try:
            result.deleted_crawl_jobs = self.db.delete_crawl_jobs_by_url(base_url)
            logger.info(f"Deleted {result.deleted_crawl_jobs} crawl_jobs")
        except Exception as e:
            result.errors.append(f"Failed to delete crawl_jobs: {e}")
            logger.error(f"Failed to delete crawl_jobs: {e}")

        # 7. Delete crawl_plans
        try:
            result.deleted_crawl_plans = self.db.delete_crawl_plans_by_source(source_id)
            logger.info(f"Deleted {result.deleted_crawl_plans} crawl_plans")
        except Exception as e:
            result.errors.append(f"Failed to delete crawl_plans: {e}")
            logger.error(f"Failed to delete crawl_plans: {e}")

        # 8. Delete url_universe_urls
        try:
            result.deleted_url_universe = self.db.delete_url_universe_by_source(source_id)
            logger.info(f"Deleted url_universe")
        except Exception as e:
            result.errors.append(f"Failed to delete url_universe: {e}")
            logger.error(f"Failed to delete url_universe: {e}")

        # 9. Delete url_universe_meta
        try:
            self.db.delete_universe_meta(source_id)
            logger.info("Deleted universe_meta")
        except Exception as e:
            result.errors.append(f"Failed to delete universe_meta: {e}")
            logger.error(f"Failed to delete universe_meta: {e}")

        # 10. Delete source record
        try:
            result.deleted_source = self.db.delete_source(source_id)
            logger.info(f"Deleted source record: {result.deleted_source}")
        except Exception as e:
            result.errors.append(f"Failed to delete source: {e}")
            logger.error(f"Failed to delete source: {e}")

        logger.info(f"Cascade delete complete. Total deleted: {result.total_deleted}, Errors: {len(result.errors)}")
        return result

    def get_chunks_by_source(
        self,
        source_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get raw chunks for a source with pagination."""
        source = self.registry.get_source(source_id)
        if not source:
            return {"chunks": [], "total": 0, "source_id": source_id}

        try:
            chunks = self.db.list_raw_chunks_by_source(source.base_url, limit=limit, offset=offset)
            total = self.db.count_raw_chunks_by_source(source.base_url)
        except Exception as e:
            logger.error(f"Failed to get chunks: {e}")
            return {"chunks": [], "total": 0, "source_id": source_id, "error": str(e)}

        return {
            "chunks": [
                {
                    "id": c.get("id") or c.get("page_url", ""),
                    "page_url": c.get("page_url", ""),
                    "title": c.get("title", ""),
                    "content_preview": (c.get("content", "") or "")[:300],
                    "content_hash": c.get("content_hash", ""),
                    "created_at": c.get("created_at", 0),
                }
                for c in chunks
            ],
            "total": total,
            "source_id": source_id,
        }

    def get_crawl_history_by_source(
        self,
        source_id: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get crawl job history for a source."""
        source = self.registry.get_source(source_id)
        if not source:
            return {"jobs": [], "total": 0, "source_id": source_id}

        try:
            jobs = self.db.list_crawl_jobs_by_url(source.base_url, limit=limit)
            total = self.db.count_crawl_jobs_by_url(source.base_url)
        except Exception as e:
            logger.error(f"Failed to get crawl history: {e}")
            return {"jobs": [], "total": 0, "source_id": source_id, "error": str(e)}

        return {
            "jobs": [
                {
                    "id": j.id,
                    "url": j.url,
                    "status": j.status.value if j.status else "unknown",
                    "method": j.method.value if j.method else None,
                    "chunks_count": j.chunks_count or 0,
                    "error_message": j.error_message,
                    "started_at": j.started_at,
                    "completed_at": j.completed_at,
                    "created_at": j.created_at,
                }
                for j in jobs
            ],
            "total": total,
            "source_id": source_id,
        }

    def delete_crawl_history_by_source(self, source_id: str) -> int:
        """Delete all crawl history for a source."""
        source = self.registry.get_source(source_id)
        if not source:
            return 0
        try:
            return self.db.delete_crawl_jobs_by_url(source.base_url)
        except Exception as e:
            logger.error(f"Failed to delete crawl history: {e}")
            return 0

    def get_all_sources_stats(self) -> List[SourceDataStats]:
        """Get stats for all sources."""
        try:
            sources = self.registry.list_sources()
        except Exception as e:
            logger.error(f"Failed to list sources: {e}")
            return []

        stats = []
        for source in sources:
            try:
                source_stats = self.get_source_stats(source.id)
                if source_stats:
                    stats.append(source_stats)
            except Exception as e:
                logger.warning(f"Failed to get stats for source {source.id}: {e}")
                continue

        return stats

    def delete_all_data(self) -> Dict[str, Any]:
        """Nuclear option - delete ALL data for tenant including orphaned data."""
        results = []
        errors = []

        # 1. Delete all sources via cascade
        try:
            sources = self.registry.list_sources()
            for source in sources:
                result = self.delete_source_cascade(source.id)
                results.append({
                    "source_id": source.id,
                    "source_name": source.name,
                    "deleted": result.total_deleted,
                    "errors": result.errors,
                })
        except Exception as e:
            logger.error(f"Failed to delete sources: {e}")
            errors.append(f"Failed to delete sources: {e}")

        # 2. Delete ALL vectors from Pinecone (catches orphans)
        deleted_vectors = 0
        try:
            deleted_vectors = self.embed_manager.delete_all()
            logger.info(f"Deleted all vectors from Pinecone: {deleted_vectors}")
        except Exception as e:
            logger.error(f"Failed to delete all vectors: {e}")
            errors.append(f"Failed to delete all vectors: {e}")

        # 3. Delete ALL indexed_docs (catches orphans)
        deleted_indexed = 0
        try:
            deleted_indexed = self.db.delete_all_indexed_docs()
            logger.info(f"Deleted all indexed_docs: {deleted_indexed}")
        except Exception as e:
            logger.error(f"Failed to delete all indexed_docs: {e}")
            errors.append(f"Failed to delete all indexed_docs: {e}")

        # 4. Delete ALL raw_chunks (catches orphans)
        deleted_chunks = 0
        try:
            deleted_chunks = self.db.delete_all_raw_chunks()
            logger.info(f"Deleted all raw_chunks: {deleted_chunks}")
        except Exception as e:
            logger.error(f"Failed to delete all raw_chunks: {e}")
            errors.append(f"Failed to delete all raw_chunks: {e}")

        # 5. Delete ALL crawl_jobs (catches orphans)
        deleted_jobs = 0
        try:
            deleted_jobs = self.db.delete_all_crawl_jobs()
            logger.info(f"Deleted all crawl_jobs: {deleted_jobs}")
        except Exception as e:
            logger.error(f"Failed to delete all crawl_jobs: {e}")
            errors.append(f"Failed to delete all crawl_jobs: {e}")

        return {
            "sources_deleted": len(results),
            "results": results,
            "orphaned_cleanup": {
                "vectors": deleted_vectors,
                "indexed_docs": deleted_indexed,
                "raw_chunks": deleted_chunks,
                "crawl_jobs": deleted_jobs,
            },
            "errors": errors,
        }


# Singleton per tenant
_managers: Dict[str, DataManager] = {}


def get_data_manager(tenant_id: Optional[str] = None) -> DataManager:
    """Get the data manager instance for a tenant."""
    key = tenant_id or "default"
    if key not in _managers:
        _managers[key] = DataManager(tenant_id)
    return _managers[key]
