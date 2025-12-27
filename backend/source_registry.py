"""
ContextPilot Source Registry Manager

Manages documentation sources for crawling, including:
- Loading curated sources from YAML
- Syncing curated sources to tenant
- Adding/managing user sources
- Health tracking and scheduling
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

from .firestore_db import get_firestore_db
from .models import (
    Source,
    CrawlFrequency,
    SourceHealthStatus,
    SourceCreatedBy,
)
from .tenant_context import get_tenant_id

logger = logging.getLogger("contextpilot.source_registry")


class SourceRegistry:
    """
    Manages the source registry for ContextPilot.

    Responsibilities:
    - Load and parse curated sources from YAML
    - Sync curated sources to a tenant's registry
    - Add user-submitted sources
    - Track source health and scheduling
    """

    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id or get_tenant_id()
        self.db = get_firestore_db(self.tenant_id)
        self.curated_path = Path(__file__).parent / "data" / "curated_sources.yaml"

    def load_curated_sources(self) -> List[Dict[str, Any]]:
        """
        Load curated sources from the YAML file.

        Returns:
            List of source dictionaries from the curated sources file.
        """
        if not HAS_YAML:
            logger.warning("PyYAML not installed, cannot load curated sources")
            return []

        if not self.curated_path.exists():
            logger.warning(f"Curated sources file not found: {self.curated_path}")
            return []

        try:
            with open(self.curated_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            sources = data.get("sources", [])
            logger.info(f"Loaded {len(sources)} curated sources from {self.curated_path}")
            return sources
        except Exception as e:
            logger.error(f"Failed to load curated sources: {e}")
            return []

    def _curated_dict_to_source(self, data: Dict[str, Any]) -> Source:
        """Convert a curated source dict to a Source dataclass."""
        freq_str = data.get("crawl_frequency", "weekly").lower()
        try:
            frequency = CrawlFrequency(freq_str)
        except ValueError:
            frequency = CrawlFrequency.WEEKLY

        return Source(
            name=data.get("name", ""),
            base_url=data.get("base_url", ""),
            sitemap_url=data.get("sitemap_url"),
            priority_paths=data.get("priority_paths", []),
            exclude_paths=data.get("exclude_paths", []),
            crawl_frequency=frequency,
            max_pages=data.get("max_pages", 500),
            is_enabled=True,
            is_curated=True,
            created_by=SourceCreatedBy.CURATED,
            health_status=SourceHealthStatus.UNKNOWN,
            tags=data.get("tags", []),
            description=data.get("description"),
        )

    def sync_curated_to_tenant(self, overwrite: bool = False) -> Dict[str, int]:
        """
        Sync curated sources to the tenant's registry.

        Args:
            overwrite: If True, update existing sources. If False, skip existing.

        Returns:
            Dict with counts: {"added": X, "updated": Y, "skipped": Z}
        """
        curated = self.load_curated_sources()
        results = {"added": 0, "updated": 0, "skipped": 0}

        for data in curated:
            base_url = data.get("base_url", "")
            if not base_url:
                continue

            existing = self.db.get_source_by_url(base_url)

            if existing:
                if overwrite:
                    # Update existing source with curated data
                    updates = {
                        "name": data.get("name", existing.name),
                        "sitemap_url": data.get("sitemap_url"),
                        "priority_paths": data.get("priority_paths", []),
                        "exclude_paths": data.get("exclude_paths", []),
                        "max_pages": data.get("max_pages", 500),
                        "tags": data.get("tags", []),
                        "description": data.get("description"),
                        "is_curated": True,
                    }
                    self.db.update_source(existing.id, updates)
                    results["updated"] += 1
                else:
                    results["skipped"] += 1
            else:
                # Add new source
                source = self._curated_dict_to_source(data)
                self.db.create_source(source)
                results["added"] += 1

        logger.info(
            f"Synced curated sources: {results['added']} added, "
            f"{results['updated']} updated, {results['skipped']} skipped"
        )
        return results

    def add_user_source(
        self,
        name: str,
        base_url: str,
        sitemap_url: Optional[str] = None,
        priority_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        crawl_frequency: str = "weekly",
        max_pages: int = 500,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> Source:
        """
        Add a user-submitted source.

        Args:
            name: Display name for the source
            base_url: Base URL of the documentation
            sitemap_url: Optional sitemap URL for discovery
            priority_paths: URL paths to prioritize
            exclude_paths: URL paths to exclude
            crawl_frequency: How often to crawl (daily/weekly/monthly/manual)
            max_pages: Maximum pages to crawl
            tags: Optional tags for categorization
            description: Optional description

        Returns:
            The created Source object.

        Raises:
            ValueError: If a source with this URL already exists.
        """
        # Check for duplicate
        existing = self.db.get_source_by_url(base_url)
        if existing:
            raise ValueError(f"Source already exists for URL: {base_url}")

        # Parse frequency
        try:
            frequency = CrawlFrequency(crawl_frequency.lower())
        except ValueError:
            frequency = CrawlFrequency.WEEKLY

        source = Source(
            name=name,
            base_url=base_url,
            sitemap_url=sitemap_url,
            priority_paths=priority_paths or [],
            exclude_paths=exclude_paths or [],
            crawl_frequency=frequency,
            max_pages=max_pages,
            is_enabled=True,
            is_curated=False,
            created_by=SourceCreatedBy.USER,
            health_status=SourceHealthStatus.UNKNOWN,
            tags=tags or [],
            description=description,
        )

        return self.db.create_source(source)

    def add_discovered_source(
        self,
        name: str,
        base_url: str,
        sitemap_url: Optional[str] = None,
        description: Optional[str] = None,
        confidence: float = 0.0,
    ) -> Source:
        """
        Add a source discovered by the discovery agent.

        Args:
            name: Display name for the source
            base_url: Base URL of the documentation
            sitemap_url: Optional sitemap URL
            description: Description from discovery
            confidence: Discovery confidence score (0-1)

        Returns:
            The created Source object.

        Raises:
            ValueError: If a source with this URL already exists.
        """
        existing = self.db.get_source_by_url(base_url)
        if existing:
            raise ValueError(f"Source already exists for URL: {base_url}")

        source = Source(
            name=name,
            base_url=base_url,
            sitemap_url=sitemap_url,
            crawl_frequency=CrawlFrequency.WEEKLY,
            max_pages=500,
            is_enabled=True,
            is_curated=False,
            created_by=SourceCreatedBy.DISCOVERY,
            health_status=SourceHealthStatus.UNKNOWN,
            description=description,
        )

        return self.db.create_source(source)

    def list_sources(
        self,
        is_enabled: Optional[bool] = None,
        created_by: Optional[str] = None,
        health_status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Source]:
        """
        List sources with optional filtering.

        Args:
            is_enabled: Filter by enabled status
            created_by: Filter by creator (curated/user/discovery)
            health_status: Filter by health status
            limit: Maximum number of sources to return

        Returns:
            List of Source objects.
        """
        created_by_enum = None
        if created_by:
            try:
                created_by_enum = SourceCreatedBy(created_by.lower())
            except ValueError:
                pass

        health_enum = None
        if health_status:
            try:
                health_enum = SourceHealthStatus(health_status.lower())
            except ValueError:
                pass

        return self.db.list_sources(
            is_enabled=is_enabled,
            created_by=created_by_enum,
            health_status=health_enum,
            limit=limit,
        )

    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        return self.db.get_source(source_id)

    def update_source(self, source_id: str, updates: Dict[str, Any]) -> Optional[Source]:
        """
        Update a source.

        Args:
            source_id: ID of the source to update
            updates: Dictionary of fields to update

        Returns:
            Updated Source object, or None if not found.
        """
        # Convert string enums to proper types
        if "crawl_frequency" in updates:
            try:
                updates["crawl_frequency"] = CrawlFrequency(updates["crawl_frequency"].lower())
            except (ValueError, AttributeError):
                del updates["crawl_frequency"]

        return self.db.update_source(source_id, updates)

    def delete_source(self, source_id: str) -> bool:
        """Delete a source."""
        return self.db.delete_source(source_id)

    def toggle_source(self, source_id: str, enabled: bool) -> Optional[Source]:
        """Enable or disable a source."""
        return self.db.update_source(source_id, {"is_enabled": enabled})

    def get_sources_due_for_crawl(self) -> List[Source]:
        """Get sources that are due for scheduled crawling."""
        return self.db.list_sources_due_for_crawl()

    def mark_crawl_complete(
        self,
        source_id: str,
        success: bool,
        chunks_count: int = 0,
        content_hash: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Source]:
        """Update source after a crawl completes."""
        return self.db.update_source_after_crawl(
            source_id=source_id,
            success=success,
            chunks_count=chunks_count,
            content_hash=content_hash,
            error_message=error_message,
        )

    def mark_source_stale(self, source_id: str) -> Optional[Source]:
        """Mark a source as stale (needs re-crawling)."""
        return self.db.mark_source_stale(source_id)

    def get_stats(self) -> Dict[str, int]:
        """Get source registry statistics."""
        return self.db.count_sources()

    def get_curated_source_names(self) -> List[str]:
        """Get list of curated source names for quick reference."""
        curated = self.load_curated_sources()
        return [s.get("name", "") for s in curated if s.get("name")]

    def check_source_health(self) -> Dict[str, List[str]]:
        """
        Check health of all sources and identify issues.

        Returns:
            Dict with lists of source IDs by issue type.
        """
        issues = {
            "stale": [],      # Sources that haven't been crawled in 2x their frequency
            "error": [],      # Sources with errors
            "never_crawled": [],  # Sources that have never been crawled
        }

        sources = self.db.list_sources(is_enabled=True)
        now = time.time()

        for source in sources:
            if source.health_status == SourceHealthStatus.ERROR:
                issues["error"].append(source.id)
            elif source.last_crawled_at is None:
                issues["never_crawled"].append(source.id)
            else:
                # Check if stale (2x the crawl frequency)
                age = now - source.last_crawled_at
                stale_threshold = {
                    CrawlFrequency.DAILY: 2 * 86400,
                    CrawlFrequency.WEEKLY: 2 * 604800,
                    CrawlFrequency.MONTHLY: 2 * 2592000,
                    CrawlFrequency.MANUAL: float("inf"),
                }.get(source.crawl_frequency, 604800 * 2)

                if age > stale_threshold:
                    issues["stale"].append(source.id)
                    # Also update the source status
                    self.db.update_source(source.id, {
                        "health_status": SourceHealthStatus.STALE.value
                    })

        return issues


# Singleton instances per tenant
_registries: Dict[str, SourceRegistry] = {}


def get_source_registry(tenant_id: Optional[str] = None) -> SourceRegistry:
    """Get the source registry instance for a tenant."""
    key = tenant_id or get_tenant_id()
    if key not in _registries:
        _registries[key] = SourceRegistry(tenant_id=key)
    return _registries[key]
