"""
Shared data models and enums for ContextPilot.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


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


class CrawlFrequency(str, Enum):
    """Frequency for scheduled crawling."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    MANUAL = "manual"


class SourceHealthStatus(str, Enum):
    """Health status of a source."""
    HEALTHY = "healthy"
    STALE = "stale"
    ERROR = "error"
    UNKNOWN = "unknown"


class SourceCreatedBy(str, Enum):
    """How the source was added."""
    CURATED = "curated"
    USER = "user"
    DISCOVERY = "discovery"


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


@dataclass
class RawChunk:
    """Represents a raw chunk stored for normalization."""
    id: Optional[str] = None
    source_url: str = ""
    page_url: str = ""
    title: str = ""
    content: str = ""
    content_hash: str = ""
    created_at: float = 0.0


@dataclass
class Source:
    """Represents a documentation source for crawling."""
    id: Optional[str] = None
    name: str = ""
    base_url: str = ""
    sitemap_url: Optional[str] = None
    priority_paths: List[str] = None
    exclude_paths: List[str] = None
    crawl_frequency: CrawlFrequency = CrawlFrequency.WEEKLY
    max_pages: int = 500
    is_enabled: bool = True
    is_curated: bool = False
    created_by: SourceCreatedBy = SourceCreatedBy.USER
    health_status: SourceHealthStatus = SourceHealthStatus.UNKNOWN
    last_crawled_at: Optional[float] = None
    last_content_hash: Optional[str] = None
    next_crawl_at: Optional[float] = None
    chunks_count: int = 0
    error_message: Optional[str] = None
    tags: List[str] = None
    description: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        if self.priority_paths is None:
            self.priority_paths = []
        if self.exclude_paths is None:
            self.exclude_paths = []
        if self.tags is None:
            self.tags = []

    def is_due_for_crawl(self) -> bool:
        """Check if this source is due for a scheduled crawl."""
        if not self.is_enabled:
            return False
        if self.crawl_frequency == CrawlFrequency.MANUAL:
            return False
        if self.next_crawl_at is None:
            return True
        return time.time() >= self.next_crawl_at

    def compute_next_crawl_time(self) -> float:
        """Compute the next crawl time based on frequency."""
        now = time.time()
        if self.crawl_frequency == CrawlFrequency.DAILY:
            return now + 86400  # 24 hours
        if self.crawl_frequency == CrawlFrequency.WEEKLY:
            return now + 604800  # 7 days
        if self.crawl_frequency == CrawlFrequency.MONTHLY:
            return now + 2592000  # 30 days
        return now
