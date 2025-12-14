"""
ContextPilot Crawl Manager
Handles web crawling with Firecrawl (primary) and trafilatura (fallback).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from urllib.parse import urlsplit

import trafilatura

from .config import get_config
from .firestore_db import get_firestore_db as get_db, CrawlStatus, CrawlMethod
from .embed_manager import get_embed_manager, Chunk

logger = logging.getLogger("contextpilot.crawl")


# Try to import Firecrawl
try:
    from firecrawl import FirecrawlApp
    HAS_FIRECRAWL = True
except ImportError:
    try:
        from firecrawl import Firecrawl as FirecrawlApp
        HAS_FIRECRAWL = True
    except ImportError:
        FirecrawlApp = None
        HAS_FIRECRAWL = False


@dataclass
class CrawlResult:
    """Result of a crawl operation."""
    success: bool
    url: str
    method: CrawlMethod
    pages: List["PageContent"] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def total_pages(self) -> int:
        return len(self.pages)


@dataclass
class PageContent:
    """Content extracted from a single page."""
    url: str
    title: str
    markdown: str


class CrawlManager:
    """
    Manages web crawling with automatic fallback.
    
    Primary: Firecrawl (better for JS-heavy sites, handles multi-page)
    Fallback: trafilatura (local, no API credits needed)
    """
    
    def __init__(self):
        self.config = get_config()
        self.db = get_db()
        self.embed_manager = get_embed_manager()
        
        # Initialize Firecrawl if available
        self._firecrawl_client = None
        if HAS_FIRECRAWL and self.config.has_firecrawl:
            try:
                self._firecrawl_client = FirecrawlApp(
                    api_key=self.config.firecrawl.api_key
                )
                logger.info("Firecrawl client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Firecrawl: {e}")
    
    @property
    def has_firecrawl(self) -> bool:
        """Check if Firecrawl is available."""
        return self._firecrawl_client is not None
    
    def crawl(
        self,
        url: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> CrawlResult:
        """
        Crawl a URL using the best available method.
        
        Args:
            url: The URL to crawl
            callback: Optional progress callback(message, progress_0_to_1)
        
        Returns:
            CrawlResult with extracted content
        """
        # Create crawl job
        job = self.db.create_crawl_job(url)
        self.db.update_crawl_job(
            job.id,
            status=CrawlStatus.RUNNING,
            started_at=time.time(),
        )
        
        self._notify(callback, f"Starting crawl of {url}", 0.1)
        
        result: Optional[CrawlResult] = None
        
        # Try Firecrawl first
        if self._firecrawl_client:
            self._notify(callback, "Using Firecrawl...", 0.2)
            result = self._crawl_firecrawl(url, callback)
            
            if not result.success:
                logger.info(f"Firecrawl failed, falling back to local: {result.error}")
                self._notify(callback, "Firecrawl failed, trying local fallback...", 0.4)
        
        # Fallback to trafilatura
        if result is None or not result.success:
            self._notify(callback, "Using local crawler...", 0.5)
            result = self._crawl_local(url, callback)
        
        # Update job status
        if result.success:
            self.db.update_crawl_job(
                job.id,
                status=CrawlStatus.COMPLETED,
                method=result.method,
                completed_at=time.time(),
                chunks_count=result.total_pages,
            )
            self._notify(callback, f"Crawl completed: {result.total_pages} pages", 1.0)
        else:
            self.db.update_crawl_job(
                job.id,
                status=CrawlStatus.FAILED,
                method=result.method,
                completed_at=time.time(),
                error_message=result.error,
            )
            self._notify(callback, f"Crawl failed: {result.error}", 1.0)
        
        return result
    
    def _crawl_firecrawl(
        self,
        url: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> CrawlResult:
        """Crawl using Firecrawl API."""
        try:
            # Simple scrape call - no 11-attempt cascade
            result = self._firecrawl_client.scrape(
                url,
                formats=["markdown"],
            )
            
            pages = self._extract_pages_from_firecrawl(result, url)
            
            if not pages:
                return CrawlResult(
                    success=False,
                    url=url,
                    method=CrawlMethod.FIRECRAWL,
                    error="No content extracted from Firecrawl response",
                )
            
            return CrawlResult(
                success=True,
                url=url,
                method=CrawlMethod.FIRECRAWL,
                pages=pages,
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Firecrawl error: {error_msg}")
            return CrawlResult(
                success=False,
                url=url,
                method=CrawlMethod.FIRECRAWL,
                error=error_msg,
            )
    
    def _extract_pages_from_firecrawl(self, result, url: str) -> List[PageContent]:
        """Extract pages from Firecrawl response (handles various formats)."""
        pages = []
        
        if result is None:
            return pages
        
        # Handle different Firecrawl response formats
        if hasattr(result, "markdown") and result.markdown:
            title = getattr(result, "title", "") or self._title_from_url(url)
            page_url = getattr(result, "url", url) or url
            pages.append(PageContent(
                url=page_url,
                title=title,
                markdown=result.markdown,
            ))
        
        # Handle data list format
        if hasattr(result, "data") and isinstance(result.data, list):
            for item in result.data:
                markdown = None
                if hasattr(item, "markdown"):
                    markdown = item.markdown
                elif isinstance(item, dict):
                    markdown = item.get("markdown") or item.get("content")
                
                if markdown:
                    if hasattr(item, "metadata"):
                        title = item.metadata.get("title", "") if isinstance(item.metadata, dict) else ""
                        page_url = item.metadata.get("url", url) if isinstance(item.metadata, dict) else url
                    elif isinstance(item, dict):
                        title = item.get("title", "")
                        page_url = item.get("url", url)
                    else:
                        title = ""
                        page_url = url
                    
                    pages.append(PageContent(
                        url=page_url or url,
                        title=title or self._title_from_url(page_url or url),
                        markdown=markdown,
                    ))
        
        # Handle dict format
        if isinstance(result, dict):
            if result.get("markdown"):
                pages.append(PageContent(
                    url=result.get("url", url),
                    title=result.get("title", self._title_from_url(url)),
                    markdown=result["markdown"],
                ))
            elif result.get("data") and isinstance(result["data"], list):
                for item in result["data"]:
                    if isinstance(item, dict) and item.get("markdown"):
                        pages.append(PageContent(
                            url=item.get("url", url),
                            title=item.get("title", ""),
                            markdown=item["markdown"],
                        ))
        
        return pages
    
    def _crawl_local(
        self,
        url: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> CrawlResult:
        """Crawl using trafilatura (local fallback)."""
        try:
            self._notify(callback, "Fetching page...", 0.6)
            
            # Fetch the page
            downloaded = trafilatura.fetch_url(url)
            
            if not downloaded:
                return CrawlResult(
                    success=False,
                    url=url,
                    method=CrawlMethod.LOCAL,
                    error="Failed to fetch URL",
                )
            
            self._notify(callback, "Extracting content...", 0.8)
            
            # Extract markdown-like content
            markdown = trafilatura.extract(
                downloaded,
                include_formatting=True,
                include_tables=True,
                include_links=True,
                output_format="markdown",
            )
            
            if not markdown or len(markdown.strip()) < 50:
                return CrawlResult(
                    success=False,
                    url=url,
                    method=CrawlMethod.LOCAL,
                    error="No meaningful content extracted",
                )
            
            # Extract title
            title = trafilatura.extract(downloaded, output_format="txt")
            title = title.split("\n")[0][:100] if title else self._title_from_url(url)
            
            return CrawlResult(
                success=True,
                url=url,
                method=CrawlMethod.LOCAL,
                pages=[PageContent(
                    url=url,
                    title=title,
                    markdown=markdown,
                )],
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Local crawl error: {error_msg}")
            return CrawlResult(
                success=False,
                url=url,
                method=CrawlMethod.LOCAL,
                error=error_msg,
            )
    
    def _title_from_url(self, url: str) -> str:
        """Extract a title from URL path."""
        parsed = urlsplit(url)
        path = parsed.path.strip("/")
        if path:
            # Take last path segment and clean it up
            segment = path.split("/")[-1]
            # Remove extension and convert to title case
            segment = segment.rsplit(".", 1)[0]
            segment = segment.replace("-", " ").replace("_", " ")
            return segment.title()
        return parsed.netloc
    
    def _notify(
        self,
        callback: Optional[Callable[[str, float], None]],
        message: str,
        progress: float,
    ) -> None:
        """Send progress notification if callback is provided."""
        if callback:
            try:
                callback(message, progress)
            except Exception:
                pass
        logger.debug(f"[{progress:.0%}] {message}")
    
    def crawl_and_index(
        self,
        url: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> int:
        """
        Crawl a URL and index all extracted content.
        
        Returns the number of chunks indexed.
        """
        # Crawl
        result = self.crawl(url, callback)
        
        if not result.success:
            logger.error(f"Crawl failed for {url}: {result.error}")
            return 0
        
        # Chunk and index
        chunks = []
        for page in result.pages:
            if self._is_noise_page(page.markdown):
                continue
            
            page_chunks = self._chunk_markdown(page.markdown)
            for chunk_content in page_chunks:
                chunks.append(Chunk(
                    content=chunk_content,
                    url=url,  # Source URL
                    page_url=page.url,  # Actual page URL
                    title=page.title,
                ))
        
        if not chunks:
            logger.warning(f"No valid chunks extracted from {url}")
            return 0
        
        self._notify(callback, f"Indexing {len(chunks)} chunks...", 0.9)
        
        # Index
        indexed = self.embed_manager.index_chunks(chunks)
        
        # Update job with final count
        jobs = self.db.list_crawl_jobs(limit=1)
        if jobs:
            self.db.update_crawl_job(jobs[0].id, chunks_count=indexed)
        
        self._notify(callback, f"Indexed {indexed} chunks", 1.0)
        return indexed
    
    def _chunk_markdown(
        self,
        markdown: str,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> List[str]:
        """
        Split markdown into chunks, preserving headings.
        
        Simpler than the old version - just heading-aware splitting.
        """
        text = markdown.strip()
        if not text:
            return []
        
        lines = text.splitlines()
        blocks: List[str] = []
        current: List[str] = []
        
        # Split on headings
        for line in lines:
            if line.strip().startswith("#") and current:
                blocks.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        
        if current:
            blocks.append("\n".join(current).strip())
        
        # Chunk each block
        chunks: List[str] = []
        for block in blocks:
            if not block:
                continue
            
            # If block is small enough, keep it whole
            if len(block) <= chunk_size:
                chunks.append(block)
                continue
            
            # Otherwise, chunk with overlap
            start = 0
            while start < len(block):
                end = min(len(block), start + chunk_size)
                chunks.append(block[start:end])
                start += chunk_size - overlap
        
        return [c for c in chunks if c.strip()]
    
    def _is_noise_page(self, text: str) -> bool:
        """Check if a page is noise (error page, empty, etc.)."""
        if not text:
            return True
        
        lowered = text.lower()
        
        # Common noise indicators
        noise_phrases = [
            "page not found",
            "404 error",
            "enable javascript",
            "please enable cookies",
            "access denied",
        ]
        
        for phrase in noise_phrases:
            if phrase in lowered:
                return True
        
        # Too short to be useful
        if len(text.split()) < 30:
            return True
        
        return False


# Singleton instance
_crawl_manager: Optional[CrawlManager] = None


def get_crawl_manager() -> CrawlManager:
    """Get the singleton crawl manager instance."""
    global _crawl_manager
    if _crawl_manager is None:
        _crawl_manager = CrawlManager()
    return _crawl_manager

