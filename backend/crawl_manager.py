"""
ContextPilot Crawl Manager
Handles web crawling with adaptive 3-tier escalation:
  1. trafilatura (fast, local) → 2. Firecrawl (API, JS support) → 3. Playwright (full browser)
Supports batch crawling, sitemap-based discovery, freshness detection, and quality gates.
"""

import asyncio
import hashlib
import logging
import random
import re
import time
import threading
import os
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import List, Optional, Callable, Set, Dict, Any, Tuple
from urllib.parse import urlsplit, urljoin

import trafilatura

from .config import get_config
from .firestore_db import get_firestore_db as get_db
from .models import CrawlStatus, CrawlMethod, Source
from .embed_manager import get_embed_manager, Chunk
from .tenant_context import get_tenant_id

logger = logging.getLogger("contextpilot.crawl")


# ---------------------------------------------------------------------------
# Quality Gate Configuration
# ---------------------------------------------------------------------------
class QualityGateConfig:
    """Configuration for content quality gates."""
    min_content_length: int = 200  # Minimum characters
    max_link_density: float = 0.5  # Max ratio of link chars to total
    min_alpha_ratio: float = 0.6  # Min ratio of alphabetic chars
    min_word_count: int = 30  # Minimum words
    max_boilerplate_ratio: float = 0.4  # Max ratio of boilerplate phrases


class EscalationReason(Enum):
    """Reasons for escalating to next crawl tier."""
    FETCH_FAILED = "fetch_failed"
    EMPTY_CONTENT = "empty_content"
    QUALITY_GATE_FAILED = "quality_gate_failed"
    JS_REQUIRED = "js_required"
    BLOCKED = "blocked"


@dataclass
class QualityReport:
    """Report from quality gate checks."""
    passed: bool
    content_length: int
    word_count: int
    alpha_ratio: float
    link_density: float
    boilerplate_ratio: float
    issues: List[str] = field(default_factory=list)
    escalation_reason: Optional[EscalationReason] = None


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """
    Decorator for retry with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        jitter = delay * 0.2 * random.random()
                        sleep_time = delay + jitter
                        logger.debug(f"Retry {attempt + 1}/{max_retries} after {sleep_time:.1f}s: {e}")
                        time.sleep(sleep_time)
            raise last_exception
        return wrapper
    return decorator


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

# Try to import Playwright (optional tier 3)
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    sync_playwright = None
    HAS_PLAYWRIGHT = False


# ---------------------------------------------------------------------------
# JS Detection Patterns (used to detect if page requires JavaScript)
# ---------------------------------------------------------------------------
JS_REQUIRED_PATTERNS = [
    r"please enable javascript",
    r"javascript is required",
    r"this page requires javascript",
    r"enable javascript to view",
    r"you need to enable javascript",
    r"<noscript>.*?(enable|requires?|need).*?javascript",
    r"loading\.\.\.",
    r"__NEXT_DATA__",  # Next.js app (may need JS)
    r"react-root",
    r"ng-app",  # Angular
    r"data-reactroot",
]

BLOCKED_PATTERNS = [
    r"access denied",
    r"403 forbidden",
    r"rate limit",
    r"too many requests",
    r"captcha",
    r"cloudflare",
    r"please verify you are human",
    r"bot detected",
]

BOILERPLATE_PHRASES = [
    "skip to main content",
    "skip to content",
    "accept all cookies",
    "cookie settings",
    "privacy policy",
    "terms of service",
    "sign in",
    "sign up",
    "log in",
    "create account",
    "subscribe to newsletter",
    "follow us on",
    "share this",
    "related articles",
    "recommended for you",
    "advertisement",
    "sponsored content",
    "copyright ©",
    "all rights reserved",
]


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
    content_hash: Optional[str] = None


@dataclass
class SourceCrawlResult:
    """Result of crawling an entire source (sitemap-based)."""
    success: bool
    source_id: str
    source_url: str
    pages_crawled: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    chunks_indexed: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    is_incremental: bool = False


class CrawlManager:
    """
    Manages web crawling with adaptive 3-tier escalation.

    Tier 1: trafilatura (fast, local, no API costs)
    Tier 2: Firecrawl (API-based, handles JS-heavy sites)
    Tier 3: Playwright (full browser rendering, optional)

    Each tier is tried in order, escalating when quality gates fail
    or when the fetcher detects JS requirements.
    """

    def __init__(self, tenant_id: Optional[str] = None):
        self.config = get_config()
        self.tenant_id = tenant_id or get_tenant_id()
        self.db = get_db(self.tenant_id)
        self.embed_manager = get_embed_manager(self.tenant_id)

        # Quality gate configuration
        self.quality_config = QualityGateConfig()

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

        # Check Playwright availability
        self._playwright_available = HAS_PLAYWRIGHT
        if self._playwright_available:
            logger.info("Playwright available for tier 3 crawling")

        self._firecrawl_lock = threading.Lock()
        self._firecrawl_last_at = 0.0
        self._local_semaphore = threading.Semaphore(int(os.getenv("LOCAL_CRAWL_MAX_CONCURRENT", "3")))
        self._playwright_semaphore = threading.Semaphore(int(os.getenv("PLAYWRIGHT_MAX_CONCURRENT", "2")))
    
    @property
    def has_firecrawl(self) -> bool:
        """Check if Firecrawl is available."""
        return self._firecrawl_client is not None

    @property
    def has_playwright(self) -> bool:
        """Check if Playwright is available."""
        return self._playwright_available

    # ---------------------------------------------------------------------------
    # Quality Gates
    # ---------------------------------------------------------------------------

    def _check_quality_gates(self, content: str, url: str = "") -> QualityReport:
        """
        Check content against quality gates.

        Args:
            content: The extracted content to check
            url: URL for context in logging

        Returns:
            QualityReport with pass/fail status and metrics
        """
        issues: List[str] = []
        escalation_reason: Optional[EscalationReason] = None

        if not content or not content.strip():
            return QualityReport(
                passed=False,
                content_length=0,
                word_count=0,
                alpha_ratio=0.0,
                link_density=0.0,
                boilerplate_ratio=0.0,
                issues=["Empty content"],
                escalation_reason=EscalationReason.EMPTY_CONTENT,
            )

        content_length = len(content)
        words = content.split()
        word_count = len(words)

        # Calculate alpha ratio
        alpha_chars = sum(1 for c in content if c.isalpha())
        alpha_ratio = alpha_chars / content_length if content_length > 0 else 0.0

        # Calculate link density
        link_chars = len(re.findall(r'https?://\S+|\[.*?\]\(.*?\)', content))
        link_density = link_chars / content_length if content_length > 0 else 0.0

        # Calculate boilerplate ratio
        boilerplate_ratio = self._calculate_boilerplate_ratio(content)

        # Check for JS requirements
        if self._detect_js_required(content):
            issues.append("JavaScript required")
            escalation_reason = EscalationReason.JS_REQUIRED

        # Check for blocked indicators
        if self._detect_blocked(content):
            issues.append("Access blocked")
            escalation_reason = EscalationReason.BLOCKED

        # Check minimum content length
        if content_length < self.quality_config.min_content_length:
            issues.append(f"Content too short ({content_length} < {self.quality_config.min_content_length})")
            if not escalation_reason:
                escalation_reason = EscalationReason.QUALITY_GATE_FAILED

        # Check minimum word count
        if word_count < self.quality_config.min_word_count:
            issues.append(f"Too few words ({word_count} < {self.quality_config.min_word_count})")
            if not escalation_reason:
                escalation_reason = EscalationReason.QUALITY_GATE_FAILED

        # Check alpha ratio
        if alpha_ratio < self.quality_config.min_alpha_ratio:
            issues.append(f"Low alpha ratio ({alpha_ratio:.2f} < {self.quality_config.min_alpha_ratio})")
            if not escalation_reason:
                escalation_reason = EscalationReason.QUALITY_GATE_FAILED

        # Check link density
        if link_density > self.quality_config.max_link_density:
            issues.append(f"High link density ({link_density:.2f} > {self.quality_config.max_link_density})")
            if not escalation_reason:
                escalation_reason = EscalationReason.QUALITY_GATE_FAILED

        # Check boilerplate ratio
        if boilerplate_ratio > self.quality_config.max_boilerplate_ratio:
            issues.append(f"High boilerplate ({boilerplate_ratio:.2f} > {self.quality_config.max_boilerplate_ratio})")
            if not escalation_reason:
                escalation_reason = EscalationReason.QUALITY_GATE_FAILED

        passed = len(issues) == 0

        if not passed and url:
            logger.debug(f"Quality gate failed for {url}: {', '.join(issues)}")

        return QualityReport(
            passed=passed,
            content_length=content_length,
            word_count=word_count,
            alpha_ratio=alpha_ratio,
            link_density=link_density,
            boilerplate_ratio=boilerplate_ratio,
            issues=issues,
            escalation_reason=escalation_reason,
        )

    def _detect_js_required(self, content: str) -> bool:
        """Detect if content indicates JavaScript is required."""
        content_lower = content.lower()
        for pattern in JS_REQUIRED_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE | re.DOTALL):
                return True
        return False

    def _detect_blocked(self, content: str) -> bool:
        """Detect if content indicates the request was blocked."""
        content_lower = content.lower()
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return True
        return False

    def _calculate_boilerplate_ratio(self, content: str) -> float:
        """Calculate ratio of boilerplate content."""
        content_lower = content.lower()
        boilerplate_chars = 0

        for phrase in BOILERPLATE_PHRASES:
            count = content_lower.count(phrase.lower())
            if count > 0:
                boilerplate_chars += len(phrase) * count

        return boilerplate_chars / len(content) if content else 0.0

    # ---------------------------------------------------------------------------
    # Adaptive 3-Tier Crawling
    # ---------------------------------------------------------------------------

    def _crawl_with_escalation(
        self,
        url: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[CrawlResult, List[str]]:
        """
        Crawl URL using adaptive 3-tier escalation.

        Tier 1: trafilatura (fast, local)
        Tier 2: Firecrawl (API, JS support)
        Tier 3: Playwright (full browser)

        Args:
            url: URL to crawl
            callback: Progress callback

        Returns:
            Tuple of (CrawlResult, list of escalation reasons)
        """
        escalation_history: List[str] = []

        # Tier 1: Local (trafilatura)
        self._notify(callback, "Tier 1: Using local crawler...", 0.2)
        result = self._crawl_local_with_retry(url, callback)

        if result.success:
            # Check quality gates
            for page in result.pages:
                quality = self._check_quality_gates(page.markdown, url)
                if not quality.passed:
                    result.success = False
                    reason = quality.escalation_reason.value if quality.escalation_reason else "quality_failed"
                    escalation_history.append(f"tier1:{reason}")
                    logger.info(f"Tier 1 quality gate failed for {url}: {quality.issues}")
                    break

        if result.success:
            logger.debug(f"Tier 1 success for {url}")
            return result, escalation_history

        # Record tier 1 failure
        if not escalation_history:
            escalation_history.append(f"tier1:{result.error or 'failed'}")

        # Tier 2: Firecrawl (if available)
        if self._firecrawl_client:
            self._notify(callback, "Tier 2: Escalating to Firecrawl...", 0.4)
            result = self._crawl_firecrawl_with_retry(url, callback)

            if result.success:
                # Check quality gates
                for page in result.pages:
                    quality = self._check_quality_gates(page.markdown, url)
                    if not quality.passed:
                        result.success = False
                        reason = quality.escalation_reason.value if quality.escalation_reason else "quality_failed"
                        escalation_history.append(f"tier2:{reason}")
                        logger.info(f"Tier 2 quality gate failed for {url}: {quality.issues}")
                        break

            if result.success:
                logger.debug(f"Tier 2 success for {url}")
                return result, escalation_history

            # Record tier 2 failure
            if len(escalation_history) < 2:
                escalation_history.append(f"tier2:{result.error or 'failed'}")

        # Tier 3: Playwright (if available and needed)
        if self._playwright_available:
            # Only use Playwright for JS-required or blocked scenarios
            needs_browser = any(
                "js_required" in h or "blocked" in h
                for h in escalation_history
            )

            if needs_browser or not result.success:
                self._notify(callback, "Tier 3: Escalating to Playwright browser...", 0.6)
                result = self._crawl_playwright(url, callback)

                if result.success:
                    logger.debug(f"Tier 3 success for {url}")
                    return result, escalation_history

                escalation_history.append(f"tier3:{result.error or 'failed'}")

        return result, escalation_history

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _crawl_local_with_retry(
        self,
        url: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> CrawlResult:
        """Crawl using trafilatura with retry."""
        return self._crawl_local(url, callback)

    @retry_with_backoff(max_retries=2, base_delay=2.0)
    def _crawl_firecrawl_with_retry(
        self,
        url: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> CrawlResult:
        """Crawl using Firecrawl with retry."""
        return self._crawl_firecrawl(url, callback)

    def _crawl_playwright(
        self,
        url: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> CrawlResult:
        """
        Crawl using Playwright (full browser rendering).

        This is the fallback for JS-heavy sites that trafilatura
        and Firecrawl cannot handle.
        """
        if not self._playwright_available or sync_playwright is None:
            return CrawlResult(
                success=False,
                url=url,
                method=CrawlMethod.LOCAL,  # No PLAYWRIGHT enum, use LOCAL
                error="Playwright not available",
            )

        acquired = self._playwright_semaphore.acquire(timeout=60)
        if not acquired:
            return CrawlResult(
                success=False,
                url=url,
                method=CrawlMethod.LOCAL,
                error="Playwright concurrency limit reached",
            )

        try:
            self._notify(callback, "Launching browser...", 0.65)

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    context = browser.new_context(
                        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                    page = context.new_page()

                    # Navigate with timeout
                    self._notify(callback, "Loading page...", 0.7)
                    page.goto(url, wait_until="networkidle", timeout=30000)

                    # Wait for content to load
                    page.wait_for_load_state("domcontentloaded")

                    # Extract title
                    title = page.title() or self._title_from_url(url)

                    # Extract main content using various strategies
                    self._notify(callback, "Extracting content...", 0.8)

                    # Try to find main content area
                    content_selectors = [
                        "main",
                        "article",
                        "[role='main']",
                        ".content",
                        ".main-content",
                        "#content",
                        "#main",
                        ".documentation",
                        ".docs-content",
                    ]

                    html_content = None
                    for selector in content_selectors:
                        try:
                            element = page.query_selector(selector)
                            if element:
                                html_content = element.inner_html()
                                break
                        except Exception:
                            continue

                    # Fall back to body
                    if not html_content:
                        html_content = page.content()

                    # Convert HTML to markdown using trafilatura
                    markdown = trafilatura.extract(
                        html_content,
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
                            error="No meaningful content extracted from Playwright",
                        )

                    return CrawlResult(
                        success=True,
                        url=url,
                        method=CrawlMethod.LOCAL,  # Report as LOCAL since no PLAYWRIGHT enum
                        pages=[PageContent(
                            url=url,
                            title=title,
                            markdown=markdown,
                        )],
                    )

                finally:
                    browser.close()

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Playwright crawl error for {url}: {error_msg}")
            return CrawlResult(
                success=False,
                url=url,
                method=CrawlMethod.LOCAL,
                error=f"Playwright error: {error_msg}",
            )
        finally:
            try:
                self._playwright_semaphore.release()
            except Exception:
                pass

    def crawl(
        self,
        url: str,
        callback: Optional[Callable[[str, float], None]] = None,
        use_escalation: bool = True,
    ) -> CrawlResult:
        """
        Crawl a URL using adaptive 3-tier escalation.

        Tier order: trafilatura (local) → Firecrawl (API) → Playwright (browser)

        Args:
            url: The URL to crawl
            callback: Optional progress callback(message, progress_0_to_1)
            use_escalation: If True, use adaptive escalation; if False, use old behavior

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

        if use_escalation:
            # Use new adaptive 3-tier escalation
            result, escalation_history = self._crawl_with_escalation(url, callback)

            if escalation_history:
                logger.info(f"Crawl escalation for {url}: {' → '.join(escalation_history)}")
        else:
            # Legacy behavior: Firecrawl first, then local
            result = None

            if self._firecrawl_client:
                self._notify(callback, "Using Firecrawl...", 0.2)
                result = self._crawl_firecrawl(url, callback)

                if not result.success:
                    logger.info(f"Firecrawl failed, falling back to local: {result.error}")
                    self._notify(callback, "Firecrawl failed, trying local fallback...", 0.4)

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
            self._throttle_firecrawl()
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

    def _throttle_firecrawl(self) -> None:
        """Throttle Firecrawl calls to respect rate limits."""
        min_interval = max(self.config.firecrawl.min_interval_seconds, 0.0)
        if min_interval <= 0:
            return
        with self._firecrawl_lock:
            elapsed = time.time() - self._firecrawl_last_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._firecrawl_last_at = time.time()
    
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
        acquired = self._local_semaphore.acquire(timeout=120)
        if not acquired:
            return CrawlResult(
                success=False,
                url=url,
                method=CrawlMethod.LOCAL,
                error="Local crawl concurrency limit reached",
            )

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
        finally:
            try:
                self._local_semaphore.release()
            except Exception:
                pass
    
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
                if not self._is_noise_chunk(block):
                    chunks.append(block)
                continue
            
            # Otherwise, chunk with overlap
            start = 0
            while start < len(block):
                end = min(len(block), start + chunk_size)
                chunk = block[start:end]
                if not self._is_noise_chunk(chunk):
                    chunks.append(chunk)
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
            "sign in",
            "sign up",
            "privacy policy",
            "terms of service",
            "cookie policy",
        ]

        for phrase in noise_phrases:
            if phrase in lowered:
                return True

        # Too short to be useful
        if len(text.split()) < 50:
            return True

        return False

    def _is_noise_chunk(self, text: str) -> bool:
        """Check if a chunk is mostly boilerplate or links."""
        stripped = text.strip()
        if not stripped:
            return True

        words = stripped.split()
        if len(words) < 40:
            return True

        # Basic link-density heuristic
        link_tokens = stripped.count("http://") + stripped.count("https://") + stripped.count("](")
        if link_tokens > 8:
            return True

        alpha_chars = sum(1 for c in stripped if c.isalpha())
        if alpha_chars < 80:
            return True

        return False

    def _merge_exclude_paths(self, source_excludes: List[str]) -> List[str]:
        """Combine default and source-specific exclusions."""
        defaults = self.config.default_exclude_paths or []
        merged: List[str] = []
        for item in defaults + source_excludes:
            if not item:
                continue
            if not item.startswith("/"):
                item = f"/{item}"
            if item not in merged:
                merged.append(item)
        return merged

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content for freshness detection."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _check_freshness(self, url: str, content: str, stored_hash: Optional[str]) -> bool:
        """
        Check if content has changed since last crawl.

        Returns:
            True if content is new or changed, False if unchanged.
        """
        if not stored_hash:
            return True  # No previous hash, treat as new

        current_hash = self._compute_content_hash(content)
        return current_hash != stored_hash

    async def crawl_source(
        self,
        source: Source,
        incremental: bool = True,
        max_urls: Optional[int] = None,
        max_concurrent: int = 5,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> SourceCrawlResult:
        """
        Crawl an entire documentation source using its sitemap.

        Args:
            source: Source to crawl
            incremental: If True, skip unchanged pages
            max_urls: Maximum URLs to crawl
            max_concurrent: Maximum concurrent crawl requests
            callback: Progress callback

        Returns:
            SourceCrawlResult with crawl statistics
        """
        from .sitemap_parser import get_sitemap_parser

        start_time = time.time()
        result = SourceCrawlResult(
            success=False,
            source_id=source.id,
            source_url=source.base_url,
            is_incremental=incremental,
        )

        self._notify(callback, f"Starting crawl of {source.name}", 0.05)

        if max_urls is None:
            max_urls = source.max_pages or 200

        # Discover URLs from sitemap
        try:
            from urllib.parse import urlparse
            parser = get_sitemap_parser()
            sitemap_url = source.sitemap_url or f"{source.base_url.rstrip('/')}/sitemap.xml"

            self._notify(callback, f"Parsing sitemap: {sitemap_url}", 0.1)

            exclude_paths = self._merge_exclude_paths(source.exclude_paths or [])
            urls = await parser.parse_recursive(
                sitemap_url=sitemap_url,
                priority_paths=source.priority_paths or [],
                exclude_paths=exclude_paths,
                max_urls=max_urls * 3,  # Over-fetch since we'll filter
            )

            if not urls:
                # Fallback: try robots.txt
                self._notify(callback, "No sitemap found, checking robots.txt", 0.15)
                sitemaps = await parser.discover_from_robots(source.base_url)
                if sitemaps:
                    for sm_url in sitemaps[:3]:
                        more_urls = await parser.parse_recursive(
                            sitemap_url=sm_url,
                            priority_paths=source.priority_paths or [],
                            exclude_paths=exclude_paths,
                            max_urls=max_urls * 3 - len(urls),
                        )
                        urls.extend(more_urls)
                        if len(urls) >= max_urls * 3:
                            break

            # Filter URLs to only those under the base_url path
            parsed_base = urlparse(source.base_url)
            base_path = parsed_base.path.rstrip("/") or ""
            if base_path:
                pre_filter_count = len(urls)
                filtered_urls = []
                for url in urls:
                    try:
                        parsed_url = urlparse(url)
                        # Must match same host
                        if parsed_url.netloc != parsed_base.netloc:
                            continue
                        # Must be under the base path
                        url_path = parsed_url.path
                        if url_path == base_path or url_path.startswith(base_path + "/"):
                            filtered_urls.append(url)
                    except Exception:
                        continue
                urls = filtered_urls[:max_urls]
                if pre_filter_count != len(urls):
                    logger.info(f"Filtered {pre_filter_count} URLs to {len(urls)} under base path {base_path}")
            else:
                urls = urls[:max_urls]

            if not urls:
                # Last resort: just crawl the base URL
                urls = [source.base_url]

            logger.info(f"Discovered {len(urls)} URLs for {source.name}")
            self._notify(callback, f"Found {len(urls)} pages to crawl", 0.2)

        except Exception as e:
            result.errors.append(f"Sitemap parsing failed: {str(e)}")
            logger.error(f"Sitemap error for {source.name}: {e}")
            # Fall back to base URL
            urls = [source.base_url]

        # Crawl URLs in batches
        batch_result = await self.crawl_urls_batch(
            urls=urls,
            source=source,
            incremental=incremental,
            max_concurrent=max_concurrent,
            callback=callback,
        )

        # Update result
        result.pages_crawled = batch_result["pages_crawled"]
        result.pages_skipped = batch_result["pages_skipped"]
        result.pages_failed = batch_result["pages_failed"]
        result.chunks_indexed = batch_result["chunks_indexed"]
        result.errors.extend(batch_result.get("errors", []))
        result.success = result.pages_crawled > 0
        result.duration_seconds = time.time() - start_time

        self._notify(
            callback,
            f"Completed: {result.pages_crawled} pages, {result.chunks_indexed} chunks",
            1.0
        )

        return result

    async def crawl_urls_batch(
        self,
        urls: List[str],
        source: Source,
        incremental: bool = True,
        max_concurrent: int = 5,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Crawl multiple URLs in parallel with concurrency control.

        Args:
            urls: List of URLs to crawl
            source: Source these URLs belong to
            incremental: Skip unchanged content if True
            max_concurrent: Maximum concurrent requests
            callback: Progress callback

        Returns:
            Dict with crawl statistics
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        result = {
            "pages_crawled": 0,
            "pages_skipped": 0,
            "pages_failed": 0,
            "chunks_indexed": 0,
            "errors": [],
        }

        all_chunks: List[Chunk] = []
        content_hashes: Dict[str, str] = {}

        # Get stored hashes for incremental crawling
        if incremental:
            try:
                content_hashes = self.db.get_content_hashes(source.id) or {}
            except Exception:
                content_hashes = {}

        async def crawl_one(url: str, index: int):
            async with semaphore:
                try:
                    # Update progress
                    progress = 0.2 + (0.7 * index / len(urls))
                    self._notify(callback, f"Crawling {index + 1}/{len(urls)}: {url[:50]}...", progress)

                    # Run synchronous crawl in thread pool
                    loop = asyncio.get_event_loop()
                    crawl_result = await loop.run_in_executor(
                        None,
                        lambda: self.crawl(url)
                    )

                    if not crawl_result.success:
                        result["pages_failed"] += 1
                        result["errors"].append(f"{url}: {crawl_result.error}")
                        return

                    # Process each page
                    for page in crawl_result.pages:
                        if self._is_noise_page(page.markdown):
                            result["pages_skipped"] += 1
                            continue

                        # Compute hash
                        content_hash = self._compute_content_hash(page.markdown)

                        # Check freshness for incremental crawl
                        if incremental:
                            stored_hash = content_hashes.get(page.url)
                            if stored_hash == content_hash:
                                result["pages_skipped"] += 1
                                continue

                        # Store new hash
                        content_hashes[page.url] = content_hash

                        # Chunk content
                        page_chunks = self._chunk_markdown(page.markdown)
                        for chunk_content in page_chunks:
                            all_chunks.append(Chunk(
                                content=chunk_content,
                                url=source.base_url,
                                page_url=page.url,
                                title=page.title,
                            ))

                        result["pages_crawled"] += 1

                except Exception as e:
                    result["pages_failed"] += 1
                    result["errors"].append(f"{url}: {str(e)}")
                    logger.error(f"Batch crawl error for {url}: {e}")

        # Run all crawls concurrently
        tasks = [crawl_one(url, i) for i, url in enumerate(urls)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Index all chunks at once
        if all_chunks:
            self._notify(callback, f"Indexing {len(all_chunks)} chunks...", 0.92)
            try:
                indexed = self.embed_manager.index_chunks(all_chunks)
                result["chunks_indexed"] = indexed
            except Exception as e:
                result["errors"].append(f"Indexing error: {str(e)}")
                logger.error(f"Failed to index chunks: {e}")

        # Store updated hashes
        if content_hashes:
            try:
                self.db.store_content_hashes(source.id, content_hashes)
            except Exception as e:
                logger.warning(f"Failed to store content hashes: {e}")

        return result

    async def crawl_url_async(
        self,
        url: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> CrawlResult:
        """
        Async wrapper for crawl() method.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.crawl(url, callback)
        )


# Singleton instances (keyed by tenant id).
_crawl_managers: dict[str, CrawlManager] = {}


def get_crawl_manager(tenant_id: Optional[str] = None) -> CrawlManager:
    """Get the crawl manager instance for the current tenant."""
    key = tenant_id or get_tenant_id()
    if key not in _crawl_managers:
        _crawl_managers[key] = CrawlManager(tenant_id=key)
    return _crawl_managers[key]


def get_firecrawl_client(tenant_id: Optional[str] = None):
    """
    Get the Firecrawl client from the crawl manager.

    Returns None if Firecrawl is not configured.
    Used by discovery ladder for URL mapping.
    """
    manager = get_crawl_manager(tenant_id)
    return manager._firecrawl_client
