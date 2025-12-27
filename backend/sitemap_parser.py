"""
ContextPilot Sitemap Parser

Parses XML sitemaps to discover documentation URLs for crawling.
Supports:
- Standard sitemaps (urlset)
- Sitemap indexes (sitemapindex)
- URL filtering by priority/exclude paths
- Concurrent fetching for nested sitemaps
"""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger("contextpilot.sitemap")

# XML namespaces used in sitemaps
SITEMAP_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
}


class SitemapParser:
    """
    Async sitemap parser for discovering documentation URLs.

    Features:
    - Parses both sitemap.xml and sitemap index files
    - Filters URLs by priority and exclusion paths
    - Respects max_urls limit
    - Handles malformed XML gracefully
    """

    def __init__(self, timeout: float = 30.0, max_concurrent: int = 5):
        """
        Initialize the parser.

        Args:
            timeout: HTTP request timeout in seconds
            max_concurrent: Max concurrent requests for nested sitemaps
        """
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._client is None:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=40)
            timeout = httpx.Timeout(self.timeout, connect=60.0)
            self._client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                headers={"User-Agent": "ContextPilot/1.0 (sitemap crawler)"},
                follow_redirects=True,
            )
        return self._client

    async def parse(
        self,
        sitemap_url: str,
        priority_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        max_urls: int = 500,
    ) -> List[str]:
        """
        Fetch and parse a sitemap, returning filtered URLs.

        Args:
            sitemap_url: URL of the sitemap.xml or sitemap index
            priority_paths: URL path prefixes to prioritize (e.g., ["/api", "/docs"])
            exclude_paths: URL path prefixes to exclude (e.g., ["/blog", "/changelog"])
            max_urls: Maximum number of URLs to return

        Returns:
            List of discovered URLs, filtered and limited.
        """
        logger.info(f"Parsing sitemap: {sitemap_url}")

        try:
            content = await self._fetch(sitemap_url)
            if not content:
                return []

            # Parse XML
            urls = self._parse_xml(content, sitemap_url)
            logger.info(f"Found {len(urls)} URLs in sitemap")

            # Filter URLs
            filtered = self._filter_urls(
                urls=urls,
                priority_paths=priority_paths or [],
                exclude_paths=exclude_paths or [],
                max_urls=max_urls,
            )

            logger.info(f"Filtered to {len(filtered)} URLs")
            return filtered

        except Exception as e:
            logger.error(f"Failed to parse sitemap {sitemap_url}: {e}")
            return []

    async def _fetch(self, url: str) -> Optional[str]:
        """Fetch URL content with timeout and error handling."""
        async with self._semaphore:
            try:
                client = self._get_client()
                response = await client.get(url)
                response.raise_for_status()
                return response.text
            except httpx.TimeoutException:
                logger.warning(f"Timeout fetching {url}")
                return None
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error {e.response.status_code} for {url}")
                return None
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                return None

    def _parse_xml(self, content: str, base_url: str) -> List[str]:
        """
        Parse sitemap XML content.

        Handles both:
        - <urlset>: Contains <url><loc>...</loc></url> entries
        - <sitemapindex>: Contains nested sitemap references
        """
        urls: List[str] = []

        try:
            # Try to parse as XML
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
            # Try regex fallback for malformed XML
            return self._parse_urls_regex(content)

        # Get root tag without namespace
        root_tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

        if root_tag == "sitemapindex":
            # This is a sitemap index - extract nested sitemap URLs
            # For now, we'll just extract URLs from the index
            # A more complete implementation would recursively fetch nested sitemaps
            for sitemap in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
                if sitemap.text:
                    urls.append(sitemap.text.strip())
            # Also try without namespace
            for sitemap in root.findall(".//loc"):
                if sitemap.text and sitemap.text.strip() not in urls:
                    urls.append(sitemap.text.strip())

            logger.info(f"Found sitemap index with {len(urls)} nested sitemaps")
            # For sitemap indexes, we return the sitemap URLs themselves
            # The caller should decide whether to recursively parse them

        elif root_tag == "urlset":
            # Standard sitemap with URL entries
            for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
                if loc.text:
                    urls.append(loc.text.strip())
            # Also try without namespace
            for loc in root.findall(".//loc"):
                if loc.text and loc.text.strip() not in urls:
                    urls.append(loc.text.strip())

        else:
            # Unknown format - try regex fallback
            logger.warning(f"Unknown sitemap format: {root_tag}")
            urls = self._parse_urls_regex(content)

        return urls

    def _parse_urls_regex(self, content: str) -> List[str]:
        """Fallback URL extraction using regex for malformed XML."""
        pattern = r"<loc>\s*(https?://[^<\s]+)\s*</loc>"
        matches = re.findall(pattern, content, re.IGNORECASE)
        return list(set(matches))

    # Language codes to filter out (non-English)
    NON_ENGLISH_LANGS = {
        "ar", "bg", "bn", "cs", "da", "de", "el", "es", "es-419", "et", "fa", "fi",
        "fil", "fr", "gu", "he", "hi", "hr", "hu", "id", "it", "iw", "ja", "kn",
        "ko", "lt", "lv", "ml", "mr", "ms", "nl", "no", "pl", "pt", "pt-BR", "pt-PT",
        "ro", "ru", "sk", "sl", "sr", "sv", "sw", "ta", "te", "th", "tr", "uk", "ur",
        "vi", "zh", "zh-CN", "zh-TW", "zh-Hans", "zh-Hant",
    }

    def _is_non_english_url(self, url: str) -> bool:
        """Check if URL has a non-English language parameter."""
        parsed = urlparse(url)
        query = parsed.query

        # Check for hl=, lang=, locale= query params
        if query:
            # Parse query params
            from urllib.parse import parse_qs
            params = parse_qs(query)

            for key in ["hl", "lang", "locale", "language"]:
                if key in params:
                    lang_value = params[key][0].lower() if params[key] else ""
                    # Allow English variants
                    if lang_value in ("en", "en-us", "en-gb", "en-au", ""):
                        continue
                    # Block non-English
                    if lang_value in self.NON_ENGLISH_LANGS or lang_value.split("-")[0] in self.NON_ENGLISH_LANGS:
                        return True

        # Also check path for language segments like /ja/, /zh-cn/, etc.
        path_lower = parsed.path.lower()
        for lang in self.NON_ENGLISH_LANGS:
            if f"/{lang}/" in path_lower or path_lower.startswith(f"/{lang}/"):
                return True

        return False

    def _filter_urls(
        self,
        urls: List[str],
        priority_paths: List[str],
        exclude_paths: List[str],
        max_urls: int,
    ) -> List[str]:
        """
        Filter and sort URLs based on priority and exclusion rules.

        Algorithm:
        1. Remove non-English language URLs
        2. Remove excluded paths
        3. Sort: priority paths first, then alphabetically
        4. Limit to max_urls
        """
        filtered: List[str] = []
        priority_urls: List[str] = []
        other_urls: List[str] = []

        non_english_count = 0

        for url in urls:
            parsed = urlparse(url)
            path = parsed.path

            # Filter out non-English URLs FIRST
            if self._is_non_english_url(url):
                non_english_count += 1
                continue

            # Check exclusions
            excluded = any(path.startswith(ex) for ex in exclude_paths)
            if excluded:
                continue

            # Check priority
            is_priority = any(path.startswith(p) for p in priority_paths)
            if is_priority:
                priority_urls.append(url)
            else:
                other_urls.append(url)

        if non_english_count > 0:
            logger.info(f"Filtered out {non_english_count} non-English URLs")

        # Sort each group alphabetically for consistency
        priority_urls.sort()
        other_urls.sort()

        # Combine: priority first
        all_urls = priority_urls + other_urls

        # Apply limit
        return all_urls[:max_urls]

    async def discover_from_robots(self, base_url: str) -> List[str]:
        """
        Discover sitemap URLs from robots.txt.

        Args:
            base_url: Base URL of the site (e.g., "https://docs.example.com")

        Returns:
            List of sitemap URLs found in robots.txt
        """
        robots_url = urljoin(base_url, "/robots.txt")
        logger.info(f"Checking robots.txt: {robots_url}")

        content = await self._fetch(robots_url)
        if not content:
            return []

        sitemaps: List[str] = []
        for line in content.split("\n"):
            line = line.strip().lower()
            if line.startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[1].strip()
                if sitemap_url:
                    sitemaps.append(sitemap_url)

        logger.info(f"Found {len(sitemaps)} sitemaps in robots.txt")
        return sitemaps

    async def parse_recursive(
        self,
        sitemap_url: str,
        priority_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        max_urls: int = 500,
        max_depth: int = 2,
    ) -> List[str]:
        """
        Recursively parse sitemap indexes to discover all URLs.

        Args:
            sitemap_url: Starting sitemap URL
            priority_paths: URL path prefixes to prioritize
            exclude_paths: URL path prefixes to exclude
            max_urls: Maximum total URLs to return
            max_depth: Maximum recursion depth for nested sitemaps

        Returns:
            List of discovered documentation URLs.
        """
        all_urls: Set[str] = set()
        visited: Set[str] = set()

        async def _parse_level(url: str, depth: int):
            if depth > max_depth or url in visited or len(all_urls) >= max_urls:
                return

            visited.add(url)
            content = await self._fetch(url)
            if not content:
                return

            try:
                root = ET.fromstring(content)
            except ET.ParseError:
                # Try regex fallback
                urls = self._parse_urls_regex(content)
                all_urls.update(urls[:max_urls - len(all_urls)])
                return

            root_tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

            if root_tag == "sitemapindex":
                # Extract nested sitemap URLs and parse them
                nested_urls = []
                for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
                    if loc.text:
                        nested_urls.append(loc.text.strip())
                for loc in root.findall(".//loc"):
                    if loc.text and loc.text.strip() not in nested_urls:
                        nested_urls.append(loc.text.strip())

                # Parse nested sitemaps concurrently
                tasks = [_parse_level(u, depth + 1) for u in nested_urls[:50]]
                await asyncio.gather(*tasks, return_exceptions=True)

            else:
                # Extract URLs
                for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
                    if loc.text and len(all_urls) < max_urls:
                        all_urls.add(loc.text.strip())
                for loc in root.findall(".//loc"):
                    if loc.text and len(all_urls) < max_urls:
                        all_urls.add(loc.text.strip())

        await _parse_level(sitemap_url, 0)

        # Filter the collected URLs
        return self._filter_urls(
            urls=list(all_urls),
            priority_paths=priority_paths or [],
            exclude_paths=exclude_paths or [],
            max_urls=max_urls,
        )

    # =========================================================================
    # 4-Tier Discovery Ladder (Phase 1)
    # =========================================================================

    async def discover_with_ladder(
        self,
        base_url: str,
        priority_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        max_urls: int = 500,
        firecrawl_client: Optional[any] = None,
    ) -> List[str]:
        """
        Discover URLs using a 4-tier fallback ladder.

        Tier 1: Sitemap.xml (fastest, most reliable)
        Tier 2: robots.txt sitemaps
        Tier 3: Firecrawl map API (if available)
        Tier 4: Local BFS crawl (slowest, most thorough)

        IMPORTANT: All discovered URLs are filtered to only include those
        under the base_url path prefix. For example, if base_url is
        "https://example.com/docs/api", only URLs starting with "/docs/api"
        will be returned.

        Args:
            base_url: Base URL of the site (including path prefix to crawl)
            priority_paths: URL path prefixes to prioritize
            exclude_paths: URL path prefixes to exclude
            max_urls: Maximum URLs to discover
            firecrawl_client: Optional Firecrawl client for tier 3

        Returns:
            List of discovered URLs under the base_url path
        """
        parsed = urlparse(base_url)
        root_url = f"{parsed.scheme}://{parsed.netloc}"
        base_path = parsed.path.rstrip("/") or ""

        # Helper to filter URLs to only those under base_url path
        def filter_to_base_path(urls: List[str]) -> List[str]:
            if not base_path:
                return urls  # No path restriction
            filtered = []
            for url in urls:
                url_parsed = urlparse(url)
                # Must match same host
                if url_parsed.netloc != parsed.netloc:
                    continue
                # Must be under the base path
                url_path = url_parsed.path
                if url_path == base_path or url_path.startswith(base_path + "/"):
                    filtered.append(url)
            if len(filtered) < len(urls):
                logger.info(f"Filtered {len(urls)} URLs to {len(filtered)} under {base_path}")
            return filtered

        # Tier 1: Try direct sitemap URLs
        logger.info(f"Discovery ladder tier 1: Trying sitemap.xml for {base_url}")
        sitemap_candidates = [
            f"{base_url.rstrip('/')}/sitemap.xml",
            f"{base_url.rstrip('/')}/sitemap_index.xml",
            f"{root_url}/sitemap.xml",
            f"{root_url}/sitemap_index.xml",
        ]

        for sitemap_url in sitemap_candidates:
            try:
                urls = await self.parse_recursive(
                    sitemap_url=sitemap_url,
                    priority_paths=priority_paths,
                    exclude_paths=exclude_paths,
                    max_urls=max_urls * 10,  # Fetch more, then filter
                )
                if urls:
                    # Filter to only URLs under the base path
                    urls = filter_to_base_path(urls)
                    if urls:
                        logger.info(f"Tier 1 success: Found {len(urls)} URLs from {sitemap_url}")
                        return urls[:max_urls]
            except Exception as e:
                logger.debug(f"Tier 1: Sitemap {sitemap_url} failed: {e}")
                continue

        # Tier 2: Try robots.txt sitemaps
        logger.info(f"Discovery ladder tier 2: Checking robots.txt for {base_url}")
        try:
            from .robots_parser import get_robots_parser
            robots_parser = get_robots_parser()
            sitemaps = robots_parser.get_sitemaps(base_url)

            for sitemap_url in sitemaps[:5]:  # Limit to first 5
                try:
                    urls = await self.parse_recursive(
                        sitemap_url=sitemap_url,
                        priority_paths=priority_paths,
                        exclude_paths=exclude_paths,
                        max_urls=max_urls * 10,  # Fetch more, then filter
                    )
                    if urls:
                        # Filter to only URLs under the base path
                        urls = filter_to_base_path(urls)
                        if urls:
                            logger.info(f"Tier 2 success: Found {len(urls)} URLs from {sitemap_url}")
                            return urls[:max_urls]
                except Exception as e:
                    logger.debug(f"Tier 2: Sitemap {sitemap_url} failed: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Tier 2: robots.txt parsing failed: {e}")

        # Tier 3: Try Firecrawl map API (if available)
        if firecrawl_client:
            logger.info(f"Discovery ladder tier 3: Using Firecrawl map for {base_url}")
            try:
                urls = await self._discover_with_firecrawl(
                    base_url=base_url,
                    firecrawl_client=firecrawl_client,
                    max_urls=max_urls * 10,  # Fetch more, then filter
                    exclude_paths=exclude_paths or [],
                )
                if urls:
                    # Filter to only URLs under the base path
                    urls = filter_to_base_path(urls)
                    if urls:
                        logger.info(f"Tier 3 success: Found {len(urls)} URLs via Firecrawl")
                        return self._filter_urls(urls, priority_paths or [], exclude_paths or [], max_urls)
            except Exception as e:
                logger.debug(f"Tier 3: Firecrawl map failed: {e}")

        # Tier 4: Local BFS crawl (last resort)
        logger.info(f"Discovery ladder tier 4: Using local BFS for {base_url}")
        try:
            urls = await self._bfs_crawl_links(
                base_url=base_url,
                max_depth=2,
                max_urls=max_urls,
                exclude_paths=exclude_paths or [],
            )
            if urls:
                # BFS already respects base_url, but filter to be safe
                urls = filter_to_base_path(urls)
                if urls:
                    logger.info(f"Tier 4 success: Found {len(urls)} URLs via BFS")
                    return self._filter_urls(urls, priority_paths or [], exclude_paths or [], max_urls)
        except Exception as e:
            logger.warning(f"Tier 4: BFS crawl failed: {e}")

        # Final fallback: just the base URL
        logger.warning(f"All discovery tiers failed for {base_url}, returning base URL only")
        return [base_url]

    async def _discover_with_firecrawl(
        self,
        base_url: str,
        firecrawl_client: any,
        max_urls: int = 500,
        exclude_paths: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Use Firecrawl's map endpoint to discover URLs.

        This is useful for JavaScript-heavy sites where sitemaps may not exist.
        """
        try:
            # Firecrawl map endpoint
            result = firecrawl_client.map(base_url, limit=max_urls)

            urls = []
            if hasattr(result, "links"):
                urls = list(result.links or [])
            elif isinstance(result, dict) and "links" in result:
                urls = list(result["links"] or [])
            elif isinstance(result, list):
                urls = result

            # Filter out excluded paths
            if exclude_paths:
                filtered = []
                for url in urls:
                    parsed = urlparse(url)
                    if not any(parsed.path.startswith(ex) for ex in exclude_paths):
                        filtered.append(url)
                urls = filtered

            return urls[:max_urls]

        except Exception as e:
            logger.warning(f"Firecrawl map error: {e}")
            return []

    async def _bfs_crawl_links(
        self,
        base_url: str,
        max_depth: int = 2,
        max_urls: int = 500,
        exclude_paths: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Discover URLs via breadth-first link crawling.

        This is the slowest but most thorough discovery method.
        Used as a last resort when sitemaps are unavailable.
        """
        import re
        from collections import deque

        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        discovered: Set[str] = set()
        visited: Set[str] = set()
        queue: deque = deque([(base_url, 0)])

        # Regex to find links
        link_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)

        while queue and len(discovered) < max_urls:
            url, depth = queue.popleft()

            if url in visited or depth > max_depth:
                continue

            visited.add(url)

            try:
                content = await self._fetch(url)
                if not content:
                    continue

                # Extract links
                for match in link_pattern.findall(content):
                    link = match.strip()

                    # Skip non-HTTP links
                    if link.startswith(("#", "javascript:", "mailto:", "tel:")):
                        continue

                    # Resolve relative URLs
                    if not link.startswith(("http://", "https://")):
                        link = urljoin(url, link)

                    # Parse and validate
                    parsed_link = urlparse(link)

                    # Must be same domain
                    if parsed_link.netloc != base_domain:
                        continue

                    # Check exclusions
                    if exclude_paths:
                        if any(parsed_link.path.startswith(ex) for ex in exclude_paths):
                            continue

                    # Skip common non-doc paths
                    skip_patterns = [
                        "/login", "/logout", "/signup", "/register",
                        "/cart", "/checkout", "/account",
                        ".css", ".js", ".png", ".jpg", ".gif", ".svg",
                        ".pdf", ".zip", ".tar", ".gz",
                    ]
                    if any(pattern in parsed_link.path.lower() for pattern in skip_patterns):
                        continue

                    # Normalize URL (remove fragment)
                    normalized = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path}"
                    if parsed_link.query:
                        normalized += f"?{parsed_link.query}"

                    if normalized not in discovered and normalized not in visited:
                        discovered.add(normalized)
                        if depth < max_depth:
                            queue.append((normalized, depth + 1))

            except Exception as e:
                logger.debug(f"BFS crawl error for {url}: {e}")
                continue

        return list(discovered)[:max_urls]


# Singleton instance
_parser: Optional[SitemapParser] = None


def get_sitemap_parser() -> SitemapParser:
    """Get the sitemap parser instance."""
    global _parser
    if _parser is None:
        _parser = SitemapParser()
    return _parser
