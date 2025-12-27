"""
ContextPilot Discovery Agent

AI-powered discovery of documentation sources.
Uses LLM knowledge to find relevant documentation for a topic,
validates URLs, and extracts metadata.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, urljoin

import httpx
from .config import get_config
from .tenant_context import get_tenant_id
from .genai_client import get_genai_client

logger = logging.getLogger("contextpilot.discovery_agent")


@dataclass
class DiscoveredSource:
    """A discovered documentation source."""
    name: str
    base_url: str
    sitemap_url: Optional[str] = None
    description: str = ""
    confidence: float = 0.7
    category: str = "general"
    suggested_paths: List[str] = field(default_factory=list)
    validation_status: str = "pending"  # pending, valid, invalid
    validation_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "sitemap_url": self.sitemap_url,
            "description": self.description,
            "confidence": self.confidence,
            "category": self.category,
            "suggested_paths": self.suggested_paths,
            "validation_status": self.validation_status,
            "validation_error": self.validation_error,
        }


@dataclass
class DiscoveryResult:
    """Result from a discovery operation."""
    topic: str
    sources: List[DiscoveredSource]
    discovery_time_ms: float
    llm_suggestions: int
    validated_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "sources": [s.to_dict() for s in self.sources],
            "discovery_time_ms": self.discovery_time_ms,
            "llm_suggestions": self.llm_suggestions,
            "validated_count": self.validated_count,
        }


class DiscoveryAgent:
    """
    AI-powered documentation source discovery.

    Uses LLM knowledge to discover documentation sources for a topic,
    validates URLs, and attempts to find sitemaps.
    """

    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id or get_tenant_id()
        self.config = get_config()
        self._model = None

        # Initialize Gemini model
        if self.config.has_gemini:
            try:
                self._model = get_genai_client()
                logger.info("Discovery agent initialized with Gemini")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini for discovery: {e}")

    @property
    def is_ready(self) -> bool:
        """Check if LLM is available for discovery."""
        return self._model is not None

    async def discover_for_topic(
        self,
        topic: str,
        existing_sources: Optional[List[str]] = None,
        max_results: int = 10,
        validate: bool = True,
    ) -> DiscoveryResult:
        """
        Discover documentation sources for a topic.

        Args:
            topic: The topic to find documentation for
            existing_sources: URLs of already-added sources to exclude
            max_results: Maximum number of sources to return
            validate: Whether to validate discovered URLs

        Returns:
            DiscoveryResult with discovered sources
        """
        start_time = time.time()
        existing_sources = existing_sources or []

        if not self._model:
            return DiscoveryResult(
                topic=topic,
                sources=[],
                discovery_time_ms=0,
                llm_suggestions=0,
                validated_count=0,
            )

        # Step 1: Use LLM to discover sources
        suggestions = await self._llm_discover(topic, existing_sources, max_results)

        # Step 2: Validate and enrich sources
        if validate and suggestions:
            validated = await self._validate_sources(suggestions)
        else:
            validated = suggestions

        elapsed_ms = (time.time() - start_time) * 1000

        return DiscoveryResult(
            topic=topic,
            sources=validated,
            discovery_time_ms=elapsed_ms,
            llm_suggestions=len(suggestions),
            validated_count=sum(1 for s in validated if s.validation_status == "valid"),
        )

    async def _llm_discover(
        self,
        topic: str,
        existing_sources: List[str],
        max_results: int,
    ) -> List[DiscoveredSource]:
        """Use LLM to discover documentation sources."""
        existing_list = "\n".join(f"- {url}" for url in existing_sources[:20]) if existing_sources else "None"

        prompt = f"""You are a documentation expert. Find official documentation sources for the following topic.

Topic: "{topic}"

Already indexed sources (do NOT include these):
{existing_list}

Find {max_results} documentation sources. For each source, provide:
1. name: Short name of the documentation
2. base_url: The base URL of the documentation site
3. sitemap_url: The sitemap.xml URL if known (often at /sitemap.xml)
4. description: Brief description of what the documentation covers
5. category: One of: api_reference, sdk, framework, library, platform, tool, language, general
6. suggested_paths: Important paths within the docs (e.g., /api/, /guide/, /reference/)

Focus on:
- Official documentation (not tutorials or blog posts)
- Active and maintained sources
- High-quality technical documentation
- API references, SDK docs, framework guides

Respond with a JSON array:
[
  {{
    "name": "Example Docs",
    "base_url": "https://docs.example.com",
    "sitemap_url": "https://docs.example.com/sitemap.xml",
    "description": "Official API documentation for Example",
    "category": "api_reference",
    "suggested_paths": ["/api/", "/reference/", "/guides/"]
  }},
  ...
]

Respond ONLY with the JSON array, no other text."""

        try:
            text = self._model.generate(prompt, temperature=0.3, max_output_tokens=2000)
            return self._parse_discovery_response(text)

        except Exception as e:
            logger.error(f"LLM discovery failed: {e}")
            return []

    def _parse_discovery_response(self, response_text: str) -> List[DiscoveredSource]:
        """Parse LLM response into DiscoveredSource objects."""
        sources = []

        try:
            # Clean up response
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"```json?\s*", "", cleaned)
                cleaned = re.sub(r"```\s*$", "", cleaned)

            data = json.loads(cleaned)

            for item in data:
                if not item.get("base_url"):
                    continue

                # Validate URL format
                parsed = urlparse(item["base_url"])
                if not parsed.scheme or not parsed.netloc:
                    continue

                sources.append(DiscoveredSource(
                    name=item.get("name", "Unknown"),
                    base_url=item["base_url"].rstrip("/"),
                    sitemap_url=item.get("sitemap_url"),
                    description=item.get("description", ""),
                    category=item.get("category", "general"),
                    suggested_paths=item.get("suggested_paths", []),
                    confidence=0.8,  # LLM suggestions start at high confidence
                ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse discovery response: {e}")

        return sources

    async def _validate_sources(
        self,
        sources: List[DiscoveredSource],
    ) -> List[DiscoveredSource]:
        """Validate discovered sources by checking URLs."""
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=40)
        timeout = httpx.Timeout(10.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, limits=limits) as client:
            tasks = [self._validate_source(source, client) for source in sources]
            return await asyncio.gather(*tasks)

    async def _validate_source(
        self,
        source: DiscoveredSource,
        client: httpx.AsyncClient,
    ) -> DiscoveredSource:
        """Validate a single source."""
        try:
            # Check base URL
            response = await client.get(source.base_url)
            if response.status_code != 200:
                source.validation_status = "invalid"
                source.validation_error = f"HTTP {response.status_code}"
                source.confidence *= 0.3
                return source

            source.validation_status = "valid"

            # Try to find sitemap if not provided
            if not source.sitemap_url:
                source.sitemap_url = await self._find_sitemap(source.base_url, client)

            # If we found a sitemap, boost confidence
            if source.sitemap_url:
                source.confidence = min(source.confidence + 0.1, 1.0)

        except httpx.TimeoutException:
            source.validation_status = "invalid"
            source.validation_error = "Connection timeout"
            source.confidence *= 0.5

        except Exception as e:
            source.validation_status = "invalid"
            source.validation_error = str(e)[:100]
            source.confidence *= 0.5

        return source

    async def _find_sitemap(
        self,
        base_url: str,
        client: httpx.AsyncClient,
    ) -> Optional[str]:
        """Try to find a sitemap for a base URL."""
        common_paths = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemap-index.xml",
            "/sitemaps/sitemap.xml",
        ]

        for path in common_paths:
            url = urljoin(base_url, path)
            try:
                response = await client.head(url)
                if response.status_code == 200:
                    return url
            except Exception:
                continue

        return None

    async def discover_related_docs(
        self,
        base_url: str,
        max_links: int = 20,
    ) -> List[str]:
        """
        Discover related documentation from a base URL by crawling links.

        Args:
            base_url: The starting URL to crawl from
            max_links: Maximum number of links to return

        Returns:
            List of discovered documentation URLs
        """
        discovered = set()
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc

        try:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=40)
            timeout = httpx.Timeout(15.0, connect=30.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, limits=limits) as client:
                response = await client.get(base_url)
                if response.status_code != 200:
                    return []

                # Simple link extraction
                html = response.text
                link_pattern = r'href=["\']([^"\']+)["\']'
                links = re.findall(link_pattern, html)

                for link in links:
                    if len(discovered) >= max_links:
                        break

                    # Normalize link
                    if link.startswith("/"):
                        full_url = f"{parsed_base.scheme}://{base_domain}{link}"
                    elif link.startswith("http"):
                        full_url = link
                    else:
                        continue

                    # Filter for documentation-like URLs
                    if self._is_doc_url(full_url, base_domain):
                        discovered.add(full_url)

        except Exception as e:
            logger.warning(f"Failed to crawl {base_url}: {e}")

        return list(discovered)[:max_links]

    def _is_doc_url(self, url: str, base_domain: str) -> bool:
        """Check if a URL looks like documentation."""
        parsed = urlparse(url)

        # Must be same domain or subdomain
        if not parsed.netloc.endswith(base_domain.replace("www.", "")):
            return False

        # Skip common non-doc paths
        skip_patterns = [
            r"/blog/",
            r"/news/",
            r"/careers/",
            r"/about/",
            r"/contact/",
            r"/pricing/",
            r"/login",
            r"/signup",
            r"/auth/",
            r"\.(css|js|png|jpg|gif|svg|ico)$",
        ]

        path = parsed.path.lower()
        for pattern in skip_patterns:
            if re.search(pattern, path):
                return False

        # Prefer doc-like paths
        doc_indicators = [
            "/docs/",
            "/documentation/",
            "/api/",
            "/reference/",
            "/guide/",
            "/manual/",
            "/handbook/",
            "/sdk/",
        ]

        for indicator in doc_indicators:
            if indicator in path:
                return True

        # Accept root paths of docs subdomains
        if "docs" in parsed.netloc:
            return True

        return len(path) > 1  # Accept most paths as potential docs


# Singleton instances (keyed by tenant id)
_discovery_agents: Dict[str, DiscoveryAgent] = {}


def get_discovery_agent(tenant_id: Optional[str] = None) -> DiscoveryAgent:
    """Get the discovery agent instance for the current tenant."""
    key = tenant_id or get_tenant_id()
    if key not in _discovery_agents:
        _discovery_agents[key] = DiscoveryAgent(tenant_id=key)
    return _discovery_agents[key]
