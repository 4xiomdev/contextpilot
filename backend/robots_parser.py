"""
ContextPilot Robots.txt Parser

Parses robots.txt files to:
- Extract sitemap URLs
- Check if URLs are allowed for crawling
- Respect crawl-delay directives
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from functools import lru_cache

import httpx

logger = logging.getLogger("contextpilot.robots_parser")


@dataclass
class RobotsRule:
    """A single allow/disallow rule."""
    path: str
    is_allow: bool

    def matches(self, url_path: str) -> bool:
        """Check if this rule matches the given URL path."""
        # Handle wildcards
        if "*" in self.path:
            pattern = self.path.replace("*", ".*")
            if self.path.endswith("$"):
                pattern = pattern[:-1] + "$"
            else:
                pattern = pattern + ".*"
            try:
                return bool(re.match(pattern, url_path))
            except re.error:
                return url_path.startswith(self.path.replace("*", ""))

        # Handle end-of-string anchor
        if self.path.endswith("$"):
            return url_path == self.path[:-1]

        # Standard prefix matching
        return url_path.startswith(self.path)


@dataclass
class UserAgentRules:
    """Rules for a specific user agent."""
    user_agent: str
    rules: List[RobotsRule] = field(default_factory=list)
    crawl_delay: Optional[float] = None

    def is_allowed(self, url_path: str) -> bool:
        """
        Check if a URL path is allowed.

        Rules are checked in order of specificity (longer paths first).
        """
        if not self.rules:
            return True

        # Sort rules by path length (most specific first)
        sorted_rules = sorted(self.rules, key=lambda r: len(r.path), reverse=True)

        for rule in sorted_rules:
            if rule.matches(url_path):
                return rule.is_allow

        # Default to allow if no rules match
        return True


@dataclass
class RobotsTxt:
    """Parsed robots.txt file."""
    sitemaps: List[str] = field(default_factory=list)
    user_agent_rules: Dict[str, UserAgentRules] = field(default_factory=dict)
    default_rules: Optional[UserAgentRules] = None
    crawl_delay: Optional[float] = None
    fetched_at: float = 0.0

    def get_rules_for_agent(self, user_agent: str) -> UserAgentRules:
        """Get rules for a specific user agent."""
        # Check for exact match
        ua_lower = user_agent.lower()
        for ua, rules in self.user_agent_rules.items():
            if ua.lower() == ua_lower:
                return rules

        # Check for partial match (e.g., "Googlebot" matches "Googlebot/2.1")
        for ua, rules in self.user_agent_rules.items():
            if ua.lower() in ua_lower or ua_lower in ua.lower():
                return rules

        # Fall back to wildcard rules
        if "*" in self.user_agent_rules:
            return self.user_agent_rules["*"]

        # Fall back to default (allow all)
        if self.default_rules:
            return self.default_rules

        return UserAgentRules(user_agent="*")

    def is_allowed(self, url_path: str, user_agent: str = "*") -> bool:
        """Check if a URL path is allowed for a user agent."""
        rules = self.get_rules_for_agent(user_agent)
        return rules.is_allowed(url_path)


class RobotsParser:
    """
    Parser for robots.txt files.

    Features:
    - Extracts sitemap URLs
    - Parses allow/disallow rules
    - Respects crawl-delay
    - Caches parsed results
    """

    DEFAULT_USER_AGENT = "ContextPilotBot"
    DEFAULT_TIMEOUT = 10.0
    CACHE_TTL = 3600  # 1 hour

    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self._cache: Dict[str, Tuple[RobotsTxt, float]] = {}
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.DEFAULT_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
        return self._client

    def _get_robots_url(self, base_url: str) -> str:
        """Get the robots.txt URL for a base URL."""
        parsed = urlparse(base_url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def fetch_and_parse(self, base_url: str, force_refresh: bool = False) -> RobotsTxt:
        """
        Fetch and parse robots.txt for a domain.

        Args:
            base_url: Base URL of the site
            force_refresh: Force refresh even if cached

        Returns:
            Parsed RobotsTxt object
        """
        robots_url = self._get_robots_url(base_url)

        # Check cache
        if not force_refresh and robots_url in self._cache:
            cached, fetched_at = self._cache[robots_url]
            if time.time() - fetched_at < self.CACHE_TTL:
                return cached

        # Fetch robots.txt
        try:
            client = self._get_client()
            response = client.get(robots_url)

            if response.status_code == 200:
                content = response.text
            else:
                # No robots.txt or error - allow all
                logger.debug(f"No robots.txt at {robots_url} (status {response.status_code})")
                content = ""

        except Exception as e:
            logger.warning(f"Failed to fetch robots.txt from {robots_url}: {e}")
            content = ""

        # Parse
        result = self.parse(content, base_url)
        result.fetched_at = time.time()

        # Cache
        self._cache[robots_url] = (result, result.fetched_at)

        return result

    def parse(self, content: str, base_url: str = "") -> RobotsTxt:
        """
        Parse robots.txt content.

        Args:
            content: Raw robots.txt content
            base_url: Base URL for resolving relative sitemap URLs

        Returns:
            Parsed RobotsTxt object
        """
        result = RobotsTxt()

        if not content or not content.strip():
            return result

        current_user_agents: List[str] = []

        for line in content.splitlines():
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Remove inline comments
            if "#" in line:
                line = line[:line.index("#")].strip()

            # Parse directive
            if ":" not in line:
                continue

            directive, value = line.split(":", 1)
            directive = directive.strip().lower()
            value = value.strip()

            if not value:
                continue

            if directive == "sitemap":
                # Resolve relative URLs
                if base_url and not value.startswith(("http://", "https://")):
                    value = urljoin(base_url, value)
                if value not in result.sitemaps:
                    result.sitemaps.append(value)

            elif directive == "user-agent":
                current_user_agents = [value]
                # Ensure rules exist for this user agent
                if value not in result.user_agent_rules:
                    result.user_agent_rules[value] = UserAgentRules(user_agent=value)

            elif directive == "disallow" and current_user_agents:
                if not value:
                    # Empty disallow = allow all
                    continue
                for ua in current_user_agents:
                    if ua in result.user_agent_rules:
                        result.user_agent_rules[ua].rules.append(
                            RobotsRule(path=value, is_allow=False)
                        )

            elif directive == "allow" and current_user_agents:
                for ua in current_user_agents:
                    if ua in result.user_agent_rules:
                        result.user_agent_rules[ua].rules.append(
                            RobotsRule(path=value, is_allow=True)
                        )

            elif directive == "crawl-delay" and current_user_agents:
                try:
                    delay = float(value)
                    for ua in current_user_agents:
                        if ua in result.user_agent_rules:
                            result.user_agent_rules[ua].crawl_delay = delay
                    if "*" in current_user_agents:
                        result.crawl_delay = delay
                except ValueError:
                    pass

        # Set default rules if wildcard exists
        if "*" in result.user_agent_rules:
            result.default_rules = result.user_agent_rules["*"]

        return result

    def is_allowed(self, url: str, user_agent: Optional[str] = None) -> bool:
        """
        Check if a URL is allowed to be crawled.

        Args:
            url: Full URL to check
            user_agent: User agent to check for (default: configured user agent)

        Returns:
            True if allowed, False if disallowed
        """
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        robots = self.fetch_and_parse(base_url)
        ua = user_agent or self.user_agent

        return robots.is_allowed(parsed.path or "/", ua)

    def get_sitemaps(self, base_url: str) -> List[str]:
        """
        Get sitemap URLs from robots.txt.

        Args:
            base_url: Base URL of the site

        Returns:
            List of sitemap URLs
        """
        robots = self.fetch_and_parse(base_url)
        return robots.sitemaps

    def get_crawl_delay(self, base_url: str, user_agent: Optional[str] = None) -> Optional[float]:
        """
        Get crawl delay for a site.

        Args:
            base_url: Base URL of the site
            user_agent: User agent to check for

        Returns:
            Crawl delay in seconds, or None if not specified
        """
        robots = self.fetch_and_parse(base_url)
        ua = user_agent or self.user_agent
        rules = robots.get_rules_for_agent(ua)
        return rules.crawl_delay or robots.crawl_delay

    def filter_allowed_urls(
        self,
        urls: List[str],
        user_agent: Optional[str] = None,
    ) -> List[str]:
        """
        Filter a list of URLs to only include allowed ones.

        Args:
            urls: List of URLs to filter
            user_agent: User agent to check for

        Returns:
            List of allowed URLs
        """
        if not urls:
            return []

        # Group URLs by domain
        by_domain: Dict[str, List[str]] = {}
        for url in urls:
            parsed = urlparse(url)
            domain = f"{parsed.scheme}://{parsed.netloc}"
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(url)

        # Check each domain
        allowed: List[str] = []
        ua = user_agent or self.user_agent

        for domain, domain_urls in by_domain.items():
            robots = self.fetch_and_parse(domain)
            for url in domain_urls:
                parsed = urlparse(url)
                if robots.is_allowed(parsed.path or "/", ua):
                    allowed.append(url)

        return allowed

    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Singleton instance
_parser: Optional[RobotsParser] = None


def get_robots_parser() -> RobotsParser:
    """Get the singleton robots parser instance."""
    global _parser
    if _parser is None:
        _parser = RobotsParser()
    return _parser
