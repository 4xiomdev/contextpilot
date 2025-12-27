"""
ContextPilot Crawl Planner

Builds a URL universe for a source, then uses an LLM to propose a crawl plan:
- Tree summary + keep/drop/maybe decisions
- Derived allow/deny prefixes and query param policy

The system collects the URL universe deterministically; Gemini curates it.
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from .config import get_config
from .firestore_db import get_firestore_db as get_db
from .genai_client import get_genai_client
from .models import Source
from .sitemap_parser import get_sitemap_parser
from .path_patterns import (
    PathPatternAnalysis,
    PathPatternDetector,
    get_pattern_detector,
    extract_title_from_html,
)

logger = logging.getLogger("contextpilot.crawl_planner")


@dataclass(frozen=True)
class UniverseUrl:
    url_id: str
    url: str
    canonical_url: str
    path: str
    depth: int
    query_keys: List[str]


@dataclass(frozen=True)
class QueryParamPolicy:
    keep_keys: List[str]
    drop_keys: List[str]
    default: str  # "drop_unknown_keys" | "keep_unknown_keys"


@dataclass(frozen=True)
class CrawlPlanRules:
    allow_prefixes: List[str]
    deny_prefixes: List[str]
    maybe_prefixes: List[str]
    query_param_policy: QueryParamPolicy


def _url_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def _path_depth(path: str) -> int:
    stripped = path.strip("/")
    return 0 if not stripped else len([p for p in stripped.split("/") if p])


def canonicalize_url(url: str, policy: Optional[QueryParamPolicy] = None) -> str:
    """
    Canonicalize a URL for dedupe/stability.

    - Drops fragments
    - Drops common tracking/language params by default
    - Applies optional query param policy for keeps/drops
    """
    parsed = urlparse(url)

    default_drop = {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "gclid", "fbclid", "mc_cid", "mc_eid",
        "hl", "lang", "locale",
    }

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    keys = [k for k, _ in query_pairs]

    keep_keys = set((policy.keep_keys if policy else []))
    drop_keys = set((policy.drop_keys if policy else [])) | default_drop
    default_mode = (policy.default if policy else "drop_unknown_keys")

    filtered: List[Tuple[str, str]] = []
    for k, v in query_pairs:
        if k in keep_keys:
            filtered.append((k, v))
            continue
        if k in drop_keys:
            continue
        if default_mode == "keep_unknown_keys":
            filtered.append((k, v))

    new_query = urlencode(filtered, doseq=True)
    canonical = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, ""))
    return canonical


def _common_prefix_segments(paths: List[str]) -> List[str]:
    if not paths:
        return []
    split_paths = [[p for p in path.split("/") if p] for path in paths]
    prefix = split_paths[0]
    for parts in split_paths[1:]:
        max_len = min(len(prefix), len(parts))
        i = 0
        while i < max_len and prefix[i] == parts[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def _normalize_prefix(prefix: str) -> str:
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    if not prefix.endswith("/"):
        prefix = f"{prefix}/"
    return prefix


def build_branch_summaries(
    urls: List[UniverseUrl],
    max_branches: int = 80,
    base_url: Optional[str] = None,
    enriched_titles: Optional[Dict[str, Dict[str, Any]]] = None,
    pattern_analysis: Optional[PathPatternAnalysis] = None,
) -> List[Dict[str, Any]]:
    """
    Build a compact branch summary suitable for an LLM prompt.

    Groups by leading path segments to approximate a tree.

    If `base_url` is provided and has a non-root path (e.g. `/docs/`), we treat that as a
    "base prefix" and group by the next segment (e.g. `/docs/guides/`, `/docs/api-reference/`),
    plus we include a base-prefix summary node (e.g. `/docs/`) for rule validation/repair.

    Args:
        urls: List of UniverseUrl objects
        max_branches: Maximum branches to return
        base_url: Optional base URL for prefix detection
        enriched_titles: Optional dict mapping url_id -> {"title": str, ...}
        pattern_analysis: Optional PathPatternAnalysis for pattern context
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    enriched_titles = enriched_titles or {}

    base_segments: List[str] = []
    if base_url:
        base_path = urlparse(base_url).path or ""
        base_path_segments = [p for p in base_path.split("/") if p]
        if base_path_segments:
            base_segments = base_path_segments
    if not base_segments and urls:
        base_segments = _common_prefix_segments([u.path for u in urls])

    base_prefix = "/" + "/".join(base_segments) + "/" if base_segments else "/"
    group_depth = 1 if base_segments else 2

    if base_segments:
        # Add an explicit base-prefix node; this helps the LLM (and our validator)
        # propose broad allowlists like `/docs/` or `/gemini-api/docs/`.
        base_bucket = buckets.setdefault(base_prefix, {"path_prefix": base_prefix, "count": 0, "examples": []})
        for u in urls:
            if u.path.startswith(base_prefix.rstrip("/")):
                base_bucket["count"] += 1
                if len(base_bucket["examples"]) < 5:
                    base_bucket["examples"].append(u.url_id)

    for u in urls:
        parts = [p for p in u.path.split("/") if p]
        if base_segments and parts[: len(base_segments)] == base_segments:
            rel_parts = parts[len(base_segments) :]
            prefix_parts = base_segments + rel_parts[:group_depth]
        else:
            prefix_parts = parts[:2]
        prefix = "/" + "/".join(prefix_parts) + "/" if prefix_parts else "/"
        b = buckets.setdefault(prefix, {
            "path_prefix": prefix,
            "count": 0,
            "examples": [],
            "sample_titles": [],  # Enriched titles for this branch
        })
        b["count"] += 1
        if len(b["examples"]) < 5:
            b["examples"].append(u.url_id)
            # Add title if available
            if u.url_id in enriched_titles:
                title = enriched_titles[u.url_id].get("title", "")
                if title and len(b["sample_titles"]) < 3:
                    b["sample_titles"].append(title)

    # Sort by count desc
    branches = sorted(buckets.values(), key=lambda x: x["count"], reverse=True)
    result = branches[:max_branches]

    # Add pattern analysis summary to first branch if available
    if result and pattern_analysis:
        result[0]["pattern_summary"] = pattern_analysis.summary_for_prompt()

    return result


def _merge_prefixes(prefixes: List[str]) -> List[str]:
    out: List[str] = []
    for p in prefixes:
        if not p:
            continue
        if not p.startswith("/"):
            p = f"/{p}"
        if not p.endswith("/"):
            p = f"{p}/"
        if p not in out:
            out.append(p)
    return out


def apply_plan_rules(urls: List[UniverseUrl], rules: CrawlPlanRules) -> Dict[str, List[UniverseUrl]]:
    allow = _merge_prefixes(rules.allow_prefixes)
    deny = _merge_prefixes(rules.deny_prefixes)
    maybe = _merge_prefixes(rules.maybe_prefixes)

    keep_list: List[UniverseUrl] = []
    drop_list: List[UniverseUrl] = []
    maybe_list: List[UniverseUrl] = []

    for u in urls:
        path = u.path if u.path.startswith("/") else f"/{u.path}"

        if any(path.startswith(d) for d in deny):
            drop_list.append(u)
            continue

        if any(path.startswith(m) for m in maybe):
            maybe_list.append(u)
            continue

        if allow:
            if any(path.startswith(a) for a in allow):
                keep_list.append(u)
            else:
                drop_list.append(u)
        else:
            keep_list.append(u)

    return {"keep": keep_list, "drop": drop_list, "maybe": maybe_list}


@dataclass
class SafetyRailResult:
    """Result of safety rail checks."""
    passed: bool
    warnings: List[str]
    errors: List[str]
    coverage_percent: float
    drop_percent: float


@dataclass
class ExplorationResult:
    """Result from exploration pass."""
    site_structure: str
    candidate_patterns: List[str]
    confidence_scores: Dict[str, float]
    reasoning: str


class CrawlPlanner:
    """
    LLM-powered crawl planner with two-pass planning and safety rails.

    Pass 1 (Exploration): Analyze site structure, identify patterns
    Pass 2 (Refinement): Generate rules with coverage validation
    """

    # Safety rail thresholds
    MIN_COVERAGE_PERCENT = 10.0  # Minimum % of URLs to keep
    MAX_DROP_PERCENT = 90.0  # Maximum % of URLs to drop
    MAX_DENY_RULE_COVERAGE = 90.0  # Single deny rule can't drop >90% of URLs

    # URL Universe TTL (default 24 hours)
    DEFAULT_UNIVERSE_TTL_HOURS = 24

    def __init__(self, tenant_id: Optional[str] = None):
        self.config = get_config()
        self.db = get_db(tenant_id)
        self._genai = get_genai_client()

    def _is_universe_stale(
        self,
        source_id: str,
        ttl_hours: Optional[int] = None,
    ) -> bool:
        """
        Check if the URL universe for a source needs to be rebuilt.

        Args:
            source_id: Source ID to check
            ttl_hours: Override TTL (defaults to DEFAULT_UNIVERSE_TTL_HOURS)

        Returns:
            True if universe is stale or doesn't exist
        """
        ttl = ttl_hours or self.DEFAULT_UNIVERSE_TTL_HOURS
        return self.db.is_universe_stale(source_id, default_ttl_hours=ttl)

    def _load_cached_universe(self, source_id: str) -> Optional[List[UniverseUrl]]:
        """
        Load cached URL universe from database.

        Returns:
            List of UniverseUrl objects if cache exists, None otherwise
        """
        try:
            rows = self.db.list_url_universe(source_id)
            if not rows:
                return None

            universe = []
            for row in rows:
                universe.append(UniverseUrl(
                    url_id=row["url_id"],
                    url=row["url"],
                    canonical_url=row["canonical_url"],
                    path=row["path"],
                    depth=row["depth"],
                    query_keys=row.get("query_keys", []),
                ))
            return universe
        except Exception as e:
            logger.warning(f"Failed to load cached universe: {e}")
            return None

    async def enrich_universe_with_titles(
        self,
        universe: List[UniverseUrl],
        samples_per_branch: int = 3,
        max_total_samples: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Sample URLs from each branch and fetch their titles for better LLM context.

        Args:
            universe: List of UniverseUrl objects
            samples_per_branch: Max samples per path prefix
            max_total_samples: Total max samples to fetch

        Returns:
            Dict mapping url_id -> {"title": str, "path": str}
        """
        import httpx

        if not universe:
            return {}

        # Group by branch prefix (first 2 path segments)
        branches: Dict[str, List[UniverseUrl]] = {}
        for u in universe:
            parts = [p for p in u.path.split("/") if p]
            prefix = "/" + "/".join(parts[:2]) + "/" if len(parts) >= 2 else "/"
            if prefix not in branches:
                branches[prefix] = []
            branches[prefix].append(u)

        # Select samples from each branch
        samples: List[UniverseUrl] = []
        for prefix, urls in sorted(branches.items(), key=lambda x: -len(x[1])):
            # Take up to samples_per_branch from each branch
            branch_samples = urls[:samples_per_branch]
            samples.extend(branch_samples)
            if len(samples) >= max_total_samples:
                break

        samples = samples[:max_total_samples]
        logger.info(f"Sampling titles from {len(samples)} URLs across {len(branches)} branches")

        # Fetch titles concurrently
        enriched: Dict[str, Dict[str, Any]] = {}

        async with httpx.AsyncClient(
            timeout=5.0,
            follow_redirects=True,
            headers={"User-Agent": "ContextPilotBot/1.0"},
        ) as client:
            for u in samples:
                try:
                    response = await client.get(u.url)
                    if response.status_code == 200:
                        title = extract_title_from_html(response.text)
                        if title:
                            enriched[u.url_id] = {
                                "title": title,
                                "path": u.path,
                                "url": u.url,
                            }
                except Exception as e:
                    logger.debug(f"Failed to fetch title for {u.url}: {e}")

        logger.info(f"Enriched {len(enriched)} URLs with titles")
        return enriched

    def analyze_path_patterns(self, universe: List[UniverseUrl]) -> PathPatternAnalysis:
        """
        Analyze URL paths to detect common patterns (versions, languages, modules).

        Args:
            universe: List of UniverseUrl objects

        Returns:
            PathPatternAnalysis with detected patterns
        """
        detector = get_pattern_detector()
        urls = [u.url for u in universe]
        return detector.analyze_paths(urls)

    async def build_url_universe(
        self,
        source: Source,
        max_urls: Optional[int] = None,
        force_refresh: bool = False,
        ttl_hours: Optional[int] = None,
    ) -> List[UniverseUrl]:
        """
        Discover URLs for a source via sitemap/robots and store them.

        Uses TTL-based caching to avoid redundant discovery.

        Args:
            source: Source to build universe for
            max_urls: Maximum URLs to discover
            force_refresh: Force rebuild even if cache is valid
            ttl_hours: Override default TTL (24 hours)

        Returns:
            List of discovered URLs
        """
        if max_urls is None:
            max_urls = source.max_pages or 2000

        # Check cache unless force refresh requested
        if not force_refresh and not self._is_universe_stale(source.id, ttl_hours):
            cached = self._load_cached_universe(source.id)
            if cached:
                logger.info(f"Using cached URL universe for {source.base_url} ({len(cached)} URLs)")
                return cached
            logger.debug(f"Cache check passed but no URLs found, rebuilding universe")

        logger.info(f"Building URL universe for {source.base_url}")
        parser = get_sitemap_parser()
        exclude_paths = self._merge_exclude_paths(source.exclude_paths or [])

        # Use 4-tier discovery ladder for reliable URL discovery
        # Tier 1: Sitemap.xml (fastest)
        # Tier 2: robots.txt sitemaps
        # Tier 3: Firecrawl map API (JS-heavy sites)
        # Tier 4: BFS link crawling (last resort, most thorough)
        firecrawl_client = None
        try:
            from .crawl_manager import get_firecrawl_client
            firecrawl_client = get_firecrawl_client()
        except Exception:
            logger.debug("Firecrawl client not available for discovery")

        urls = await parser.discover_with_ladder(
            base_url=source.base_url,
            priority_paths=source.priority_paths or [],
            exclude_paths=exclude_paths,
            max_urls=max_urls,
            firecrawl_client=firecrawl_client,
        )

        # Canonicalize, dedupe
        seen: set[str] = set()
        universe: List[UniverseUrl] = []
        for url in urls:
            canonical = canonicalize_url(url)
            if canonical in seen:
                continue
            seen.add(canonical)

            parsed = urlparse(canonical)
            path = parsed.path or "/"
            depth = _path_depth(path)
            query_keys = sorted(set([k for k, _ in parse_qsl(parsed.query, keep_blank_values=True)]))

            uid = _url_id(canonical)
            universe.append(UniverseUrl(
                url_id=uid,
                url=url,
                canonical_url=canonical,
                path=path,
                depth=depth,
                query_keys=query_keys,
            ))

        # Determine discovery method used
        discovery_method = "ladder"  # Using 4-tier discovery ladder
        if not urls or urls == [source.base_url]:
            discovery_method = "fallback"
        elif len(universe) > 1:
            discovery_method = "ladder"  # Successfully discovered via ladder

        # Persist to DB (for later confirmation/crawl)
        try:
            self.db.replace_url_universe(source.id, [u.__dict__ for u in universe])
            # Store universe metadata for TTL caching
            self.db.set_universe_meta(
                source_id=source.id,
                url_count=len(universe),
                ttl_hours=ttl_hours or self.DEFAULT_UNIVERSE_TTL_HOURS,
                discovery_method=discovery_method,
            )
            logger.info(f"Persisted URL universe: {len(universe)} URLs (TTL: {ttl_hours or self.DEFAULT_UNIVERSE_TTL_HOURS}h)")
        except Exception as e:
            logger.warning(f"Failed to persist url universe: {e}")

        return universe

    def validate_rules(
        self,
        rules: CrawlPlanRules,
        branches: List[Dict[str, Any]],
    ) -> CrawlPlanRules:
        known = [_normalize_prefix(b.get("path_prefix") or "/") for b in branches if b.get("path_prefix")]
        known_counts = { _normalize_prefix(b.get("path_prefix") or "/"): b.get("count", 0) for b in branches }

        def _is_valid(prefix: str) -> bool:
            prefix = _normalize_prefix(prefix)
            return any(k.startswith(prefix) or prefix.startswith(k) for k in known)

        def _repair(prefix: str) -> Optional[str]:
            prefix = _normalize_prefix(prefix)
            segments = [p for p in prefix.split("/") if p]
            if not segments:
                return prefix
            candidates = []
            for k in known:
                k_segments = [p for p in k.split("/") if p]
                if len(segments) <= len(k_segments) and k_segments[-len(segments):] == segments:
                    candidates.append(k)
            if candidates:
                return max(candidates, key=lambda p: known_counts.get(p, 0))
            return None

        def _clean_list(prefixes: List[str], allow_empty: bool) -> List[str]:
            cleaned: List[str] = []
            for p in prefixes:
                if not p:
                    continue
                p_norm = _normalize_prefix(p)
                if _is_valid(p_norm):
                    cleaned.append(p_norm)
                    continue
                repaired = _repair(p_norm)
                if repaired and repaired not in cleaned:
                    cleaned.append(repaired)
            if not cleaned and not allow_empty:
                cleaned = [p for p in sorted(known, key=lambda x: known_counts.get(x, 0), reverse=True)[:3]]
            return cleaned

        allow_empty = not bool(rules.allow_prefixes)
        allow = _clean_list(rules.allow_prefixes, allow_empty=allow_empty)
        deny = _clean_list(rules.deny_prefixes, allow_empty=True)
        maybe = _clean_list(rules.maybe_prefixes, allow_empty=True)
        return CrawlPlanRules(
            allow_prefixes=allow,
            deny_prefixes=deny,
            maybe_prefixes=maybe,
            query_param_policy=rules.query_param_policy,
        )

    def propose_rules(self, base_url: str, branches: List[Dict[str, Any]]) -> CrawlPlanRules:
        """Ask Gemini to propose allow/deny prefixes and query param policy."""
        prompt = f"""You are curating a documentation crawl plan.

RULES:
- Use ONLY the provided branches and examples. Do not invent URLs.
- Output STRICT JSON only.
- Prefer allowlisting documentation-like surfaces and deny obvious noise.
- Avoid overly-broad deny rules that might delete real docs.

SITE: {base_url}

BRANCH SUMMARY (path_prefix, count, examples are url_ids):
{json.dumps(branches, ensure_ascii=False)}

Return JSON exactly in this shape:
{{
  "allow_prefixes": ["/docs/", "/api/"],
  "deny_prefixes": ["/blog/", "/pricing/"],
  "maybe_prefixes": ["/changelog/"],
  "query_param_policy": {{
    "keep_keys": ["version", "tab"],
    "drop_keys": ["utm_source", "gclid", "hl", "lang"],
    "default": "drop_unknown_keys"
  }}
}}
"""

        text = self._genai.generate(prompt, temperature=0.2, max_output_tokens=1200)
        try:
            stripped = _strip_code_fences(text)
            data = json.loads(stripped)
            logger.info(f"Successfully parsed crawl plan rules: {list(data.keys())}")
        except Exception as e:
            logger.warning(f"Failed to parse crawl plan rules JSON: {e}")
            logger.debug(f"Raw response (first 500 chars): {text[:500] if text else 'empty'}")
            data = {
                "allow_prefixes": ["/docs/", "/api/"],
                "deny_prefixes": self.config.default_exclude_paths,
                "maybe_prefixes": [],
                "query_param_policy": {
                    "keep_keys": [],
                    "drop_keys": ["utm_source", "utm_medium", "utm_campaign", "gclid", "fbclid", "hl", "lang"],
                    "default": "drop_unknown_keys",
                },
            }

        q = data.get("query_param_policy") or {}
        policy = QueryParamPolicy(
            keep_keys=list(q.get("keep_keys") or []),
            drop_keys=list(q.get("drop_keys") or []),
            default=str(q.get("default") or "drop_unknown_keys"),
        )
        return CrawlPlanRules(
            allow_prefixes=list(data.get("allow_prefixes") or []),
            deny_prefixes=list(data.get("deny_prefixes") or []),
            maybe_prefixes=list(data.get("maybe_prefixes") or []),
            query_param_policy=policy,
        )

    # =========================================================================
    # Two-Pass Planning (Phase 3)
    # =========================================================================

    def _explore_pass(
        self,
        base_url: str,
        branches: List[Dict[str, Any]],
        universe: List[UniverseUrl],
    ) -> ExplorationResult:
        """
        Pass 1: Explore site structure and identify candidate patterns.

        This is a broad analysis pass that:
        - Identifies the site's documentation structure
        - Finds candidate allow/deny patterns
        - Assigns confidence scores to patterns
        """
        # Build a compact representation of the URL structure
        path_patterns = {}
        for u in universe[:500]:  # Sample first 500 URLs
            parts = [p for p in u.path.split("/") if p]
            for i in range(1, min(4, len(parts) + 1)):
                prefix = "/" + "/".join(parts[:i]) + "/"
                if prefix not in path_patterns:
                    path_patterns[prefix] = 0
                path_patterns[prefix] += 1

        # Top patterns by frequency
        top_patterns = sorted(path_patterns.items(), key=lambda x: -x[1])[:20]

        prompt = f"""You are analyzing a documentation site to understand its structure.

SITE: {base_url}

URL PATH PATTERNS (prefix, count):
{json.dumps(top_patterns, ensure_ascii=False)}

BRANCH SUMMARY (path_prefix, count):
{json.dumps(branches[:30], ensure_ascii=False)}

Analyze this site and provide:
1. A brief description of the site structure
2. Which path patterns are likely documentation (vs blog, marketing, etc)
3. Confidence scores (0-1) for each documentation-like pattern

Return JSON:
{{
  "site_structure": "Brief description of site layout",
  "documentation_patterns": ["/docs/", "/api/", ...],
  "marketing_patterns": ["/blog/", "/pricing/", ...],
  "confidence_scores": {{"/docs/": 0.95, "/api/": 0.9, ...}},
  "reasoning": "Brief explanation of analysis"
}}
"""

        text = self._genai.generate(prompt, temperature=0.3, max_output_tokens=1200)
        try:
            data = json.loads(_strip_code_fences(text))
        except Exception:
            logger.warning("Failed to parse exploration JSON; using defaults")
            data = {
                "site_structure": "Unknown structure",
                "documentation_patterns": ["/docs/", "/api/"],
                "marketing_patterns": self.config.default_exclude_paths,
                "confidence_scores": {},
                "reasoning": "Fallback defaults",
            }

        candidate_patterns = list(data.get("documentation_patterns") or [])
        candidate_patterns.extend(data.get("marketing_patterns") or [])

        return ExplorationResult(
            site_structure=data.get("site_structure", ""),
            candidate_patterns=candidate_patterns,
            confidence_scores=data.get("confidence_scores", {}),
            reasoning=data.get("reasoning", ""),
        )

    def _refine_pass(
        self,
        base_url: str,
        branches: List[Dict[str, Any]],
        universe: List[UniverseUrl],
        exploration: ExplorationResult,
    ) -> CrawlPlanRules:
        """
        Pass 2: Refine rules based on exploration and coverage analysis.

        Uses exploration insights to generate more targeted rules,
        then validates coverage before returning.
        """
        prompt = f"""You are refining a crawl plan based on site analysis.

SITE: {base_url}

EXPLORATION FINDINGS:
- Structure: {exploration.site_structure}
- Documentation patterns: {exploration.candidate_patterns[:10]}
- Confidence scores: {json.dumps(exploration.confidence_scores)}
- Reasoning: {exploration.reasoning}

BRANCH SUMMARY:
{json.dumps(branches[:40], ensure_ascii=False)}

TOTAL URLS: {len(universe)}

Generate refined crawl rules that:
1. Maximize documentation coverage
2. Minimize noise (blogs, marketing, auth pages)
3. Use high-confidence patterns preferentially
4. NEVER create a deny rule that would drop >80% of URLs

Return JSON:
{{
  "allow_prefixes": ["/docs/", ...],
  "deny_prefixes": ["/blog/", ...],
  "maybe_prefixes": ["/changelog/", ...],
  "query_param_policy": {{
    "keep_keys": ["version", "tab"],
    "drop_keys": ["utm_source", "gclid", "hl", "lang"],
    "default": "drop_unknown_keys"
  }},
  "coverage_estimate": 0.85
}}
"""

        text = self._genai.generate(prompt, temperature=0.2, max_output_tokens=1200)
        try:
            data = json.loads(_strip_code_fences(text))
        except Exception:
            logger.warning("Failed to parse refinement JSON; using exploration patterns")
            # Fall back to using exploration patterns
            doc_patterns = [p for p in exploration.candidate_patterns
                           if exploration.confidence_scores.get(p, 0) > 0.5]
            data = {
                "allow_prefixes": doc_patterns[:5] or ["/docs/", "/api/"],
                "deny_prefixes": self.config.default_exclude_paths,
                "maybe_prefixes": [],
                "query_param_policy": {
                    "keep_keys": [],
                    "drop_keys": ["utm_source", "gclid", "hl", "lang"],
                    "default": "drop_unknown_keys",
                },
            }

        q = data.get("query_param_policy") or {}
        policy = QueryParamPolicy(
            keep_keys=list(q.get("keep_keys") or []),
            drop_keys=list(q.get("drop_keys") or []),
            default=str(q.get("default") or "drop_unknown_keys"),
        )

        return CrawlPlanRules(
            allow_prefixes=list(data.get("allow_prefixes") or []),
            deny_prefixes=list(data.get("deny_prefixes") or []),
            maybe_prefixes=list(data.get("maybe_prefixes") or []),
            query_param_policy=policy,
        )

    def two_pass_planning(
        self,
        base_url: str,
        branches: List[Dict[str, Any]],
        universe: List[UniverseUrl],
    ) -> Tuple[CrawlPlanRules, SafetyRailResult]:
        """
        Execute two-pass planning with safety rails.

        Pass 1: Explore site structure
        Pass 2: Generate refined rules
        Validation: Check coverage and safety rails

        Returns:
            Tuple of (rules, safety_result)
        """
        # Pass 1: Exploration
        logger.info(f"Two-pass planning for {base_url}: starting exploration pass")
        exploration = self._explore_pass(base_url, branches, universe)
        logger.debug(f"Exploration result: {exploration.site_structure}")

        # Pass 2: Refinement
        logger.info(f"Two-pass planning for {base_url}: starting refinement pass")
        rules = self._refine_pass(base_url, branches, universe, exploration)

        # Validate rules against branches
        rules = self.validate_rules(rules, branches)

        # Check safety rails
        safety = self.check_safety_rails(rules, universe)

        if not safety.passed:
            logger.warning(f"Safety rails failed for {base_url}: {safety.errors}")
            # Attempt to fix by relaxing deny rules
            if safety.drop_percent > self.MAX_DROP_PERCENT:
                rules = self._relax_deny_rules(rules, universe)
                safety = self.check_safety_rails(rules, universe)

        logger.info(
            f"Two-pass planning complete for {base_url}: "
            f"coverage={safety.coverage_percent:.1f}%, drop={safety.drop_percent:.1f}%"
        )

        return rules, safety

    def check_safety_rails(
        self,
        rules: CrawlPlanRules,
        universe: List[UniverseUrl],
    ) -> SafetyRailResult:
        """
        Check rules against safety rails.

        Safety rails prevent:
        - "Drop all" rules (deny prefix matching >90% of URLs)
        - Too low coverage (<10% of URLs kept)
        - Too high drop rate (>90% of URLs dropped)
        """
        warnings: List[str] = []
        errors: List[str] = []

        if not universe:
            return SafetyRailResult(
                passed=True,
                warnings=["Empty URL universe"],
                errors=[],
                coverage_percent=0.0,
                drop_percent=0.0,
            )

        # Apply rules to calculate coverage
        classified = apply_plan_rules(universe, rules)
        keep_count = len(classified["keep"])
        drop_count = len(classified["drop"])
        maybe_count = len(classified["maybe"])
        total = len(universe)

        coverage_percent = (keep_count / total) * 100 if total > 0 else 0
        drop_percent = (drop_count / total) * 100 if total > 0 else 0

        # Check minimum coverage
        if coverage_percent < self.MIN_COVERAGE_PERCENT:
            errors.append(
                f"Coverage too low: {coverage_percent:.1f}% < {self.MIN_COVERAGE_PERCENT}% minimum"
            )

        # Check maximum drop rate
        if drop_percent > self.MAX_DROP_PERCENT:
            errors.append(
                f"Drop rate too high: {drop_percent:.1f}% > {self.MAX_DROP_PERCENT}% maximum"
            )

        # Check individual deny rules for "drop all" scenarios
        for deny_prefix in rules.deny_prefixes:
            deny_norm = _normalize_prefix(deny_prefix)
            matching = sum(1 for u in universe if u.path.startswith(deny_norm.rstrip("/")))
            deny_coverage = (matching / total) * 100 if total > 0 else 0

            if deny_coverage > self.MAX_DENY_RULE_COVERAGE:
                errors.append(
                    f"Deny rule '{deny_prefix}' would drop {deny_coverage:.1f}% of URLs"
                )

        # Warn about maybe URLs (need manual review)
        if maybe_count > total * 0.2:
            warnings.append(
                f"High 'maybe' count: {maybe_count} URLs ({(maybe_count/total)*100:.1f}%) need review"
            )

        # Warn about empty allow list
        if not rules.allow_prefixes:
            warnings.append("No allow prefixes specified; all non-denied URLs will be kept")

        passed = len(errors) == 0

        return SafetyRailResult(
            passed=passed,
            warnings=warnings,
            errors=errors,
            coverage_percent=coverage_percent,
            drop_percent=drop_percent,
        )

    def _relax_deny_rules(
        self,
        rules: CrawlPlanRules,
        universe: List[UniverseUrl],
    ) -> CrawlPlanRules:
        """
        Relax deny rules that cause safety rail violations.

        Removes or narrows deny prefixes that drop too many URLs.
        """
        total = len(universe)
        if total == 0:
            return rules

        relaxed_deny: List[str] = []

        for deny_prefix in rules.deny_prefixes:
            deny_norm = _normalize_prefix(deny_prefix)
            matching = sum(1 for u in universe if u.path.startswith(deny_norm.rstrip("/")))
            deny_coverage = (matching / total) * 100 if total > 0 else 0

            if deny_coverage <= self.MAX_DENY_RULE_COVERAGE:
                relaxed_deny.append(deny_prefix)
            else:
                logger.warning(
                    f"Removing deny rule '{deny_prefix}' (would drop {deny_coverage:.1f}%)"
                )

        return CrawlPlanRules(
            allow_prefixes=rules.allow_prefixes,
            deny_prefixes=relaxed_deny,
            maybe_prefixes=rules.maybe_prefixes,
            query_param_policy=rules.query_param_policy,
        )

    def generate_report(
        self,
        base_url: str,
        branches: List[Dict[str, Any]],
        rules: CrawlPlanRules,
        counts: Dict[str, int],
        samples: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Ask Gemini to generate the 'ocean deep' report JSON."""
        prompt = f"""You are producing an 'ocean-deep' crawl planning report for a documentation site.

RULES:
- Output STRICT JSON only.
- Do not invent URLs. Use only url_ids provided in the inputs.
- Use the provided counts as authoritative.

SITE: {base_url}

BRANCH SUMMARY:
{json.dumps(branches, ensure_ascii=False)}

DERIVED RULES:
{json.dumps({
  "allow_prefixes": rules.allow_prefixes,
  "deny_prefixes": rules.deny_prefixes,
  "maybe_prefixes": rules.maybe_prefixes,
  "query_param_policy": {
    "keep_keys": rules.query_param_policy.keep_keys,
    "drop_keys": rules.query_param_policy.drop_keys,
    "default": rules.query_param_policy.default,
  },
}, ensure_ascii=False)}

CLASSIFICATION COUNTS:
{json.dumps(counts, ensure_ascii=False)}

REPRESENTATIVE URL IDS (samples):
{json.dumps(samples, ensure_ascii=False)}

Return JSON with keys:
- report_version
- summary (include total_urls_seen, recommended_strategy)
- tree (can reuse branch summary; keep it compact)
- decisions (keep/drop/maybe with prefix targets + why + evidence_examples as url_ids)
- derived_rules (mirror the derived rules above)
- crawl_phases (at least 1 phase)
- audit (risks + spot checks)
"""

        text = self._genai.generate(prompt, temperature=0.3, max_output_tokens=2000)
        try:
            return json.loads(_strip_code_fences(text))
        except Exception as e:
            logger.warning(f"Failed to parse report JSON: {e}")
            return {
                "report_version": "1.0",
                "summary": {
                    "total_urls_seen": counts.get("total_urls_seen", 0),
                    "recommended_strategy": "prefix_allowlist_with_exceptions",
                },
                "tree": branches[:20],
                "derived_rules": {
                    "allow_prefixes": rules.allow_prefixes,
                    "deny_prefixes": rules.deny_prefixes,
                    "maybe_prefixes": rules.maybe_prefixes,
                    "query_param_policy": {
                        "keep_keys": rules.query_param_policy.keep_keys,
                        "drop_keys": rules.query_param_policy.drop_keys,
                        "default": rules.query_param_policy.default,
                    },
                },
                "decisions": {},
                "crawl_phases": [],
                "audit": {},
            }

    def store_plan(
        self,
        source_id: str,
        rules: CrawlPlanRules,
        report: Dict[str, Any],
        total_urls_seen: int,
        kept_urls: int,
        dropped_urls: int,
        maybe_urls: int,
        status: str = "ready",
    ) -> Any:
        """Persist the plan."""
        return self.db.upsert_crawl_plan(
            source_id=source_id,
            status=status,
            rules={
                "allow_prefixes": rules.allow_prefixes,
                "deny_prefixes": rules.deny_prefixes,
                "maybe_prefixes": rules.maybe_prefixes,
                "query_param_policy": {
                    "keep_keys": rules.query_param_policy.keep_keys,
                    "drop_keys": rules.query_param_policy.drop_keys,
                    "default": rules.query_param_policy.default,
                },
            },
            report=report,
            total_urls_seen=total_urls_seen,
            kept_urls=kept_urls,
            dropped_urls=dropped_urls,
            maybe_urls=maybe_urls,
        )

    def _merge_exclude_paths(self, source_excludes: List[str]) -> List[str]:
        defaults = self.config.default_exclude_paths or []
        merged = []
        for item in defaults + source_excludes:
            if not item:
                continue
            if not item.startswith("/"):
                item = f"/{item}"
            if item not in merged:
                merged.append(item)
        return merged


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    t = (text or "").strip()
    if t.startswith("```"):
        # Remove opening fence (```json or just ```)
        t = re.sub(r"^```json\s*\n?", "", t)
        t = re.sub(r"^```\s*\n?", "", t)
        # Remove closing fence
        t = re.sub(r"\n?```\s*$", "", t)
    # Also handle case where response is wrapped in code block mid-text
    if "```json" in t:
        match = re.search(r"```json\s*\n?(.*?)\n?```", t, re.DOTALL)
        if match:
            t = match.group(1)
    return t.strip()
