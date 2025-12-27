"""
ContextPilot Path Patterns

Regex-based pattern extraction for URL paths:
- Detect version patterns: /v1/, /v2.0/, /latest/
- Detect language patterns: /en/, /ja/, /zh-cn/
- Detect module patterns: /api/, /guides/, /reference/
- Detect dated patterns: /2024/, /2024-01/
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("contextpilot.path_patterns")


@dataclass
class PathPatternMatch:
    """A detected pattern in a URL path."""
    pattern_type: str  # "version", "language", "module", "date"
    pattern: str  # The regex pattern
    value: str  # The matched value (e.g., "v2", "en", "api")
    position: int  # Position in path (0-indexed segment)
    count: int = 1  # How many URLs match this pattern


@dataclass
class PathPatternAnalysis:
    """Analysis of path patterns across a URL set."""
    patterns: List[PathPatternMatch] = field(default_factory=list)
    version_segments: List[str] = field(default_factory=list)  # e.g., ["v1", "v2", "latest"]
    language_segments: List[str] = field(default_factory=list)  # e.g., ["en", "ja", "zh-cn"]
    module_segments: List[str] = field(default_factory=list)  # e.g., ["api", "guides"]
    date_segments: List[str] = field(default_factory=list)  # e.g., ["2024", "2024-01"]
    common_prefix: str = ""  # Common path prefix across all URLs
    depth_distribution: Dict[int, int] = field(default_factory=dict)  # depth -> count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM prompt."""
        return {
            "version_segments": self.version_segments,
            "language_segments": self.language_segments,
            "module_segments": self.module_segments,
            "date_segments": self.date_segments,
            "common_prefix": self.common_prefix,
            "depth_distribution": self.depth_distribution,
            "patterns": [
                {
                    "type": p.pattern_type,
                    "value": p.value,
                    "position": p.position,
                    "count": p.count,
                }
                for p in self.patterns[:20]  # Limit for prompt size
            ],
        }

    def summary_for_prompt(self) -> str:
        """Generate a concise summary for LLM prompts."""
        parts = []

        if self.version_segments:
            parts.append(f"Versions: {', '.join(self.version_segments[:5])}")
        if self.language_segments:
            parts.append(f"Languages: {', '.join(self.language_segments[:5])}")
        if self.module_segments:
            parts.append(f"Modules: {', '.join(self.module_segments[:10])}")
        if self.common_prefix:
            parts.append(f"Common prefix: {self.common_prefix}")

        return "; ".join(parts) if parts else "No significant patterns detected"


class PathPatternDetector:
    """
    Detects common patterns in URL paths.

    Patterns detected:
    - Versions: v1, v2.0, latest, stable, beta
    - Languages: en, ja, zh-cn, de-de
    - Modules: api, guides, reference, tutorials
    - Dates: 2024, 2024-01, 2024/01
    """

    # Version patterns
    VERSION_PATTERNS = [
        (r"^v\d+(?:\.\d+)*$", "semver"),  # v1, v2.0, v1.2.3
        (r"^v\d+$", "major"),  # v1, v2
        (r"^latest$", "alias"),
        (r"^stable$", "alias"),
        (r"^beta$", "alias"),
        (r"^alpha$", "alias"),
        (r"^next$", "alias"),
        (r"^master$", "alias"),
        (r"^main$", "alias"),
        (r"^\d+\.\d+(?:\.\d+)?$", "bare_semver"),  # 1.0, 2.3.1
    ]

    # Language patterns (ISO 639-1 and common variants)
    LANGUAGE_PATTERNS = [
        r"^[a-z]{2}$",  # en, ja, de
        r"^[a-z]{2}-[a-z]{2}$",  # en-us, zh-cn
        r"^[a-z]{2}_[A-Z]{2}$",  # en_US, zh_CN
        r"^[a-z]{2}-[A-Z]{2}$",  # en-US, zh-CN
    ]

    # Common language codes to match
    KNOWN_LANGUAGES = {
        "en", "ja", "zh", "ko", "de", "fr", "es", "pt", "ru", "it",
        "en-us", "en-gb", "zh-cn", "zh-tw", "pt-br", "es-es", "es-mx",
    }

    # Module/section patterns
    MODULE_PATTERNS = [
        r"^api$",
        r"^docs$",
        r"^guide(?:s)?$",
        r"^tutorial(?:s)?$",
        r"^reference$",
        r"^manual$",
        r"^getting-started$",
        r"^quickstart$",
        r"^faq$",
        r"^example(?:s)?$",
        r"^sample(?:s)?$",
        r"^changelog$",
        r"^release(?:s)?$",
        r"^blog$",
        r"^learn$",
        r"^concepts?$",
        r"^cookbook$",
        r"^advanced$",
        r"^fundamentals$",
    ]

    # Date patterns
    DATE_PATTERNS = [
        (r"^\d{4}$", "year"),  # 2024
        (r"^\d{4}-\d{2}$", "year_month"),  # 2024-01
        (r"^\d{4}/\d{2}$", "year_month_slash"),  # 2024/01
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self._version_re = [(re.compile(p, re.IGNORECASE), t) for p, t in self.VERSION_PATTERNS]
        self._lang_re = [re.compile(p, re.IGNORECASE) for p in self.LANGUAGE_PATTERNS]
        self._module_re = [re.compile(p, re.IGNORECASE) for p in self.MODULE_PATTERNS]
        self._date_re = [(re.compile(p), t) for p, t in self.DATE_PATTERNS]

    def analyze_paths(self, urls: List[str]) -> PathPatternAnalysis:
        """
        Analyze a list of URLs for common patterns.

        Args:
            urls: List of URLs to analyze

        Returns:
            PathPatternAnalysis with detected patterns
        """
        analysis = PathPatternAnalysis()

        if not urls:
            return analysis

        # Extract paths and segments
        all_segments: List[List[str]] = []
        segment_counts: Dict[Tuple[int, str], int] = {}  # (position, value) -> count

        for url in urls:
            parsed = urlparse(url)
            path = parsed.path.strip("/")
            if not path:
                continue

            segments = [s for s in path.split("/") if s]
            all_segments.append(segments)

            # Track depth distribution
            depth = len(segments)
            analysis.depth_distribution[depth] = analysis.depth_distribution.get(depth, 0) + 1

            # Count segment occurrences by position
            for i, seg in enumerate(segments):
                key = (i, seg.lower())
                segment_counts[key] = segment_counts.get(key, 0) + 1

        # Find common prefix
        if all_segments:
            analysis.common_prefix = self._find_common_prefix(all_segments)

        # Detect patterns
        version_set: Set[str] = set()
        language_set: Set[str] = set()
        module_set: Set[str] = set()
        date_set: Set[str] = set()
        pattern_matches: Dict[str, PathPatternMatch] = {}

        for (pos, seg), count in segment_counts.items():
            # Only consider segments that appear in >5% of URLs or >3 times
            if count < 3 and count < len(urls) * 0.05:
                continue

            # Check for version
            if self._is_version(seg):
                version_set.add(seg)
                key = f"version:{pos}:{seg}"
                pattern_matches[key] = PathPatternMatch(
                    pattern_type="version",
                    pattern=r"/v\d+(?:\.\d+)*/",
                    value=seg,
                    position=pos,
                    count=count,
                )

            # Check for language
            if self._is_language(seg):
                language_set.add(seg)
                key = f"language:{pos}:{seg}"
                pattern_matches[key] = PathPatternMatch(
                    pattern_type="language",
                    pattern=r"/[a-z]{2}(?:-[a-z]{2})?/",
                    value=seg,
                    position=pos,
                    count=count,
                )

            # Check for module
            if self._is_module(seg):
                module_set.add(seg)
                key = f"module:{pos}:{seg}"
                pattern_matches[key] = PathPatternMatch(
                    pattern_type="module",
                    pattern=f"/{seg}/",
                    value=seg,
                    position=pos,
                    count=count,
                )

            # Check for date
            if self._is_date(seg):
                date_set.add(seg)
                key = f"date:{pos}:{seg}"
                pattern_matches[key] = PathPatternMatch(
                    pattern_type="date",
                    pattern=r"/\d{4}(?:-\d{2})?/",
                    value=seg,
                    position=pos,
                    count=count,
                )

        # Sort and assign results
        analysis.version_segments = sorted(version_set)
        analysis.language_segments = sorted(language_set)
        analysis.module_segments = sorted(module_set)
        analysis.date_segments = sorted(date_set)
        analysis.patterns = sorted(
            pattern_matches.values(),
            key=lambda p: (-p.count, p.position),
        )

        return analysis

    def _is_version(self, segment: str) -> bool:
        """Check if segment looks like a version."""
        seg_lower = segment.lower()
        for pattern, _ in self._version_re:
            if pattern.match(seg_lower):
                return True
        return False

    def _is_language(self, segment: str) -> bool:
        """Check if segment looks like a language code."""
        seg_lower = segment.lower()

        # Check known languages first
        if seg_lower in self.KNOWN_LANGUAGES:
            return True

        # Check patterns
        for pattern in self._lang_re:
            if pattern.match(seg_lower):
                return True

        return False

    def _is_module(self, segment: str) -> bool:
        """Check if segment looks like a module/section name."""
        seg_lower = segment.lower()
        for pattern in self._module_re:
            if pattern.match(seg_lower):
                return True
        return False

    def _is_date(self, segment: str) -> bool:
        """Check if segment looks like a date."""
        for pattern, _ in self._date_re:
            if pattern.match(segment):
                return True
        return False

    def _find_common_prefix(self, segments_list: List[List[str]]) -> str:
        """Find the common prefix across all segment lists."""
        if not segments_list:
            return ""

        # Find minimum length
        min_len = min(len(s) for s in segments_list)
        if min_len == 0:
            return ""

        # Find common prefix length
        prefix_len = 0
        for i in range(min_len):
            first_seg = segments_list[0][i].lower()
            if all(s[i].lower() == first_seg for s in segments_list):
                prefix_len = i + 1
            else:
                break

        if prefix_len == 0:
            return ""

        return "/" + "/".join(segments_list[0][:prefix_len]) + "/"


def extract_title_from_html(html: str) -> Optional[str]:
    """
    Extract title from HTML content.

    Tries in order:
    1. <title> tag
    2. <h1> tag
    3. og:title meta tag
    """
    if not html:
        return None

    # Try <title>
    title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        if title:
            return _clean_title(title)

    # Try <h1>
    h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE | re.DOTALL)
    if h1_match:
        title = re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()
        if title:
            return _clean_title(title)

    # Try og:title
    og_match = re.search(r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if og_match:
        title = og_match.group(1).strip()
        if title:
            return _clean_title(title)

    return None


def _clean_title(title: str) -> str:
    """Clean up extracted title."""
    # Remove common suffixes
    suffixes = [
        " | Documentation",
        " - Documentation",
        " — Documentation",
        " | Docs",
        " - Docs",
        " | API",
        " - API",
        " | Reference",
        " - Reference",
    ]
    for suffix in suffixes:
        if title.endswith(suffix):
            title = title[:-len(suffix)]

    # Decode HTML entities
    title = re.sub(r"&amp;", "&", title)
    title = re.sub(r"&lt;", "<", title)
    title = re.sub(r"&gt;", ">", title)
    title = re.sub(r"&quot;", '"', title)
    title = re.sub(r"&#39;", "'", title)

    return title.strip()


# Singleton instance
_detector: Optional[PathPatternDetector] = None


def get_pattern_detector() -> PathPatternDetector:
    """Get the singleton path pattern detector."""
    global _detector
    if _detector is None:
        _detector = PathPatternDetector()
    return _detector
