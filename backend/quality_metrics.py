"""
ContextPilot Quality Metrics

Provides quality analysis and metrics for normalization and content processing:
- Source coverage: % of source chunks used
- Coherence scoring: LLM-judged coherence
- Code example and API reference counts
- Warnings and issue detection
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("contextpilot.quality_metrics")


@dataclass
class NormalizationQuality:
    """Quality metrics for a normalized document."""

    # Coverage metrics
    source_coverage: float = 0.0  # % of source chunks used (0-1)
    total_source_chunks: int = 0
    used_source_chunks: int = 0

    # Content metrics
    coherence_score: float = 0.0  # LLM-judged coherence (0-1)
    code_example_count: int = 0
    api_reference_count: int = 0
    heading_count: int = 0
    word_count: int = 0

    # Quality indicators
    has_introduction: bool = False
    has_conclusion: bool = False
    has_code_examples: bool = False
    has_api_references: bool = False

    # Issues
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if quality passes minimum thresholds."""
        return len(self.errors) == 0 and self.source_coverage >= 0.3

    @property
    def quality_score(self) -> float:
        """Compute overall quality score (0-1)."""
        scores = [
            self.source_coverage * 0.3,  # 30% weight
            self.coherence_score * 0.3,  # 30% weight
            (1.0 if self.has_code_examples else 0.5) * 0.2,  # 20% weight
            (1.0 if self.heading_count >= 3 else 0.5) * 0.1,  # 10% weight
            (1.0 if self.has_introduction else 0.5) * 0.1,  # 10% weight
        ]
        return sum(scores)


@dataclass
class LineageEntry:
    """Tracks which source chunks contributed to a normalized section."""

    section_title: str
    section_index: int
    source_chunk_ids: List[str] = field(default_factory=list)
    source_page_urls: List[str] = field(default_factory=list)
    contribution_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class NormalizationLineage:
    """Full lineage tracking for a normalized document."""

    normalized_doc_id: str
    url_prefix: str
    source_chunk_count: int
    used_chunk_count: int
    sections: List[LineageEntry] = field(default_factory=list)
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "normalized_doc_id": self.normalized_doc_id,
            "url_prefix": self.url_prefix,
            "source_chunk_count": self.source_chunk_count,
            "used_chunk_count": self.used_chunk_count,
            "sections": [
                {
                    "section_title": s.section_title,
                    "section_index": s.section_index,
                    "source_chunk_ids": s.source_chunk_ids,
                    "source_page_urls": s.source_page_urls,
                    "contribution_scores": s.contribution_scores,
                }
                for s in self.sections
            ],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationLineage":
        """Create from dictionary."""
        sections = [
            LineageEntry(
                section_title=s.get("section_title", ""),
                section_index=s.get("section_index", 0),
                source_chunk_ids=s.get("source_chunk_ids", []),
                source_page_urls=s.get("source_page_urls", []),
                contribution_scores=s.get("contribution_scores", {}),
            )
            for s in data.get("sections", [])
        ]
        return cls(
            normalized_doc_id=data.get("normalized_doc_id", ""),
            url_prefix=data.get("url_prefix", ""),
            source_chunk_count=data.get("source_chunk_count", 0),
            used_chunk_count=data.get("used_chunk_count", 0),
            sections=sections,
            created_at=data.get("created_at", 0.0),
        )


class QualityAnalyzer:
    """
    Analyzes quality of normalized documents.

    Checks:
    - Source coverage (how much of the input was used)
    - Structural quality (headings, code blocks)
    - Content coherence
    - Potential hallucination markers
    """

    # Hallucination markers - phrases that suggest LLM made things up
    HALLUCINATION_MARKERS = [
        r"I don't have access to",
        r"As an AI",
        r"I cannot",
        r"I'm not sure",
        r"I think",
        r"probably",
        r"might be",
        r"I assume",
        r"[Note: .*]",
        r"[Placeholder.*]",
        r"TODO:?",
        r"FIXME:?",
        r"XXX:?",
    ]

    # API reference patterns
    API_PATTERNS = [
        r"`[A-Z][a-zA-Z]*\([^)]*\)`",  # Function calls like `CreateUser()`
        r"`[a-z_]+\.[a-z_]+`",  # Method calls like `user.create`
        r"```(?:python|javascript|typescript|go|rust|java)",  # Language-specific code
        r"(?:GET|POST|PUT|DELETE|PATCH)\s+/[a-zA-Z/{}]+",  # REST endpoints
    ]

    def __init__(self, genai_client: Optional[Any] = None):
        """
        Initialize the quality analyzer.

        Args:
            genai_client: Optional GenAI client for coherence scoring
        """
        self._genai = genai_client

    def analyze_normalization(
        self,
        normalized_content: str,
        source_chunks: List[Dict[str, Any]],
        used_chunk_ids: Optional[List[str]] = None,
    ) -> NormalizationQuality:
        """
        Analyze the quality of a normalized document.

        Args:
            normalized_content: The normalized markdown content
            source_chunks: List of source chunks that were available
            used_chunk_ids: Optional list of chunk IDs that were actually used

        Returns:
            NormalizationQuality with detailed metrics
        """
        quality = NormalizationQuality()

        # Calculate source coverage
        quality.total_source_chunks = len(source_chunks)
        if used_chunk_ids:
            quality.used_source_chunks = len(used_chunk_ids)
        else:
            # Estimate based on content matching
            quality.used_source_chunks = self._estimate_used_chunks(
                normalized_content, source_chunks
            )

        if quality.total_source_chunks > 0:
            quality.source_coverage = (
                quality.used_source_chunks / quality.total_source_chunks
            )

        # Analyze structure
        quality.heading_count = len(re.findall(r"^#+\s+.+$", normalized_content, re.MULTILINE))
        quality.code_example_count = len(re.findall(r"```[\s\S]*?```", normalized_content))
        quality.word_count = len(normalized_content.split())

        # Check for API references
        api_matches = 0
        for pattern in self.API_PATTERNS:
            api_matches += len(re.findall(pattern, normalized_content))
        quality.api_reference_count = api_matches

        # Check content indicators
        quality.has_code_examples = quality.code_example_count > 0
        quality.has_api_references = quality.api_reference_count > 0
        quality.has_introduction = self._has_introduction(normalized_content)
        quality.has_conclusion = self._has_conclusion(normalized_content)

        # Check for hallucination markers
        hallucination_warnings = self._check_hallucination_markers(normalized_content)
        quality.warnings.extend(hallucination_warnings)

        # Check for common issues
        issues = self._check_common_issues(normalized_content)
        quality.warnings.extend(issues.get("warnings", []))
        quality.errors.extend(issues.get("errors", []))

        # Calculate coherence (if GenAI available)
        if self._genai:
            quality.coherence_score = self._calculate_coherence(normalized_content)
        else:
            # Heuristic coherence based on structure
            quality.coherence_score = self._heuristic_coherence(quality)

        return quality

    def _estimate_used_chunks(
        self,
        normalized_content: str,
        source_chunks: List[Dict[str, Any]],
    ) -> int:
        """Estimate how many source chunks were used based on content matching."""
        used = 0
        normalized_lower = normalized_content.lower()

        for chunk in source_chunks:
            chunk_content = chunk.get("content", "")
            if not chunk_content:
                continue

            # Check if significant portions of the chunk appear in normalized content
            words = chunk_content.split()
            if len(words) < 10:
                continue

            # Sample key phrases (every 5th word, groups of 3)
            sample_phrases = []
            for i in range(0, len(words) - 3, 5):
                phrase = " ".join(words[i : i + 3]).lower()
                sample_phrases.append(phrase)

            # Count how many sample phrases appear
            matches = sum(1 for p in sample_phrases if p in normalized_lower)
            if matches > len(sample_phrases) * 0.3:  # 30% threshold
                used += 1

        return used

    def _has_introduction(self, content: str) -> bool:
        """Check if content has an introduction section."""
        # Look for common intro patterns
        intro_patterns = [
            r"^#\s+.*\n\n[A-Z]",  # Title followed by paragraph
            r"## (?:Introduction|Overview|About|Getting Started)",
            r"^This (?:guide|document|page|section)",
        ]
        for pattern in intro_patterns:
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                return True
        return False

    def _has_conclusion(self, content: str) -> bool:
        """Check if content has a conclusion or summary."""
        conclusion_patterns = [
            r"## (?:Conclusion|Summary|Next Steps|What's Next)",
            r"(?:In summary|To summarize|In conclusion)",
        ]
        for pattern in conclusion_patterns:
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                return True
        return False

    def _check_hallucination_markers(self, content: str) -> List[str]:
        """Check for potential hallucination markers."""
        warnings = []
        for pattern in self.HALLUCINATION_MARKERS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                warnings.append(f"Potential hallucination marker: '{matches[0]}'")
        return warnings

    def _check_common_issues(self, content: str) -> Dict[str, List[str]]:
        """Check for common quality issues."""
        warnings = []
        errors = []

        # Check for broken code blocks
        code_block_count = content.count("```")
        if code_block_count % 2 != 0:
            errors.append("Unbalanced code blocks (missing closing ```)")

        # Check for empty sections
        empty_sections = re.findall(r"^##+ .+\n\n##", content, re.MULTILINE)
        if empty_sections:
            warnings.append(f"Found {len(empty_sections)} empty sections")

        # Check for duplicate headings
        headings = re.findall(r"^(#+\s+.+)$", content, re.MULTILINE)
        unique_headings = set(headings)
        if len(headings) != len(unique_headings):
            warnings.append("Found duplicate headings")

        # Check for very short content
        if len(content.split()) < 100:
            warnings.append("Content is very short (<100 words)")

        # Check for broken links
        broken_links = re.findall(r"\[([^\]]+)\]\(\s*\)", content)
        if broken_links:
            errors.append(f"Found {len(broken_links)} broken links (empty URLs)")

        # Check for incomplete sentences (ending with ...)
        incomplete = re.findall(r"\.\.\.\s*$", content, re.MULTILINE)
        if len(incomplete) > 2:
            warnings.append("Multiple incomplete sentences found")

        return {"warnings": warnings, "errors": errors}

    def _calculate_coherence(self, content: str) -> float:
        """Use LLM to calculate coherence score."""
        if not self._genai:
            return 0.5

        prompt = f"""Rate the coherence of this documentation on a scale of 0-10.
Consider:
- Logical flow between sections
- Consistent terminology
- Clear explanations
- Proper structure

Content:
{content[:3000]}

Return only a number between 0 and 10."""

        try:
            response = self._genai.generate(prompt, temperature=0.1, max_output_tokens=10)
            score = float(re.search(r"(\d+(?:\.\d+)?)", response).group(1))
            return min(1.0, score / 10.0)
        except Exception as e:
            logger.warning(f"Failed to calculate coherence: {e}")
            return 0.5

    def _heuristic_coherence(self, quality: NormalizationQuality) -> float:
        """Calculate heuristic coherence score based on structural metrics."""
        scores = []

        # Heading structure (0-1)
        if quality.heading_count >= 5:
            scores.append(1.0)
        elif quality.heading_count >= 3:
            scores.append(0.7)
        elif quality.heading_count >= 1:
            scores.append(0.5)
        else:
            scores.append(0.2)

        # Code examples (0-1)
        if quality.code_example_count >= 3:
            scores.append(1.0)
        elif quality.code_example_count >= 1:
            scores.append(0.7)
        else:
            scores.append(0.4)

        # Has intro/conclusion (0-1)
        intro_score = 1.0 if quality.has_introduction else 0.5
        conclusion_score = 1.0 if quality.has_conclusion else 0.5
        scores.append((intro_score + conclusion_score) / 2)

        # Word count (0-1)
        if quality.word_count >= 500:
            scores.append(1.0)
        elif quality.word_count >= 200:
            scores.append(0.7)
        else:
            scores.append(0.4)

        # No errors (0-1)
        if not quality.errors:
            scores.append(1.0)
        else:
            scores.append(0.3)

        return sum(scores) / len(scores)


def validate_normalized_output(content: str) -> Tuple[bool, List[str]]:
    """
    Validate normalized output for common issues.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for hallucination markers
    for pattern in QualityAnalyzer.HALLUCINATION_MARKERS:
        if re.search(pattern, content, re.IGNORECASE):
            issues.append(f"Contains potential hallucination marker: {pattern}")

    # Check code block validity
    code_blocks = re.findall(r"```(\w*)\n([\s\S]*?)```", content)
    for lang, code in code_blocks:
        if not code.strip():
            issues.append(f"Empty code block with language: {lang}")
        if "undefined" in code or "null" in code.lower():
            # Check for placeholder values
            if re.search(r"(undefined|null|TODO|FIXME)", code):
                issues.append("Code block may contain placeholder values")

    # Check heading structure
    headings = re.findall(r"^(#+)\s+(.+)$", content, re.MULTILINE)
    if headings:
        prev_level = 1
        for hash_marks, title in headings:
            level = len(hash_marks)
            if level > prev_level + 1:
                issues.append(f"Heading level jump: {prev_level} to {level} at '{title}'")
            prev_level = level

    # Check for duplicate sections
    heading_texts = [h[1].lower() for h in headings]
    duplicates = [h for h in set(heading_texts) if heading_texts.count(h) > 1]
    if duplicates:
        issues.append(f"Duplicate headings: {duplicates}")

    return len(issues) == 0, issues


def post_process_output(content: str) -> str:
    """
    Post-process normalized output to clean up common issues.

    Args:
        content: Raw normalized content

    Returns:
        Cleaned content
    """
    # Remove LLM artifacts
    content = re.sub(r"^\s*(?:Here(?:'s| is).*?:?\s*\n)", "", content)
    content = re.sub(r"\n*(?:I hope this helps|Let me know if.*?)$", "", content, flags=re.IGNORECASE)

    # Fix markdown formatting
    # Remove multiple consecutive blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Ensure code blocks have language hints
    content = re.sub(r"```\n([^`]+)```", lambda m: _infer_code_language(m.group(1)), content)

    # Fix heading spacing
    content = re.sub(r"(#+\s+.+)\n([^#\n])", r"\1\n\n\2", content)

    # Remove trailing whitespace
    content = "\n".join(line.rstrip() for line in content.split("\n"))

    # Ensure single trailing newline
    content = content.strip() + "\n"

    return content


def _infer_code_language(code: str) -> str:
    """Infer programming language from code content."""
    code_lower = code.lower().strip()

    if code_lower.startswith("def ") or "import " in code_lower:
        return f"```python\n{code}```"
    elif code_lower.startswith("func ") or "package main" in code_lower:
        return f"```go\n{code}```"
    elif "function" in code_lower or "const " in code_lower or "=>" in code:
        return f"```javascript\n{code}```"
    elif code_lower.startswith("curl ") or code_lower.startswith("$"):
        return f"```bash\n{code}```"
    elif code_lower.startswith("{") and code_lower.endswith("}"):
        return f"```json\n{code}```"
    else:
        return f"```\n{code}```"
