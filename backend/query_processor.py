"""
ContextPilot Query Processor

LLM-powered query understanding and rewriting for improved search accuracy.
Extracts intent, entities, and generates optimized search queries.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from .config import get_config
from .genai_client import get_genai_client

logger = logging.getLogger("contextpilot.query")


class QueryIntent(str, Enum):
    """Types of user query intents."""
    HOW_TO = "how_to"           # How to do something
    REFERENCE = "reference"      # API reference lookup
    CONCEPTUAL = "conceptual"    # Understanding a concept
    TROUBLESHOOT = "troubleshoot"  # Fixing an error/issue
    COMPARISON = "comparison"    # Comparing options
    EXAMPLE = "example"          # Looking for code examples
    GENERAL = "general"          # General information


@dataclass
class ProcessedQuery:
    """Result of query processing."""
    original_query: str
    rewritten_query: str
    intent: QueryIntent
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    url_filter: Optional[str] = None
    source_hints: List[str] = field(default_factory=list)
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "intent": self.intent.value,
            "entities": self.entities,
            "keywords": self.keywords,
            "url_filter": self.url_filter,
            "source_hints": self.source_hints,
            "confidence": self.confidence,
        }


class QueryProcessor:
    """
    Processes and enhances search queries using LLM.

    Features:
    - Query rewriting for better retrieval
    - Intent classification
    - Entity extraction (APIs, libraries, etc.)
    - Keyword expansion
    """

    def __init__(self):
        self.config = get_config()
        self._model = None

        # Initialize Gemini model
        if self.config.has_gemini:
            try:
                self._model = get_genai_client()
                logger.info("Query processor initialized with Gemini")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini for query processing: {e}")

    @property
    def is_ready(self) -> bool:
        """Check if LLM is available for query processing."""
        return self._model is not None

    async def process(self, query: str, context: Optional[str] = None) -> ProcessedQuery:
        """
        Process a search query to extract intent and improve it.

        Args:
            query: The user's search query
            context: Optional context about what they're searching for

        Returns:
            ProcessedQuery with enhanced query and metadata
        """
        # If LLM not available, use fallback processing
        if not self._model:
            return self._fallback_process(query)

        try:
            return await self._llm_process(query, context)
        except Exception as e:
            logger.warning(f"LLM query processing failed: {e}")
            return self._fallback_process(query)

    async def _llm_process(self, query: str, context: Optional[str] = None) -> ProcessedQuery:
        """Process query using LLM."""
        prompt = self._build_prompt(query, context)

        text = self._model.generate(prompt, temperature=0.3, max_output_tokens=500)
        return self._parse_response(query, text)

    def _build_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Build the LLM prompt for query processing."""
        context_section = f"\nContext: {context}" if context else ""

        return f"""Analyze this documentation search query and provide structured information to improve search results.

Query: "{query}"{context_section}

Respond with a JSON object containing:
{{
  "rewritten_query": "Improved/expanded version of the query for better retrieval",
  "intent": "one of: how_to, reference, conceptual, troubleshoot, comparison, example, general",
  "entities": ["list", "of", "specific", "APIs", "libraries", "tools", "mentioned"],
  "keywords": ["additional", "relevant", "search", "terms"],
  "source_hints": ["hints about which documentation sources might be relevant"],
  "confidence": 0.0 to 1.0
}}

Rules:
- Expand abbreviations (e.g., "auth" -> "authentication")
- Add related technical terms
- Identify specific APIs, SDKs, or tools mentioned
- Keep the rewritten query concise but more specific
- Source hints should be general (e.g., "OpenAI API", "React docs")

Respond ONLY with the JSON object, no other text."""

    def _parse_response(self, original_query: str, response_text: str) -> ProcessedQuery:
        """Parse LLM response into ProcessedQuery."""
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"```json?\s*", "", cleaned)
                cleaned = re.sub(r"```\s*$", "", cleaned)

            data = json.loads(cleaned)

            # Map intent string to enum
            intent_str = data.get("intent", "general").lower()
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                intent = QueryIntent.GENERAL

            return ProcessedQuery(
                original_query=original_query,
                rewritten_query=data.get("rewritten_query", original_query),
                intent=intent,
                entities=data.get("entities", []),
                keywords=data.get("keywords", []),
                source_hints=data.get("source_hints", []),
                confidence=data.get("confidence", 0.8),
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse query processor response: {e}")
            return self._fallback_process(original_query)

    def _fallback_process(self, query: str) -> ProcessedQuery:
        """
        Fallback query processing without LLM.
        Uses heuristics and pattern matching.
        """
        query_lower = query.lower()

        # Detect intent from keywords
        intent = QueryIntent.GENERAL
        if any(w in query_lower for w in ["how to", "how do i", "how can i"]):
            intent = QueryIntent.HOW_TO
        elif any(w in query_lower for w in ["error", "fix", "issue", "problem", "bug", "failed"]):
            intent = QueryIntent.TROUBLESHOOT
        elif any(w in query_lower for w in ["example", "sample", "code"]):
            intent = QueryIntent.EXAMPLE
        elif any(w in query_lower for w in ["what is", "explain", "understand", "concept"]):
            intent = QueryIntent.CONCEPTUAL
        elif any(w in query_lower for w in ["vs", "versus", "compare", "difference", "or"]):
            intent = QueryIntent.COMPARISON
        elif any(w in query_lower for w in ["api", "function", "method", "parameter", "reference"]):
            intent = QueryIntent.REFERENCE

        # Extract potential entities (capitalized words, technical terms)
        entities = []
        # Look for common API/library patterns
        api_patterns = [
            r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b",  # CamelCase
            r"\b([a-z]+(?:-[a-z]+)+)\b",           # kebab-case libraries
            r"\b([a-z]+(?:_[a-z]+)+)\b",           # snake_case
        ]
        for pattern in api_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches[:5])  # Limit to 5

        # Extract keywords (remove common words)
        stop_words = {"the", "a", "an", "is", "are", "how", "to", "do", "i", "can", "what", "in", "for", "with"}
        keywords = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]

        return ProcessedQuery(
            original_query=query,
            rewritten_query=query,  # No rewriting without LLM
            intent=intent,
            entities=list(set(entities)),
            keywords=keywords[:10],
            confidence=0.5,  # Lower confidence for fallback
        )

    def expand_query(self, processed: ProcessedQuery) -> str:
        """
        Create an expanded query string for vector search.
        Combines rewritten query with keywords.
        """
        parts = [processed.rewritten_query]

        # Add unique keywords not already in query
        query_lower = processed.rewritten_query.lower()
        for kw in processed.keywords:
            if kw.lower() not in query_lower:
                parts.append(kw)

        return " ".join(parts)


# Singleton instance
_query_processor: Optional[QueryProcessor] = None


def get_query_processor() -> QueryProcessor:
    """Get the query processor instance."""
    global _query_processor
    if _query_processor is None:
        _query_processor = QueryProcessor()
    return _query_processor
