"""
ContextPilot Reranker

LLM-based reranking of search results for improved relevance.
Uses batch scoring to efficiently rerank multiple results in a single call.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .config import get_config
from .genai_client import get_genai_client

logger = logging.getLogger("contextpilot.reranker")


@dataclass
class RankedResult:
    """A search result with reranking metadata."""
    url: str
    page_url: Optional[str]
    title: str
    content: str
    original_score: float
    rerank_score: float
    relevance_explanation: str = ""
    position: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "page_url": self.page_url,
            "title": self.title,
            "content": self.content,
            "original_score": self.original_score,
            "rerank_score": self.rerank_score,
            "relevance_explanation": self.relevance_explanation,
            "position": self.position,
        }


@dataclass
class SearchResult:
    """Input search result for reranking."""
    url: str
    page_url: Optional[str]
    title: str
    content: str
    score: float


class Reranker:
    """
    LLM-based result reranker.

    Uses a single LLM call to batch-score multiple results,
    providing both relevance scores and explanations.
    """

    def __init__(self):
        self.config = get_config()
        self._model = None

        # Initialize Gemini model
        if self.config.has_gemini:
            try:
                self._model = get_genai_client()
                logger.info("Reranker initialized with Gemini")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini for reranking: {e}")

    @property
    def is_ready(self) -> bool:
        """Check if LLM is available for reranking."""
        return self._model is not None

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10,
    ) -> List[RankedResult]:
        """
        Rerank search results based on relevance to the query.

        Args:
            query: The user's search query
            results: List of search results to rerank
            top_k: Number of top results to return

        Returns:
            List of RankedResult sorted by rerank_score descending
        """
        if not results:
            return []

        # If LLM not available or few results, use original scores
        if not self._model or len(results) <= 3:
            return self._fallback_rerank(results, top_k)

        try:
            return await self._llm_rerank(query, results, top_k)
        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}")
            return self._fallback_rerank(results, top_k)

    async def _llm_rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[RankedResult]:
        """Rerank using LLM batch scoring."""
        # Limit to reasonable number for LLM context
        candidates = results[:20]

        prompt = self._build_prompt(query, candidates)

        text = self._model.generate(prompt, temperature=0.2, max_output_tokens=1500)
        ranked = self._parse_response(candidates, text)

        # Sort by rerank score and return top_k
        ranked.sort(key=lambda r: r.rerank_score, reverse=True)
        return ranked[:top_k]

    def _build_prompt(self, query: str, results: List[SearchResult]) -> str:
        """Build the LLM prompt for batch reranking."""
        results_text = ""
        for i, r in enumerate(results):
            # Truncate content to save tokens
            content_preview = r.content[:500] + "..." if len(r.content) > 500 else r.content
            results_text += f"""
[{i}] Title: {r.title}
URL: {r.page_url or r.url}
Content: {content_preview}
---"""

        return f"""You are a search result reranker. Score how relevant each result is to the user's query.

Query: "{query}"

Results to score:
{results_text}

For each result, provide a relevance score from 0.0 to 1.0 and a brief explanation.

Respond with a JSON array:
[
  {{"index": 0, "score": 0.95, "reason": "Directly answers the query about X"}},
  {{"index": 1, "score": 0.7, "reason": "Related but focuses on Y instead"}},
  ...
]

Scoring guidelines:
- 0.9-1.0: Directly answers the query, highly relevant
- 0.7-0.89: Very relevant, addresses the core question
- 0.5-0.69: Somewhat relevant, related topic
- 0.3-0.49: Tangentially related
- 0.0-0.29: Not relevant to the query

Respond ONLY with the JSON array, no other text."""

    def _parse_response(
        self,
        results: List[SearchResult],
        response_text: str,
    ) -> List[RankedResult]:
        """Parse LLM response into RankedResults."""
        ranked: List[RankedResult] = []

        try:
            # Clean up response
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"```json?\s*", "", cleaned)
                cleaned = re.sub(r"```\s*$", "", cleaned)

            scores_data = json.loads(cleaned)

            # Create a map of index to score data
            score_map: Dict[int, Dict] = {}
            for item in scores_data:
                idx = item.get("index", -1)
                if 0 <= idx < len(results):
                    score_map[idx] = item

            # Create ranked results
            for i, result in enumerate(results):
                score_data = score_map.get(i, {"score": result.score, "reason": ""})
                ranked.append(RankedResult(
                    url=result.url,
                    page_url=result.page_url,
                    title=result.title,
                    content=result.content,
                    original_score=result.score,
                    rerank_score=float(score_data.get("score", result.score)),
                    relevance_explanation=score_data.get("reason", ""),
                    position=i,
                ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reranker response: {e}")
            # Fall back to original scores
            return self._fallback_rerank(results, len(results))

        return ranked

    def _fallback_rerank(
        self,
        results: List[SearchResult],
        top_k: int,
    ) -> List[RankedResult]:
        """Fallback: use original scores without LLM reranking."""
        ranked = [
            RankedResult(
                url=r.url,
                page_url=r.page_url,
                title=r.title,
                content=r.content,
                original_score=r.score,
                rerank_score=r.score,  # Use original score
                position=i,
            )
            for i, r in enumerate(results)
        ]
        ranked.sort(key=lambda r: r.rerank_score, reverse=True)
        return ranked[:top_k]


# Singleton instance
_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """Get the reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
