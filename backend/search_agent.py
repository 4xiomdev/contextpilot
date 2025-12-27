"""
ContextPilot Search Agent

Agentic search system with two modes:
- Quick: Fast retrieval with query rewriting and reranking
- Deep: Full synthesis with citations, confidence scores, and follow-ups
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from .config import get_config
from .embed_manager import get_embed_manager, SearchResult as EmbedSearchResult
from .query_processor import get_query_processor, ProcessedQuery, QueryIntent
from .reranker import get_reranker, SearchResult as RerankerInput, RankedResult
from .tenant_context import get_tenant_id
from .genai_client import get_genai_client

logger = logging.getLogger("contextpilot.search_agent")


class SearchMode(str, Enum):
    """Search mode selection."""
    QUICK = "quick"
    DEEP = "deep"


@dataclass
class Citation:
    """A citation reference in the synthesized answer."""
    index: int
    url: str
    title: str
    excerpt: str


@dataclass
class QuickSearchResult:
    """Result from quick search mode."""
    query: str
    processed_query: ProcessedQuery
    results: List[RankedResult]
    total_results: int
    search_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "processed_query": self.processed_query.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "search_time_ms": self.search_time_ms,
        }


@dataclass
class DeepSearchResult:
    """Result from deep search mode with synthesis."""
    query: str
    processed_query: ProcessedQuery
    answer: str
    citations: List[Citation]
    confidence: float
    follow_up_questions: List[str]
    sources_used: int
    search_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "processed_query": self.processed_query.to_dict(),
            "answer": self.answer,
            "citations": [
                {
                    "index": c.index,
                    "url": c.url,
                    "title": c.title,
                    "excerpt": c.excerpt,
                }
                for c in self.citations
            ],
            "confidence": self.confidence,
            "follow_up_questions": self.follow_up_questions,
            "sources_used": self.sources_used,
            "search_time_ms": self.search_time_ms,
        }


class SearchAgent:
    """
    Agentic documentation search with quick and deep modes.

    Quick mode:
    1. Process query (rewrite, extract intent)
    2. Vector search
    3. Rerank results
    4. Return top results

    Deep mode:
    1. Process query
    2. Extended vector search (more results)
    3. Rerank
    4. Synthesize answer with citations
    5. Generate follow-up questions
    """

    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id or get_tenant_id()
        self.config = get_config()
        self.embed_manager = get_embed_manager(self.tenant_id)
        self.query_processor = get_query_processor()
        self.reranker = get_reranker()
        self._model = None

        # Initialize Gemini model for synthesis
        if self.config.has_gemini:
            try:
                self._model = get_genai_client()
                logger.info("Search agent initialized with Gemini")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini for search agent: {e}")

    async def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.QUICK,
        limit: int = 10,
        url_filter: str = "",
    ) -> QuickSearchResult | DeepSearchResult:
        """
        Execute a search in the specified mode.

        Args:
            query: User's search query
            mode: QUICK or DEEP
            limit: Number of results (quick) or sources to consider (deep)
            url_filter: Optional URL prefix filter

        Returns:
            QuickSearchResult or DeepSearchResult based on mode
        """
        if mode == SearchMode.DEEP:
            return await self.deep_search(query, max_sources=limit, url_filter=url_filter)
        else:
            return await self.quick_search(query, limit=limit, url_filter=url_filter)

    async def quick_search(
        self,
        query: str,
        limit: int = 10,
        url_filter: str = "",
    ) -> QuickSearchResult:
        """
        Fast search with query enhancement and reranking.

        Args:
            query: User's search query
            limit: Maximum results to return
            url_filter: Optional URL prefix filter

        Returns:
            QuickSearchResult with ranked results
        """
        start_time = time.time()

        # Step 1: Process query
        processed = await self.query_processor.process(query)
        search_query = self.query_processor.expand_query(processed)

        # Step 2: Vector search
        raw_results = self.embed_manager.search(
            query=search_query,
            limit=limit * 2,  # Get more for reranking
            url_filter=url_filter or processed.url_filter or "",
        )

        # Step 3: Rerank
        reranker_input = [
            RerankerInput(
                url=r.url,
                page_url=r.page_url,
                title=r.title,
                content=r.content,
                score=r.score,
            )
            for r in raw_results
        ]
        ranked = await self.reranker.rerank(query, reranker_input, top_k=limit)

        elapsed_ms = (time.time() - start_time) * 1000

        return QuickSearchResult(
            query=query,
            processed_query=processed,
            results=ranked,
            total_results=len(raw_results),
            search_time_ms=elapsed_ms,
        )

    async def deep_search(
        self,
        query: str,
        max_sources: int = 20,
        url_filter: str = "",
    ) -> DeepSearchResult:
        """
        Deep search with synthesis, citations, and follow-ups.

        Args:
            query: User's search query
            max_sources: Maximum sources to consider for synthesis
            url_filter: Optional URL prefix filter

        Returns:
            DeepSearchResult with synthesized answer
        """
        start_time = time.time()

        # Step 1: Process query
        processed = await self.query_processor.process(query)
        search_query = self.query_processor.expand_query(processed)

        # Step 2: Extended vector search
        raw_results = self.embed_manager.search(
            query=search_query,
            limit=max_sources * 2,
            url_filter=url_filter or processed.url_filter or "",
        )

        # Also search normalized docs
        normalized_results = self.embed_manager.search_normalized(
            query=search_query,
            limit=max_sources,
            url_filter=url_filter or processed.url_filter or "",
        )

        # Combine results (prefer normalized)
        all_results = list(normalized_results) + list(raw_results)

        # Step 3: Rerank
        reranker_input = [
            RerankerInput(
                url=r.url,
                page_url=r.page_url,
                title=r.title,
                content=r.content,
                score=r.score,
            )
            for r in all_results
        ]
        ranked = await self.reranker.rerank(query, reranker_input, top_k=max_sources)

        # Step 4: Synthesize answer
        if self._model and ranked:
            synthesis = await self._synthesize(query, processed, ranked)
        else:
            # Fallback: no synthesis
            synthesis = {
                "answer": "Unable to synthesize answer. Please review the search results.",
                "citations": [],
                "confidence": 0.3,
                "follow_ups": [],
            }

        elapsed_ms = (time.time() - start_time) * 1000

        return DeepSearchResult(
            query=query,
            processed_query=processed,
            answer=synthesis["answer"],
            citations=synthesis["citations"],
            confidence=synthesis["confidence"],
            follow_up_questions=synthesis["follow_ups"],
            sources_used=len(ranked),
            search_time_ms=elapsed_ms,
        )

    async def _synthesize(
        self,
        query: str,
        processed: ProcessedQuery,
        sources: List[RankedResult],
    ) -> Dict[str, Any]:
        """
        Synthesize an answer from search results using LLM.

        Returns dict with: answer, citations, confidence, follow_ups
        """
        # Build context from sources
        context_parts = []
        for i, source in enumerate(sources[:15]):  # Limit to top 15
            context_parts.append(f"""[{i + 1}] {source.title}
URL: {source.page_url or source.url}
{source.content[:1500]}
---""")

        context = "\n".join(context_parts)

        # Build synthesis prompt
        intent_guidance = self._get_intent_guidance(processed.intent)

        prompt = f"""You are a documentation expert. Answer the user's question using ONLY the provided documentation sources.

Question: "{query}"

{intent_guidance}

Documentation Sources:
{context}

Instructions:
1. Provide a comprehensive, accurate answer based on the sources
2. Include inline citations like [1], [2] referencing source numbers
3. Be specific and include code examples when relevant
4. If the sources don't fully answer the question, say what's missing

After your answer, provide:
- Confidence level (0.0-1.0) based on how well sources answer the question
- 2-3 follow-up questions the user might want to ask

Format your response as:

ANSWER:
[Your detailed answer with [1] citations]

CONFIDENCE: 0.X

FOLLOW_UPS:
- Question 1?
- Question 2?
- Question 3?"""

        try:
            text = self._model.generate(prompt, temperature=0.4, max_output_tokens=2000)
            return self._parse_synthesis(text, sources)

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "answer": f"Error synthesizing answer: {str(e)}",
                "citations": [],
                "confidence": 0.0,
                "follow_ups": [],
            }

    def _get_intent_guidance(self, intent: QueryIntent) -> str:
        """Get intent-specific guidance for synthesis."""
        guidance = {
            QueryIntent.HOW_TO: "Focus on step-by-step instructions and practical implementation details.",
            QueryIntent.REFERENCE: "Provide precise API signatures, parameters, return types, and usage notes.",
            QueryIntent.CONCEPTUAL: "Explain the concept clearly, including how it relates to the broader system.",
            QueryIntent.TROUBLESHOOT: "Focus on identifying the cause and providing specific solutions with code fixes.",
            QueryIntent.COMPARISON: "Compare the options objectively, highlighting pros/cons and use cases for each.",
            QueryIntent.EXAMPLE: "Provide working code examples with explanations of key parts.",
            QueryIntent.GENERAL: "Provide a balanced overview with relevant details.",
        }
        return f"User Intent: {intent.value}\n{guidance.get(intent, '')}"

    def _parse_synthesis(
        self,
        response_text: str,
        sources: List[RankedResult],
    ) -> Dict[str, Any]:
        """Parse the synthesis response into structured data."""
        import re

        result = {
            "answer": "",
            "citations": [],
            "confidence": 0.7,
            "follow_ups": [],
        }

        # Split response into sections
        text = response_text.strip()

        # Extract answer
        answer_match = re.search(r"ANSWER:\s*(.+?)(?=CONFIDENCE:|$)", text, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        else:
            # No ANSWER section, use whole text up to CONFIDENCE
            conf_pos = text.find("CONFIDENCE:")
            if conf_pos > 0:
                result["answer"] = text[:conf_pos].strip()
            else:
                result["answer"] = text

        # Extract confidence
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        # Extract follow-ups
        followup_match = re.search(r"FOLLOW_UPS?:\s*(.+?)$", text, re.DOTALL)
        if followup_match:
            followup_text = followup_match.group(1)
            # Extract bullet points
            questions = re.findall(r"[-•*]\s*(.+?)(?:\n|$)", followup_text)
            result["follow_ups"] = [q.strip().rstrip("?") + "?" for q in questions if q.strip()][:5]

        # Extract citations from answer
        citation_refs = re.findall(r"\[(\d+)\]", result["answer"])
        seen_refs = set()
        for ref in citation_refs:
            idx = int(ref) - 1
            if idx not in seen_refs and 0 <= idx < len(sources):
                seen_refs.add(idx)
                source = sources[idx]
                result["citations"].append(Citation(
                    index=int(ref),
                    url=source.page_url or source.url,
                    title=source.title,
                    excerpt=source.content[:200] + "..." if len(source.content) > 200 else source.content,
                ))

        return result


# Singleton instances (keyed by tenant id)
_search_agents: Dict[str, SearchAgent] = {}


def get_search_agent(tenant_id: Optional[str] = None) -> SearchAgent:
    """Get the search agent instance for the current tenant."""
    key = tenant_id or get_tenant_id()
    if key not in _search_agents:
        _search_agents[key] = SearchAgent(tenant_id=key)
    return _search_agents[key]
