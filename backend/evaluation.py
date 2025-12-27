"""
ContextPilot Evaluation Harness

Provides quality evaluation through golden queries and metrics:
- Golden query management (expected results for test queries)
- Precision, recall, MRR metrics
- Quality scoring
- Regression detection
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("contextpilot.evaluation")


@dataclass
class GoldenQuery:
    """A golden query with expected results for evaluation."""

    id: str
    query: str
    source_id: str
    expected_urls: List[str] = field(default_factory=list)  # URLs that should be returned
    expected_keywords: List[str] = field(default_factory=list)  # Keywords that should appear
    min_relevance_score: float = 0.7  # Minimum relevance threshold
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "source_id": self.source_id,
            "expected_urls": self.expected_urls,
            "expected_keywords": self.expected_keywords,
            "min_relevance_score": self.min_relevance_score,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenQuery":
        return cls(
            id=data.get("id", ""),
            query=data.get("query", ""),
            source_id=data.get("source_id", ""),
            expected_urls=data.get("expected_urls", []),
            expected_keywords=data.get("expected_keywords", []),
            min_relevance_score=data.get("min_relevance_score", 0.7),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
        )


@dataclass
class QueryResult:
    """Result from running a single query."""

    query_id: str
    query: str
    returned_urls: List[str] = field(default_factory=list)
    returned_scores: List[float] = field(default_factory=list)
    returned_content: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class EvaluationResult:
    """Result of evaluating a golden query."""

    query_id: str
    query: str
    passed: bool = False

    # Retrieval metrics
    precision: float = 0.0  # Relevant returned / Total returned
    recall: float = 0.0  # Relevant returned / Total relevant
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain

    # Keyword metrics
    keyword_coverage: float = 0.0  # Expected keywords found / Total expected

    # Details
    expected_urls: List[str] = field(default_factory=list)
    returned_urls: List[str] = field(default_factory=list)
    matched_urls: List[str] = field(default_factory=list)
    missing_urls: List[str] = field(default_factory=list)
    found_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "passed": self.passed,
            "precision": self.precision,
            "recall": self.recall,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "keyword_coverage": self.keyword_coverage,
            "matched_urls": self.matched_urls,
            "missing_urls": self.missing_urls,
            "found_keywords": self.found_keywords,
            "missing_keywords": self.missing_keywords,
            "latency_ms": self.latency_ms,
        }


@dataclass
class EvaluationSummary:
    """Summary of an evaluation run."""

    source_id: str
    run_id: str
    timestamp: float
    total_queries: int = 0
    passed_queries: int = 0
    failed_queries: int = 0

    # Aggregate metrics
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_mrr: float = 0.0
    avg_keyword_coverage: float = 0.0
    avg_latency_ms: float = 0.0

    # Quality score (0-100)
    quality_score: float = 0.0

    # Individual results
    results: List[EvaluationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_queries": self.total_queries,
            "passed_queries": self.passed_queries,
            "failed_queries": self.failed_queries,
            "avg_precision": self.avg_precision,
            "avg_recall": self.avg_recall,
            "avg_mrr": self.avg_mrr,
            "avg_keyword_coverage": self.avg_keyword_coverage,
            "avg_latency_ms": self.avg_latency_ms,
            "quality_score": self.quality_score,
            "results": [r.to_dict() for r in self.results],
        }


class EvaluationHarness:
    """
    Evaluation harness for testing retrieval quality.

    Usage:
        harness = EvaluationHarness(db, embed_manager)
        harness.add_golden_query(query, source_id, expected_urls, expected_keywords)
        summary = await harness.run_evaluation(source_id)
        print(f"Quality score: {summary.quality_score}")
    """

    def __init__(self, db: Any, embed_manager: Any):
        """
        Initialize the evaluation harness.

        Args:
            db: Database instance for storing golden queries
            embed_manager: EmbedManager for running queries
        """
        self.db = db
        self.embed_manager = embed_manager
        self._golden_queries: Dict[str, List[GoldenQuery]] = {}  # source_id -> queries

    def add_golden_query(
        self,
        query: str,
        source_id: str,
        expected_urls: List[str],
        expected_keywords: Optional[List[str]] = None,
        min_relevance_score: float = 0.7,
    ) -> GoldenQuery:
        """
        Add a golden query for evaluation.

        Args:
            query: The test query
            source_id: Source to evaluate against
            expected_urls: URLs that should be returned
            expected_keywords: Keywords that should appear in results
            min_relevance_score: Minimum relevance threshold

        Returns:
            Created GoldenQuery
        """
        query_id = hashlib.sha256(f"{source_id}:{query}".encode()).hexdigest()[:12]
        now = time.time()

        golden = GoldenQuery(
            id=query_id,
            query=query,
            source_id=source_id,
            expected_urls=expected_urls,
            expected_keywords=expected_keywords or [],
            min_relevance_score=min_relevance_score,
            created_at=now,
            updated_at=now,
        )

        # Store in memory
        if source_id not in self._golden_queries:
            self._golden_queries[source_id] = []
        self._golden_queries[source_id].append(golden)

        # Store in database if available
        if hasattr(self.db, "store_golden_query"):
            try:
                self.db.store_golden_query(golden.to_dict())
            except Exception as e:
                logger.warning(f"Failed to store golden query: {e}")

        return golden

    def get_golden_queries(self, source_id: str) -> List[GoldenQuery]:
        """Get all golden queries for a source."""
        # Try database first
        if hasattr(self.db, "list_golden_queries"):
            try:
                rows = self.db.list_golden_queries(source_id)
                return [GoldenQuery.from_dict(r) for r in rows]
            except Exception as e:
                logger.warning(f"Failed to load golden queries from DB: {e}")

        # Fall back to in-memory
        return self._golden_queries.get(source_id, [])

    async def run_evaluation(
        self,
        source_id: str,
        top_k: int = 10,
    ) -> EvaluationSummary:
        """
        Run evaluation for all golden queries of a source.

        Args:
            source_id: Source to evaluate
            top_k: Number of results to retrieve per query

        Returns:
            EvaluationSummary with aggregate metrics
        """
        queries = self.get_golden_queries(source_id)
        if not queries:
            logger.warning(f"No golden queries found for source {source_id}")
            return EvaluationSummary(
                source_id=source_id,
                run_id=hashlib.sha256(f"{source_id}:{time.time()}".encode()).hexdigest()[:12],
                timestamp=time.time(),
            )

        results: List[EvaluationResult] = []

        for golden in queries:
            result = await self._evaluate_query(golden, top_k)
            results.append(result)

        # Compute summary
        summary = self._compute_summary(source_id, results)

        # Store evaluation result
        if hasattr(self.db, "store_evaluation_result"):
            try:
                self.db.store_evaluation_result(summary.to_dict())
            except Exception as e:
                logger.warning(f"Failed to store evaluation result: {e}")

        return summary

    async def _evaluate_query(
        self,
        golden: GoldenQuery,
        top_k: int,
    ) -> EvaluationResult:
        """Evaluate a single golden query."""
        result = EvaluationResult(
            query_id=golden.id,
            query=golden.query,
            expected_urls=golden.expected_urls,
        )

        try:
            # Run the query
            start_time = time.time()
            search_results = await self.embed_manager.search(
                query=golden.query,
                source_id=golden.source_id,
                top_k=top_k,
            )
            result.latency_ms = (time.time() - start_time) * 1000

            # Extract returned URLs and content
            result.returned_urls = [r.get("url", "") for r in search_results]
            returned_content = " ".join(r.get("content", "") for r in search_results)

            # Calculate URL matching
            expected_set = set(golden.expected_urls)
            returned_set = set(result.returned_urls)
            result.matched_urls = list(expected_set & returned_set)
            result.missing_urls = list(expected_set - returned_set)

            # Calculate precision and recall
            if result.returned_urls:
                result.precision = len(result.matched_urls) / len(result.returned_urls)
            if golden.expected_urls:
                result.recall = len(result.matched_urls) / len(golden.expected_urls)

            # Calculate MRR (Mean Reciprocal Rank)
            for i, url in enumerate(result.returned_urls):
                if url in expected_set:
                    result.mrr = 1.0 / (i + 1)
                    break

            # Check keywords
            returned_content_lower = returned_content.lower()
            for keyword in golden.expected_keywords:
                if keyword.lower() in returned_content_lower:
                    result.found_keywords.append(keyword)
                else:
                    result.missing_keywords.append(keyword)

            if golden.expected_keywords:
                result.keyword_coverage = len(result.found_keywords) / len(golden.expected_keywords)

            # Determine pass/fail
            result.passed = (
                result.recall >= golden.min_relevance_score or
                (result.precision >= 0.5 and result.keyword_coverage >= 0.7)
            )

        except Exception as e:
            logger.error(f"Error evaluating query '{golden.query}': {e}")
            result.passed = False

        return result

    def _compute_summary(
        self,
        source_id: str,
        results: List[EvaluationResult],
    ) -> EvaluationSummary:
        """Compute aggregate summary from individual results."""
        run_id = hashlib.sha256(f"{source_id}:{time.time()}".encode()).hexdigest()[:12]

        summary = EvaluationSummary(
            source_id=source_id,
            run_id=run_id,
            timestamp=time.time(),
            total_queries=len(results),
            passed_queries=sum(1 for r in results if r.passed),
            failed_queries=sum(1 for r in results if not r.passed),
            results=results,
        )

        if results:
            summary.avg_precision = sum(r.precision for r in results) / len(results)
            summary.avg_recall = sum(r.recall for r in results) / len(results)
            summary.avg_mrr = sum(r.mrr for r in results) / len(results)
            summary.avg_keyword_coverage = sum(r.keyword_coverage for r in results) / len(results)
            summary.avg_latency_ms = sum(r.latency_ms for r in results) / len(results)

            # Compute quality score (0-100)
            summary.quality_score = (
                summary.avg_precision * 25 +
                summary.avg_recall * 25 +
                summary.avg_mrr * 25 +
                summary.avg_keyword_coverage * 15 +
                (summary.passed_queries / summary.total_queries) * 10
            )

        return summary

    def compute_quality_score(self, summary: EvaluationSummary) -> float:
        """Compute a 0-100 quality score from evaluation summary."""
        return summary.quality_score


@dataclass
class RegressionSnapshot:
    """A snapshot of evaluation metrics for regression tracking."""

    source_id: str
    run_id: str
    timestamp: float
    quality_score: float
    avg_precision: float
    avg_recall: float
    avg_mrr: float
    query_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "quality_score": self.quality_score,
            "avg_precision": self.avg_precision,
            "avg_recall": self.avg_recall,
            "avg_mrr": self.avg_mrr,
            "query_count": self.query_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegressionSnapshot":
        return cls(
            source_id=data.get("source_id", ""),
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", 0.0),
            quality_score=data.get("quality_score", 0.0),
            avg_precision=data.get("avg_precision", 0.0),
            avg_recall=data.get("avg_recall", 0.0),
            avg_mrr=data.get("avg_mrr", 0.0),
            query_count=data.get("query_count", 0),
        )

    @classmethod
    def from_summary(cls, summary: EvaluationSummary) -> "RegressionSnapshot":
        return cls(
            source_id=summary.source_id,
            run_id=summary.run_id,
            timestamp=summary.timestamp,
            quality_score=summary.quality_score,
            avg_precision=summary.avg_precision,
            avg_recall=summary.avg_recall,
            avg_mrr=summary.avg_mrr,
            query_count=summary.total_queries,
        )


class RegressionTracker:
    """
    Tracks evaluation metrics over time to detect regressions.

    Usage:
        tracker = RegressionTracker(db)
        tracker.record_snapshot(summary)
        regression = tracker.detect_regression(source_id)
        if regression:
            print(f"Regression detected: {regression}")
    """

    # Default threshold for regression detection (10% drop)
    REGRESSION_THRESHOLD = 0.10

    def __init__(self, db: Any):
        self.db = db
        self._snapshots: Dict[str, List[RegressionSnapshot]] = {}

    def record_snapshot(self, summary: EvaluationSummary) -> RegressionSnapshot:
        """Record a new evaluation snapshot."""
        snapshot = RegressionSnapshot.from_summary(summary)

        # Store in memory
        if summary.source_id not in self._snapshots:
            self._snapshots[summary.source_id] = []
        self._snapshots[summary.source_id].append(snapshot)

        # Keep only last 50 snapshots per source
        if len(self._snapshots[summary.source_id]) > 50:
            self._snapshots[summary.source_id] = self._snapshots[summary.source_id][-50:]

        # Store in database if available
        if hasattr(self.db, "store_regression_snapshot"):
            try:
                self.db.store_regression_snapshot(snapshot.to_dict())
            except Exception as e:
                logger.warning(f"Failed to store regression snapshot: {e}")

        return snapshot

    def get_history(
        self,
        source_id: str,
        limit: int = 20,
    ) -> List[RegressionSnapshot]:
        """Get historical snapshots for a source."""
        # Try database first
        if hasattr(self.db, "list_regression_snapshots"):
            try:
                rows = self.db.list_regression_snapshots(source_id, limit)
                return [RegressionSnapshot.from_dict(r) for r in rows]
            except Exception as e:
                logger.warning(f"Failed to load regression snapshots: {e}")

        # Fall back to in-memory
        snapshots = self._snapshots.get(source_id, [])
        return sorted(snapshots, key=lambda s: -s.timestamp)[:limit]

    def detect_regression(
        self,
        source_id: str,
        threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if there's been a regression in quality.

        Args:
            source_id: Source to check
            threshold: Regression threshold (default 10% drop)

        Returns:
            Dict with regression details if detected, None otherwise
        """
        threshold = threshold or self.REGRESSION_THRESHOLD
        history = self.get_history(source_id, limit=5)

        if len(history) < 2:
            return None

        latest = history[0]
        previous = history[1]

        # Check for quality score regression
        if previous.quality_score > 0:
            score_drop = (previous.quality_score - latest.quality_score) / previous.quality_score
            if score_drop > threshold:
                return {
                    "type": "quality_score",
                    "previous_value": previous.quality_score,
                    "current_value": latest.quality_score,
                    "drop_percent": score_drop * 100,
                    "threshold_percent": threshold * 100,
                    "previous_run_id": previous.run_id,
                    "current_run_id": latest.run_id,
                }

        # Check for precision regression
        if previous.avg_precision > 0:
            precision_drop = (previous.avg_precision - latest.avg_precision) / previous.avg_precision
            if precision_drop > threshold:
                return {
                    "type": "precision",
                    "previous_value": previous.avg_precision,
                    "current_value": latest.avg_precision,
                    "drop_percent": precision_drop * 100,
                    "threshold_percent": threshold * 100,
                    "previous_run_id": previous.run_id,
                    "current_run_id": latest.run_id,
                }

        # Check for recall regression
        if previous.avg_recall > 0:
            recall_drop = (previous.avg_recall - latest.avg_recall) / previous.avg_recall
            if recall_drop > threshold:
                return {
                    "type": "recall",
                    "previous_value": previous.avg_recall,
                    "current_value": latest.avg_recall,
                    "drop_percent": recall_drop * 100,
                    "threshold_percent": threshold * 100,
                    "previous_run_id": previous.run_id,
                    "current_run_id": latest.run_id,
                }

        return None

    def get_trend(
        self,
        source_id: str,
        metric: str = "quality_score",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get trend data for a metric over time.

        Args:
            source_id: Source to get trend for
            metric: Metric to track (quality_score, avg_precision, avg_recall, avg_mrr)
            limit: Number of data points

        Returns:
            List of {timestamp, value} dicts
        """
        history = self.get_history(source_id, limit)

        trend = []
        for snapshot in reversed(history):
            value = getattr(snapshot, metric, 0.0)
            trend.append({
                "timestamp": snapshot.timestamp,
                "run_id": snapshot.run_id,
                "value": value,
            })

        return trend
