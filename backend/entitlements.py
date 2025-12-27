"""
ContextPilot Entitlements

Manages tenant quotas and usage limits:
- Max sources per tenant
- Max pages per source
- Monthly crawl limits
- Vector storage limits
- LLM call limits
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("contextpilot.entitlements")


class PlanTier(Enum):
    """Subscription plan tiers."""
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TenantQuotas:
    """Quota limits for a tenant."""

    tier: PlanTier = PlanTier.FREE

    # Source limits
    max_sources: int = 3
    max_pages_per_source: int = 100

    # Crawl limits
    max_monthly_crawls: int = 10
    max_concurrent_crawls: int = 1

    # Vector limits
    max_vectors: int = 10_000
    max_namespaces: int = 3

    # LLM limits
    llm_calls_per_month: int = 100
    max_input_tokens_per_call: int = 50_000

    # Feature flags
    can_use_firecrawl: bool = False
    can_use_playwright: bool = False
    can_use_normalization: bool = False
    can_use_evaluation: bool = False

    @classmethod
    def for_tier(cls, tier: PlanTier) -> "TenantQuotas":
        """Get default quotas for a plan tier."""
        if tier == PlanTier.FREE:
            return cls(
                tier=tier,
                max_sources=3,
                max_pages_per_source=100,
                max_monthly_crawls=10,
                max_concurrent_crawls=1,
                max_vectors=10_000,
                max_namespaces=3,
                llm_calls_per_month=100,
                max_input_tokens_per_call=50_000,
                can_use_firecrawl=False,
                can_use_playwright=False,
                can_use_normalization=False,
                can_use_evaluation=False,
            )
        elif tier == PlanTier.STARTER:
            return cls(
                tier=tier,
                max_sources=10,
                max_pages_per_source=500,
                max_monthly_crawls=50,
                max_concurrent_crawls=2,
                max_vectors=100_000,
                max_namespaces=10,
                llm_calls_per_month=500,
                max_input_tokens_per_call=100_000,
                can_use_firecrawl=True,
                can_use_playwright=False,
                can_use_normalization=True,
                can_use_evaluation=False,
            )
        elif tier == PlanTier.PRO:
            return cls(
                tier=tier,
                max_sources=50,
                max_pages_per_source=2000,
                max_monthly_crawls=200,
                max_concurrent_crawls=5,
                max_vectors=1_000_000,
                max_namespaces=50,
                llm_calls_per_month=2000,
                max_input_tokens_per_call=200_000,
                can_use_firecrawl=True,
                can_use_playwright=True,
                can_use_normalization=True,
                can_use_evaluation=True,
            )
        else:  # ENTERPRISE
            return cls(
                tier=tier,
                max_sources=999999,
                max_pages_per_source=10000,
                max_monthly_crawls=999999,
                max_concurrent_crawls=20,
                max_vectors=10_000_000,
                max_namespaces=999,
                llm_calls_per_month=999999,
                max_input_tokens_per_call=500_000,
                can_use_firecrawl=True,
                can_use_playwright=True,
                can_use_normalization=True,
                can_use_evaluation=True,
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "max_sources": self.max_sources,
            "max_pages_per_source": self.max_pages_per_source,
            "max_monthly_crawls": self.max_monthly_crawls,
            "max_concurrent_crawls": self.max_concurrent_crawls,
            "max_vectors": self.max_vectors,
            "max_namespaces": self.max_namespaces,
            "llm_calls_per_month": self.llm_calls_per_month,
            "max_input_tokens_per_call": self.max_input_tokens_per_call,
            "can_use_firecrawl": self.can_use_firecrawl,
            "can_use_playwright": self.can_use_playwright,
            "can_use_normalization": self.can_use_normalization,
            "can_use_evaluation": self.can_use_evaluation,
        }


@dataclass
class TenantUsage:
    """Current usage for a tenant."""

    tenant_id: str
    period_start: float = 0.0  # Start of current billing period

    # Current counts
    source_count: int = 0
    vector_count: int = 0
    namespace_count: int = 0

    # Monthly usage
    crawls_this_month: int = 0
    llm_calls_this_month: int = 0
    pages_crawled_this_month: int = 0

    # Running totals
    concurrent_crawls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "period_start": self.period_start,
            "source_count": self.source_count,
            "vector_count": self.vector_count,
            "namespace_count": self.namespace_count,
            "crawls_this_month": self.crawls_this_month,
            "llm_calls_this_month": self.llm_calls_this_month,
            "pages_crawled_this_month": self.pages_crawled_this_month,
            "concurrent_crawls": self.concurrent_crawls,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TenantUsage":
        return cls(
            tenant_id=data.get("tenant_id", ""),
            period_start=data.get("period_start", 0.0),
            source_count=data.get("source_count", 0),
            vector_count=data.get("vector_count", 0),
            namespace_count=data.get("namespace_count", 0),
            crawls_this_month=data.get("crawls_this_month", 0),
            llm_calls_this_month=data.get("llm_calls_this_month", 0),
            pages_crawled_this_month=data.get("pages_crawled_this_month", 0),
            concurrent_crawls=data.get("concurrent_crawls", 0),
        )


@dataclass
class QuotaCheckResult:
    """Result of a quota check."""

    allowed: bool
    quota_type: str
    current_usage: int
    limit: int
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "quota_type": self.quota_type,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "message": self.message,
        }


class EntitlementManager:
    """
    Manages tenant entitlements and quota checking.

    Usage:
        manager = EntitlementManager(db)
        quotas = manager.get_quotas(tenant_id)
        check = manager.check_quota(tenant_id, "sources")
        if check.allowed:
            # proceed with operation
        manager.increment_usage(tenant_id, "crawls")
    """

    def __init__(self, db: Any):
        self.db = db
        self._quotas_cache: Dict[str, TenantQuotas] = {}
        self._usage_cache: Dict[str, TenantUsage] = {}

    def get_quotas(self, tenant_id: str) -> TenantQuotas:
        """Get quota limits for a tenant."""
        # Check cache first
        if tenant_id in self._quotas_cache:
            return self._quotas_cache[tenant_id]

        # Try to load from database
        if hasattr(self.db, "get_tenant_quotas"):
            try:
                data = self.db.get_tenant_quotas(tenant_id)
                if data:
                    tier = PlanTier(data.get("tier", "free"))
                    quotas = TenantQuotas.for_tier(tier)
                    # Override with any custom values
                    for key, value in data.items():
                        if hasattr(quotas, key) and key != "tier":
                            setattr(quotas, key, value)
                    self._quotas_cache[tenant_id] = quotas
                    return quotas
            except Exception as e:
                logger.warning(f"Failed to load tenant quotas: {e}")

        # Default to free tier
        quotas = TenantQuotas.for_tier(PlanTier.FREE)
        self._quotas_cache[tenant_id] = quotas
        return quotas

    def get_usage(self, tenant_id: str) -> TenantUsage:
        """Get current usage for a tenant."""
        # Check cache first
        if tenant_id in self._usage_cache:
            usage = self._usage_cache[tenant_id]
            # Check if we need to reset monthly counters
            if self._should_reset_monthly(usage):
                usage = self._reset_monthly_usage(usage)
            return usage

        # Try to load from database
        if hasattr(self.db, "get_tenant_usage"):
            try:
                data = self.db.get_tenant_usage(tenant_id)
                if data:
                    usage = TenantUsage.from_dict(data)
                    if self._should_reset_monthly(usage):
                        usage = self._reset_monthly_usage(usage)
                    self._usage_cache[tenant_id] = usage
                    return usage
            except Exception as e:
                logger.warning(f"Failed to load tenant usage: {e}")

        # Create new usage record
        usage = TenantUsage(tenant_id=tenant_id, period_start=time.time())
        self._usage_cache[tenant_id] = usage
        return usage

    def check_quota(
        self,
        tenant_id: str,
        quota_type: str,
        increment: int = 1,
    ) -> QuotaCheckResult:
        """
        Check if an operation is allowed within quotas.

        Args:
            tenant_id: Tenant to check
            quota_type: Type of quota (sources, crawls, vectors, llm_calls, etc.)
            increment: Amount to increment by

        Returns:
            QuotaCheckResult indicating if allowed
        """
        quotas = self.get_quotas(tenant_id)
        usage = self.get_usage(tenant_id)

        quota_map = {
            "sources": (usage.source_count, quotas.max_sources),
            "pages": (usage.pages_crawled_this_month, quotas.max_pages_per_source),
            "crawls": (usage.crawls_this_month, quotas.max_monthly_crawls),
            "concurrent_crawls": (usage.concurrent_crawls, quotas.max_concurrent_crawls),
            "vectors": (usage.vector_count, quotas.max_vectors),
            "namespaces": (usage.namespace_count, quotas.max_namespaces),
            "llm_calls": (usage.llm_calls_this_month, quotas.llm_calls_per_month),
        }

        if quota_type not in quota_map:
            return QuotaCheckResult(
                allowed=True,
                quota_type=quota_type,
                current_usage=0,
                limit=0,
                message="Unknown quota type",
            )

        current, limit = quota_map[quota_type]
        allowed = (current + increment) <= limit

        message = ""
        if not allowed:
            message = f"Quota exceeded: {quota_type} ({current + increment}/{limit})"

        return QuotaCheckResult(
            allowed=allowed,
            quota_type=quota_type,
            current_usage=current,
            limit=limit,
            message=message,
        )

    def check_feature(self, tenant_id: str, feature: str) -> bool:
        """Check if a tenant has access to a feature."""
        quotas = self.get_quotas(tenant_id)

        feature_map = {
            "firecrawl": quotas.can_use_firecrawl,
            "playwright": quotas.can_use_playwright,
            "normalization": quotas.can_use_normalization,
            "evaluation": quotas.can_use_evaluation,
        }

        return feature_map.get(feature, False)

    def increment_usage(
        self,
        tenant_id: str,
        usage_type: str,
        amount: int = 1,
    ) -> TenantUsage:
        """
        Increment usage counter.

        Args:
            tenant_id: Tenant to update
            usage_type: Type of usage (sources, crawls, vectors, llm_calls, etc.)
            amount: Amount to increment by

        Returns:
            Updated TenantUsage
        """
        usage = self.get_usage(tenant_id)

        usage_map = {
            "sources": "source_count",
            "crawls": "crawls_this_month",
            "pages": "pages_crawled_this_month",
            "vectors": "vector_count",
            "namespaces": "namespace_count",
            "llm_calls": "llm_calls_this_month",
            "concurrent_crawls": "concurrent_crawls",
        }

        if usage_type in usage_map:
            attr = usage_map[usage_type]
            setattr(usage, attr, getattr(usage, attr) + amount)

        # Persist to database
        self._save_usage(usage)

        return usage

    def decrement_usage(
        self,
        tenant_id: str,
        usage_type: str,
        amount: int = 1,
    ) -> TenantUsage:
        """Decrement usage counter (e.g., when deleting a source)."""
        usage = self.get_usage(tenant_id)

        usage_map = {
            "sources": "source_count",
            "vectors": "vector_count",
            "namespaces": "namespace_count",
            "concurrent_crawls": "concurrent_crawls",
        }

        if usage_type in usage_map:
            attr = usage_map[usage_type]
            current = getattr(usage, attr)
            setattr(usage, attr, max(0, current - amount))

        # Persist to database
        self._save_usage(usage)

        return usage

    def reset_monthly_usage(self, tenant_id: str) -> TenantUsage:
        """Reset monthly usage counters for a tenant."""
        usage = self.get_usage(tenant_id)
        usage = self._reset_monthly_usage(usage)
        self._save_usage(usage)
        return usage

    def _should_reset_monthly(self, usage: TenantUsage) -> bool:
        """Check if monthly counters should be reset."""
        if usage.period_start == 0:
            return True

        # Reset on the 1st of each month
        import datetime
        period_start = datetime.datetime.fromtimestamp(usage.period_start)
        now = datetime.datetime.now()

        return (now.year > period_start.year or
                (now.year == period_start.year and now.month > period_start.month))

    def _reset_monthly_usage(self, usage: TenantUsage) -> TenantUsage:
        """Reset monthly counters."""
        usage.crawls_this_month = 0
        usage.llm_calls_this_month = 0
        usage.pages_crawled_this_month = 0
        usage.period_start = time.time()
        return usage

    def _save_usage(self, usage: TenantUsage) -> None:
        """Persist usage to database."""
        self._usage_cache[usage.tenant_id] = usage

        if hasattr(self.db, "store_tenant_usage"):
            try:
                self.db.store_tenant_usage(usage.to_dict())
            except Exception as e:
                logger.warning(f"Failed to save tenant usage: {e}")

    def get_usage_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Get a summary of usage vs quotas for display."""
        quotas = self.get_quotas(tenant_id)
        usage = self.get_usage(tenant_id)

        return {
            "tier": quotas.tier.value,
            "sources": {
                "used": usage.source_count,
                "limit": quotas.max_sources,
                "percent": (usage.source_count / quotas.max_sources * 100) if quotas.max_sources > 0 else 0,
            },
            "vectors": {
                "used": usage.vector_count,
                "limit": quotas.max_vectors,
                "percent": (usage.vector_count / quotas.max_vectors * 100) if quotas.max_vectors > 0 else 0,
            },
            "crawls_this_month": {
                "used": usage.crawls_this_month,
                "limit": quotas.max_monthly_crawls,
                "percent": (usage.crawls_this_month / quotas.max_monthly_crawls * 100) if quotas.max_monthly_crawls > 0 else 0,
            },
            "llm_calls_this_month": {
                "used": usage.llm_calls_this_month,
                "limit": quotas.llm_calls_per_month,
                "percent": (usage.llm_calls_this_month / quotas.llm_calls_per_month * 100) if quotas.llm_calls_per_month > 0 else 0,
            },
            "features": {
                "firecrawl": quotas.can_use_firecrawl,
                "playwright": quotas.can_use_playwright,
                "normalization": quotas.can_use_normalization,
                "evaluation": quotas.can_use_evaluation,
            },
        }


# Singleton instance
_manager: Optional[EntitlementManager] = None


def get_entitlement_manager(db: Optional[Any] = None) -> EntitlementManager:
    """Get the singleton entitlement manager."""
    global _manager
    if _manager is None:
        if db is None:
            from .firestore_db import get_firestore_db
            db = get_firestore_db()
        _manager = EntitlementManager(db)
    return _manager
