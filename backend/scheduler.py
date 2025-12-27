"""
ContextPilot Crawl Scheduler

APScheduler-based scheduling for automated documentation crawling.
Checks sources periodically and triggers crawls when due.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    AsyncIOScheduler = None
    IntervalTrigger = None
    CronTrigger = None

from .config import get_config
from .firestore_db import get_firestore_db
from .models import Source, SourceHealthStatus
from .source_registry import get_source_registry
from .tenant_context import get_tenant_id, set_tenant_id, reset_tenant_id

logger = logging.getLogger("contextpilot.scheduler")


class JobStatus(str, Enum):
    """Status of a scheduled job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledJob:
    """Represents a scheduled crawl job."""
    id: str
    source_id: str
    tenant_id: str
    status: JobStatus = JobStatus.PENDING
    scheduled_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    chunks_indexed: int = 0


class CrawlScheduler:
    """
    Manages scheduled crawling of documentation sources.

    Features:
    - Hourly check for sources due for crawling
    - Per-tenant source isolation
    - Concurrent crawl execution with limits
    - Health status updates
    """

    def __init__(
        self,
        check_interval_minutes: int = 60,
        max_concurrent_crawls: int = 3,
    ):
        """
        Initialize the scheduler.

        Args:
            check_interval_minutes: How often to check for due sources
            max_concurrent_crawls: Maximum concurrent crawl jobs
        """
        self.check_interval = check_interval_minutes
        self.max_concurrent = max_concurrent_crawls
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._running_jobs: Dict[str, ScheduledJob] = {}
        self._job_history: List[ScheduledJob] = []
        self._crawl_callback: Optional[Callable] = None
        self._is_running = False
        self._semaphore = asyncio.Semaphore(max_concurrent_crawls)

    def set_crawl_callback(self, callback: Callable[[str, str], Any]):
        """
        Set the callback function for executing crawls.

        Args:
            callback: Async function(tenant_id, source_id) that performs the crawl
        """
        self._crawl_callback = callback

    def start(self):
        """Start the scheduler."""
        if not HAS_APSCHEDULER:
            logger.warning("APScheduler not installed, scheduler disabled")
            return

        if self._is_running:
            logger.warning("Scheduler already running")
            return

        self._scheduler = AsyncIOScheduler()

        # Add the main check job
        self._scheduler.add_job(
            self._check_and_crawl_due_sources,
            IntervalTrigger(minutes=self.check_interval),
            id="source_check",
            name="Check sources for scheduled crawls",
            replace_existing=True,
        )

        # Also run immediately on start
        self._scheduler.add_job(
            self._check_and_crawl_due_sources,
            id="source_check_initial",
            name="Initial source check",
            replace_existing=True,
        )

        self._scheduler.start()
        self._is_running = True
        logger.info(f"Scheduler started (check interval: {self.check_interval} minutes)")

    def stop(self):
        """Stop the scheduler."""
        if self._scheduler and self._is_running:
            self._scheduler.shutdown(wait=False)
            self._is_running = False
            logger.info("Scheduler stopped")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running

    async def _check_and_crawl_due_sources(self):
        """
        Check all tenants for sources due for crawling and trigger crawls.

        This runs periodically based on check_interval.
        """
        logger.info("Checking for sources due for crawling...")

        try:
            # Get all tenants with sources
            # For now, we'll check the default tenant
            # In a full multi-tenant setup, we'd iterate over all tenants
            tenant_ids = await self._get_active_tenants()

            for tenant_id in tenant_ids:
                await self._check_tenant_sources(tenant_id)

        except Exception as e:
            logger.error(f"Error in scheduled source check: {e}")

    async def _get_active_tenants(self) -> List[str]:
        """
        Get list of active tenant IDs.

        For now, returns just the default tenant.
        In production, this would query Firestore for all tenants.
        """
        # TODO: Query all tenants from Firestore
        return ["default"]

    async def _check_tenant_sources(self, tenant_id: str):
        """Check and crawl due sources for a specific tenant."""
        token = set_tenant_id(tenant_id)

        try:
            registry = get_source_registry(tenant_id)
            due_sources = registry.get_sources_due_for_crawl()

            logger.info(f"Tenant {tenant_id}: {len(due_sources)} sources due for crawl")

            for source in due_sources:
                await self._schedule_crawl(tenant_id, source)

        finally:
            reset_tenant_id(token)

    async def _schedule_crawl(self, tenant_id: str, source: Source):
        """
        Schedule a crawl for a source.

        Respects concurrency limits and avoids duplicate jobs.
        """
        job_key = f"{tenant_id}:{source.id}"

        # Check if already running
        if job_key in self._running_jobs:
            logger.debug(f"Skipping {source.name}: already crawling")
            return

        # Create job
        job = ScheduledJob(
            id=f"crawl_{source.id}_{int(time.time())}",
            source_id=source.id,
            tenant_id=tenant_id,
        )

        self._running_jobs[job_key] = job
        logger.info(f"Scheduling crawl for {source.name} ({source.base_url})")

        # Execute crawl with semaphore for concurrency control
        asyncio.create_task(self._execute_crawl(job, source))

    async def _execute_crawl(self, job: ScheduledJob, source: Source):
        """Execute a crawl job."""
        job_key = f"{job.tenant_id}:{job.source_id}"

        async with self._semaphore:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()

            try:
                if self._crawl_callback:
                    # Use the provided callback
                    result = await self._crawl_callback(job.tenant_id, job.source_id)
                    job.chunks_indexed = getattr(result, "chunks_indexed", 0) if result else 0
                    job.status = JobStatus.COMPLETED
                else:
                    # Fallback: use crawl manager directly
                    await self._default_crawl(job, source)

                job.completed_at = time.time()
                logger.info(
                    f"Completed crawl for {source.name}: "
                    f"{job.chunks_indexed} chunks in "
                    f"{job.completed_at - job.started_at:.1f}s"
                )

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = time.time()
                logger.error(f"Failed crawl for {source.name}: {e}")

            finally:
                # Update source health
                try:
                    registry = get_source_registry(job.tenant_id)
                    registry.mark_crawl_complete(
                        source_id=job.source_id,
                        success=(job.status == JobStatus.COMPLETED),
                        chunks_count=job.chunks_indexed,
                        error_message=job.error_message,
                    )
                except Exception as e:
                    logger.error(f"Failed to update source health: {e}")

                # Move to history
                self._job_history.append(job)
                if len(self._job_history) > 100:
                    self._job_history = self._job_history[-100:]

                # Remove from running
                self._running_jobs.pop(job_key, None)

    async def _default_crawl(self, job: ScheduledJob, source: Source):
        """Default crawl implementation using CrawlManager."""
        from .crawl_manager import get_crawl_manager

        token = set_tenant_id(job.tenant_id)
        try:
            crawl_manager = get_crawl_manager(job.tenant_id)
            result = await crawl_manager.crawl_source(source)

            if result and result.success:
                job.chunks_indexed = result.chunks_indexed
                job.status = JobStatus.COMPLETED
            else:
                job.status = JobStatus.FAILED
                job.error_message = "Crawl returned no results"
        finally:
            reset_tenant_id(token)

    def get_running_jobs(self) -> List[ScheduledJob]:
        """Get list of currently running jobs."""
        return list(self._running_jobs.values())

    def get_job_history(self, limit: int = 50) -> List[ScheduledJob]:
        """Get recent job history."""
        return self._job_history[-limit:]

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status information."""
        return {
            "is_running": self._is_running,
            "check_interval_minutes": self.check_interval,
            "max_concurrent_crawls": self.max_concurrent,
            "running_jobs": len(self._running_jobs),
            "jobs_in_history": len(self._job_history),
            "next_check": self._get_next_check_time(),
        }

    def _get_next_check_time(self) -> Optional[float]:
        """Get timestamp of next scheduled check."""
        if not self._scheduler or not self._is_running:
            return None

        job = self._scheduler.get_job("source_check")
        if job and job.next_run_time:
            return job.next_run_time.timestamp()
        return None

    async def trigger_source_crawl(self, tenant_id: str, source_id: str) -> ScheduledJob:
        """
        Manually trigger a crawl for a specific source.

        Args:
            tenant_id: Tenant ID
            source_id: Source ID to crawl

        Returns:
            The scheduled job
        """
        registry = get_source_registry(tenant_id)
        source = registry.get_source(source_id)

        if not source:
            raise ValueError(f"Source not found: {source_id}")

        job = ScheduledJob(
            id=f"manual_{source_id}_{int(time.time())}",
            source_id=source_id,
            tenant_id=tenant_id,
        )

        job_key = f"{tenant_id}:{source_id}"
        if job_key in self._running_jobs:
            raise ValueError("Source is already being crawled")

        self._running_jobs[job_key] = job
        asyncio.create_task(self._execute_crawl(job, source))

        return job

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: ID of the job to cancel

        Returns:
            True if job was found and cancelled
        """
        for key, job in list(self._running_jobs.items()):
            if job.id == job_id:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
                self._running_jobs.pop(key, None)
                self._job_history.append(job)
                logger.info(f"Cancelled job {job_id}")
                return True
        return False


# Singleton instance
_scheduler: Optional[CrawlScheduler] = None


def get_scheduler() -> CrawlScheduler:
    """Get the scheduler instance."""
    global _scheduler
    if _scheduler is None:
        config = get_config()
        _scheduler = CrawlScheduler(
            check_interval_minutes=60,  # Check every hour
            max_concurrent_crawls=3,
        )
    return _scheduler


def start_scheduler():
    """Start the global scheduler."""
    scheduler = get_scheduler()
    scheduler.start()


def stop_scheduler():
    """Stop the global scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
