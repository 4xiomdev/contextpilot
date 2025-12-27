"""
ContextPilot Job Queue

Simple async job queue with Firestore persistence for background tasks.
Handles crawl jobs, normalization jobs, and other async operations.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Awaitable

from .firestore_db import get_firestore_db
from .tenant_context import get_tenant_id, set_tenant_id, reset_tenant_id

logger = logging.getLogger("contextpilot.job_queue")


class JobType(str, Enum):
    """Types of background jobs."""
    CRAWL_URL = "crawl_url"
    CRAWL_SOURCE = "crawl_source"
    NORMALIZE = "normalize"
    REINDEX = "reindex"
    DISCOVERY = "discovery"
    HEALTH_CHECK = "health_check"


class JobPriority(int, Enum):
    """Job priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class JobStatus(str, Enum):
    """Job status values."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class Job:
    """Represents a background job."""
    id: str
    type: JobType
    tenant_id: str
    payload: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    retries: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firestore."""
        return {
            "id": self.id,
            "type": self.type.value,
            "tenant_id": self.tenant_id,
            "payload": self.payload,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "result": self.result,
            "retries": self.retries,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=JobType(data["type"]),
            tenant_id=data["tenant_id"],
            payload=data.get("payload", {}),
            status=JobStatus(data.get("status", "pending")),
            priority=JobPriority(data.get("priority", 1)),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
            result=data.get("result"),
            retries=data.get("retries", 0),
            max_retries=data.get("max_retries", 3),
        )


# Type for job handlers
JobHandler = Callable[[Job], Awaitable[Dict[str, Any]]]


class JobQueue:
    """
    Async job queue with Firestore persistence.

    Features:
    - Priority-based processing
    - Automatic retries with backoff
    - Firestore persistence for durability
    - Concurrent worker execution
    - Job status tracking
    """

    def __init__(
        self,
        num_workers: int = 3,
        persist_to_firestore: bool = True,
    ):
        """
        Initialize the job queue.

        Args:
            num_workers: Number of concurrent worker tasks
            persist_to_firestore: Whether to persist jobs to Firestore
        """
        self.num_workers = num_workers
        self.persist = persist_to_firestore
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._handlers: Dict[JobType, JobHandler] = {}
        self._workers: List[asyncio.Task] = []
        self._running_jobs: Dict[str, Job] = {}
        self._is_running = False
        self._shutdown_event = asyncio.Event()

    def register_handler(self, job_type: JobType, handler: JobHandler):
        """
        Register a handler for a job type.

        Args:
            job_type: Type of job to handle
            handler: Async function that processes the job
        """
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for {job_type.value}")

    async def enqueue(
        self,
        job_type: JobType,
        tenant_id: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
    ) -> Job:
        """
        Add a job to the queue.

        Args:
            job_type: Type of job
            tenant_id: Tenant ID for the job
            payload: Job-specific data
            priority: Job priority

        Returns:
            The created Job object
        """
        job = Job(
            id=f"{job_type.value}_{uuid.uuid4().hex[:12]}",
            type=job_type,
            tenant_id=tenant_id,
            payload=payload,
            priority=priority,
            status=JobStatus.QUEUED,
        )

        # Persist to Firestore
        if self.persist:
            await self._persist_job(job)

        # Add to in-memory queue (negative priority for max-heap behavior)
        await self._queue.put((-priority.value, job.created_at, job))

        logger.info(f"Enqueued job {job.id} (type={job_type.value}, priority={priority.name})")
        return job

    async def _persist_job(self, job: Job):
        """Persist job to Firestore."""
        try:
            db = get_firestore_db(job.tenant_id)
            if hasattr(db, "_col"):
                db._col("jobs").document(job.id).set(job.to_dict())
        except Exception as e:
            logger.warning(f"Failed to persist job {job.id}: {e}")

    async def _update_job(self, job: Job):
        """Update job in Firestore."""
        if self.persist:
            await self._persist_job(job)

    async def start(self):
        """Start the job queue workers."""
        if self._is_running:
            logger.warning("Job queue already running")
            return

        self._is_running = True
        self._shutdown_event.clear()

        # Load pending jobs from Firestore
        if self.persist:
            await self._load_pending_jobs()

        # Start workers
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        logger.info(f"Job queue started with {self.num_workers} workers")

    async def stop(self, wait: bool = True):
        """
        Stop the job queue.

        Args:
            wait: Whether to wait for current jobs to complete
        """
        if not self._is_running:
            return

        self._is_running = False
        self._shutdown_event.set()

        if wait:
            # Wait for workers to finish current jobs
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        logger.info("Job queue stopped")

    async def _load_pending_jobs(self):
        """Load pending jobs from Firestore on startup."""
        try:
            # Load jobs for default tenant
            # In production, would iterate over all tenants
            db = get_firestore_db("default")
            if not hasattr(db, "_col"):
                return

            docs = db._col("jobs").where("status", "in", ["pending", "queued"]).stream()
            count = 0
            for doc in docs:
                job = Job.from_dict(doc.to_dict())
                await self._queue.put((-job.priority.value, job.created_at, job))
                count += 1

            if count > 0:
                logger.info(f"Loaded {count} pending jobs from Firestore")
        except Exception as e:
            logger.warning(f"Failed to load pending jobs: {e}")

    async def _worker(self, worker_id: int):
        """Worker coroutine that processes jobs from the queue."""
        logger.debug(f"Worker {worker_id} started")

        while self._is_running:
            try:
                # Wait for job with timeout
                try:
                    _, _, job = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the job
                await self._process_job(job, worker_id)
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.debug(f"Worker {worker_id} stopped")

    async def _process_job(self, job: Job, worker_id: int):
        """Process a single job."""
        handler = self._handlers.get(job.type)
        if not handler:
            logger.warning(f"No handler for job type {job.type.value}")
            job.status = JobStatus.FAILED
            job.error_message = f"No handler registered for {job.type.value}"
            await self._update_job(job)
            return

        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self._running_jobs[job.id] = job
        await self._update_job(job)

        logger.info(f"Worker {worker_id} processing job {job.id}")

        # Set tenant context
        token = set_tenant_id(job.tenant_id)

        try:
            result = await handler(job)
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = time.time()

            duration = job.completed_at - job.started_at
            logger.info(f"Job {job.id} completed in {duration:.2f}s")

        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            job.error_message = str(e)
            job.retries += 1

            if job.retries < job.max_retries:
                # Retry with exponential backoff
                job.status = JobStatus.RETRYING
                await self._update_job(job)

                backoff = min(300, 2 ** job.retries * 5)  # Max 5 minutes
                logger.info(f"Retrying job {job.id} in {backoff}s (attempt {job.retries + 1})")

                await asyncio.sleep(backoff)
                await self._queue.put((-job.priority.value, time.time(), job))
            else:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()

        finally:
            reset_tenant_id(token)
            self._running_jobs.pop(job.id, None)
            await self._update_job(job)

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        # Check running jobs first
        if job_id in self._running_jobs:
            return self._running_jobs[job_id]

        # Check Firestore
        if self.persist:
            try:
                db = get_firestore_db()
                doc = db._col("jobs").document(job_id).get()
                if doc.exists:
                    return Job.from_dict(doc.to_dict())
            except Exception as e:
                logger.warning(f"Failed to get job {job_id}: {e}")

        return None

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        if job_id in self._running_jobs:
            job = self._running_jobs[job_id]
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            await self._update_job(job)
            self._running_jobs.pop(job_id, None)
            return True

        return False

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "is_running": self._is_running,
            "num_workers": self.num_workers,
            "queue_size": self._queue.qsize(),
            "running_jobs": len(self._running_jobs),
            "registered_handlers": list(self._handlers.keys()),
        }

    def get_running_jobs(self) -> List[Job]:
        """Get list of currently running jobs."""
        return list(self._running_jobs.values())


# Singleton instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue(num_workers=3, persist_to_firestore=True)
    return _job_queue


async def start_job_queue():
    """Start the global job queue."""
    queue = get_job_queue()
    await queue.start()


async def stop_job_queue():
    """Stop the global job queue."""
    global _job_queue
    if _job_queue:
        await _job_queue.stop()
