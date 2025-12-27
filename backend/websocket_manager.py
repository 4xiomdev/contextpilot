"""
ContextPilot WebSocket Manager
Real-time event broadcasting for the dashboard.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from weakref import WeakSet

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("contextpilot.websocket")


class EventType(str, Enum):
    """Types of real-time events."""
    # Connection events
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"

    # Crawl events
    CRAWL_STARTED = "crawl:started"
    CRAWL_PROGRESS = "crawl:progress"
    CRAWL_COMPLETED = "crawl:completed"
    CRAWL_FAILED = "crawl:failed"

    # Indexing events
    INDEX_CHUNK = "index:chunk"
    INDEX_BATCH = "index:batch"
    INDEX_COMPLETED = "index:completed"

    # Search events
    SEARCH_STARTED = "search:started"
    SEARCH_COMPLETED = "search:completed"

    # System events
    HEALTH_UPDATE = "system:health"
    STATS_UPDATE = "system:stats"


@dataclass
class Event:
    """A real-time event."""
    type: EventType
    data: Dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        })


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts events.

    Features:
    - Connection pooling with automatic cleanup
    - Broadcast to all connected clients
    - Tenant-specific broadcasts
    - Event history for reconnection
    """

    def __init__(self, max_history: int = 100):
        self._connections: Dict[str, Set[WebSocket]] = {}  # tenant_id -> connections
        self._all_connections: Set[WebSocket] = set()
        self._event_history: List[Event] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        tenant_id: str = "default",
    ) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()

        async with self._lock:
            self._all_connections.add(websocket)

            if tenant_id not in self._connections:
                self._connections[tenant_id] = set()
            self._connections[tenant_id].add(websocket)

        logger.info(f"WebSocket connected: tenant={tenant_id}, total={len(self._all_connections)}")

        # Send connection confirmation with recent history
        await self._send_to_socket(websocket, Event(
            type=EventType.CONNECTED,
            data={
                "message": "Connected to ContextPilot",
                "tenant_id": tenant_id,
                "history_count": len(self._event_history),
            },
        ))

    async def disconnect(
        self,
        websocket: WebSocket,
        tenant_id: str = "default",
    ) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._all_connections.discard(websocket)

            if tenant_id in self._connections:
                self._connections[tenant_id].discard(websocket)
                if not self._connections[tenant_id]:
                    del self._connections[tenant_id]

        logger.info(f"WebSocket disconnected: tenant={tenant_id}, total={len(self._all_connections)}")

    async def broadcast(
        self,
        event: Event,
        tenant_id: Optional[str] = None,
    ) -> None:
        """
        Broadcast an event to connected clients.

        Args:
            event: The event to broadcast
            tenant_id: If specified, only broadcast to this tenant
        """
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Determine target connections
        if tenant_id:
            connections = self._connections.get(tenant_id, set())
        else:
            connections = self._all_connections

        if not connections:
            return

        # Broadcast to all connections
        disconnected = set()

        for websocket in connections:
            try:
                await self._send_to_socket(websocket, event)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.add(websocket)

        # Clean up disconnected sockets
        for ws in disconnected:
            await self.disconnect(ws, tenant_id or "default")

    async def _send_to_socket(self, websocket: WebSocket, event: Event) -> None:
        """Send an event to a single socket."""
        await websocket.send_text(event.to_json())

    def get_recent_events(self, count: int = 50) -> List[Event]:
        """Get recent events from history."""
        return self._event_history[-count:]

    @property
    def connection_count(self) -> int:
        """Get total number of connections."""
        return len(self._all_connections)

    @property
    def tenant_counts(self) -> Dict[str, int]:
        """Get connection count per tenant."""
        return {k: len(v) for k, v in self._connections.items()}


# Singleton instance
_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get the singleton connection manager."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager


# Helper functions for broadcasting events

async def broadcast_crawl_started(
    url: str,
    job_id: str,
    tenant_id: Optional[str] = None,
) -> None:
    """Broadcast crawl started event."""
    manager = get_connection_manager()
    await manager.broadcast(
        Event(
            type=EventType.CRAWL_STARTED,
            data={"url": url, "job_id": job_id},
        ),
        tenant_id=tenant_id,
    )


async def broadcast_crawl_progress(
    url: str,
    job_id: str,
    message: str,
    progress: float,
    tenant_id: Optional[str] = None,
) -> None:
    """Broadcast crawl progress event."""
    manager = get_connection_manager()
    await manager.broadcast(
        Event(
            type=EventType.CRAWL_PROGRESS,
            data={
                "url": url,
                "job_id": job_id,
                "message": message,
                "progress": progress,
            },
        ),
        tenant_id=tenant_id,
    )


async def broadcast_crawl_completed(
    url: str,
    job_id: str,
    chunks_indexed: int,
    tenant_id: Optional[str] = None,
) -> None:
    """Broadcast crawl completed event."""
    manager = get_connection_manager()
    await manager.broadcast(
        Event(
            type=EventType.CRAWL_COMPLETED,
            data={
                "url": url,
                "job_id": job_id,
                "chunks_indexed": chunks_indexed,
            },
        ),
        tenant_id=tenant_id,
    )


async def broadcast_crawl_failed(
    url: str,
    job_id: str,
    error: str,
    tenant_id: Optional[str] = None,
) -> None:
    """Broadcast crawl failed event."""
    manager = get_connection_manager()
    await manager.broadcast(
        Event(
            type=EventType.CRAWL_FAILED,
            data={
                "url": url,
                "job_id": job_id,
                "error": error,
            },
        ),
        tenant_id=tenant_id,
    )


async def broadcast_index_progress(
    chunks_indexed: int,
    total_chunks: int,
    current_url: str,
    tenant_id: Optional[str] = None,
) -> None:
    """Broadcast indexing progress event."""
    manager = get_connection_manager()
    await manager.broadcast(
        Event(
            type=EventType.INDEX_BATCH,
            data={
                "chunks_indexed": chunks_indexed,
                "total_chunks": total_chunks,
                "current_url": current_url,
                "progress": chunks_indexed / total_chunks if total_chunks > 0 else 0,
            },
        ),
        tenant_id=tenant_id,
    )


async def broadcast_stats_update(
    stats: Dict[str, Any],
    tenant_id: Optional[str] = None,
) -> None:
    """Broadcast stats update event."""
    manager = get_connection_manager()
    await manager.broadcast(
        Event(
            type=EventType.STATS_UPDATE,
            data=stats,
        ),
        tenant_id=tenant_id,
    )
