"""
Vector store abstraction for ContextPilot.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import time

from pinecone import Pinecone, ServerlessSpec

from .config import Config

logger = logging.getLogger("contextpilot.vector_store")


@dataclass
class VectorMatch:
    """Normalized search match."""
    id: str
    score: float
    metadata: Dict[str, Any]


class VectorStore:
    """Base vector store interface."""

    def is_ready(self) -> bool:
        raise NotImplementedError

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str) -> None:
        raise NotImplementedError

    def query(
        self,
        vector: List[float],
        top_k: int,
        namespace: str,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[VectorMatch]:
        raise NotImplementedError

    def delete(self, ids: List[str], namespace: str) -> None:
        raise NotImplementedError

    def delete_all(self, namespace: str) -> None:
        """Delete all vectors in a namespace."""
        raise NotImplementedError

    def describe_stats(self) -> Dict[str, Any]:
        raise NotImplementedError


class PineconeVectorStore(VectorStore):
    """Pinecone-backed vector store."""

    def __init__(self, config: Config):
        self.config = config
        self._client: Optional[Pinecone] = None
        self._index = None
        self._init_pinecone()

    def _init_pinecone(self) -> None:
        try:
            self._client = Pinecone(api_key=self.config.pinecone.api_key)
            existing_indexes = [idx.name for idx in self._client.list_indexes()]
            if self.config.pinecone.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.config.pinecone.index_name}")
                self._client.create_index(
                    name=self.config.pinecone.index_name,
                    dimension=self.config.pinecone.dimension,
                    metric=self.config.pinecone.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.config.pinecone.environment,
                    ),
                )
                time.sleep(5)
            self._index = self._client.Index(self.config.pinecone.index_name)
            logger.info(f"Connected to Pinecone index: {self.config.pinecone.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self._client = None
            self._index = None

    def is_ready(self) -> bool:
        return self._index is not None

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str) -> None:
        if not self._index:
            raise RuntimeError("Pinecone not initialized")
        self._index.upsert(vectors=vectors, namespace=namespace)

    def query(
        self,
        vector: List[float],
        top_k: int,
        namespace: str,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[VectorMatch]:
        if not self._index:
            raise RuntimeError("Pinecone not initialized")
        results = self._index.query(
            vector=vector,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True,
            namespace=namespace,
        )
        matches = []
        for match in results.get("matches", []):
            matches.append(VectorMatch(
                id=match["id"],
                score=match["score"],
                metadata=match.get("metadata", {}),
            ))
        return matches

    def delete(self, ids: List[str], namespace: str) -> None:
        if not self._index:
            return
        self._index.delete(ids=ids, namespace=namespace)

    def delete_all(self, namespace: str) -> None:
        """Delete all vectors in a namespace."""
        if not self._index:
            return
        self._index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Deleted all vectors in namespace '{namespace}'")

    def describe_stats(self) -> Dict[str, Any]:
        if not self._index:
            return {"error": "Pinecone not initialized"}
        stats = self._index.describe_index_stats()
        namespaces = {}
        if hasattr(stats, "namespaces") and stats.namespaces:
            for ns_name, ns_data in stats.namespaces.items():
                namespaces[ns_name] = {
                    "vector_count": getattr(ns_data, "vector_count", 0)
                }
        return {
            "total_vectors": getattr(stats, "total_vector_count", 0),
            "namespaces": namespaces,
            "dimension": getattr(stats, "dimension", self.config.pinecone.dimension),
        }


class QdrantVectorStore(VectorStore):
    """Qdrant local vector store (embedded)."""

    def __init__(self, config: Config, tenant_id: str):
        self.config = config
        self.tenant_id = tenant_id
        self._client = None
        self._init_qdrant()

    def _init_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(path=str(self.config.qdrant.path))
            logger.info("Connected to local Qdrant store")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            self._client = None

    def _collection(self, namespace: str) -> str:
        if self.config.multi_tenant.enabled:
            prefix = f"{self.tenant_id}__"
            if namespace.startswith(prefix):
                return namespace
            return f"{prefix}{namespace}"
        return namespace

    def _ensure_collection(self, namespace: str) -> bool:
        if not self._client:
            return False
        collection = self._collection(namespace)
        try:
            from qdrant_client.http import models as rest
            existing = self._client.get_collections().collections
            if not any(c.name == collection for c in existing):
                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=rest.VectorParams(
                        size=self.config.pinecone.dimension,
                        distance=rest.Distance.COSINE,
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {e}")
            return False

    def is_ready(self) -> bool:
        return self._client is not None

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str) -> None:
        if not self._client or not self._ensure_collection(namespace):
            raise RuntimeError("Qdrant not initialized")
        from qdrant_client.http import models as rest
        points = [
            rest.PointStruct(
                id=v["id"],
                vector=v["values"],
                payload=v.get("metadata", {}),
            )
            for v in vectors
        ]
        self._client.upsert(collection_name=self._collection(namespace), points=points)

    def query(
        self,
        vector: List[float],
        top_k: int,
        namespace: str,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[VectorMatch]:
        if not self._client or not self._ensure_collection(namespace):
            raise RuntimeError("Qdrant not initialized")

        # Extract URL prefix filter if present
        url_prefix_filter = None
        if filter_dict and "url" in filter_dict:
            url_filter = filter_dict["url"]
            # Handle Pinecone-style prefix filter: {"$gte": prefix, "$lt": prefix + "\uffff"}
            if isinstance(url_filter, dict) and "$gte" in url_filter:
                url_prefix_filter = url_filter["$gte"]

        # Over-fetch if we need to post-filter by URL prefix
        fetch_limit = top_k * 3 if url_prefix_filter else top_k

        result = self._client.search(
            collection_name=self._collection(namespace),
            query_vector=vector,
            limit=fetch_limit,
            with_payload=True,
            score_threshold=None,
        )

        matches = []
        for match in result:
            metadata = match.payload or {}

            # Post-filter by URL prefix if specified
            if url_prefix_filter:
                url = metadata.get("url", "")
                if not url.startswith(url_prefix_filter):
                    continue

            matches.append(VectorMatch(
                id=str(match.id),
                score=match.score,
                metadata=metadata,
            ))

            # Stop once we have enough matches
            if len(matches) >= top_k:
                break

        return matches

    def delete(self, ids: List[str], namespace: str) -> None:
        if not self._client or not ids:
            return
        from qdrant_client.http import models as rest
        self._client.delete(
            collection_name=self._collection(namespace),
            points_selector=rest.PointIdsList(points=ids),
        )

    def describe_stats(self) -> Dict[str, Any]:
        if not self._client:
            return {"error": "Qdrant not initialized"}
        try:
            collections = self._client.get_collections().collections
            namespaces = {}
            for c in collections:
                info = self._client.get_collection(c.name)
                namespaces[c.name] = {"vector_count": info.points_count or 0}
            return {
                "total_vectors": sum(v["vector_count"] for v in namespaces.values()),
                "namespaces": namespaces,
                "dimension": self.config.pinecone.dimension,
            }
        except Exception as e:
            return {"error": str(e)}


def get_vector_store(config: Config, tenant_id: str) -> VectorStore:
    """Select the best available vector store."""
    provider = (config.vector_store_provider or "").lower()
    if provider == "pinecone" or (not provider and config.has_pinecone):
        return PineconeVectorStore(config)
    return QdrantVectorStore(config, tenant_id)
