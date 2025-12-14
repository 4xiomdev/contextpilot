"""
ContextPilot Embedding Manager
Handles Google embeddings and Pinecone vector storage.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import uuid

from google import genai
from pinecone import Pinecone, ServerlessSpec

from .config import get_config
from .firestore_db import get_firestore_db as get_db

logger = logging.getLogger("contextpilot.embed")


@dataclass
class SearchResult:
    """A search result from Pinecone."""
    id: str
    score: float
    url: str
    page_url: str
    title: str
    content: str
    namespace: str


@dataclass
class Chunk:
    """A document chunk ready for embedding."""
    content: str
    url: str
    page_url: str
    title: str
    content_hash: str = ""
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()[:32]


class EmbedManager:
    """
    Manages embeddings and Pinecone vector storage.
    
    Responsibilities:
    - Generate embeddings using Google's text-embedding-004
    - Store/retrieve vectors from Pinecone
    - Handle deduplication via content hashing
    """
    
    def __init__(self):
        self.config = get_config()
        self.db = get_db()
        
        # Initialize Google GenAI client
        self._genai_client = genai.Client(api_key=self.config.google.api_key)
        
        # Initialize Pinecone client
        self._pinecone_client: Optional[Pinecone] = None
        self._index = None
        
        if self.config.has_pinecone:
            self._init_pinecone()
    
    def _init_pinecone(self) -> None:
        """Initialize Pinecone client and ensure index exists."""
        try:
            self._pinecone_client = Pinecone(api_key=self.config.pinecone.api_key)
            
            # Check if index exists, create if not
            existing_indexes = [idx.name for idx in self._pinecone_client.list_indexes()]
            
            if self.config.pinecone.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.config.pinecone.index_name}")
                self._pinecone_client.create_index(
                    name=self.config.pinecone.index_name,
                    dimension=self.config.pinecone.dimension,
                    metric=self.config.pinecone.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.config.pinecone.environment,
                    )
                )
                # Wait for index to be ready
                time.sleep(5)
            
            self._index = self._pinecone_client.Index(self.config.pinecone.index_name)
            logger.info(f"Connected to Pinecone index: {self.config.pinecone.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self._pinecone_client = None
            self._index = None
    
    @property
    def is_ready(self) -> bool:
        """Check if the embed manager is ready to use."""
        return self._index is not None
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Returns a zero vector if the text is empty or embedding fails.
        """
        if not text or not text.strip():
            return self._zero_vector()
        
        try:
            response = self._genai_client.models.embed_content(
                model=self.config.google.embedding_model,
                contents=text
            )
            
            # Handle new API format: response.embeddings[0].values
            if hasattr(response, "embeddings") and response.embeddings:
                first_embedding = response.embeddings[0]
                if hasattr(first_embedding, "values"):
                    return list(first_embedding.values)
            
            # Legacy format: response.embedding.values
            if hasattr(response, "embedding") and hasattr(response.embedding, "values"):
                return list(response.embedding.values)
            
            # Dict format
            if isinstance(response, dict):
                embeddings = response.get("embeddings", [])
                if embeddings and isinstance(embeddings[0], dict):
                    values = embeddings[0].get("values")
                    if values:
                        return list(values)
            
            logger.warning("Unexpected embedding response format")
            return self._zero_vector()
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return self._zero_vector()
    
    def _zero_vector(self) -> List[float]:
        """Return a zero vector of the correct dimension."""
        return [0.0] * self.config.pinecone.dimension
    
    def _is_zero_vector(self, vector: List[float]) -> bool:
        """Check if a vector is all zeros."""
        return all(v == 0.0 for v in vector)
    
    def index_chunk(self, chunk: Chunk, namespace: str = "docs") -> Optional[str]:
        """
        Index a single chunk into Pinecone.
        
        Returns the Pinecone ID if successful, None if skipped (duplicate) or failed.
        """
        if not self._index:
            logger.error("Pinecone not initialized")
            return None
        
        # Check for duplicate via content hash
        existing = self.db.get_indexed_doc_by_hash(chunk.content_hash)
        if existing:
            logger.debug(f"Skipping duplicate chunk: {chunk.content_hash[:8]}...")
            return existing.pinecone_id
        
        # Generate embedding
        embedding = self.get_embedding(chunk.content)
        if self._is_zero_vector(embedding):
            logger.warning(f"Zero embedding for chunk, skipping")
            return None
        
        # Generate unique Pinecone ID
        pinecone_id = f"{namespace}_{uuid.uuid4().hex[:16]}"
        
        # Prepare metadata (Pinecone has metadata size limits)
        content_preview = chunk.content[:500] if len(chunk.content) > 500 else chunk.content
        
        metadata = {
            "url": chunk.url,
            "page_url": chunk.page_url,
            "title": chunk.title,
            "content": chunk.content,  # Full content for retrieval
            "content_hash": chunk.content_hash,
            "indexed_at": time.time(),
        }
        
        try:
            # Upsert to Pinecone
            self._index.upsert(
                vectors=[{
                    "id": pinecone_id,
                    "values": embedding,
                    "metadata": metadata,
                }],
                namespace=namespace,
            )
            
            # Track in SQLite
            self.db.upsert_indexed_doc(
                url=chunk.url,
                page_url=chunk.page_url,
                source_url=chunk.url,
                title=chunk.title,
                content_hash=chunk.content_hash,
                pinecone_id=pinecone_id,
                content_preview=content_preview,
            )
            
            logger.debug(f"Indexed chunk: {pinecone_id}")
            return pinecone_id
            
        except Exception as e:
            logger.error(f"Failed to index chunk: {e}")
            return None
    
    def index_chunks(
        self, 
        chunks: List[Chunk], 
        namespace: str = "docs",
        batch_size: int = 100
    ) -> int:
        """
        Index multiple chunks in batches.
        
        Returns the number of successfully indexed chunks.
        """
        if not self._index:
            logger.error("Pinecone not initialized")
            return 0
        
        indexed_count = 0
        
        # Process in batches for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = []
            
            for chunk in batch:
                # Skip duplicates
                existing = self.db.get_indexed_doc_by_hash(chunk.content_hash)
                if existing:
                    continue
                
                # Generate embedding
                embedding = self.get_embedding(chunk.content)
                if self._is_zero_vector(embedding):
                    continue
                
                pinecone_id = f"{namespace}_{uuid.uuid4().hex[:16]}"
                content_preview = chunk.content[:500]
                
                vectors.append({
                    "id": pinecone_id,
                    "values": embedding,
                    "metadata": {
                        "url": chunk.url,
                        "page_url": chunk.page_url,
                        "title": chunk.title,
                        "content": chunk.content,
                        "content_hash": chunk.content_hash,
                        "indexed_at": time.time(),
                    },
                })
                
                # Track in SQLite
                self.db.upsert_indexed_doc(
                    url=chunk.url,
                    page_url=chunk.page_url,
                    source_url=chunk.url,
                    title=chunk.title,
                    content_hash=chunk.content_hash,
                    pinecone_id=pinecone_id,
                    content_preview=content_preview,
                )
            
            if vectors:
                try:
                    self._index.upsert(vectors=vectors, namespace=namespace)
                    indexed_count += len(vectors)
                    logger.info(f"Indexed batch of {len(vectors)} chunks")
                except Exception as e:
                    logger.error(f"Failed to index batch: {e}")
        
        return indexed_count
    
    def search(
        self,
        query: str,
        limit: int = 10,
        url_filter: str = "",
        namespace: str = "docs",
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: The search query
            limit: Maximum number of results
            url_filter: Optional URL prefix to filter results
            namespace: Pinecone namespace to search
        
        Returns:
            List of SearchResult objects
        """
        if not self._index:
            logger.error("Pinecone not initialized")
            return []
        
        # Generate query embedding
        query_embedding = self.get_embedding(query)
        if self._is_zero_vector(query_embedding):
            logger.warning("Zero embedding for query")
            return []
        
        # Build filter if needed
        filter_dict = None
        if url_filter:
            # Pinecone filter for URL prefix matching
            filter_dict = {
                "url": {"$gte": url_filter, "$lt": url_filter + "\uffff"}
            }
        
        try:
            # Query Pinecone
            results = self._index.query(
                vector=query_embedding,
                top_k=limit * 2,  # Over-fetch for deduplication
                filter=filter_dict,
                include_metadata=True,
                namespace=namespace,
            )
            
            # Convert to SearchResult objects
            search_results = []
            seen_content_hashes = set()
            
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                content_hash = metadata.get("content_hash", "")
                
                # Deduplicate by content
                if content_hash in seen_content_hashes:
                    continue
                seen_content_hashes.add(content_hash)
                
                search_results.append(SearchResult(
                    id=match["id"],
                    score=match["score"],
                    url=metadata.get("url", ""),
                    page_url=metadata.get("page_url", ""),
                    title=metadata.get("title", ""),
                    content=metadata.get("content", ""),
                    namespace=namespace,
                ))
                
                if len(search_results) >= limit:
                    break
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_normalized(
        self,
        query: str,
        limit: int = 5,
        url_filter: str = "",
    ) -> List[SearchResult]:
        """Search normalized documents namespace."""
        return self.search(
            query=query,
            limit=limit,
            url_filter=url_filter,
            namespace=self.config.pinecone.normalized_namespace,
        )
    
    def delete_by_source(self, source_url: str) -> int:
        """
        Delete all vectors for a given source URL.
        
        Returns the number of deleted vectors.
        """
        if not self._index:
            return 0
        
        # Get Pinecone IDs from SQLite
        pinecone_ids = self.db.delete_indexed_docs_by_source(source_url)
        
        if pinecone_ids:
            try:
                # Delete from Pinecone in batches
                for i in range(0, len(pinecone_ids), 100):
                    batch = pinecone_ids[i:i + 100]
                    self._index.delete(ids=batch, namespace="docs")
                logger.info(f"Deleted {len(pinecone_ids)} vectors for {source_url}")
            except Exception as e:
                logger.error(f"Failed to delete vectors: {e}")
        
        return len(pinecone_ids)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        if not self._index:
            return {"error": "Pinecone not initialized"}
        
        try:
            stats = self._index.describe_index_stats()
            # Convert to serializable format
            namespaces = {}
            if hasattr(stats, 'namespaces') and stats.namespaces:
                for ns_name, ns_data in stats.namespaces.items():
                    namespaces[ns_name] = {
                        "vector_count": getattr(ns_data, 'vector_count', 0)
                    }
            
            return {
                "total_vectors": getattr(stats, 'total_vector_count', 0),
                "namespaces": namespaces,
                "dimension": getattr(stats, 'dimension', 768),
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}


# Singleton instance
_embed_manager: Optional[EmbedManager] = None


def get_embed_manager() -> EmbedManager:
    """Get the singleton embed manager instance."""
    global _embed_manager
    if _embed_manager is None:
        _embed_manager = EmbedManager()
    return _embed_manager

