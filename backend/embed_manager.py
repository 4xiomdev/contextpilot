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

from .config import get_config
from .firestore_db import get_firestore_db as get_db
from .tenant_context import get_tenant_id
from .vector_store import get_vector_store, VectorStore
from .genai_client import get_genai_client

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
    
    def __init__(self, tenant_id: Optional[str] = None):
        self.config = get_config()
        self.tenant_id = tenant_id or get_tenant_id()
        self.db = get_db(self.tenant_id)

        self._genai = get_genai_client()

        # Initialize vector store (Pinecone or local fallback)
        self._store: Optional[VectorStore] = None
        self._store = get_vector_store(self.config, self.tenant_id)

    def _ns(self, base: str) -> str:
        """Resolve the namespace for the current tenant."""
        if self.config.multi_tenant.enabled:
            return f"{self.tenant_id}__{base}"
        return base
    
    @property
    def is_ready(self) -> bool:
        """Check if the embed manager is ready to use."""
        return bool(self._store and self._store.is_ready())
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.

        Returns a zero vector if the text is empty or embedding fails.
        """
        if not text or not text.strip():
            return self._zero_vector()

        try:
            # Use google-generativeai API
            embedding = self._genai.embed(text, task_type="retrieval_document")
            return list(embedding) if embedding else self._zero_vector()

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
        if not self._store or not self._store.is_ready():
            logger.error("Vector store not initialized")
            return None
        
        # Check for duplicate via content hash
        existing = self.db.get_indexed_doc_by_hash(chunk.content_hash)
        if existing:
            logger.debug(f"Skipping duplicate chunk: {chunk.content_hash[:8]}...")
            return existing.pinecone_id

        try:
            self.db.upsert_raw_chunk(
                source_url=chunk.url,
                page_url=chunk.page_url,
                title=chunk.title,
                content=chunk.content,
                content_hash=chunk.content_hash,
            )
        except Exception as e:
            logger.debug(f"Failed to store raw chunk: {e}")
        
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
            self._store.upsert(
                vectors=[{
                    "id": pinecone_id,
                    "values": embedding,
                    "metadata": metadata,
                }],
                namespace=self._ns(namespace),
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
        if not self._store or not self._store.is_ready():
            logger.error("Vector store not initialized")
            return 0
        
        indexed_count = 0
        
        # Process in batches for efficiency
        ns = self._ns(namespace)
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

                try:
                    self.db.upsert_raw_chunk(
                        source_url=chunk.url,
                        page_url=chunk.page_url,
                        title=chunk.title,
                        content=chunk.content,
                        content_hash=chunk.content_hash,
                    )
                except Exception as e:
                    logger.debug(f"Failed to store raw chunk: {e}")
                
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
                    self._store.upsert(vectors=vectors, namespace=ns)
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
        if not self._store or not self._store.is_ready():
            logger.error("Vector store not initialized")
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
            ns = self._ns(namespace)
            results = self._store.query(
                vector=query_embedding,
                top_k=limit * 2,
                namespace=ns,
                filter_dict=filter_dict,
            )
            
            # Convert to SearchResult objects
            search_results = []
            seen_content_hashes = set()
            
            for match in results:
                metadata = match.metadata or {}
                if url_filter and not metadata.get("url", "").startswith(url_filter):
                    continue
                content_hash = metadata.get("content_hash", "")
                
                # Deduplicate by content
                if content_hash in seen_content_hashes:
                    continue
                seen_content_hashes.add(content_hash)
                
                search_results.append(SearchResult(
                    id=match.id,
                    score=match.score,
                    url=metadata.get("url", ""),
                    page_url=metadata.get("page_url", ""),
                    title=metadata.get("title", ""),
                    content=metadata.get("content", ""),
                    namespace=ns,
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

    def index_normalized_doc(
        self,
        content: str,
        url_prefix: str,
        title: str,
        raw_chunk_count: int,
    ) -> Optional[str]:
        """Index a normalized document into the normalized namespace."""
        if not self._store or not self._store.is_ready():
            logger.error("Vector store not initialized")
            return None

        embedding = self.get_embedding(content)
        if self._is_zero_vector(embedding):
            return None

        pinecone_id = f"normalized_{uuid.uuid4().hex[:16]}"
        metadata = {
            "url": url_prefix,
            "page_url": url_prefix,
            "title": title,
            "content": content,
            "type": "normalized",
            "raw_chunk_count": raw_chunk_count,
            "created_at": time.time(),
        }

        try:
            self._store.upsert(
                vectors=[{"id": pinecone_id, "values": embedding, "metadata": metadata}],
                namespace=self._ns(self.config.pinecone.normalized_namespace),
            )
            return pinecone_id
        except Exception as e:
            logger.error(f"Failed to index normalized doc: {e}")
            return None

    def hybrid_search(
        self,
        query: str,
        keywords: List[str],
        limit: int = 10,
        url_filter: str = "",
        namespace: str = "docs",
        keyword_boost: float = 0.2,
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity with keyword matching.

        Args:
            query: The search query for embedding
            keywords: Additional keywords to boost matching results
            limit: Maximum number of results
            url_filter: Optional URL prefix to filter results
            namespace: Pinecone namespace to search
            keyword_boost: Score boost for keyword matches (0.0-1.0)

        Returns:
            List of SearchResult objects with boosted scores
        """
        if not self._store or not self._store.is_ready():
            logger.error("Vector store not initialized")
            return []

        # Get base vector search results
        base_results = self.search(
            query=query,
            limit=limit * 2,  # Over-fetch for re-scoring
            url_filter=url_filter,
            namespace=namespace,
        )

        if not keywords or not base_results:
            return base_results[:limit]

        # Apply keyword boosting
        keywords_lower = [kw.lower() for kw in keywords]
        boosted_results = []

        for result in base_results:
            # Check for keyword matches in title and content
            text_lower = f"{result.title} {result.content}".lower()
            matches = sum(1 for kw in keywords_lower if kw in text_lower)

            # Calculate boost based on keyword matches
            if matches > 0:
                # Boost proportional to number of matches
                boost = min(keyword_boost * (matches / len(keywords)), keyword_boost)
                boosted_score = min(result.score + boost, 1.0)
            else:
                boosted_score = result.score

            # Create new result with boosted score
            boosted_results.append(SearchResult(
                id=result.id,
                score=boosted_score,
                url=result.url,
                page_url=result.page_url,
                title=result.title,
                content=result.content,
                namespace=result.namespace,
            ))

        # Sort by boosted score and return top results
        boosted_results.sort(key=lambda r: r.score, reverse=True)
        return boosted_results[:limit]

    def multi_query_search(
        self,
        queries: List[str],
        limit: int = 10,
        url_filter: str = "",
        namespace: str = "docs",
    ) -> List[SearchResult]:
        """
        Search with multiple query variants and merge results.

        Useful for query expansion where the original query is rewritten
        into multiple forms.

        Args:
            queries: List of query variants to search
            limit: Maximum total results
            url_filter: Optional URL prefix filter
            namespace: Pinecone namespace

        Returns:
            Merged and deduplicated search results
        """
        if not queries:
            return []

        all_results: Dict[str, SearchResult] = {}

        for query in queries[:5]:  # Limit to 5 query variants
            results = self.search(
                query=query,
                limit=limit,
                url_filter=url_filter,
                namespace=namespace,
            )

            for result in results:
                key = result.id
                if key not in all_results:
                    all_results[key] = result
                else:
                    # Keep the higher score
                    if result.score > all_results[key].score:
                        all_results[key] = result

        # Sort by score and return
        merged = list(all_results.values())
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:limit]

    # =========================================================================
    # Incremental Chunk Updates (Phase 5)
    # =========================================================================

    def update_chunks_incremental(
        self,
        source_id: str,
        page_url: str,
        chunks: List[Chunk],
        namespace: str = "docs",
    ) -> Dict[str, int]:
        """
        Incrementally update chunks for a page using chunk-level hashing.

        Compares new chunks against stored hashes to determine:
        - Which chunks are new (need to be indexed)
        - Which chunks changed (need re-indexing)
        - Which chunks were removed (need deletion from vector store)
        - Which chunks are unchanged (skip)

        Args:
            source_id: Source ID for hash tracking
            page_url: URL of the page being updated
            chunks: New chunks for this page
            namespace: Vector store namespace

        Returns:
            Dict with counts: {"added": N, "updated": N, "deleted": N, "unchanged": N}
        """
        if not self._store or not self._store.is_ready():
            logger.error("Vector store not initialized")
            return {"added": 0, "updated": 0, "deleted": 0, "unchanged": 0, "errors": 1}

        # Build current chunk hashes
        current_hashes: Dict[int, str] = {}
        chunk_by_index: Dict[int, Chunk] = {}
        for i, chunk in enumerate(chunks):
            current_hashes[i] = chunk.content_hash
            chunk_by_index[i] = chunk

        # Compare with stored hashes
        diff = self.db.get_stale_chunks(source_id, page_url, current_hashes)

        result = {
            "added": 0,
            "updated": 0,
            "deleted": 0,
            "unchanged": len(diff["unchanged"]),
            "errors": 0,
        }

        ns = self._ns(namespace)

        # Delete removed chunks from vector store
        if diff["deleted"]:
            stored_hashes = self.db.get_chunk_hashes(source_id, page_url)
            pinecone_ids_to_delete = []
            for idx in diff["deleted"]:
                if idx in stored_hashes and stored_hashes[idx].get("pinecone_id"):
                    pinecone_ids_to_delete.append(stored_hashes[idx]["pinecone_id"])

            if pinecone_ids_to_delete:
                try:
                    self._store.delete(ids=pinecone_ids_to_delete, namespace=ns)
                    result["deleted"] = len(pinecone_ids_to_delete)
                    logger.debug(f"Deleted {len(pinecone_ids_to_delete)} chunks from {page_url}")
                except Exception as e:
                    logger.error(f"Failed to delete chunks: {e}")
                    result["errors"] += 1

            # Delete hash records for removed chunks
            self.db.delete_chunk_hashes(source_id, page_url, diff["deleted"])

        # Index new and modified chunks
        chunks_to_index = diff["added"] + diff["modified"]
        if chunks_to_index:
            new_hash_records = []

            for idx in chunks_to_index:
                chunk = chunk_by_index.get(idx)
                if not chunk:
                    continue

                # Generate embedding
                embedding = self.get_embedding(chunk.content)
                if self._is_zero_vector(embedding):
                    logger.warning(f"Zero embedding for chunk {idx} in {page_url}")
                    result["errors"] += 1
                    continue

                pinecone_id = f"{namespace}_{uuid.uuid4().hex[:16]}"
                content_preview = chunk.content[:500]

                metadata = {
                    "url": chunk.url,
                    "page_url": chunk.page_url,
                    "title": chunk.title,
                    "content": chunk.content,
                    "content_hash": chunk.content_hash,
                    "chunk_index": idx,
                    "indexed_at": time.time(),
                }

                try:
                    # Upsert to vector store
                    self._store.upsert(
                        vectors=[{
                            "id": pinecone_id,
                            "values": embedding,
                            "metadata": metadata,
                        }],
                        namespace=ns,
                    )

                    # Track hash record
                    new_hash_records.append({
                        "index": idx,
                        "hash": chunk.content_hash,
                        "pinecone_id": pinecone_id,
                    })

                    # Store raw chunk
                    try:
                        self.db.upsert_raw_chunk(
                            source_url=chunk.url,
                            page_url=chunk.page_url,
                            title=chunk.title,
                            content=chunk.content,
                            content_hash=chunk.content_hash,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to store raw chunk: {e}")

                    # Track in indexed_docs
                    self.db.upsert_indexed_doc(
                        url=chunk.url,
                        page_url=chunk.page_url,
                        source_url=chunk.url,
                        title=chunk.title,
                        content_hash=chunk.content_hash,
                        pinecone_id=pinecone_id,
                        content_preview=content_preview,
                    )

                    if idx in diff["added"]:
                        result["added"] += 1
                    else:
                        result["updated"] += 1

                except Exception as e:
                    logger.error(f"Failed to index chunk {idx}: {e}")
                    result["errors"] += 1

            # Store new hash records
            if new_hash_records:
                self.db.upsert_chunk_hashes_batch(source_id, page_url, new_hash_records)

        logger.info(
            f"Incremental update for {page_url}: "
            f"+{result['added']} ~{result['updated']} -{result['deleted']} ={result['unchanged']}"
        )

        return result

    def delete_by_source(self, source_url: str) -> int:
        """
        Delete all vectors for a given source URL.
        
        Returns the number of deleted vectors.
        """
        if not self._store or not self._store.is_ready():
            return 0
        
        # Get Pinecone IDs from SQLite
        pinecone_ids = self.db.delete_indexed_docs_by_source(source_url)
        
        if pinecone_ids:
            try:
                # Delete from Pinecone in batches
                for i in range(0, len(pinecone_ids), 100):
                    batch = pinecone_ids[i:i + 100]
                    self._store.delete(ids=batch, namespace=self._ns(self.config.pinecone.docs_namespace))
                logger.info(f"Deleted {len(pinecone_ids)} vectors for {source_url}")
            except Exception as e:
                logger.error(f"Failed to delete vectors: {e}")

        try:
            self.db.delete_raw_chunks_by_source(source_url)
        except Exception as e:
            logger.warning(f"Failed to delete raw chunks for {source_url}: {e}")
        
        return len(pinecone_ids)

    def delete_all(self) -> int:
        """
        Delete ALL vectors from the index (nuclear option).

        Returns estimated count of deleted vectors.
        """
        if not self._store or not self._store.is_ready():
            return 0

        try:
            # Get stats before delete to know count
            stats = self._store.describe_stats()
            total_vectors = stats.get("total_vectors", 0)

            # Delete all in the docs namespace
            namespace = self._ns(self.config.pinecone.docs_namespace)
            self._store.delete_all(namespace=namespace)
            logger.info(f"Deleted all vectors from namespace '{namespace}' (~{total_vectors} vectors)")

            return total_vectors
        except Exception as e:
            logger.error(f"Failed to delete all vectors: {e}")
            return 0

    def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        if not self._store or not self._store.is_ready():
            return {"error": "Vector store not initialized"}
        
        try:
            return self._store.describe_stats()
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}


# Singleton instances (keyed by tenant id).
_embed_managers: Dict[str, EmbedManager] = {}


def get_embed_manager(tenant_id: Optional[str] = None) -> EmbedManager:
    """Get the embed manager instance for the current tenant."""
    key = tenant_id or get_tenant_id()
    if key not in _embed_managers:
        _embed_managers[key] = EmbedManager(tenant_id=key)
    return _embed_managers[key]
