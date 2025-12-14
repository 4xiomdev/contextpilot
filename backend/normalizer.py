"""
ContextPilot Document Normalizer
Synthesizes raw chunks into clean, embedding-friendly documentation.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid

from google import genai

from .config import get_config
from .firestore_db import get_firestore_db as get_db
from .embed_manager import get_embed_manager

logger = logging.getLogger("contextpilot.normalizer")


@dataclass
class NormalizeResult:
    """Result of a normalization operation."""
    success: bool
    url_prefix: str
    title: str
    content: str = ""
    raw_chunk_count: int = 0
    pinecone_id: str = ""
    error: Optional[str] = None


class Normalizer:
    """
    Normalizes raw documentation chunks into clean, structured documents.
    
    Uses Gemini to synthesize multiple chunks into a single, well-organized
    markdown document that's optimized for embedding and retrieval.
    """
    
    def __init__(self):
        self.config = get_config()
        self.db = get_db()
        self.embed_manager = get_embed_manager()
        
        # Initialize Gemini client
        self._genai_client = genai.Client(api_key=self.config.google.api_key)
        
        # Output directory for markdown files
        self.output_dir = self.config.base_dir / "normalized_docs"
        self.output_dir.mkdir(exist_ok=True)
    
    def normalize(
        self,
        url_prefix: str,
        title: str,
        max_input_chars: int = 120_000,
    ) -> NormalizeResult:
        """
        Normalize all chunks matching a URL prefix into a single document.
        
        Args:
            url_prefix: URL prefix to filter chunks
            title: Title for the normalized document
            max_input_chars: Maximum input size per LLM call
        
        Returns:
            NormalizeResult with the synthesized content
        """
        logger.info(f"Normalizing docs for: {url_prefix}")
        
        # Search for matching chunks
        chunks = self.embed_manager.search(
            query=f"documentation for {title}",
            limit=500,  # Get a lot of chunks
            url_filter=url_prefix,
        )
        
        if not chunks:
            return NormalizeResult(
                success=False,
                url_prefix=url_prefix,
                title=title,
                error="No chunks found matching URL prefix",
            )
        
        # Collect unique content
        seen_hashes = set()
        corpus_parts = []
        
        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()[:16]
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            
            # Clean up the content
            cleaned = self._clean_content(chunk.content)
            if cleaned:
                corpus_parts.append(cleaned)
        
        if not corpus_parts:
            return NormalizeResult(
                success=False,
                url_prefix=url_prefix,
                title=title,
                error="No valid content after deduplication",
            )
        
        raw_chunk_count = len(corpus_parts)
        corpus = "\n\n---\n\n".join(corpus_parts)
        
        logger.info(f"Found {raw_chunk_count} unique chunks, {len(corpus)} chars")
        
        # Generate normalized document
        try:
            if len(corpus) > max_input_chars:
                # Split into parts and merge
                normalized = self._normalize_large_corpus(corpus, title, url_prefix, max_input_chars)
            else:
                normalized = self._generate_normalized(corpus, title, url_prefix)
            
            if not normalized:
                return NormalizeResult(
                    success=False,
                    url_prefix=url_prefix,
                    title=title,
                    error="LLM returned empty response",
                )
            
            # Index the normalized document
            embedding = self.embed_manager.get_embedding(normalized)
            pinecone_id = f"normalized_{uuid.uuid4().hex[:16]}"
            
            # Store in Pinecone (normalized namespace)
            if self.embed_manager._index:
                self.embed_manager._index.upsert(
                    vectors=[{
                        "id": pinecone_id,
                        "values": embedding,
                        "metadata": {
                            "url": url_prefix,
                            "page_url": url_prefix,
                            "title": title,
                            "content": normalized,
                            "type": "normalized",
                            "raw_chunk_count": raw_chunk_count,
                            "created_at": time.time(),
                        },
                    }],
                    namespace=self.config.pinecone.normalized_namespace,
                )
            
            # Track in SQLite
            doc_hash = hashlib.sha256(normalized.encode()).hexdigest()[:32]
            self.db.upsert_normalized_doc(
                url_prefix=url_prefix,
                title=title,
                doc_hash=doc_hash,
                pinecone_id=pinecone_id,
                raw_chunk_count=raw_chunk_count,
                content_preview=normalized[:500],
            )
            
            # Save to file
            self._save_to_file(title, normalized)
            
            return NormalizeResult(
                success=True,
                url_prefix=url_prefix,
                title=title,
                content=normalized,
                raw_chunk_count=raw_chunk_count,
                pinecone_id=pinecone_id,
            )
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return NormalizeResult(
                success=False,
                url_prefix=url_prefix,
                title=title,
                error=str(e),
            )
    
    def _generate_normalized(
        self,
        corpus: str,
        title: str,
        url_prefix: str,
    ) -> str:
        """Generate a normalized document using Gemini."""
        prompt = f"""You are creating a clean, embedding-friendly developer documentation artifact.

RULES:
- Use ONLY the provided SOURCE TEXT. Do not invent APIs or details.
- If information is unclear, say "See official documentation".
- Output clean Markdown.
- Keep code blocks intact.
- Focus on: model names, endpoints/methods, core workflows, code examples, parameters.
- Be concise but complete.

TITLE: {title}
URL: {url_prefix}

SOURCE TEXT:
{corpus}

---

Generate a well-structured API documentation document. Use clear headings, tables for parameters/models, and include any code examples from the source.
"""
        
        response = self._genai_client.models.generate_content(
            model=self.config.google.generation_model,
            contents=prompt,
        )
        
        if hasattr(response, "text"):
            return response.text.strip()
        elif isinstance(response, dict) and response.get("text"):
            return response["text"].strip()
        
        return ""
    
    def _normalize_large_corpus(
        self,
        corpus: str,
        title: str,
        url_prefix: str,
        max_input_chars: int,
    ) -> str:
        """Handle large corpus by splitting and merging."""
        # Split into manageable parts
        parts = []
        start = 0
        while start < len(corpus):
            end = min(len(corpus), start + max_input_chars)
            parts.append(corpus[start:end])
            start = end
        
        logger.info(f"Splitting corpus into {len(parts)} parts")
        
        # Normalize each part
        part_summaries = []
        for i, part in enumerate(parts):
            logger.info(f"Processing part {i + 1}/{len(parts)}")
            summary = self._generate_normalized(part, f"{title} (Part {i + 1})", url_prefix)
            if summary:
                part_summaries.append(summary)
        
        if len(part_summaries) == 1:
            return part_summaries[0]
        
        # Merge parts
        merge_prompt = f"""You are merging partial documentation summaries into one canonical document.

RULES:
- Combine the PARTIAL DOCS below into a single, coherent document.
- Remove duplicates and redundancy.
- Keep the best code examples.
- Maintain clear structure with headings.

TITLE: {title}

PARTIAL DOCS:
{"---".join(part_summaries)}

---

Generate a unified, well-structured documentation document.
"""
        
        response = self._genai_client.models.generate_content(
            model=self.config.google.generation_model,
            contents=merge_prompt,
        )
        
        if hasattr(response, "text"):
            return response.text.strip()
        elif isinstance(response, dict) and response.get("text"):
            return response["text"].strip()
        
        # Fallback: just concatenate
        return "\n\n---\n\n".join(part_summaries)
    
    def _clean_content(self, text: str) -> str:
        """Clean up content by removing boilerplate."""
        if not text:
            return ""
        
        lines = text.splitlines()
        cleaned = []
        
        skip_phrases = [
            "skip to main content",
            "we use cookies",
            "accept all",
            "reject all",
            "log in",
            "sign up",
            "sign in",
            "send feedback",
        ]
        
        for line in lines:
            lowered = line.strip().lower()
            if not lowered:
                cleaned.append("")
                continue
            
            # Skip common boilerplate
            if any(phrase in lowered for phrase in skip_phrases):
                continue
            
            cleaned.append(line)
        
        return "\n".join(cleaned).strip()
    
    def _save_to_file(self, title: str, content: str) -> Path:
        """Save normalized content to a markdown file."""
        # Create safe filename
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)
        safe_name = safe_name[:80].strip().replace(" ", "_")
        filename = f"{safe_name}.md"
        
        path = self.output_dir / filename
        path.write_text(content, encoding="utf-8")
        
        logger.info(f"Saved normalized doc to: {path}")
        return path
    
    def list_normalized_docs(self) -> List[Dict[str, Any]]:
        """List all normalized documents."""
        docs = self.db.list_normalized_docs()
        return [
            {
                "id": doc.id,
                "url_prefix": doc.url_prefix,
                "title": doc.title,
                "raw_chunk_count": doc.raw_chunk_count,
                "created_at": doc.created_at,
            }
            for doc in docs
        ]


# Singleton instance
_normalizer: Optional[Normalizer] = None


def get_normalizer() -> Normalizer:
    """Get the singleton normalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = Normalizer()
    return _normalizer


