"""
ContextPilot Document Normalizer
Synthesizes raw chunks into clean, embedding-friendly documentation.

Features:
- LLM-powered content synthesis
- Quality validation and post-processing
- Lineage tracking (source chunks → normalized sections)
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from .config import get_config
from .firestore_db import get_firestore_db as get_db
from .embed_manager import get_embed_manager
from .tenant_context import get_tenant_id
from .genai_client import get_genai_client
from .quality_metrics import (
    NormalizationQuality,
    NormalizationLineage,
    LineageEntry,
    QualityAnalyzer,
    validate_normalized_output,
    post_process_output,
)

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
    quality: Optional[NormalizationQuality] = None
    lineage: Optional[NormalizationLineage] = None
    validation_issues: List[str] = field(default_factory=list)


class Normalizer:
    """
    Normalizes raw documentation chunks into clean, structured documents.
    
    Uses Gemini to synthesize multiple chunks into a single, well-organized
    markdown document that's optimized for embedding and retrieval.
    """
    
    def __init__(self, tenant_id: Optional[str] = None):
        self.config = get_config()
        self.tenant_id = tenant_id or get_tenant_id()
        self.db = get_db(self.tenant_id)
        self.embed_manager = get_embed_manager(self.tenant_id)

        self._genai = get_genai_client()
        self._quality_analyzer = QualityAnalyzer(genai_client=self._genai)

        # Output directory for markdown files
        self.output_dir = self.config.base_dir / "normalized_docs"
        self.output_dir.mkdir(exist_ok=True)
    
    def normalize(
        self,
        url_prefix: str,
        title: str,
        max_input_chars: int = 120_000,
        validate_output: bool = True,
        track_lineage: bool = True,
    ) -> NormalizeResult:
        """
        Normalize all chunks matching a URL prefix into a single document.

        Args:
            url_prefix: URL prefix to filter chunks
            title: Title for the normalized document
            max_input_chars: Maximum input size per LLM call
            validate_output: Whether to validate and post-process output
            track_lineage: Whether to track source chunk lineage

        Returns:
            NormalizeResult with the synthesized content
        """
        logger.info(f"Normalizing docs for: {url_prefix}")

        # Load raw chunks by URL prefix (more reliable than vector search)
        try:
            chunks = self.db.list_raw_chunks_by_prefix(url_prefix=url_prefix, limit=2000)
        except Exception as e:
            logger.warning(f"Failed to load raw chunks from DB: {e}")
            chunks = []

        if not chunks:
            return NormalizeResult(
                success=False,
                url_prefix=url_prefix,
                title=title,
                error="No raw chunks found matching URL prefix",
            )

        # Collect unique content with tracking
        seen_hashes = set()
        corpus_parts = []
        chunk_map: Dict[str, Dict[str, Any]] = {}  # hash -> chunk info for lineage

        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()[:16]
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # Clean up the content
            cleaned = self._clean_content(chunk.content)
            if cleaned:
                corpus_parts.append(cleaned)
                chunk_map[content_hash] = {
                    "id": str(chunk.id),
                    "page_url": chunk.page_url,
                    "title": chunk.title,
                    "content_hash": content_hash,
                }

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

            # Validate and post-process output
            validation_issues: List[str] = []
            if validate_output:
                is_valid, issues = validate_normalized_output(normalized)
                validation_issues = issues
                if issues:
                    logger.warning(f"Validation issues: {issues}")

                # Post-process to clean up common issues
                normalized = post_process_output(normalized)

            # Analyze quality
            source_chunks_for_analysis = [
                {"id": v["id"], "content": corpus_parts[i]}
                for i, v in enumerate(chunk_map.values())
                if i < len(corpus_parts)
            ]
            quality = self._quality_analyzer.analyze_normalization(
                normalized_content=normalized,
                source_chunks=source_chunks_for_analysis,
                used_chunk_ids=list(chunk_map.keys()),
            )

            if quality.errors:
                logger.warning(f"Quality errors: {quality.errors}")

            # Build lineage tracking
            lineage: Optional[NormalizationLineage] = None
            if track_lineage:
                lineage = self._build_lineage(
                    normalized_doc_id="",  # Will be set after indexing
                    url_prefix=url_prefix,
                    normalized_content=normalized,
                    chunk_map=chunk_map,
                )

            # Index the normalized document
            pinecone_id = self.embed_manager.index_normalized_doc(
                content=normalized,
                url_prefix=url_prefix,
                title=title,
                raw_chunk_count=raw_chunk_count,
            )
            if not pinecone_id:
                logger.warning("Failed to index normalized document")

            # Update lineage with doc ID
            if lineage:
                lineage.normalized_doc_id = pinecone_id or ""

            # Track in database
            doc_hash = hashlib.sha256(normalized.encode()).hexdigest()[:32]
            self.db.upsert_normalized_doc(
                url_prefix=url_prefix,
                title=title,
                doc_hash=doc_hash,
                pinecone_id=pinecone_id,
                raw_chunk_count=raw_chunk_count,
                content_preview=normalized[:500],
            )

            # Store lineage if available
            if lineage and hasattr(self.db, "store_normalization_lineage"):
                try:
                    self.db.store_normalization_lineage(lineage.to_dict())
                except Exception as e:
                    logger.warning(f"Failed to store lineage: {e}")

            # Save to file
            self._save_to_file(title, normalized)

            return NormalizeResult(
                success=True,
                url_prefix=url_prefix,
                title=title,
                content=normalized,
                raw_chunk_count=raw_chunk_count,
                pinecone_id=pinecone_id,
                quality=quality,
                lineage=lineage,
                validation_issues=validation_issues,
            )

        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return NormalizeResult(
                success=False,
                url_prefix=url_prefix,
                title=title,
                error=str(e),
            )

    def _build_lineage(
        self,
        normalized_doc_id: str,
        url_prefix: str,
        normalized_content: str,
        chunk_map: Dict[str, Dict[str, Any]],
    ) -> NormalizationLineage:
        """
        Build lineage tracking for a normalized document.

        Maps sections in the normalized doc to source chunks.
        """
        import re

        # Extract sections from normalized content
        sections: List[LineageEntry] = []
        section_pattern = r"^(#{1,3})\s+(.+)$"
        current_section_idx = 0

        for match in re.finditer(section_pattern, normalized_content, re.MULTILINE):
            section_title = match.group(2).strip()

            # Find which source chunks likely contributed to this section
            # by checking for content overlap
            contributing_chunks = []
            contributing_urls = []

            section_start = match.end()
            next_match = re.search(section_pattern, normalized_content[section_start:], re.MULTILINE)
            section_end = section_start + next_match.start() if next_match else len(normalized_content)
            section_content = normalized_content[section_start:section_end].lower()

            for chunk_hash, chunk_info in chunk_map.items():
                # Simple overlap check - look for shared phrases
                chunk_id = chunk_info["id"]
                page_url = chunk_info["page_url"]

                # Add if section title or content matches chunk title
                chunk_title = (chunk_info.get("title") or "").lower()
                if chunk_title and chunk_title in section_content:
                    if chunk_id not in contributing_chunks:
                        contributing_chunks.append(chunk_id)
                        if page_url not in contributing_urls:
                            contributing_urls.append(page_url)

            sections.append(LineageEntry(
                section_title=section_title,
                section_index=current_section_idx,
                source_chunk_ids=contributing_chunks,
                source_page_urls=contributing_urls,
            ))
            current_section_idx += 1

        # Count total used chunks
        all_used_chunks = set()
        for section in sections:
            all_used_chunks.update(section.source_chunk_ids)

        return NormalizationLineage(
            normalized_doc_id=normalized_doc_id,
            url_prefix=url_prefix,
            source_chunk_count=len(chunk_map),
            used_chunk_count=len(all_used_chunks),
            sections=sections,
            created_at=time.time(),
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
        
        return self._genai.generate(prompt, temperature=0.4, max_output_tokens=2000)
    
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
        
        merged = self._genai.generate(merge_prompt, temperature=0.4, max_output_tokens=2000)
        if merged:
            return merged
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


# Singleton instances (keyed by tenant id).
_normalizers: dict[str, Normalizer] = {}


def get_normalizer(tenant_id: Optional[str] = None) -> Normalizer:
    """Get the normalizer instance for the current tenant."""
    key = tenant_id or get_tenant_id()
    if key not in _normalizers:
        _normalizers[key] = Normalizer(tenant_id=key)
    return _normalizers[key]
