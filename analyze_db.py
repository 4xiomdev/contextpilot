#!/usr/bin/env python3
"""
ContextPilot Database Analysis Script
Deep analysis of database, vectors, and document quality.
"""

import os
import sys
import warnings
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import get_config
from backend.firestore_db import get_firestore_db
from backend.embed_manager import get_embed_manager

def format_timestamp(ts):
    """Format Unix timestamp to readable date."""
    if not ts:
        return "Never"
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')

def analyze():
    config = get_config()

    print("="*70)
    print("CONTEXTPILOT DATABASE DEEP ANALYSIS")
    print("="*70)

    # ==================== CONFIGURATION ====================
    print("\n[1] CONFIGURATION STATUS")
    print("-"*50)
    print(f"  Pinecone configured: {config.has_pinecone}")
    print(f"  Firestore configured: {config.has_firestore}")
    print(f"  Firecrawl configured: {config.has_firecrawl}")
    print(f"  Gemini configured: {config.has_gemini}")
    print(f"  Pinecone Index: {config.pinecone.index_name}")
    print(f"  Embedding Model: {config.google.embedding_model}")
    print(f"  Generation Model: {config.google.generation_model}")
    print(f"  Embedding Dimension: {config.pinecone.dimension}")

    # Get database instance
    db = get_firestore_db()
    is_firestore = config.has_firestore
    print(f"\n  Database backend: {'Firestore' if is_firestore else 'SQLite'}")
    print(f"  Database ready: True")  # SQLite is always ready if initialized

    # ==================== DATABASE STATS ====================
    print("\n[2] DATABASE STATISTICS")
    print("-"*50)
    try:
        stats = db.get_stats()

        print(f"\n  CRAWL JOBS:")
        print(f"    Total jobs: {stats['crawl_jobs']['total']}")
        print(f"    Completed: {stats['crawl_jobs']['completed']}")
        print(f"    Failed: {stats['crawl_jobs']['failed']}")
        print(f"    Running: {stats['crawl_jobs']['running']}")

        print(f"\n  INDEXED DOCUMENTS:")
        print(f"    Total chunks: {stats['indexed_docs']['total']}")
        print(f"    Unique sources: {stats['indexed_docs']['sources']}")

        print(f"\n  NORMALIZED DOCUMENTS:")
        print(f"    Total: {stats['normalized_docs']['total']}")

        if 'sources' in stats:
            print(f"\n  SOURCES REGISTRY:")
            for k, v in stats['sources'].items():
                print(f"    {k}: {v}")
        else:
            print(f"\n  SOURCES REGISTRY: (SQLite mode - no sources table)")

    except Exception as e:
        print(f"  Error getting stats: {e}")

    # ==================== INDEXED SOURCES DETAIL ====================
    print("\n[3] INDEXED SOURCES BREAKDOWN")
    print("-"*50)
    try:
        sources = db.list_indexed_sources()
        total_chunks = 0

        if not sources:
            print("  No indexed sources found!")
        else:
            for i, src in enumerate(sources, 1):
                total_chunks += src['chunks']
                last_idx = format_timestamp(src['last_indexed'])
                url = src['source_url']

                # Categorize URL
                if 'google' in url or 'ai.google' in url:
                    category = "Google AI"
                elif 'openai' in url:
                    category = "OpenAI"
                elif 'anthropic' in url:
                    category = "Anthropic"
                elif 'github' in url:
                    category = "GitHub"
                elif 'firebase' in url:
                    category = "Firebase"
                else:
                    category = "Other"

                print(f"\n  {i}. [{category}] {url[:65]}{'...' if len(url) > 65 else ''}")
                print(f"     Chunks: {src['chunks']}")
                print(f"     Last indexed: {last_idx}")

            print(f"\n  TOTAL: {len(sources)} sources, {total_chunks} chunks")

    except Exception as e:
        print(f"  Error: {e}")

    # ==================== NORMALIZED DOCS ====================
    print("\n[4] NORMALIZED DOCUMENTS")
    print("-"*50)
    try:
        normalized = db.list_normalized_docs()

        if not normalized:
            print("  ⚠️  NO NORMALIZED DOCUMENTS FOUND!")
            print("\n  This is a problem - normalization synthesizes raw chunks")
            print("  into clean, embedding-friendly documents.")
            print("\n  To normalize, call: build_normalized_doc MCP tool")
        else:
            for doc in normalized:
                created = format_timestamp(doc.created_at)
                print(f"\n  ✓ {doc.title}")
                print(f"    URL Prefix: {doc.url_prefix[:55]}...")
                print(f"    Raw chunks synthesized: {doc.raw_chunk_count}")
                print(f"    Created: {created}")

    except Exception as e:
        print(f"  Error: {e}")

    # ==================== PINECONE STATS ====================
    print("\n[5] PINECONE VECTOR INDEX")
    print("-"*50)
    try:
        embed_mgr = get_embed_manager()

        if not embed_mgr.is_ready:
            print("  ⚠️  Pinecone not initialized!")
        else:
            stats = embed_mgr.get_index_stats()

            if 'error' in stats:
                print(f"  Error: {stats['error']}")
            else:
                print(f"  Total vectors: {stats.get('total_vectors', 0):,}")
                print(f"  Dimension: {stats.get('dimension', 768)}")

                namespaces = stats.get('namespaces', {})
                print(f"\n  NAMESPACES:")

                docs_count = 0
                normalized_count = 0

                for ns, data in namespaces.items():
                    count = data.get('vector_count', 0)
                    print(f"    - {ns}: {count:,} vectors")

                    if 'docs' in ns:
                        docs_count += count
                    if 'normalized' in ns:
                        normalized_count += count

                # Analysis
                print(f"\n  ANALYSIS:")
                print(f"    Raw doc vectors: {docs_count:,}")
                print(f"    Normalized vectors: {normalized_count:,}")

                if docs_count > 0 and normalized_count == 0:
                    print(f"\n  ⚠️  WARNING: You have {docs_count:,} raw chunks but 0 normalized docs!")
                    print("     Normalized docs improve retrieval quality significantly.")

    except Exception as e:
        print(f"  Error: {e}")

    # ==================== CRAWL JOBS ANALYSIS ====================
    print("\n[6] RECENT CRAWL JOBS")
    print("-"*50)
    try:
        jobs = db.list_crawl_jobs(limit=10)

        if not jobs:
            print("  No crawl jobs found")
        else:
            for job in jobs:
                created = format_timestamp(job.created_at)
                status_icon = {
                    'completed': '✓',
                    'failed': '✗',
                    'running': '⟳',
                    'pending': '○'
                }.get(job.status.value, '?')

                print(f"\n  {status_icon} {job.url[:55]}{'...' if len(job.url) > 55 else ''}")
                print(f"    Status: {job.status.value}")
                print(f"    Chunks: {job.chunks_count}")
                print(f"    Created: {created}")
                if job.error_message:
                    print(f"    Error: {job.error_message[:60]}...")

    except Exception as e:
        print(f"  Error: {e}")

    # ==================== SOURCES REGISTRY ====================
    print("\n[7] SOURCES REGISTRY (Crawl Schedule)")
    print("-"*50)
    try:
        if hasattr(db, 'list_sources') or (hasattr(db, '_impl') and hasattr(db._impl, 'list_sources')):
            sources_list = db.list_sources(limit=20)

            if not sources_list:
                print("  No sources registered")
            else:
                healthy = 0
                stale = 0
                error = 0

                for src in sources_list:
                    status_icon = {
                        'healthy': '✓',
                        'stale': '⚠',
                        'error': '✗',
                        'unknown': '?'
                    }.get(src.health_status.value, '?')

                    if src.health_status.value == 'healthy':
                        healthy += 1
                    elif src.health_status.value == 'stale':
                        stale += 1
                    elif src.health_status.value == 'error':
                        error += 1

                    last_crawl = format_timestamp(src.last_crawled_at)
                    next_crawl = format_timestamp(src.next_crawl_at)

                    print(f"\n  {status_icon} {src.name}")
                    print(f"    URL: {src.base_url[:50]}...")
                    print(f"    Frequency: {src.crawl_frequency.value}")
                    print(f"    Chunks: {src.chunks_count}")
                    print(f"    Last crawl: {last_crawl}")
                    print(f"    Next crawl: {next_crawl}")
                    print(f"    Enabled: {src.is_enabled}")

                print(f"\n  HEALTH SUMMARY:")
                print(f"    Healthy: {healthy}")
                print(f"    Stale: {stale}")
                print(f"    Error: {error}")
        else:
            print("  (Sources registry not available in SQLite mode)")
            print("  The crawl job history shows what was crawled.")

    except Exception as e:
        print(f"  Error: {e}")

    # ==================== CHUNK QUALITY SAMPLE ====================
    print("\n[8] CHUNK QUALITY ANALYSIS")
    print("-"*50)
    try:
        # Sample some chunks from Pinecone to analyze quality
        embed_mgr = get_embed_manager()

        if embed_mgr.is_ready:
            # Do a test search
            test_results = embed_mgr.search(
                query="API documentation guide",
                limit=5,
            )

            if test_results:
                print(f"  Sample of {len(test_results)} chunks:\n")

                content_lengths = []
                for i, r in enumerate(test_results, 1):
                    content_len = len(r.content)
                    content_lengths.append(content_len)

                    print(f"  {i}. Score: {r.score:.3f}")
                    print(f"     Title: {r.title[:50]}...")
                    print(f"     URL: {r.page_url[:50]}...")
                    print(f"     Content length: {content_len} chars")
                    print(f"     Preview: {r.content[:100]}...")
                    print()

                avg_len = sum(content_lengths) / len(content_lengths)
                print(f"  CHUNK SIZE STATS:")
                print(f"    Average length: {avg_len:.0f} chars")
                print(f"    Min: {min(content_lengths)} chars")
                print(f"    Max: {max(content_lengths)} chars")

                if avg_len < 200:
                    print(f"\n  ⚠️  WARNING: Chunks are small (avg {avg_len:.0f} chars)")
                    print("     Small chunks may lack context for good retrieval.")
                elif avg_len > 2000:
                    print(f"\n  ⚠️  WARNING: Chunks are large (avg {avg_len:.0f} chars)")
                    print("     Large chunks may be inefficient and dilute relevance.")
                else:
                    print(f"\n  ✓ Chunk sizes look reasonable")
            else:
                print("  No results from test search")

    except Exception as e:
        print(f"  Error: {e}")

    # ==================== RECOMMENDATIONS ====================
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    recommendations = []

    # Check normalized docs
    try:
        normalized = db.list_normalized_docs()
        sources = db.list_indexed_sources()

        if not normalized and sources:
            recommendations.append({
                "priority": "HIGH",
                "issue": "No normalized documents",
                "impact": "Raw chunks are harder to retrieve accurately",
                "fix": "Run build_normalized_doc for each major doc source"
            })
        elif len(normalized) < len(sources) * 0.5:
            recommendations.append({
                "priority": "MEDIUM",
                "issue": f"Only {len(normalized)} normalized docs for {len(sources)} sources",
                "impact": "Some documentation isn't optimized for retrieval",
                "fix": "Normalize remaining sources"
            })
    except:
        pass

    # Check Pinecone stats
    try:
        embed_mgr = get_embed_manager()
        if embed_mgr.is_ready:
            stats = embed_mgr.get_index_stats()
            namespaces = stats.get('namespaces', {})

            normalized_ns = [ns for ns in namespaces if 'normalized' in ns]
            if not normalized_ns or all(namespaces[ns].get('vector_count', 0) == 0 for ns in normalized_ns):
                recommendations.append({
                    "priority": "HIGH",
                    "issue": "No vectors in normalized namespace",
                    "impact": "Deep search won't find synthesized docs",
                    "fix": "Run normalization to populate normalized namespace"
                })
    except:
        pass

    # Check sources health (only for Firestore)
    try:
        if hasattr(db, 'list_sources') or (hasattr(db, '_impl') and hasattr(db._impl, 'list_sources')):
            sources_list = db.list_sources()
            stale = [s for s in sources_list if s.health_status.value == 'stale']
            error = [s for s in sources_list if s.health_status.value == 'error']

            if stale:
                recommendations.append({
                    "priority": "MEDIUM",
                    "issue": f"{len(stale)} sources are stale",
                    "impact": "Documentation may be outdated",
                    "fix": "Re-crawl stale sources"
                })

            if error:
                recommendations.append({
                    "priority": "HIGH",
                    "issue": f"{len(error)} sources have errors",
                    "impact": "These sources aren't being indexed",
                    "fix": "Check error messages and fix crawl issues"
                })
    except:
        pass

    # Print recommendations
    if recommendations:
        for i, rec in enumerate(sorted(recommendations, key=lambda x: x['priority']), 1):
            priority_icon = '🔴' if rec['priority'] == 'HIGH' else '🟡' if rec['priority'] == 'MEDIUM' else '🟢'
            print(f"\n{priority_icon} [{rec['priority']}] {rec['issue']}")
            print(f"   Impact: {rec['impact']}")
            print(f"   Fix: {rec['fix']}")
    else:
        print("\n✓ No critical issues found!")

    print("\n" + "="*70)


if __name__ == "__main__":
    analyze()
