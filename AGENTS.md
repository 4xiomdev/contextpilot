# ContextPilot - Agent Guidelines

> Guidelines for AI agents using ContextPilot as an MCP retrieval layer

## Project Architecture

```
contextpilot/
├── backend/
│   ├── config.py              # Configuration management
│   ├── mcp_server.py          # FastMCP + FastAPI server (main entry)
│   ├── embed_manager.py       # Google embeddings + Pinecone vector storage
│   ├── crawl_manager.py       # Web crawling (Firecrawl + trafilatura)
│   ├── normalizer.py          # Gemini-based doc synthesis
│   ├── search_agent.py        # Agentic search (quick/deep modes)
│   ├── discovery_agent.py     # AI-powered source discovery
│   ├── source_registry.py     # Source management + scheduling
│   ├── firestore_db.py        # Database layer
│   ├── websocket_manager.py   # Real-time event broadcasting
│   ├── scheduler.py           # APScheduler for periodic crawls
│   └── job_queue.py           # Background job processing
├── frontend/
│   ├── src/app/(app)/         # Dashboard pages
│   │   ├── dashboard/         # Overview stats
│   │   ├── crawl/             # Crawl manager
│   │   ├── search/            # Semantic search
│   │   ├── docs/              # Normalized docs
│   │   ├── playground/        # MCP tool testing
│   │   ├── chat/              # LLM chat interface
│   │   └── vectors/           # Vector visualization
│   ├── src/lib/api.ts         # API client
│   └── src-tauri/             # Desktop app wrapper
├── .env                       # Environment variables
└── requirements.txt           # Python dependencies
```

## MCP Tools Available

### `search_documentation`
Search indexed documentation with semantic understanding.

```json
{
  "query": "How to use streaming with Gemini API",
  "mode": "quick",      // "quick" (fast) or "deep" (synthesized)
  "limit": 10,
  "url_filter": "https://ai.google.dev"
}
```

**Tips:**
- Use `url_filter` to scope searches to specific docs (e.g., `https://ai.google.dev` for Gemini)
- Use `mode: "deep"` when you need a synthesized answer with citations
- Use `mode: "quick"` for fast retrieval of relevant chunks

### `crawl_url`
Crawl and index a URL.

```json
{
  "url": "https://ai.google.dev/gemini-api/docs"
}
```

### `build_normalized_doc`
Synthesize a clean document from indexed chunks.

```json
{
  "url_prefix": "https://ai.google.dev/gemini-api/docs/models",
  "title": "Gemini Models Documentation"
}
```

### `health_status`
Get system health and statistics.

```json
{}
```

### `discover_documentation`
AI-powered discovery of documentation sources.

```json
{
  "topic": "React hooks",
  "max_results": 10
}
```

## Agent Usage Guidelines

### 1. Always Use MCP for Documentation Retrieval

**Before coding against any API or framework**, query ContextPilot:

```
Wrong: Guessing API structure based on training data
Right: Call search_documentation first, then code based on results
```

### 2. Query Rewriting

For better retrieval, rewrite queries to be more specific:

```
User: "How do I stream?"
Better query: "Gemini API streaming response generate_content_stream Python"
```

Include:
- Product/framework name
- Specific feature or method
- Programming language

### 3. URL Filtering for Precision

When you know which documentation to search:

```json
{
  "query": "authentication",
  "url_filter": "https://firebase.google.com"
}
```

Common prefixes:
- Google AI: `https://ai.google.dev`
- Firebase: `https://firebase.google.com`
- React: `https://react.dev`
- Next.js: `https://nextjs.org/docs`

### 4. Deep vs Quick Mode

**Quick Mode** (default):
- Fast retrieval with reranking
- Returns relevant chunks directly
- Best for: specific lookups, code examples

**Deep Mode**:
- Full synthesis with citations
- Returns a coherent answer
- Best for: conceptual questions, "how does X work?"

### 5. Handling Missing Documentation

If search returns no results:
1. Try broader query terms
2. Check if the source is indexed using `health_status`
3. Use `discover_documentation` to find the right source
4. Crawl the documentation using `crawl_url`

## REST API Endpoints

For programmatic access outside MCP:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | GET | Dashboard statistics |
| `/api/sources` | GET | List indexed sources |
| `/api/crawl` | POST | Start crawl job |
| `/api/search` | POST | Basic search |
| `/api/search/agentic` | POST | Agentic search (quick/deep) |
| `/api/chat` | POST | Chat interface |
| `/api/vectors/sample` | POST | Get vector samples |
| `/api/mcp/tools` | GET | List MCP tools |
| `/api/mcp/execute` | POST | Execute MCP tool |
| `/ws` | WebSocket | Real-time updates |

## Configuration for Claude/Cursor

### Claude Desktop (MCP)

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "contextpilot": {
      "command": "python",
      "args": ["/path/to/contextpilot/run_mcp.py"],
      "env": {
        "GOOGLE_API_KEY": "your-key",
        "PINECONE_API_KEY": "your-key",
        "PINECONE_INDEX_NAME": "contextpilot"
      }
    }
  }
}
```

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "contextpilot": {
      "command": "python",
      "args": ["/path/to/contextpilot/run_mcp.py"],
      "env": {
        "GOOGLE_API_KEY": "your-key",
        "PINECONE_API_KEY": "your-key"
      }
    }
  }
}
```

### Remote/Hosted Mode

For cloud deployments, use HTTP transport:

```json
{
  "mcpServers": {
    "contextpilot": {
      "command": "python",
      "args": ["/path/to/contextpilot/mcp_remote_client.py"],
      "env": {
        "CONTEXTPILOT_API_URL": "https://your-cloudrun-url",
        "CONTEXTPILOT_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Environment Variables

Required:
- `GOOGLE_API_KEY` - Google AI API key (for embeddings + generation)

Recommended:
- `PINECONE_API_KEY` - Pinecone API key
- `PINECONE_INDEX_NAME` - Index name (default: contextpilot)
- `FIRECRAWL_API_KEY` - Firecrawl API key (for better crawling)

Optional (Cloud):
- `FIREBASE_PROJECT_ID` - For Firestore database
- `CONTEXTPILOT_API_KEY` - API authentication key
- `AUTH_MODE` - `none`, `api_key`, `firebase`, or `api_key_or_firebase`
- `MULTI_TENANT_ENABLED` - Enable multi-tenant mode

## Best Practices

1. **Index before you code** - Always ensure relevant docs are indexed
2. **Be specific** - Include framework names, versions, and specific features in queries
3. **Use filters** - URL filters dramatically improve precision
4. **Check freshness** - Use `health_status` to verify what's indexed
5. **Normalize important docs** - For frequently-used APIs, build normalized docs
6. **Deep mode for concepts** - Use deep search for "why" and "how" questions
7. **Quick mode for examples** - Use quick search for code snippets and specific syntax
