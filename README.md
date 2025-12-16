# ContextPilot

> **Open-source context augmentation layer for AI agents**

ContextPilot crawls web documentation, chunks and embeds it in Pinecone, and provides semantic search via **MCP** (Model Context Protocol) tools and a **REST API**. Deploy locally or to the cloud.

If you’re new, start with `OVERVIEW.md`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Dashboard                          │
│  (URL management, search interface, normalized docs viewer)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Python Backend                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  MCP Server │  │  REST API   │  │  Background Tasks   │  │
│  │  (FastMCP)  │  │  (FastAPI)  │  │  (Crawl, Normalize) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                           │                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Embed Mgr   │  │ Crawl Mgr   │  │    Normalizer       │  │
│  │ (Pinecone)  │  │ (Firecrawl) │  │    (Gemini)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Pinecone    │  │   Firestore   │  │  Firecrawl    │
│  (Vectors)    │  │  (Metadata)   │  │  (Crawling)   │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Features

- **Doc Crawling**: Crawl web docs via Firecrawl (primary) or trafilatura (fallback)
- **Semantic Search**: Vector search via Pinecone with Google embeddings
- **Doc Normalization**: Synthesize clean, embedding-friendly docs using Gemini
- **MCP Integration**: Expose tools for AI agents via FastMCP
- **REST API**: Full API for the React dashboard
- **Deduplication**: Content hashing prevents duplicate indexing
- **Cloud Deployment**: Deploy to Firebase/Cloud Run for remote access

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Firebase CLI (`npm install -g firebase-tools`)
- Google Cloud SDK (`gcloud`)

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/contextpilot.git
cd contextpilot

# Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

### 2. Environment Variables

Create a `.env` file:

```bash
# Required
GOOGLE_API_KEY=your-google-api-key

# Vector Database
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=contextpilot

# Crawling
FIRECRAWL_API_KEY=your-firecrawl-api-key

# Firebase (for cloud deployment)
FIREBASE_PROJECT_ID=your-project-id

# API Authentication (for cloud)
CONTEXTPILOT_API_KEY=your-api-key
```

### 3. Run Locally

```bash
# Backend API
python run_api.py

# Frontend (in another terminal)
cd frontend && npm run dev
```

- API: http://localhost:8000
- Dashboard: http://localhost:3000

## MCP Tools

### `search_documentation`

Search indexed documentation.

```json
{
  "query": "How to use Gemini API",
  "limit": 10,
  "url_filter": "https://ai.google.dev"
}
```

### `crawl_url`

Crawl and index a URL.

```json
{
  "url": "https://ai.google.dev/gemini-api/docs"
}
```

### `build_normalized_doc`

Create a normalized doc from chunks matching a URL prefix.

```json
{
  "url_prefix": "https://ai.google.dev/gemini-api/docs/models",
  "title": "Gemini Models Documentation"
}
```

### `health_status`

Get system health and statistics.

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/sources` | List indexed sources |
| POST | `/api/crawl` | Start a crawl job |
| GET | `/api/crawls` | List crawl jobs |
| GET | `/api/crawls/{id}` | Get crawl job status |
| POST | `/api/search` | Search documentation |
| POST | `/api/normalize` | Build normalized doc |
| GET | `/api/normalized` | List normalized docs |
| DELETE | `/api/sources/{url}` | Delete indexed source |

## Cloud Deployment

ContextPilot can be deployed to Firebase Hosting + Cloud Run for remote access.

### Deploy Backend (Cloud Run)

```bash
gcloud run deploy contextpilot-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

### Deploy Frontend (Firebase Hosting)

```bash
cd frontend
npm run build
firebase deploy --only hosting
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## Hosted Mode (Multi-tenant + API Keys)

ContextPilot supports a hosted SaaS-style deployment where users authenticate with Firebase and generate per-user API keys for MCP clients:

- Enable multi-tenancy: set `MULTI_TENANT_ENABLED=true`
- Enable auth: set `AUTH_MODE=api_key_or_firebase` (or `firebase`)
- Configure Firestore: set `FIREBASE_PROJECT_ID=<your-project-id>`
- Optional admin key: set `CONTEXTPILOT_API_KEY=<admin-key>` (for internal use)

Once authenticated (Firebase ID token), users can create API keys:
- `POST /api/api-keys` (returns plaintext key once)
- `GET /api/api-keys`
- `DELETE /api/api-keys/{digest}`

MCP clients can then use the hosted API via `mcp_remote_client.py` with:
- `CONTEXTPILOT_API_URL=https://<your-cloud-run-url>`
- `CONTEXTPILOT_API_KEY=<user-api-key>`

## Cursor Integration

Configure Cursor to use ContextPilot MCP tools:

**~/.cursor/mcp.json**:
```json
{
  "mcpServers": {
    "contextpilot": {
      "command": "python",
      "args": ["/path/to/contextpilot/run_mcp.py"],
      "env": {
        "CONTEXTPILOT_API_URL": "https://your-cloudrun-url"
      }
    }
  }
}
```

## Project Structure

```
contextpilot/
├── backend/
│   ├── config.py          # Environment and settings
│   ├── firestore_db.py    # Firestore database layer
│   ├── embed_manager.py   # Pinecone + embeddings
│   ├── crawl_manager.py   # Firecrawl + trafilatura
│   ├── normalizer.py      # Gemini-based synthesis
│   └── mcp_server.py      # MCP tools + REST API
├── frontend/              # React/Next.js dashboard
├── Dockerfile             # Container for Cloud Run
├── firebase.json          # Firebase config
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
