# ContextPilot Setup Guide

## Quick Start

### 1. Environment Setup

Create a `.env` file in the project root:

```bash
# Required
GOOGLE_API_KEY=your-google-api-key

# Vector Database
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=contextpilot

# Local Vector Store (optional fallback)
# VECTOR_STORE_PROVIDER=qdrant  # default: auto (pinecone if configured)
# QDRANT_PATH=./data/qdrant

# Crawling (optional but recommended)
FIRECRAWL_API_KEY=your-firecrawl-api-key
FIRECRAWL_MIN_INTERVAL_SECONDS=8

# Crawl filtering (optional)
# DEFAULT_EXCLUDE_PATHS=/blog,/changelog,/community,/competition,/pricing,/login,/signup

# Cloud Deployment (optional)
FIREBASE_PROJECT_ID=your-project-id
CONTEXTPILOT_API_KEY=your-api-key

# Hosted mode (optional)
AUTH_MODE=api_key_or_firebase
MULTI_TENANT_ENABLED=true
```

### 2. Install Dependencies

```bash
# Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

### 3. Run Locally

```bash
# Terminal 1: Backend API
python run_api.py

# Terminal 2: Frontend
cd frontend && npm run dev
```

- API: http://localhost:8000
- Dashboard: http://localhost:3000

## Cursor Integration

### MCP Configuration

Create/edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "contextpilot": {
      "command": "/path/to/contextpilot/.venv/bin/python",
      "args": ["/path/to/contextpilot/run_mcp.py"]
    }
  }
}
```

### Global Rules (Recommended)

Create `~/.cursor/rules/contextpilot.md`:

```markdown
# ContextPilot Integration

When the user asks about APIs or needs to write code that interfaces with external services:

1. ALWAYS search ContextPilot first: `search_documentation("relevant query")`
2. Use the returned documentation to write accurate code
3. Cite the source URL in your response

This ensures code is based on real, up-to-date documentation rather than training data.
```

### Restart Cursor

After configuration, **restart Cursor** (Cmd+Q and reopen) for MCP tools to load.

## Usage

### Via Dashboard

1. Open http://localhost:3000
2. Go to "Crawl Manager"
3. Add URLs to crawl
4. Use "Search" to test

### Via Cursor Chat

Just ask naturally:
- "Write code to use Gemini 2.5 Flash"
- "Search ContextPilot for OpenAI embeddings"
- "Crawl https://docs.example.com/api"

## Cloud Deployment

### Deploy to Cloud Run

```bash
gcloud run deploy contextpilot-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="FIREBASE_PROJECT_ID=$FIREBASE_PROJECT_ID,AUTH_MODE=api_key_or_firebase,MULTI_TENANT_ENABLED=true" \
  --set-env-vars="GOOGLE_API_KEY=$GOOGLE_API_KEY,PINECONE_API_KEY=$PINECONE_API_KEY,CONTEXTPILOT_API_KEY=$CONTEXTPILOT_API_KEY"
```

Note: `--allow-unauthenticated` is OK if you enforce `AUTH_MODE=api_key_or_firebase` (the API stays protected by Firebase tokens and/or API keys). For stricter security, remove `--allow-unauthenticated` and use Cloud Run IAM instead.

### Deploy Frontend

```bash
cd frontend
npm run build
firebase deploy --only hosting
```

### Update Cursor for Cloud

Update `~/.cursor/mcp.json` to point to cloud:

```json
{
  "mcpServers": {
    "contextpilot": {
      "command": "python",
      "args": ["/path/to/mcp_remote_client.py"],
      "env": {
        "CONTEXTPILOT_API_URL": "https://your-cloudrun-url.run.app",
        "CONTEXTPILOT_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Troubleshooting

### MCP Tools Not Showing
1. Restart Cursor completely
2. Check `~/.cursor/mcp.json` syntax
3. Verify Python path is correct

### Search Returns Empty
1. Check if documents are indexed (use dashboard)
2. Verify Pinecone connection
3. Check API key validity

### Crawl Fails
1. Check Firecrawl API key
2. Try local fallback (trafilatura)
3. Verify URL is accessible

## Next Steps

1. **Index more docs**: Add API documentation for services you use
2. **Build normalized docs**: Create clean references for frequently-used APIs
3. **Deploy to cloud**: Share indexed docs across machines
