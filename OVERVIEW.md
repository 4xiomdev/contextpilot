# ContextPilot Overview (Beginner Friendly)

ContextPilot is a “context augmentation layer” for AI agents: it **collects docs**, **turns them into searchable vectors**, and exposes tools (MCP + HTTP) so an agent can look things up instead of guessing.

## The mental model (analogy)

Think of it like a **library + librarian**:
- **Crawler** = the person who goes out and *brings books back* (downloads docs pages).
- **Chunking** = *cutting books into pages/paragraphs* so we can search precisely.
- **Embeddings** = *an index card for each paragraph* (a numeric fingerprint of meaning).
- **Pinecone** = the *index drawer* where those fingerprints live.
- **Firestore** = the *catalog + job log* (what was crawled, when, status, metadata).
- **Search tool** = the librarian: “find me the paragraphs most related to this question”.

## What runs where

### “Engine” (this repo)
This repo contains the **backend** that does crawling/embedding/search and a **dashboard** to manage it.

It can run:
- **Locally** (single-tenant): for personal use.
- **Hosted** (multi-tenant): one shared backend, multiple users isolated by tenant.

### Hosted SaaS (“Platform”)
The marketing / sign-in / billing web app is intentionally a separate product surface (a separate repo in your `Code/` folder: `contextpilot-cloud`). It talks to the backend in this repo over HTTPS.

## How MCP fits in

MCP is how an editor/agent (Codex, Cursor, etc.) calls tools like:
- `search_documentation`
- `crawl_url`
- `build_normalized_doc`

There are two ways to provide MCP tools:
1) **Local MCP**: run `run_mcp.py` and the client talks to the local process (stdio).
2) **Remote MCP client**: run `mcp_remote_client.py`, which talks to a hosted HTTP API (`CONTEXTPILOT_API_URL`) and exposes the same tools to the editor.

## “Local” vs “Hosted” (what changes)

### Local / Self-host
- You run everything for yourself.
- Usually no auth (`AUTH_MODE=none`) and single tenant (`MULTI_TENANT_ENABLED=false`).

### Hosted / Multi-tenant
- One backend serves many users.
- Users sign in (Firebase ID token) and/or use API keys.
- Each user’s data is isolated by **tenant** (Firestore subcollections + Pinecone namespaces).

In hosted mode, users can create API keys via:
- `POST /api/api-keys` (returns plaintext once)
- `GET /api/api-keys`
- `DELETE /api/api-keys/{digest}`
