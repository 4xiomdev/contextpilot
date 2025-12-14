# How ContextPilot Works in Cursor - Complete Guide

## Current Setup

Your Cursor is configured to automatically use ContextPilot for API documentation lookup.

### Configuration Files

| Location | Purpose |
|----------|---------|
| `~/.cursor/mcp.json` | Registers ContextPilot MCP server with Cursor |
| `~/.cursor/rules/contextpilot.md` | Global rule: "Search ContextPilot first" |
| `.cursorrules` | Project-specific rules |

## How It Works in Practice

### Scenario 1: You Ask About Gemini API

**You:** "Write code to call Gemini 2.5 Flash with temperature 0.9"

**What happens automatically:**

1. **Search ContextPilot**
   ```
   search_documentation("Gemini 2.5 Flash temperature Python SDK")
   ```

2. **Get Real Docs**
   ```json
   {
     "results": [{
       "content": "client.models.generate_content(model='gemini-2.5-flash', config=types.GenerateContentConfig(temperature=0.9))",
       "url": "https://ai.google.dev/gemini-api/docs/text-generation"
     }]
   }
   ```

3. **Write Code**
   ```python
   from google import genai
   from google.genai import types
   
   # Source: https://ai.google.dev/gemini-api/docs/text-generation
   client = genai.Client()
   
   response = client.models.generate_content(
       model="gemini-2.5-flash",
       contents=["Your prompt here"],
       config=types.GenerateContentConfig(temperature=0.9)
   )
   print(response.text)
   ```

### Scenario 2: Documentation Not Indexed Yet

**You:** "How do I use Anthropic Claude API?"

**What happens:**

1. Search ContextPilot → Empty results
2. AI tells you: "I don't have Anthropic docs indexed. Want me to crawl them?"
3. **You:** "Yes, crawl them"
4. AI crawls: `crawl_url("https://docs.anthropic.com/claude")`
5. Search again → Now has the docs!
6. Write code based on real docs

## MCP Tools Available

### `search_documentation`
Search indexed documentation with semantic matching.

```json
{
  "query": "Gemini function calling",
  "limit": 10,
  "url_filter": "https://ai.google.dev"
}
```

### `crawl_url`
Crawl and index a new documentation URL.

```json
{
  "url": "https://docs.example.com/api"
}
```

### `build_normalized_doc`
Synthesize multiple chunks into a clean reference document.

```json
{
  "url_prefix": "https://ai.google.dev/gemini-api/docs/models",
  "title": "Gemini Models Reference"
}
```

### `health_status`
Get system health and statistics.

## Testing It Right Now

Try these queries:

### Test 1: Gemini Function Calling
**Ask:** "Write code for Gemini function calling with weather API"

### Test 2: OpenAI Structured Outputs
**Ask:** "Show me OpenAI structured outputs with Pydantic"

### Test 3: Check What's Available
**Ask:** "Search ContextPilot for Gemini video understanding"

## Why This is Powerful

### Before ContextPilot
```
You: "Use Gemini API"
AI: *hallucinates API* or *uses outdated patterns*
```

### After ContextPilot
```
You: "Use Gemini API"
AI: 
  1. search_documentation("Gemini API usage")
  2. *finds real, current docs*
  3. *writes correct code*
  4. "Source: https://ai.google.dev/..."
```

## Cloud Deployment

ContextPilot can be deployed to Cloud Run for remote access:

1. **Backend**: Cloud Run (API + MCP over HTTP)
2. **Frontend**: Firebase Hosting (Dashboard)
3. **Database**: Firestore (Metadata)
4. **Vectors**: Pinecone (Already cloud)

This allows multiple users to share the same indexed documentation.

## Verify It's Working

After configuration, ask:
```
"Search ContextPilot for Gemini embeddings API"
```

You should see the `search_documentation` tool being used and results from indexed docs.
