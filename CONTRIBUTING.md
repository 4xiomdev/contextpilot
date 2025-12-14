# Contributing to ContextPilot

Thank you for your interest in contributing to ContextPilot! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** and clone it locally
2. **Install dependencies**:
   ```bash
   # Backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

   # Frontend
   cd frontend && npm install
   ```
3. **Set up environment variables** (see README.md)
4. **Run locally**:
   ```bash
   python run_api.py  # Backend
   cd frontend && npm run dev  # Frontend
   ```

## Development Workflow

### Branching

- Create a feature branch from `main`: `git checkout -b feature/your-feature`
- Use descriptive branch names: `feature/add-anthropic-support`, `fix/search-scoring`

### Code Style

**Python:**
- Follow PEP 8
- Use type hints for function signatures
- Write docstrings for public functions/classes
- Keep functions focused and small

**TypeScript/React:**
- Use TypeScript for all new code
- Follow the existing component patterns
- Use React Query for data fetching

### Testing

Before submitting a PR:
1. Ensure the backend starts without errors
2. Verify the frontend builds: `npm run build`
3. Test your changes manually

### Pull Requests

1. **Create a clear title**: `Add support for X` or `Fix issue with Y`
2. **Write a description** explaining:
   - What the PR does
   - Why the change is needed
   - How to test it
3. **Keep PRs focused** - one feature/fix per PR
4. **Update documentation** if needed

## Project Structure

```
contextpilot/
├── backend/
│   ├── config.py          # Configuration
│   ├── db.py              # Database layer
│   ├── embed_manager.py   # Pinecone + embeddings
│   ├── crawl_manager.py   # Web crawling
│   ├── normalizer.py      # Doc normalization
│   └── mcp_server.py      # MCP + REST API
├── frontend/              # React dashboard
├── Dockerfile             # Container
└── requirements.txt       # Python deps
```

## Areas for Contribution

### High Priority
- Additional embedding models support
- More vector database backends (Weaviate, Milvus)
- Better chunking strategies
- Improved search ranking

### Medium Priority
- Additional crawlers (Playwright, custom)
- More documentation sources
- Dashboard improvements
- Performance optimizations

### Good First Issues
- Documentation improvements
- Add tests
- Fix typos
- UI polish

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the code, not the person

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions

Thank you for contributing!

