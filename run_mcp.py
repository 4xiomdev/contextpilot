#!/usr/bin/env python3
"""Run the MCP server."""

import sys
sys.path.insert(0, ".")

from backend.mcp_server import run_mcp

if __name__ == "__main__":
    run_mcp()



