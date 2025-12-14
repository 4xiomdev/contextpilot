#!/usr/bin/env python3
"""
ContextPilot Remote MCP Client
Connects to ContextPilot API via HTTP instead of local STDIO.

This allows Cursor (or any MCP client) to use a cloud-hosted ContextPilot instance.

Usage:
    Set CONTEXTPILOT_API_URL and CONTEXTPILOT_API_KEY environment variables,
    then configure Cursor to use this script as the MCP command.

Configuration (~/.cursor/mcp.json):
{
  "mcpServers": {
    "contextpilot": {
      "command": "python",
      "args": ["/path/to/contextpilot/mcp_remote_client.py"],
      "env": {
        "CONTEXTPILOT_API_URL": "https://your-cloudrun-url.run.app",
        "CONTEXTPILOT_API_KEY": "your-api-key"
      }
    }
  }
}
"""

import os
import sys
import json
import logging
from typing import Optional

import httpx
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("contextpilot.remote")

# Configuration from environment
API_URL = os.getenv("CONTEXTPILOT_API_URL", "http://localhost:8000")
API_KEY = os.getenv("CONTEXTPILOT_API_KEY", "")

# HTTP client with timeout
client = httpx.Client(timeout=120.0)


def call_remote_tool(tool_name: str, **kwargs) -> str:
    """Call a tool on the remote ContextPilot API."""
    url = f"{API_URL}/mcp/tools/{tool_name}"
    headers = {}
    
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    
    try:
        response = client.post(url, json=kwargs, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        if "result" in result:
            return json.dumps(result["result"], ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)
        
    except httpx.HTTPStatusError as e:
        error_msg = f"API error: {e.response.status_code}"
        try:
            detail = e.response.json().get("detail", str(e))
            error_msg = f"API error: {detail}"
        except Exception:
            pass
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
        
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return json.dumps({"error": str(e)})


# Create MCP server that proxies to remote API
mcp = FastMCP("ContextPilot-Remote")


@mcp.tool()
def search_documentation(
    query: str,
    limit: int = 10,
    url_filter: str = "",
) -> str:
    """
    Search indexed documentation.
    
    Args:
        query: The search query
        limit: Maximum number of results (default 10)
        url_filter: Optional URL prefix to filter results
    
    Returns:
        JSON with search results including url, title, score, and content.
    """
    return call_remote_tool(
        "search_documentation",
        query=query,
        limit=limit,
        url_filter=url_filter,
    )


@mcp.tool()
def crawl_url(url: str) -> str:
    """
    Crawl a URL and index its content.
    
    Args:
        url: The URL to crawl
    
    Returns:
        JSON with crawl status and number of chunks indexed.
    """
    return call_remote_tool("crawl_url", url=url)


@mcp.tool()
def build_normalized_doc(url_prefix: str, title: str) -> str:
    """
    Build a normalized document from all chunks matching a URL prefix.
    
    Args:
        url_prefix: URL prefix to match (e.g., "https://docs.example.com/api")
        title: Title for the normalized document
    
    Returns:
        JSON with normalization result.
    """
    return call_remote_tool(
        "build_normalized_doc",
        url_prefix=url_prefix,
        title=title,
    )


@mcp.tool()
def health_status() -> str:
    """
    Get system health status and statistics.
    
    Returns:
        JSON with database stats, index stats, and service status.
    """
    return call_remote_tool("health_status")


if __name__ == "__main__":
    logger.info(f"Starting ContextPilot Remote Client")
    logger.info(f"API URL: {API_URL}")
    logger.info(f"API Key: {'configured' if API_KEY else 'not configured'}")
    mcp.run()

