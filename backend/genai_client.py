"""
Shared Google GenAI client wrapper using google-genai SDK.
"""

import logging
from typing import Any, List, Optional

from google import genai
from google.genai import types

from .config import get_config

logger = logging.getLogger("contextpilot.genai")


class GenAIClient:
    """Wrapper around google-genai for generation and embeddings."""

    def __init__(self):
        cfg = get_config()
        self._client = genai.Client(api_key=cfg.google.api_key)
        self._generation_model = cfg.google.generation_model
        self._embedding_model = cfg.google.embedding_model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.4,
        max_output_tokens: int = 2000,
    ) -> str:
        response = self._client.models.generate_content(
            model=self._generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return _extract_text(response)

    def embed(
        self,
        text: str,
        task_type: str = "retrieval_document",
    ) -> List[float]:
        response = self._client.models.embed_content(
            model=self._embedding_model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return _extract_embedding(response)


_client: Optional[GenAIClient] = None


def get_genai_client() -> GenAIClient:
    """Get the singleton GenAI client."""
    global _client
    if _client is None:
        _client = GenAIClient()
    return _client


def _extract_text(response: Any) -> str:
    """Extract text from a google-genai response."""
    if response is None:
        return ""
    if hasattr(response, "text") and response.text:
        return str(response.text).strip()
    if hasattr(response, "candidates") and response.candidates:
        for cand in response.candidates:
            content = getattr(cand, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        return str(text).strip()
    if isinstance(response, dict):
        text = response.get("text")
        if text:
            return str(text).strip()
    return ""


def _extract_embedding(response: Any) -> List[float]:
    """Extract embedding vector from a google-genai response."""
    if response is None:
        return []
    if hasattr(response, "embeddings") and response.embeddings:
        emb = response.embeddings[0]
        values = getattr(emb, "values", None)
        if values is not None:
            return list(values)
    if hasattr(response, "embedding"):
        emb = response.embedding
        values = getattr(emb, "values", None)
        if values is not None:
            return list(values)
    if isinstance(response, dict):
        emb = response.get("embedding") or {}
        values = emb.get("values")
        if values is not None:
            return list(values)
    return []
