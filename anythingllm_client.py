# anythingllm_client.py

import os
import json
import requests
from typing import List, Dict, Optional

ANYTHINGLLM_BASE_URL = os.getenv("ANYTHINGLLM_BASE_URL", "http://localhost:3001")
ANYTHINGLLM_API_KEY = os.getenv("ANYTHINGLLM_API_KEY")
ANYTHINGLLM_WORKSPACE_SLUG = os.getenv("ANYTHINGLLM_WORKSPACE_SLUG")

def is_anythingllm_enabled() -> bool:
    return bool(ANYTHINGLLM_API_KEY and ANYTHINGLLM_WORKSPACE_SLUG)

def query_anythingllm_books(
    question: str,
    max_snippets: int = 5,
) -> Dict[str, str]:
    """
    Ask the AnythingLLM 'books' workspace for RAG context.
    Returns a dict with:
      - 'text': the model's own answer (optional, we may or may not use it)
      - 'context': a merged string of snippet texts we can feed into Mistral
    """
    if not is_anythingllm_enabled():
        return {"text": "", "context": ""}

    url = f"{ANYTHINGLLM_BASE_URL}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/chat"

    headers = {
        "Authorization": f"Bearer {ANYTHINGLLM_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "message": question,
        "mode": "query",  # key: tells AnythingLLM to do vector-search / RAG
        # You can also pass "reset": False, "sessionId": "openadonai" if you want history there
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    text_response = data.get("textResponse", "") or data.get("text", "")

    # Many builds include a 'sources' key listing retrieved chunks
    sources = data.get("sources") or []
    snippets: List[str] = []

    for src in sources[:max_snippets]:
        # Structure can vary, but often something like:
        # { "pageContent": "...", "metadata": {...} }
        content = src.get("pageContent") or src.get("content") or ""
        meta = src.get("metadata") or {}
        title = meta.get("title") or meta.get("filename") or ""
        page = meta.get("page") or meta.get("pageNumber")
        label_parts = []
        if title:
            label_parts.append(str(title))
        if page is not None:
            label_parts.append(f"p.{page}")
        label = " – ".join(label_parts) if label_parts else "Book excerpt"

        if content:
            snippets.append(f"[{label}]\n{content.strip()}")

    book_context = "\n\n".join(snippets)

    return {
        "text": text_response,
        "context": book_context,
    }
