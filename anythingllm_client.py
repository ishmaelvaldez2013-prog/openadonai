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
) -> Dict[str, object]:
    """
    Ask the AnythingLLM 'books' workspace for RAG context.

    Returns a dict with:
      - 'text':    the model's own answer from AnythingLLM (may or may not be used)
      - 'context': a merged string of snippet texts we can feed into Mistral
      - 'sources': a structured list of source metadata:
            [
              {
                "title": ...,
                "filename": ...,
                "page": ...,
                "pageNumber": ...,
                "source": ...,
                "raw": <original source object>
              },
              ...
            ]
    """
    if not is_anythingllm_enabled():
        return {"text": "", "context": "", "sources": []}

    url = f"{ANYTHINGLLM_BASE_URL}/api/v1/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/chat"

    headers = {
        "Authorization": f"Bearer {ANYTHINGLLM_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "message": question,
        "mode": "query",  # key: tells AnythingLLM to do vector-search / RAG
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    text_response = data.get("textResponse", "") or data.get("text", "")

    sources_raw = data.get("sources") or []
    snippets: List[str] = []
    structured_sources: List[Dict[str, object]] = []

    for src in sources_raw[:max_snippets]:
        # Typical shape:
        # {
        #   "pageContent": "...",
        #   "metadata": {
        #       "title": "...",
        #       "filename": "...",
        #       "page": 3,
        #       ...
        #   }
        # }
        content = src.get("pageContent") or src.get("content") or ""
        meta = src.get("metadata") or {}

        title = meta.get("title") or meta.get("documentTitle") or ""
        filename = meta.get("filename") or meta.get("fileName") or ""
        page = meta.get("page") or meta.get("pageNumber")
        source_id = meta.get("source") or meta.get("sourceId") or meta.get("docId")

        label_parts = []
        if title:
            label_parts.append(str(title))
        elif filename:
            label_parts.append(str(filename))
        if page is not None:
            label_parts.append(f"p.{page}")
        label = " – ".join(label_parts) if label_parts else "Book excerpt"

        if content:
            snippets.append(f"[{label}]\n{content.strip()}")

        structured_sources.append(
            {
                "title": title,
                "filename": filename,
                "page": page,
                "pageNumber": page,
                "source": source_id,
                "raw": src,
            }
        )

    book_context = "\n\n".join(snippets)

    return {
        "text": text_response,
        "context": book_context,
        "sources": structured_sources,
    }
