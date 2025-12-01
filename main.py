# main.py

import os
import json
from typing import List, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# --- Load .env ---
load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "./index_data")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    content: str
    file_path: str
    chunk_index: int
    score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


def load_index():
    embeddings_path = os.path.join(INDEX_DIR, "embeddings.npy")
    metadata_path = os.path.join(INDEX_DIR, "metadata.json")

    if not os.path.isfile(embeddings_path) or not os.path.isfile(metadata_path):
        raise RuntimeError("Index not found. Run index_obsidian.py first.")

    embeddings = np.load(embeddings_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    texts = meta_data["texts"]
    metadatas = meta_data["metadata"]

    return embeddings, texts, metadatas


EMBEDDINGS, TEXTS, METADATAS = load_index()


def embed_query(query: str) -> np.ndarray:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": query},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    vec = data.get("embedding")
    if vec is None:
        raise RuntimeError(f"No 'embedding' in query response: {data}")
    return np.array(vec, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm)


app = FastAPI(
    title="Obsidian Codex RAG API",
    description="Minimal API to search Ishmael's Obsidian Codex (Ollama embeddings).",
    version="0.2.0",
)


@app.post("/search-codex", response_model=SearchResponse)
def search_codex(req: SearchRequest):
    query_vec = embed_query(req.query)
    scores = cosine_similarity(query_vec, EMBEDDINGS)

    top_k = max(1, min(req.top_k, len(scores)))
    top_idx = np.argsort(scores)[::-1][:top_k]

    results: List[SearchResult] = []
    for idx in top_idx:
        results.append(
            SearchResult(
                content=TEXTS[idx],
                file_path=METADATAS[idx]["file_path"],
                chunk_index=METADATAS[idx]["chunk_index"],
                score=float(scores[idx]),
            )
        )

    return SearchResponse(query=req.query, results=results)