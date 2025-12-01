# query_index.py

import os
import json
import time
from typing import List, Dict, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

# --- Load .env ---
load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "./index_data")

# Prefer OLLAMA_BASE_URL if present, fall back to OLLAMA_HOST, then default
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL") or os.getenv(
    "OLLAMA_HOST", "http://localhost:11434"
)

# Embedding model config: prefer OLLAMA_EMBED_MODEL, then EMBED_MODEL, then default
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL") or os.getenv(
    "EMBED_MODEL", "nomic-embed-text"
)


def log(msg: str):
    print(msg, flush=True)


# ---------- Ollama helpers ----------

def ensure_ollama_model(model: str):
    """
    Ensure the requested Ollama model is available.

    - If Ollama is not reachable → raise helpful error
    - If model missing → pull it automatically via /api/pull
    """
    base = OLLAMA_HOST.rstrip("/")

    # 1. Check Ollama server is up
    try:
        requests.get(base, timeout=3)
    except Exception as e:
        raise RuntimeError(
            f"Ollama is not reachable at {base}. "
            f"Is the Ollama app or 'ollama serve' running?\nDetails: {e}"
        )

    # 2. Check if model exists
    show_url = base + "/api/show"
    try:
        resp = requests.post(show_url, json={"name": model}, timeout=10)
    except Exception as e:
        raise RuntimeError(f"Error talking to Ollama /api/show: {e}")

    if resp.status_code == 200:
        # Model is already pulled
        return

    # 3. If not found → pull model
    log(f"⏬ Pulling Ollama embedding model '{model}' (not found locally)...")

    pull_url = base + "/api/pull"
    try:
        with requests.post(pull_url, json={"name": model}, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    status = data.get("status")
                    if status:
                        log("   " + status)
                except json.JSONDecodeError:
                    log("   " + line)
    except Exception as e:
        raise RuntimeError(f"Failed to pull Ollama model '{model}': {e}")

    log(f"✅ Embedding model '{model}' is now installed.\n")


# ---------- Index loading ----------

def load_index(index_dir: str) -> Tuple[np.ndarray, List[str], List[Dict], str]:
    """
    Load embeddings + metadata from disk.
    Returns: (embeddings, texts, metadata, embed_model_from_index)
    """
    embeddings_path = os.path.join(index_dir, "embeddings.npy")
    metadata_path = os.path.join(index_dir, "metadata.json")

    if not os.path.isfile(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    log(f"📦 Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path).astype("float32")

    log(f"📦 Loading metadata from {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    texts = meta["texts"]
    metadata = meta["metadata"]
    embed_model_from_index = meta.get("embed_model", EMBED_MODEL)

    if len(texts) != embeddings.shape[0] or len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"Embeddings rows ({embeddings.shape[0]}) != len(texts) ({len(texts)}) "
            f"or len(metadata) ({len(metadata)})"
        )

    log(f"✅ Loaded index: {embeddings.shape[0]} chunks, dim={embeddings.shape[1]}")
    return embeddings, texts, metadata, embed_model_from_index


def get_embedding_ollama(text: str, model: str) -> np.ndarray:
    """
    Call Ollama's embedding API for the query text.
    Returns a 1D numpy array.
    """
    # Ensure the embedding model exists (auto-pull if needed)
    ensure_ollama_model(model)

    url = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
    payload = {
        "model": model,
        "prompt": text,
    }
    log(f"🧠 Requesting embedding for query ({len(text)} chars) using model '{model}'")
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    emb = np.array(data["embedding"], dtype="float32")
    return emb


def cosine_similarity_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and all document vectors.
    Returns a 1D array of similarity scores.
    """
    # Normalize documents
    doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    doc_norms[doc_norms == 0] = 1e-9
    doc_matrix_norm = doc_matrix / doc_norms

    # Normalize query
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        q_norm = 1e-9
    query_vec_norm = query_vec / q_norm

    # Cosine similarity: dot(query, docs_normed.T)
    sims = doc_matrix_norm @ query_vec_norm
    return sims


def format_result(
    result: Dict,
    max_snippet_chars: int = 260,
) -> str:
    """
    Create a human-readable string for one search result.
    Expects a result dict from search_index().
    """
    rank = result["rank"]
    score = result["score"]
    text = result["text"]
    meta = result["metadata"]

    file_path = result.get("file_path", meta.get("file_path", "unknown"))
    chunk_index = result.get("chunk_index", meta.get("chunk_index", "unknown"))

    snippet = text.strip().replace("\n", " ")
    if len(snippet) > max_snippet_chars:
        snippet = snippet[: max_snippet_chars - 3] + "..."

    return (
        f"#{rank}  (score: {score:.4f})\n"
        f"   File: {file_path}\n"
        f"   Chunk: {chunk_index}\n"
        f"   Snippet: {snippet}\n"
    )


def search_index(
    query: str,
    top_k: int = 5,
    min_score: float = -1.0,
) -> List[Dict]:
    """
    High-level search:
    - loads index
    - embeds query
    - computes similarity
    - RETURNS a list of top_k result dicts, each with:
        {
          "rank": int,
          "score": float,
          "text": str,
          "metadata": dict,
          "file_path": str,
          "chunk_index": int,
        }
    """
    t0 = time.time()
    embeddings, texts, metadata, index_model = load_index(INDEX_DIR)

    # Handle model mismatch between index and .env
    if index_model != EMBED_MODEL:
        log(
            f"⚠️  embed_model in metadata.json is '{index_model}', "
            f"but EMBED_MODEL in .env is '{EMBED_MODEL}'. Using '{index_model}' for query."
        )
        model_to_use = index_model
    else:
        model_to_use = EMBED_MODEL

    # Get query embedding
    q_emb = get_embedding_ollama(query, model_to_use)
    log(f"✅ Query embedding dim = {q_emb.shape[0]}")

    # Compute cosine similarity
    sims = cosine_similarity_matrix(q_emb, embeddings)

    # Sort by similarity (descending)
    idxs = np.argsort(-sims)
    elapsed = time.time() - t0

    log("")
    log(f"🔍 Top {top_k} results for query: {query!r}")
    log(f"   (computed over {len(texts)} chunks in {elapsed:.2f}s)\n")

    results: List[Dict] = []
    shown = 0
    for rank, idx in enumerate(idxs, start=1):
        score = float(sims[idx])
        if score < min_score:
            continue
        if shown >= top_k:
            break

        res = {
            "rank": rank,
            "score": score,
            "text": texts[idx],
            "metadata": metadata[idx],
            "file_path": metadata[idx].get("file_path", "unknown"),
            "chunk_index": metadata[idx].get("chunk_index", "unknown"),
        }
        results.append(res)
        shown += 1

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Query from command line args
        query_text = " ".join(sys.argv[1:])
    else:
        # Prompt interactively
        try:
            query_text = input("Enter your query: ").strip()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    if not query_text:
        print("No query provided. Exiting.")
        sys.exit(0)

    top_k = 5
    min_score = -1.0
    results = search_index(query_text, top_k=top_k, min_score=min_score)

    if not results:
        log("⚠️ No results above the score threshold.")
        sys.exit(0)

    for res in results:
        print(format_result(res))
