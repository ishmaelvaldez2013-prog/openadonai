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
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST", "http://localhost:11434")

# You *can* override this via .env, but we'll also use the value recorded in metadata.json
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def log(msg: str):
    print(msg, flush=True)


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
    url = f"{OLLAMA_HOST}/api/embeddings"
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
    # Avoid division by zero
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
    rank: int,
    score: float,
    text: str,
    meta: Dict,
    max_snippet_chars: int = 260,
) -> str:
    """
    Create a human-readable string for one search result.
    """
    file_path = meta.get("file_path", "unknown")
    chunk_index = meta.get("chunk_index", "unknown")

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
) -> None:
    """
    High-level search:
    - loads index
    - embeds query
    - computes similarity
    - prints top_k results
    """
    t0 = time.time()
    embeddings, texts, metadata, index_model = load_index(INDEX_DIR)

    # Warn if model mismatch between index and .env
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

    shown = 0
    for rank, idx in enumerate(idxs, start=1):
        score = float(sims[idx])
        if score < min_score:
            continue
        if shown >= top_k:
            break

        result_str = format_result(rank, score, texts[idx], metadata[idx])
        print(result_str)
        shown += 1

    if shown == 0:
        log("⚠️ No results above the score threshold.")


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

    # You can tweak these defaults or add flags later
    search_index(query_text, top_k=5, min_score=-1.0)