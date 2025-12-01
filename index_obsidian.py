# index_obsidian.py

import os
import json
import time
from typing import List, Dict

import numpy as np
import requests
from dotenv import load_dotenv

# Try to import resource for memory usage (macOS/Linux only)
try:
    import resource

    def log_memory(label: str = ""):
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is in bytes, on Linux it's in KB. We'll just show both-ish.
        mb = usage / 1024 / 1024  # treat as bytes -> MB; if it's KB it's just off by 1024x but still useful trend
        print(f"   🧠 Memory [{label}]: ~{mb:.2f} MB", flush=True)

except ImportError:
    def log_memory(label: str = ""):
        # No-op on platforms without resource
        pass

# --- Load .env ---
load_dotenv()

OBSIDIAN_ROOT = os.getenv("OBSIDIAN_ROOT")
INDEX_DIR = os.getenv("INDEX_DIR", "./index_data")

# Prefer OLLAMA_BASE_URL if present, fall back to OLLAMA_HOST, then default
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST", "http://localhost:11434")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DRY_RUN_EMBED = os.getenv("DRY_RUN_EMBED", "false").lower() == "true"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def log(msg: str):
    print(msg, flush=True)


def heartbeat(label: str):
    now = time.strftime("%H:%M:%S")
    log(f"   ❤️ Heartbeat [{label}] at {now}")


def load_markdown_files(root: str) -> List[str]:
    md_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".md"):
                md_files.append(os.path.join(dirpath, fname))
    return sorted(md_files)


def read_file(path: str) -> str:
    # Extra logging for debug
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return text
    except Exception as e:
        log(f"   ❌ Error reading file {path}: {e}")
        return ""


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2]
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Safe sliding-window chunker:
    - steps forward by (chunk_size - overlap)
    - stops cleanly at the end of the text
    """
    if not text:
        return []

    chunks = []
    length = len(text)

    # step forward by (chunk_size - overlap) each time
    step = max(1, chunk_size - overlap)
    start = 0

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == length:
            # we just processed the final chunk, so stop
            break

        start += step

    return chunks


def get_embedding_ollama(text: str) -> List[float]:
    """Call Ollama's embeddings endpoint with nomic-embed-text (or other EMBED_MODEL)."""
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {
        "model": EMBED_MODEL,
        "prompt": text,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


def main():
    if not OBSIDIAN_ROOT:
        raise RuntimeError("OBSIDIAN_ROOT not set in .env")
    if not os.path.isdir(OBSIDIAN_ROOT):
        raise RuntimeError(f"OBSIDIAN_ROOT does not exist: {OBSIDIAN_ROOT}")

    os.makedirs(INDEX_DIR, exist_ok=True)

    log(f"📁 Indexing Obsidian vault at: {OBSIDIAN_ROOT}")
    log(f"📦 Saving index to: {INDEX_DIR}\n")

    log_memory("start of script")

    log(f"🔍 Scanning for markdown files in: {OBSIDIAN_ROOT}")
    md_files = load_markdown_files(OBSIDIAN_ROOT)
    log(f"📝 Found {len(md_files)} markdown files.\n")

    if not md_files:
        log("⚠️ No markdown files found. Exiting.")
        return

    log(f"🧠 Embedding backend: Ollama")
    log(f"   → Host:  {OLLAMA_HOST}")
    log(f"   → Model: {EMBED_MODEL}")
    log(f"   → DRY_RUN_EMBED = {DRY_RUN_EMBED}\n")

    if not DRY_RUN_EMBED:
        # Test Ollama connection early
        log(f"🧪 Testing Ollama embeddings with model: {EMBED_MODEL}")
        try:
            t0 = time.time()
            test_emb = get_embedding_ollama("test")
            dt = time.time() - t0
            log(f"   ✅ Ollama embedding test succeeded. Dim={len(test_emb)}, took {dt:.2f}s\n")
        except Exception as e:
            log(f"   ❌ Ollama embedding test failed: {e}")
            log("   Make sure `ollama serve` is running and the model is pulled, e.g.:")
            log("      ollama pull nomic-embed-text")
            return
    else:
        log("🧪 DRY_RUN_EMBED is true → skipping actual embedding calls.\n")

    texts: List[str] = []
    metadata: List[Dict] = []
    embeddings_list: List[List[float]] = []

    total_files = len(md_files)

    for file_idx, path in enumerate(md_files, start=1):
        rel_path = os.path.relpath(path, OBSIDIAN_ROOT)
        log(f"📄 [{file_idx}/{total_files}] Processing: {path}")
        heartbeat(f"start file {rel_path}")
        log_memory(f"before read_file {rel_path}")

        raw_text = read_file(path)
        log(f"   ↳ Raw length: {len(raw_text)} characters")

        cleaned = strip_frontmatter(raw_text)
        log(f"   ↳ Cleaned length: {len(cleaned)} characters")

        log_memory(f"before chunk_text {rel_path}")
        chunks = chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)
        log(f"   ↳ Generated {len(chunks)} chunks")
        log_memory(f"after chunk_text {rel_path}")

        if not chunks:
            log("   ⚠️ No chunks generated, skipping file.\n")
            continue

        for ci, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append(
                {
                    "file_path": rel_path,
                    "chunk_index": ci,
                }
            )

            if DRY_RUN_EMBED:
                # LIGHT DRY RUN:
                # - Do NOT allocate any embeddings
                # - Just heartbeat & memory logs so we can watch progress
                if (ci + 1) % 5 == 0 or (ci + 1) == len(chunks):
                    heartbeat(f"{rel_path} chunk {ci+1}/{len(chunks)} (DRY RUN)")
                    log_memory(f"{rel_path} chunk {ci+1}/{len(chunks)} (DRY RUN)")
                continue

            # Real embedding path
            t0 = time.time()
            try:
                emb = get_embedding_ollama(chunk)
            except Exception as e:
                log(f"   ❌ Error embedding chunk {ci+1}/{len(chunks)} for {rel_path}: {e}")
                continue
            dt = time.time() - t0
            embeddings_list.append(emb)

            if (ci + 1) % 5 == 0 or (ci + 1) == len(chunks):
                log(f"      • Embedded chunk {ci+1}/{len(chunks)} for {rel_path} (took {dt:.2f}s)")
                heartbeat(f"{rel_path} chunk {ci+1}/{len(chunks)}")
                log_memory(f"{rel_path} chunk {ci+1}/{len(chunks)}")

        log("")  # blank line between files

    # --- Saving phase ---
    metadata_path = os.path.join(INDEX_DIR, "metadata.json")

    if DRY_RUN_EMBED:
        # DRY RUN LIGHT:
        # - Do NOT save embeddings.npy
        # - Still save texts + metadata so we can inspect chunking behavior
        log("⚠️ DRY_RUN_EMBED is true → not saving embeddings.npy (only metadata.json).")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "texts": texts,
                    "metadata": metadata,
                    "embed_model": EMBED_MODEL,
                    "dry_run": True,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        log(f"✅ Saved metadata (DRY RUN) to {metadata_path}")
        log_memory("end of script (DRY RUN)")
        log("🎉 Dry run indexing complete.")
        return

    # Real embedding save path
    if not embeddings_list:
        log("⚠️ No embeddings generated. Exiting without saving.")
        return

    log_memory("before numpy save")

    embeddings = np.array(embeddings_list, dtype="float32")
    embeddings_path = os.path.join(INDEX_DIR, "embeddings.npy")

    np.save(embeddings_path, embeddings)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "texts": texts,
                "metadata": metadata,
                "embed_model": EMBED_MODEL,
                "dry_run": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log(f"✅ Saved embeddings to {embeddings_path}")
    log(f"✅ Saved metadata to {metadata_path}")
    log_memory("end of script")
    log("🎉 Indexing complete.")


if __name__ == "__main__":
    main()