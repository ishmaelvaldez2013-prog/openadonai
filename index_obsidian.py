# index_obsidian.py

import os
import json
import time
import hashlib
from typing import List, Dict, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

# Try to import resource for memory usage (macOS/Linux only)
try:
    import resource

    def log_memory(label: str = ""):
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mb = usage / 1024 / 1024
        print(f"   🧠 Memory [{label}]: ~{mb:.2f} MB", flush=True)

except ImportError:
    def log_memory(label: str = ""):
        # No-op on platforms without resource
        pass


def log(msg: str):
    print(msg, flush=True)


def heartbeat(label: str):
    now = time.strftime("%H:%M:%S")
    log(f"   ❤️ Heartbeat [{label}] at {now}")


# --- Load .env ---
load_dotenv()

OBSIDIAN_ROOT = os.getenv("OBSIDIAN_ROOT")
INDEX_DIR = os.getenv("INDEX_DIR", "./index_data")

# Prefer OLLAMA_BASE_URL if present, fall back to OLLAMA_HOST, then default
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Prefer OLLAMA_EMBED_MODEL, fall back to EMBED_MODEL, then default
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL") or os.getenv("EMBED_MODEL", "nomic-embed-text")
DRY_RUN_EMBED = os.getenv("DRY_RUN_EMBED", "false").lower() == "true"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Caching locations
FILE_CACHE_DIR = os.path.join(INDEX_DIR, "file_cache")
FILE_STATE_PATH = os.path.join(INDEX_DIR, "file_state.json")

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(FILE_CACHE_DIR, exist_ok=True)


# ---------- Basic helpers ----------

def load_markdown_files(root: str) -> List[str]:
    md_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip Obsidian internal folder
        if ".obsidian" in dirnames:
            dirnames.remove(".obsidian")
        for fname in filenames:
            if fname.lower().endswith(".md"):
                md_files.append(os.path.join(dirpath, fname))
    return sorted(md_files)


def read_file(path: str) -> str:
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
    step = max(1, chunk_size - overlap)
    start = 0

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == length:
            break

        start += step

    return chunks


def get_embedding_ollama(text: str) -> List[float]:
    """Call Ollama's embeddings endpoint."""
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {
        "model": EMBED_MODEL,
        "prompt": text,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


# ---------- Caching helpers ----------

def file_id_from_relpath(rel_path: str) -> str:
    """Stable ID for cache files based on relative path."""
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()


def get_file_state(path: str, root: str) -> Tuple[str, Dict]:
    rel_path = os.path.relpath(path, root)
    st = os.stat(path)
    return rel_path, {
        "mtime": st.st_mtime,
        "size": st.st_size,
    }


def load_previous_file_state() -> Dict[str, Dict]:
    if not os.path.exists(FILE_STATE_PATH):
        return {}
    try:
        with open(FILE_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_file_state(state: Dict[str, Dict]) -> None:
    with open(FILE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ---------- Main indexing logic with caching ----------

def main():
    if not OBSIDIAN_ROOT:
        raise RuntimeError("OBSIDIAN_ROOT not set in .env")
    if not os.path.isdir(OBSIDIAN_ROOT):
        raise RuntimeError(f"OBSIDIAN_ROOT does not exist: {OBSIDIAN_ROOT}")

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
            log(f"      ollama pull {EMBED_MODEL}")
            return
    else:
        log("🧪 DRY_RUN_EMBED is true → skipping actual embedding calls.\n")

    prev_state = load_previous_file_state()
    new_state: Dict[str, Dict] = {}

    # For DRY RUN we still build a global view of texts + metadata
    all_texts: List[str] = []
    all_meta: List[Dict] = []

    total_files = len(md_files)

    for file_idx, path in enumerate(md_files, start=1):
        rel_path, this_state = get_file_state(path, OBSIDIAN_ROOT)
        file_id = file_id_from_relpath(rel_path)
        cache_meta_path = os.path.join(FILE_CACHE_DIR, f"{file_id}.meta.json")
        cache_emb_path = os.path.join(FILE_CACHE_DIR, f"{file_id}.emb.npy")

        log(f"📄 [{file_idx}/{total_files}] Processing: {path}")
        heartbeat(f"start file {rel_path}")
        log_memory(f"before read_file {rel_path}")

        # Skip unchanged files (only when not in DRY RUN)
        prev = prev_state.get(rel_path)
        if (
            not DRY_RUN_EMBED
            and prev is not None
            and abs(prev.get("mtime", 0) - this_state["mtime"]) < 1e-6
            and prev.get("size") == this_state["size"]
            and os.path.exists(cache_meta_path)
            and os.path.exists(cache_emb_path)
        ):
            log(f"   ⏭️  Unchanged since last index, reusing cached embeddings.")
            new_state[rel_path] = this_state
            log("")
            continue

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

        # DRY RUN: no caching, just accumulate global metadata to inspect chunking
        if DRY_RUN_EMBED:
            for ci, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_meta.append(
                    {
                        "file_path": rel_path,
                        "chunk_index": ci,
                    }
                )
                if (ci + 1) % 5 == 0 or (ci + 1) == len(chunks):
                    heartbeat(f"{rel_path} chunk {ci+1}/{len(chunks)} (DRY RUN)")
                    log_memory(f"{rel_path} chunk {ci+1}/{len(chunks)} (DRY RUN)")
            log("")
            new_state[rel_path] = this_state
            continue

        # Real embedding path with per-file cache
        file_embeddings: List[np.ndarray] = []
        file_chunks_meta: List[Dict] = []

        for ci, chunk in enumerate(chunks):
            t0 = time.time()
            try:
                emb = get_embedding_ollama(chunk)
            except Exception as e:
                log(f"   ❌ Error embedding chunk {ci+1}/{len(chunks)} for {rel_path}: {e}")
                continue
            dt = time.time() - t0

            file_embeddings.append(np.array(emb, dtype="float32"))
            file_chunks_meta.append(
                {
                    "chunk_index": ci,
                    "text": chunk,
                }
            )

            if (ci + 1) % 5 == 0 or (ci + 1) == len(chunks):
                log(f"      • Embedded chunk {ci+1}/{len(chunks)} for {rel_path} (took {dt:.2f}s)")
                heartbeat(f"{rel_path} chunk {ci+1}/{len(chunks)}")
                log_memory(f"{rel_path} chunk {ci+1}/{len(chunks)}")

        if not file_embeddings:
            log(f"   ⚠️ No embeddings succeeded for {rel_path}, skipping cache.\n")
            continue

        # Save per-file cache
        np.save(cache_emb_path, np.vstack(file_embeddings))
        with open(cache_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "path": rel_path,
                    "mtime": this_state["mtime"],
                    "size": this_state["size"],
                    "chunks": file_chunks_meta,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        new_state[rel_path] = this_state
        log(f"   ✅ Cached {len(file_embeddings)} embeddings for {rel_path}\n")

    # ---------- Saving phase ----------

    metadata_path = os.path.join(INDEX_DIR, "metadata.json")
    embeddings_path = os.path.join(INDEX_DIR, "embeddings.npy")

    # DRY RUN BEHAVIOR (no caching-based rebuild; just like your original script)
    if DRY_RUN_EMBED:
        log("⚠️ DRY_RUN_EMBED is true → not saving embeddings.npy (only metadata.json).")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "texts": all_texts,
                    "metadata": all_meta,
                    "embed_model": EMBED_MODEL,
                    "dry_run": True,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        save_file_state(new_state)
        log(f"✅ Saved metadata (DRY RUN) to {metadata_path}")
        log_memory("end of script (DRY RUN)")
        log("🎉 Dry run indexing complete.")
        return

    # REAL SAVE: rebuild global index from per-file caches
    log("🧩 Rebuilding global embeddings and metadata from cache…")

    global_embeddings: List[np.ndarray] = []
    global_texts: List[str] = []
    global_meta: List[Dict] = []

    for rel_path, state in new_state.items():
        file_id = file_id_from_relpath(rel_path)
        cache_meta_path = os.path.join(FILE_CACHE_DIR, f"{file_id}.meta.json")
        cache_emb_path = os.path.join(FILE_CACHE_DIR, f"{file_id}.emb.npy")

        if not (os.path.exists(cache_meta_path) and os.path.exists(cache_emb_path)):
            log(f"   ⚠️ Missing cache for {rel_path}, skipping from global index.")
            continue

        with open(cache_meta_path, "r", encoding="utf-8") as f:
            meta_obj = json.load(f)
        file_chunks = meta_obj.get("chunks", [])
        file_emb = np.load(cache_emb_path)

        if len(file_chunks) != file_emb.shape[0]:
            log(f"   ⚠️ Mismatch between metadata and embeddings for {rel_path}, skipping.")
            continue

        global_embeddings.append(file_emb)
        for chunk_meta, emb_vec in zip(file_chunks, file_emb):
            global_texts.append(chunk_meta["text"])
            global_meta.append(
                {
                    "file_path": rel_path,
                    "chunk_index": chunk_meta["chunk_index"],
                }
            )

    if not global_embeddings:
        log("⚠️ No embeddings collected for global index. Exiting without saving.")
        return

    log_memory("before numpy save")
    embeddings_matrix = np.vstack(global_embeddings)

    np.save(embeddings_path, embeddings_matrix)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "texts": global_texts,
                "metadata": global_meta,
                "embed_model": EMBED_MODEL,
                "dry_run": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_file_state(new_state)

    log(f"✅ Saved embeddings to {embeddings_path}")
    log(f"✅ Saved metadata to {metadata_path}")
    log_memory("end of script")
    log("🎉 Indexing complete.")


if __name__ == "__main__":
    main()
