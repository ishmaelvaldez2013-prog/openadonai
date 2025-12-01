# oracle_doctor.py

import os
import json
import time
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv

# Load .env first
load_dotenv()

# --- Shared config (aligned with your other scripts) ---

INDEX_DIR = os.getenv("INDEX_DIR", "./index_data")
ORACLE_URL = os.getenv("ORACLE_URL", "http://localhost:9000/search")

# Prefer OLLAMA_BASE_URL, then OLLAMA_HOST, then default
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL") or os.getenv(
    "OLLAMA_HOST", "http://localhost:11434"
)

# Models from env
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL") or os.getenv(
    "EMBED_MODEL", "nomic-embed-text"
)


def log(msg: str):
    print(msg, flush=True)


# ---------- Helpers ----------

def check_ollama_reachable() -> bool:
    base = OLLAMA_HOST.rstrip("/")
    log(f"🔎 Checking Ollama at: {base}")
    try:
        resp = requests.get(base, timeout=3)
        log(f"   ✅ Ollama responded with status {resp.status_code}")
        return True
    except Exception as e:
        log(f"   ❌ Ollama not reachable: {e}")
        return False


def check_ollama_model(name: str) -> bool:
    """
    Check if a given Ollama model is available using /api/show.
    Returns True if present, False otherwise.
    """
    base = OLLAMA_HOST.rstrip("/")
    show_url = base + "/api/show"

    log(f"   🔍 Checking model: {name}")
    try:
        resp = requests.post(show_url, json={"name": name}, timeout=10)
    except Exception as e:
        log(f"      ❌ Error talking to /api/show: {e}")
        return False

    if resp.status_code == 200:
        log(f"      ✅ Model '{name}' is installed.")
        return True
    else:
        log(f"      ⚠️ Model '{name}' not found (status {resp.status_code}).")
        return False


def check_index_files() -> bool:
    """
    Verify that embeddings.npy and metadata.json exist and are consistent.
    Returns True if OK, False otherwise.
    """
    log(f"\n📁 Checking index directory: {INDEX_DIR}")
    embeddings_path = os.path.join(INDEX_DIR, "embeddings.npy")
    metadata_path = os.path.join(INDEX_DIR, "metadata.json")

    ok = True

    if not os.path.isfile(embeddings_path):
        log(f"   ❌ Missing embeddings file: {embeddings_path}")
        ok = False
    else:
        log(f"   ✅ Found embeddings file: {embeddings_path}")

    if not os.path.isfile(metadata_path):
        log(f"   ❌ Missing metadata file: {metadata_path}")
        ok = False
    else:
        log(f"   ✅ Found metadata file: {metadata_path}")

    if not ok:
        return False

    # Try to load and sanity-check
    try:
        log("   🔎 Loading embeddings.npy...")
        emb = np.load(embeddings_path).astype("float32")
        log(f"      ✅ embeddings shape: {emb.shape}")
    except Exception as e:
        log(f"      ❌ Failed to load embeddings.npy: {e}")
        ok = False

    try:
        log("   🔎 Loading metadata.json...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        texts = meta.get("texts", [])
        metadata = meta.get("metadata", [])
        embed_model_in_index = meta.get("embed_model", None)

        log(f"      ✅ metadata: {len(texts)} texts, {len(metadata)} metadata entries")
        if embed_model_in_index:
            log(f"      ℹ️ embed_model in index: '{embed_model_in_index}'")
        else:
            log("      ⚠️ embed_model missing in metadata.json")
    except Exception as e:
        log(f"      ❌ Failed to load metadata.json: {e}")
        return False

    # Check alignment
    if ok:
        if len(texts) != emb.shape[0] or len(metadata) != emb.shape[0]:
            log(
                f"   ❌ Mismatch: embeddings rows ({emb.shape[0]}) "
                f"!= len(texts) ({len(texts)}) or len(metadata) ({len(metadata)})"
            )
            ok = False
        else:
            log("   ✅ embeddings/texts/metadata lengths are consistent.")

        # Compare index model vs env model
        target_model = OLLAMA_EMBED_MODEL
        if embed_model_in_index and embed_model_in_index != target_model:
            log(
                f"   ⚠️ embed_model in index is '{embed_model_in_index}', "
                f"but OLLAMA_EMBED_MODEL / EMBED_MODEL in .env is '{target_model}'."
            )
            log("      → This is OK if you intentionally changed models, but it may affect search consistency.")
        elif embed_model_in_index:
            log("   ✅ Index embed_model matches the env-configured model.")

    return ok


def derive_oracle_base(url: str) -> str:
    """
    Given ORACLE_URL (likely pointing at /search), derive the base URL
    to use for /health. e.g. http://localhost:9000/search -> http://localhost:9000
    """
    if "/" not in url:
        return url
    # crude but effective: strip last path element
    parts = url.rstrip("/").rsplit("/", 1)
    return parts[0]


def check_oracle_api() -> bool:
    """
    Check FastAPI Oracle:
    - /health
    - /search test query
    """
    log(f"\n🌐 Checking FastAPI Oracle at: {ORACLE_URL}")

    base = derive_oracle_base(ORACLE_URL)
    health_url = base + "/health"

    ok = True

    # 1. /health
    log(f"   🔎 GET {health_url}")
    try:
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
            log(f"      ✅ /health OK: {resp.json()}")
        else:
            log(f"      ❌ /health returned status {resp.status_code}: {resp.text}")
            ok = False
    except Exception as e:
        log(f"      ❌ Error calling /health: {e}")
        ok = False

    # 2. /search test
    log(f"   🔎 POST {ORACLE_URL} (test query)")
    try:
        resp = requests.post(
            ORACLE_URL,
            json={"query": "test", "top_k": 2},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            log(f"      ✅ /search OK, returned {len(results)} results.")
        else:
            log(f"      ❌ /search returned status {resp.status_code}: {resp.text}")
            ok = False
    except Exception as e:
        log(f"      ❌ Error calling /search: {e}")
        ok = False

    return ok


def check_query_pipeline(sample_query: Optional[str] = None) -> bool:
    """
    Import query_index.search_index and run a sample query end-to-end
    through the local embeddings to ensure cosine search works.

    Returns True if successful.
    """
    if sample_query is None:
        sample_query = "What is the cosmic architecture of the parables?"

    log(f"\n🧪 Running end-to-end query_index search for:\n   {sample_query!r}")

    try:
        from query_index import search_index  # type: ignore
    except Exception as e:
        log(f"   ❌ Could not import query_index.search_index: {e}")
        return False

    try:
        t0 = time.time()
        results = search_index(sample_query, top_k=3, min_score=-1.0)
        dt = time.time() - t0
        if not results:
            log("   ⚠️ search_index returned no results.")
            return False

        log(f"   ✅ search_index returned {len(results)} results in {dt:.2f}s:")
        for r in results:
            log(
                f"      - [{r['score']:.4f}] {r['file_path']} "
                f"(chunk {r['chunk_index']})"
            )
        return True
    except Exception as e:
        log(f"   ❌ Error during search_index(): {e}")
        return False


# ---------- Main ----------

def main():
    log("🩺 OpenAdonAI Oracle Doctor\n")

    overall_ok = True

    # 1) Ollama & models
    log("=== 1. Ollama & models ===")
    ollama_ok = check_ollama_reachable()
    if not ollama_ok:
        overall_ok = False
    else:
        # Only check models if server is reachable
        log(f"\n   📦 Checking embedding model (OLLAMA_EMBED_MODEL): '{OLLAMA_EMBED_MODEL}'")
        emb_ok = check_ollama_model(OLLAMA_EMBED_MODEL)
        if not emb_ok:
            overall_ok = False

        log(f"\n   💬 Checking chat model (OLLAMA_CHAT_MODEL): '{OLLAMA_CHAT_MODEL}'")
        chat_ok = check_ollama_model(OLLAMA_CHAT_MODEL)
        if not chat_ok:
            overall_ok = False

    # 2) Index files
    log("\n=== 2. Index files ===")
    index_ok = check_index_files()
    if not index_ok:
        overall_ok = False

    # 3) FastAPI Oracle
    log("\n=== 3. Oracle API (FastAPI) ===")
    oracle_ok = check_oracle_api()
    if not oracle_ok:
        overall_ok = False

    # 4) Query pipeline
    log("\n=== 4. Query pipeline (query_index) ===")
    query_ok = check_query_pipeline()
    if not query_ok:
        overall_ok = False

    # Summary
    log("\n=== SUMMARY ===")
    if overall_ok:
        log("✅ All core checks passed. Your Archetype Oracle stack looks healthy. ✨")
    else:
        log("⚠️ Some checks failed. See logs above for details.\n")
        log("   Recommended next steps:")
        log("   - Fix any Ollama connectivity/model issues.")
        log("   - Ensure index_obsidian.py has been run successfully.")
        log("   - Confirm your FastAPI Oracle (app.py) is running and pointed at the right index.")
        log("   - Re-run oracle_doctor.py after fixes.")


if __name__ == "__main__":
    main()