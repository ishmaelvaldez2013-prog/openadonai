# openadonai_start.py

import os
import time
import subprocess
import requests
import argparse
from dotenv import load_dotenv

load_dotenv()

# Ollama config
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CMD = os.getenv("OLLAMA_CMD", "ollama")  # customize if needed

# Oracle / FastAPI config
ORACLE_HOST = os.getenv("ORACLE_HOST", "127.0.0.1")
ORACLE_PORT = int(os.getenv("ORACLE_PORT", "9000"))
ORACLE_RELOAD = os.getenv("ORACLE_RELOAD", "true").lower() == "true"

# Python for running uvicorn and ask_oracle (your venv)
PYTHON_BIN = os.getenv("PYTHON_BIN", ".venv/bin/python")

# Index config
INDEX_DIR = os.getenv("INDEX_DIR", "./index_data")


def log(msg: str):
    print(msg, flush=True)


# ---------- Shared helpers ----------

def parse_models(models: str):
    """Split comma-separated model list into clean names."""
    return [m.strip() for m in models.split(",") if m.strip()]


# ---------- Ollama helpers (startup) ----------

def wait_for_ollama(timeout: int = 60) -> bool:
    """Wait until Ollama responds on OLLAMA_BASE."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(OLLAMA_BASE, timeout=3)
            log(f"   ✅ Ollama responded (status {r.status_code})")
            return True
        except Exception:
            log("   … waiting for Ollama to become ready …")
            time.sleep(2)
    return False


def start_ollama_daemon():
    """
    Try to start the Ollama server.

    NOTE: On macOS with the Ollama app, the daemon may already be managed.
    This is just a best-effort call to `ollama serve`.
    """
    log(f"⚙️  Attempting to start Ollama via `{OLLAMA_CMD} serve` …")
    try:
        subprocess.Popen(
            [OLLAMA_CMD, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        log(f"   ❌ Failed to start Ollama via `{OLLAMA_CMD} serve`: {e}")


def ensure_ollama_up():
    """Ensure Ollama server is up; try to start it if not."""
    log(f"🔎 Checking Ollama server at {OLLAMA_BASE}")
    try:
        r = requests.get(OLLAMA_BASE, timeout=3)
        log(f"   ✅ Ollama is already running (status {r.status_code})")
        return
    except Exception:
        log("   ⚠️ Ollama not responding. Trying to start it…")
        start_ollama_daemon()
        if not wait_for_ollama(timeout=60):
            raise SystemExit(
                f"❌ Ollama did not become ready at {OLLAMA_BASE}. "
                f"Start the Ollama app or run `ollama serve` and try again."
            )


# ---------- Model autoload ----------

def run_autoload_models():
    """Run autoload_models.main() to ensure models are installed & warm."""
    log("\n🧠 Ensuring models are installed & warm-loaded (autoload_models.py)…")
    from autoload_models import main as autoload_main

    autoload_main()


# ---------- Oracle / FastAPI helpers ----------

def oracle_health_url() -> str:
    return f"http://{ORACLE_HOST}:{ORACLE_PORT}/health"


def oracle_already_running() -> bool:
    """Return True if the Oracle FastAPI server is already up on /health."""
    url = oracle_health_url()
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            log(f"   ✅ Oracle API already running at {url}")
            return True
    except Exception:
        pass
    return False


def start_uvicorn():
    """Start the FastAPI Oracle (uvicorn app:app) if not already running."""
    log(
        f"\n🚀 Starting OpenAdonAI Oracle API "
        f"on http://{ORACLE_HOST}:{ORACLE_PORT} "
        f"(reload={ORACLE_RELOAD})"
    )

    # Check if already running; avoid port-in-use error
    if oracle_already_running():
        log("   ℹ️ Skipping uvicorn start because Oracle is already up.")
        return

    cmd = [
        PYTHON_BIN,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        ORACLE_HOST,
        "--port",
        str(ORACLE_PORT),
    ]
    if ORACLE_RELOAD:
        cmd.append("--reload")

    log(f"   ▶ Running: {' '.join(cmd)}")
    # This process will run until Ctrl+C
    proc = subprocess.Popen(cmd)

    try:
        proc.wait()
    except KeyboardInterrupt:
        log("\n🛑 Keyboard interrupt received, stopping Oracle…")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


# ---------- Indexing pipeline ----------

def run_index_pipeline():
    """
    Rebuild the embeddings index using index_obsidian.py.
    This re-reads .env (INDEX_DIR, OBSIDIAN_ROOT, EMBED_MODEL, DRY_RUN_EMBED, etc.)
    on every run, so any changes you make are picked up automatically.
    """
    log("📚 Rebuilding Archetype index via index_obsidian.py…")

    # Only needed for *real* embeddings; if DRY_RUN_EMBED=true, this is cheap anyway.
    ensure_ollama_up()
    run_autoload_models()

    from index_obsidian import main as index_main
    index_main()


# ---------- Startup pipeline ----------

def run_start_pipeline():
    """
    Original startup behavior:
      1) Ensure Ollama is up
      2) Ensure models are installed + warm
      3) Start the Oracle API (if not already running)
    """
    log("🌌 OpenAdonAI Full Startup Pipeline\n")
    ensure_ollama_up()
    run_autoload_models()
    start_uvicorn()


# ---------- Doctor / Health Check ----------

def doctor_check_ollama() -> bool:
    log("🩺 [1/4] Checking Ollama server…")
    try:
        r = requests.get(OLLAMA_BASE, timeout=3)
        log(f"   ✅ Ollama reachable at {OLLAMA_BASE} (status {r.status_code})")
        return True
    except Exception as e:
        log(f"   ❌ Ollama not reachable at {OLLAMA_BASE}: {e}")
        return False


def doctor_model_exists(name: str) -> bool:
    """Check if a model is installed via /api/show."""
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/show", json={"name": name}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def doctor_ping_chat_model(name: str) -> bool:
    """Light check that a chat model can respond."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": name, "prompt": "ping"},
            timeout=15,
        )
        return r.status_code == 200
    except Exception:
        return False


def doctor_ping_embed_model(name: str) -> bool:
    """Light check that an embedding model can produce an embedding."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": name, "prompt": "ping"},
            timeout=15,
        )
        if r.status_code != 200:
            return False
        data = r.json()
        return bool(data.get("embedding"))
    except Exception:
        return False


def doctor_check_models() -> bool:
    log("🩺 [2/4] Checking Ollama models…")

    chat_models_raw = os.getenv("OLLAMA_CHAT_MODEL", "")
    embed_models_raw = os.getenv("OLLAMA_EMBED_MODEL", "") or os.getenv("EMBED_MODEL", "")
    extra_models_raw = os.getenv("OLLAMA_EXTRA_MODELS", "")

    chat_models = parse_models(chat_models_raw)
    embed_models = parse_models(embed_models_raw)
    extra_models = parse_models(extra_models_raw)

    if not (chat_models or embed_models or extra_models):
        log("   ⚠️ No models defined in .env (OLLAMA_CHAT_MODEL / OLLAMA_EMBED_MODEL / EMBED_MODEL / OLLAMA_EXTRA_MODELS).")
        return False

    all_ok = True

    # Chat models
    for m in chat_models:
        log(f"   🔎 Chat model: {m}")
        if not doctor_model_exists(m):
            log("      ❌ Not installed or not visible to Ollama")
            all_ok = False
            continue
        if doctor_ping_chat_model(m):
            log("      ✅ Installed & responds to /api/generate")
        else:
            log("      ⚠️ Installed but failed /api/generate ping")
            all_ok = False

    # Embed models
    for m in embed_models:
        log(f"   🔎 Embed model: {m}")
        if not doctor_model_exists(m):
            log("      ❌ Not installed or not visible to Ollama")
            all_ok = False
            continue
        if doctor_ping_embed_model(m):
            log("      ✅ Installed & responds to /api/embeddings")
        else:
            log("      ⚠️ Installed but failed /api/embeddings ping")
            all_ok = False

    # Extra models (assume chat-like)
    for m in extra_models:
        log(f"   🔎 Extra model: {m}")
        if not doctor_model_exists(m):
            log("      ❌ Not installed or not visible to Ollama")
            all_ok = False
            continue
        if doctor_ping_chat_model(m):
            log("      ✅ Installed & responds to /api/generate")
        else:
            log("      ⚠️ Installed but failed /api/generate ping")
            all_ok = False

    return all_ok


def doctor_check_index() -> bool:
    log("🩺 [3/4] Checking embeddings index…")
    embeddings_path = os.path.join(INDEX_DIR, "embeddings.npy")
    metadata_path = os.path.join(INDEX_DIR, "metadata.json")

    if not os.path.isfile(embeddings_path):
        log(f"   ❌ embeddings.npy not found at {embeddings_path}")
        return False
    if not os.path.isfile(metadata_path):
        log(f"   ❌ metadata.json not found at {metadata_path}")
        return False

    try:
        from query_index import load_index
        embeddings, texts, metadata, embed_model_from_index = load_index(INDEX_DIR)
        log(
            f"   ✅ Index OK: {embeddings.shape[0]} chunks, dim={embeddings.shape[1]}, "
            f"embed_model='{embed_model_from_index}'"
        )
        return True
    except Exception as e:
        log(f"   ❌ Error loading index from {INDEX_DIR}: {e}")
        return False


def doctor_check_oracle() -> bool:
    log("🩺 [4/4] Checking Oracle FastAPI server…")
    url = oracle_health_url()
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            log(f"   ✅ Oracle responding at {url}")
            return True
        else:
            log(f"   ❌ Oracle responded with status {r.status_code}: {r.text}")
            return False
    except Exception as e:
        log(f"   ❌ Could not connect to Oracle at {url}: {e}")
        return False


def run_doctor():
    log("🧪 OpenAdonAI Doctor — System Health Check\n")

    ok_ollama = doctor_check_ollama()
    ok_models = doctor_check_models() if ok_ollama else False
    ok_index = doctor_check_index()
    ok_oracle = doctor_check_oracle()

    log("\n📊 Summary:")
    log(f"   Ollama server : {'✅ OK' if ok_ollama else '❌ FAIL'}")
    log(f"   Ollama models : {'✅ OK' if ok_models else '❌ FAIL'}")
    log(f"   Index files   : {'✅ OK' if ok_index else '❌ FAIL'}")
    log(f"   Oracle API    : {'✅ OK' if ok_oracle else '❌ FAIL'}")

    if all([ok_ollama, ok_models, ok_index, ok_oracle]):
        log("\n🎉 All systems green. The Archetype Oracle is ready.")
    else:
        log("\n⚠️ Some checks failed. See details above.")


# ---------- ask_oracle wrapper ----------

def run_ask(extra_args):
    """
    Wrap ask_oracle.py.
    Usage:
      openadonai ask "your question"
      openadonai ask --mode scholar --backend ollama "your question"
      openadonai ask --json --mode deep "your question"
    """
    if not extra_args:
        log("⚠️ No arguments passed to 'ask'.")
        log("   Example: openadonai ask --mode scholar --backend ollama \"What is the cosmic architecture of the parables?\"")
        return

    # Check Oracle health before calling ask_oracle
    url = oracle_health_url()
    try:
        r = requests.get(url, timeout=3)
        if r.status_code != 200:
            log(f"❌ Oracle is not responding at {url} (status {r.status_code}).")
            log("   → Start it first with: openadonai start")
            return
    except Exception as e:
        log(f"❌ Could not reach Oracle at {url}: {e}")
        log("   → Make sure it's running: openadonai start")
        return

    cmd = [PYTHON_BIN, "ask_oracle.py"] + extra_args
    log(f"\n🧞 Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


# ---------- CLI ----------

def cli():
    parser = argparse.ArgumentParser(
        prog="openadonai",
        description=(
            "OpenAdonAI local orchestration command.\n\n"
            "Subcommands:\n"
            "  start   → Ensure Ollama + models + Oracle API are running (default)\n"
            "  index   → Rebuild the Archetype embeddings index\n"
            "  doctor  → Run a health check of Ollama, models, index, and Oracle\n"
            "  ask     → Ask a question via ask_oracle.py (RAG + LLM)\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="start",
        choices=["start", "index", "doctor", "ask"],
        help="Command to run (default: start)",
    )

    # Everything after the command is passed through (for `ask`)
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional args passed through when using 'ask'.",
    )

    args = parser.parse_args()

    if args.command == "start":
        run_start_pipeline()
    elif args.command == "index":
        run_index_pipeline()
    elif args.command == "doctor":
        run_doctor()
    elif args.command == "ask":
        # args.args may start with a leading '--' in some shells; safe to strip it if present
        extra = args.args
        if extra and extra[0] == "--":
            extra = extra[1:]
        run_ask(extra)
    else:
        # Shouldn’t happen due to choices=…
        parser.print_help()


if __name__ == "__main__":
    cli()