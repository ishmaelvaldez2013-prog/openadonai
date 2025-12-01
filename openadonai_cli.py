#!/usr/bin/env python
"""
OpenAdonAI CLI

Subcommands:
  start    -> run full startup pipeline (Ollama + models + Oracle API) + macOS notification
  index    -> rebuild the RAG index from Obsidian
  doctor   -> health check: Ollama, models, index, Oracle API
  logs     -> tail Oracle logs
  stop     -> stop Oracle (LaunchAgent + uvicorn fallback)
  restart  -> stop then start
  ask      -> proxy to ask_oracle.py with all its flags

Usage:
  openadonai start
  openadonai doctor
  openadonai logs
  openadonai restart
  openadonai stop
  openadonai ask "What is the cosmic architecture of the parables?"
"""

import os
import sys
import time
import argparse
import subprocess
import requests
from dotenv import load_dotenv

# Load .env for config
load_dotenv()

# --- Paths & config ---

PROJECT_ROOT = "/Users/ishmael/Developer/OpenAdonAI/Tools/rag_service"
VENV_PY = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

INDEX_DIR = os.getenv("INDEX_DIR", "./index_data")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

ORACLE_HOST = os.getenv("ORACLE_HOST", "127.0.0.1")
ORACLE_PORT = int(os.getenv("ORACLE_PORT", "9000"))
ORACLE_HEALTH_URL = f"http://{ORACLE_HOST}:{ORACLE_PORT}/health"

# LaunchAgent / logs (from the plist we created)
LAUNCH_AGENT_LABEL = "com.openadonai.oracle"
LOG_OUT = os.path.expanduser("~/Library/Logs/openadonai-oracle.log")
LOG_ERR = os.path.expanduser("~/Library/Logs/openadonai-oracle.err")


def log(msg: str):
    print(msg, flush=True)


# --- Helpers ---

def notify_mac(message: str, title: str = "OpenAdonAI Oracle"):
    """Send a macOS Notification Center alert using osascript (best-effort)."""
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}"',
            ],
            check=False,
        )
    except Exception:
        # Silently ignore if osascript isn’t available
        pass


def run_py(module_or_script: str, args=None):
    """Run a Python module or script via the project venv."""
    args = args or []
    if module_or_script.endswith(".py"):
        cmd = [VENV_PY, module_or_script] + args
    else:
        # treat as -m module
        cmd = [VENV_PY, "-m", module_or_script] + args
    log(f"   ▶ Running: {' '.join(cmd)}")
    return subprocess.run(cmd)


def check_ollama() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code == 200:
            log(f"✅ Ollama reachable at {OLLAMA_BASE}")
            return True
        else:
            log(f"❌ Ollama responded with status {r.status_code}")
            return False
    except Exception as e:
        log(f"❌ Ollama not reachable at {OLLAMA_BASE}: {e}")
        return False


def check_ollama_model(name: str) -> bool:
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/show", json={"name": name}, timeout=5)
        if r.status_code == 200:
            log(f"   ✅ Model installed: {name}")
            return True
        else:
            log(f"   ❌ Model not found (status {r.status_code}): {name}")
            return False
    except Exception as e:
        log(f"   ❌ Error checking model {name}: {e}")
        return False


def check_index() -> bool:
    from query_index import load_index

    try:
        emb, texts, meta, model = load_index(INDEX_DIR)
        log(
            f"✅ Index OK: {emb.shape[0]} chunks, dim={emb.shape[1]} "
            f"(embed_model={model})"
        )
        return True
    except Exception as e:
        log(f"❌ Index check failed: {e}")
        return False


def check_oracle() -> bool:
    try:
        r = requests.get(ORACLE_HEALTH_URL, timeout=5)
        if r.status_code == 200:
            log(f"✅ Oracle API healthy at {ORACLE_HEALTH_URL}")
            return True
        else:
            log(
                f"❌ Oracle health responded with {r.status_code}: "
                f"{r.text[:200]}"
            )
            return False
    except Exception as e:
        log(f"❌ Oracle health check failed: {e}")
        return False


# --- Commands ---

def cmd_start(args):
    """Full startup pipeline via openadonai_start.py, plus notification."""
    log("🚀 Starting OpenAdonAI Oracle pipeline (openadonai_start.py)…")
    # Delegates to your existing startup orchestrator
    result = run_py("openadonai_start.py")

    # If Oracle is healthy, ping Notification Center
    if check_oracle():
        notify_mac("Oracle is online and ready.", "OpenAdonAI")
    else:
        notify_mac("Oracle failed health check. See logs.", "OpenAdonAI")

    sys.exit(result.returncode)


def cmd_index(args):
    """Rebuild the index (index_obsidian.py)."""
    log("📚 Rebuilding Archetype index (index_obsidian.py)…")
    result = run_py("index_obsidian.py")
    sys.exit(result.returncode)


def cmd_doctor(args):
    """Doctor: check Ollama, models, index, Oracle."""
    log("🩺 OpenAdonAI Doctor – health check\n")

    # 1) Ollama
    log("1️⃣  Ollama server:")
    ok_ollama = check_ollama()
    print()

    # 2) Models
    chat_models = [
        m.strip()
        for m in os.getenv("OLLAMA_CHAT_MODEL", "").split(",")
        if m.strip()
    ]
    embed_models = [
        m.strip()
        for m in (
            os.getenv("OLLAMA_EMBED_MODEL", "")
            or os.getenv("EMBED_MODEL", "")
        ).split(",")
        if m.strip()
    ]
    extra_models = [
        m.strip()
        for m in os.getenv("OLLAMA_EXTRA_MODELS", "").split(",")
        if m.strip()
    ]
    all_models = chat_models + embed_models + extra_models
    log("2️⃣  Ollama models:")
    model_ok = True
    if not ok_ollama:
        log("   ⚠️ Skipping model checks because Ollama is down.")
        model_ok = False
    else:
        if not all_models:
            log("   ⚠️ No models defined in .env (OLLAMA_*_MODEL).")
        for name in all_models:
            if not check_ollama_model(name):
                model_ok = False
    print()

    # 3) Index
    log("3️⃣  Index:")
    index_ok = check_index()
    print()

    # 4) Oracle API
    log("4️⃣  Oracle API:")
    oracle_ok = check_oracle()
    print()

    if ok_ollama and model_ok and index_ok and oracle_ok:
        log("✅ Doctor summary: everything looks healthy.")
        sys.exit(0)
    else:
        log("❌ Doctor summary: one or more checks failed. See above.")
        sys.exit(1)


def cmd_logs(args):
    """Tail Oracle logs."""
    n = args.lines
    log(f"📜 Last {n} lines of stdout log: {LOG_OUT}")
    if os.path.exists(LOG_OUT):
        subprocess.run(["tail", "-n", str(n), LOG_OUT])
    else:
        log("   (no stdout log found)")

    print()
    log(f"📜 Last {n} lines of stderr log: {LOG_ERR}")
    if os.path.exists(LOG_ERR):
        subprocess.run(["tail", "-n", str(n), LOG_ERR])
    else:
        log("   (no stderr log found)")


def cmd_stop(args):
    """Stop Oracle (LaunchAgent + uvicorn fallback)."""
    log("🛑 Stopping OpenAdonAI Oracle…")

    # 1) If running under LaunchAgent, stop it
    if sys.platform == "darwin":
        log(f"   ▶ launchctl stop {LAUNCH_AGENT_LABEL}")
        subprocess.run(["launchctl", "stop", LAUNCH_AGENT_LABEL], check=False)

    # 2) Fallback: kill uvicorn if still running
    log('   ▶ pkill -f "uvicorn app:app" (best-effort)')
    subprocess.run(["pkill", "-f", "uvicorn app:app"], check=False)

    # 3) Check health
    time.sleep(1.0)
    if not check_oracle():
        log("✅ Oracle appears to be stopped.")
        sys.exit(0)
    else:
        log("⚠️ Oracle still responding to /health; check processes manually.")
        sys.exit(1)


def cmd_restart(args):
    """Restart Oracle (stop then start)."""
    log("🔁 Restarting Oracle…")
    cmd_stop(args)
    # If stop returns nonzero, we might still want to attempt start
    cmd_start(args)


def cmd_ask(args):
    """Proxy to ask_oracle.py with all remaining args."""
    if not args.query:
        print("Usage: openadonai ask \"your question\" [--mode ... --backend ...]")
        sys.exit(1)

    # Remaining arguments go straight to ask_oracle.py
    run_args = ["ask_oracle.py"] + args.query
    result = run_py(run_args[0], run_args[1:])
    sys.exit(result.returncode)


# --- CLI setup ---

def build_parser():
    parser = argparse.ArgumentParser(
        prog="openadonai",
        description="OpenAdonAI multi-tool CLI (start/index/doctor/logs/restart/stop/ask).",
    )
    sub = parser.add_subparsers(dest="command")

    # start
    p_start = sub.add_parser("start", help="Run full startup pipeline (Ollama + models + Oracle API).")
    p_start.set_defaults(func=cmd_start)

    # index
    p_index = sub.add_parser("index", help="Rebuild Archetype index from Obsidian.")
    p_index.set_defaults(func=cmd_index)

    # doctor
    p_doc = sub.add_parser("doctor", help="Health check: Ollama, models, index, Oracle.")
    p_doc.set_defaults(func=cmd_doctor)

    # logs
    p_logs = sub.add_parser("logs", help="Tail Oracle logs.")
    p_logs.add_argument("-n", "--lines", type=int, default=50, help="Number of log lines to show (default 50).")
    p_logs.set_defaults(func=cmd_logs)

    # stop
    p_stop = sub.add_parser("stop", help="Stop Oracle (LaunchAgent + uvicorn fallback).")
    p_stop.set_defaults(func=cmd_stop)

    # restart
    p_restart = sub.add_parser("restart", help="Restart Oracle.")
    p_restart.set_defaults(func=cmd_restart)

    # ask
    p_ask = sub.add_parser("ask", help="Ask the Archetype Oracle (proxy to ask_oracle.py).")
    # everything after 'ask' is passed thru to ask_oracle.py
    p_ask.add_argument("query", nargs=argparse.REMAINDER, help="Question and flags passed through to ask_oracle.py")
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Default to 'start' if no subcommand given: `openadonai`
    if args.command is None:
        args.command = "start"
        args.func = cmd_start

    args.func(args)


if __name__ == "__main__":
    main()