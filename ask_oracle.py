# ask_oracle.py

import os
import argparse
import requests
import json

# ---------- Config from environment ----------

ORACLE_URL = os.getenv("ORACLE_URL", "http://localhost:9000/search")

# Ollama models (configurable via .env)
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Optional: preload these models at startup (comma-separated in .env)
OLLAMA_AUTO_PULL = [
    m.strip() for m in os.getenv("OLLAMA_AUTO_PULL", "").split(",") if m.strip()
]

# Mode configuration
MODE_CONFIG = {
    "short": {
        "top_k": 3,
        "style_hint": "Give a concise, high-level answer (3–6 sentences). Focus on the core idea.",
    },
    "deep": {
        "top_k": 7,
        "style_hint": "Give a detailed, structured answer with clear sections and bullet points where helpful.",
    },
    "scholar": {
        "top_k": 12,
        "style_hint": "Give an in-depth, scholarly answer with careful reasoning, explicit mappings, and rich detail. Stay strictly within the provided context.",
    },
}

MODE_ORDER = ["short", "deep", "scholar"]


# ---------- Ollama helpers ----------

def ensure_ollama_model(model: str):
    """
    Ensure the requested Ollama model is available.

    - If Ollama is not reachable → raise helpful error
    - If model missing → pull it automatically via /api/pull
    """
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    # 1. Check Ollama server is up
    try:
        requests.get(base, timeout=3)
    except Exception as e:
        raise RuntimeError(
            f"Ollama is not reachable at {base}. Is the Ollama app or "
            f"'ollama serve' running?\nDetails: {e}"
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
    print(f"⏬ Pulling Ollama model '{model}' (not found locally)...")

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
                        print("   ", status)
                except json.JSONDecodeError:
                    print("   ", line)
    except Exception as e:
        raise RuntimeError(f"Failed to pull Ollama model '{model}': {e}")

    print(f"✅ Model '{model}' is now installed.\n")


# ---------- RAG / context helpers ----------

def fetch_context(query: str, top_k: int = 5):
    """Call Oracle API and fetch chunks."""
    resp = requests.post(
        ORACLE_URL,
        json={"query": query, "top_k": top_k},
        timeout=30,
    )
    resp.raise_for_status()

    data = resp.json()
    results = data.get("results", [])
    chunks = [r["text"] for r in results]
    context = "\n\n---\n\n".join(chunks)

    return results, context


def build_prompt(query: str, context: str, mode: str) -> str:
    mode_cfg = MODE_CONFIG.get(mode, MODE_CONFIG["deep"])
    style_hint = mode_cfg["style_hint"]

    return f"""
You are the OpenAdonAI Oracle.

Use ONLY the context below from my Archetype vault to answer the question.
If the context does not contain the answer, say so. Do not invent unrelated information.

Answer style mode: {mode.upper()}
Guidance: {style_hint}

======== BEGIN CONTEXT ========
{context}
======== END CONTEXT ==========

QUESTION:
{query}

Your answer:
""".strip()


def pretty_print_results(results):
    if not results:
        print("\n🧩 Top Matches: (none)")
        return
    print("\n🧩 Top Matches:")
    for r in results:
        print(f"- [{r['score']:.4f}] {r['file_path']} (chunk {r['chunk_index']})")


# ---------- LLM Backends ----------

def ask_ollama(prompt: str) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = base + "/api/chat"
    model = os.getenv("OLLAMA_CHAT_MODEL", OLLAMA_CHAT_MODEL)

    # Ensure the model exists (pull if needed)
    ensure_ollama_model(model)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=240)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "").strip()


def ask_openai(prompt: str) -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are the OpenAdonAI Oracle, strictly grounded in the provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ---------- Mode Helpers ----------

def downgrade_mode(mode: str) -> str | None:
    order = MODE_ORDER
    if mode not in order:
        return None
    idx = order.index(mode)
    if idx == 0:
        return None
    return order[idx - 1]


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(
        prog="ask_oracle.py",
        description=(
            "Ask the OpenAdonAI Archetype Oracle.\n\n"
            "This script:\n"
            "  1) Calls your local RAG API (ORACLE_URL, default http://localhost:9000/search)\n"
            "  2) Fetches top-k chunks from your Obsidian Archetype index\n"
            "  3) Builds an Oracle-style prompt\n"
            "  4) Optionally sends it to an LLM backend (Ollama or OpenAI)\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "query",
        nargs="+",
        help='Your question to the Oracle (e.g. "What is the cosmic architecture of the parables?")',
    )

    parser.add_argument(
        "--mode",
        choices=["short", "deep", "scholar"],
        default="deep",
        help=(
            "Answer mode (controls context size + tone):\n"
            "  short   → ~3 chunks, concise, high-level summary\n"
            "  deep    → ~7 chunks, structured & detailed (default)\n"
            "  scholar → ~12 chunks, in-depth, scholarly exposition\n"
        ),
    )

    parser.add_argument(
        "--backend",
        choices=["none", "ollama", "openai"],
        default="none",
        help=(
            "LLM backend to answer the prompt:\n"
            "  none   → just build/print prompt (no model call)\n"
            "  ollama → use local Ollama chat model (OLLAMA_CHAT_MODEL)\n"
            "  openai → use OpenAI chat completion (OPENAI_API_KEY, OPENAI_MODEL)\n"
        ),
    )

    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=None,
        help=(
            "Number of chunks to retrieve from Oracle.\n"
            "If omitted, defaults are based on --mode:\n"
            "  short=3, deep=7, scholar=12"
        ),
    )

    parser.add_argument(
        "--no-chunks",
        action="store_true",
        help="Do not print the list of top matched chunks (quieter console output).",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit machine-readable JSON instead of pretty console output.\n"
            "JSON includes: success, mode_used, top_k, prompt, answer, results, error."
        ),
    )

    return parser.parse_args()


# ---------- Core execution ----------

def run_oracle_round(query, mode, backend, top_k_arg, print_chunks):
    """Returns dict: {success, mode_used, top_k, prompt, answer, results, error}"""
    output = {
        "success": False,
        "mode_used": mode,
        "top_k": None,
        "prompt": "",
        "answer": "",
        "results": [],
        "error": "",
    }

    # Decide top_k
    top_k = top_k_arg if top_k_arg is not None else MODE_CONFIG[mode]["top_k"]
    output["top_k"] = top_k

    try:
        results, context = fetch_context(query, top_k)
        output["results"] = results
    except Exception as e:
        output["error"] = f"Error fetching context: {e}"
        return output

    if print_chunks:
        pretty_print_results(results)

    # Build prompt
    prompt = build_prompt(query, context, mode)
    output["prompt"] = prompt

    if backend == "none":
        output["success"] = True
        return output

    try:
        if backend == "ollama":
            answer = ask_ollama(prompt)
        else:
            answer = ask_openai(prompt)
        output["answer"] = answer
        output["success"] = True
        return output
    except Exception as e:
        output["error"] = f"LLM error: {e}"
        return output


def main():
    args = parse_args()
    query = " ".join(args.query)

    current_mode = args.mode
    backend = args.backend
    top_k_arg = args.top_k
    print_chunks = not args.no_chunks

    # Optional: preload models listed in OLLAMA_AUTO_PULL
    if backend == "ollama" and OLLAMA_AUTO_PULL:
        for m in OLLAMA_AUTO_PULL:
            try:
                ensure_ollama_model(m)
            except Exception as e:
                print(f"⚠️ Could not preload model '{m}': {e}")

    tried = set()

    while True:
        if current_mode in tried:
            if args.json:
                print(
                    json.dumps(
                        {
                            "success": False,
                            "error": f"Mode '{current_mode}' failed multiple times.",
                        },
                        indent=2,
                    )
                )
            else:
                print(f"❌ Mode '{current_mode}' failed multiple times.")
            return

        tried.add(current_mode)

        result = run_oracle_round(
            query, current_mode, backend, top_k_arg, print_chunks
        )

        if args.json:
            print(json.dumps(result, indent=2))
            return

        # If success → done
        if result["success"]:
            if result["answer"]:
                print("\n================= ORACLE ANSWER =================\n")
                print(result["answer"])
                print("\n=================================================\n")
            return

        # Otherwise try fallback
        next_mode = downgrade_mode(current_mode)
        if not next_mode:
            print(f"❌ No fallback modes left. Error: {result['error']}")
            return

        print(
            f"\n🔁 Mode '{current_mode}' failed → falling back to '{next_mode}'."
        )
        current_mode = next_mode


if __name__ == "__main__":
    main()
