# ask_oracle.py

import os
import sys
import argparse
import requests
import json

from dotenv import load_dotenv
load_dotenv()

ORACLE_URL = os.getenv("ORACLE_URL", "http://localhost:9000/search")


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

# -------- Env-based defaults --------

_env_mode = os.getenv("OPENADONAI_DEFAULT_MODE", "deep").strip().lower()
DEFAULT_MODE = _env_mode if _env_mode in MODE_CONFIG else "deep"

_env_backend = os.getenv("OPENADONAI_DEFAULT_BACKEND", "none").strip().lower()
DEFAULT_BACKEND = _env_backend if _env_backend in {"none", "ollama", "openai"} else "none"

# Optional env default for top_k
_env_top_k_raw = os.getenv("OPENADONAI_DEFAULT_TOP_K", "").strip()
ENV_DEFAULT_TOP_K: int | None
if _env_top_k_raw:
    try:
        val = int(_env_top_k_raw)
        ENV_DEFAULT_TOP_K = val if val > 0 else None
    except ValueError:
        ENV_DEFAULT_TOP_K = None
else:
    ENV_DEFAULT_TOP_K = None


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
    model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=240)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


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
            "  4) Optionally sends it to an LLM backend (Ollama or OpenAI)\n\n"
            "Env defaults:\n"
            "  OPENADONAI_DEFAULT_MODE      → default mode (short|deep|scholar)\n"
            "  OPENADONAI_DEFAULT_BACKEND   → default backend (none|ollama|openai)\n"
            "  OPENADONAI_DEFAULT_TOP_K     → default top_k (int > 0)\n"
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
        default=DEFAULT_MODE,
        help=(
            "Answer mode (controls context size + tone):\n"
            "  short   → ~3 chunks, concise, high-level summary\n"
            "  deep    → ~7 chunks, structured & detailed (default)\n"
            "  scholar → ~12 chunks, in-depth, scholarly exposition\n"
            f"\nDefault: {DEFAULT_MODE!r} (can override via OPENADONAI_DEFAULT_MODE)"
        ),
    )

    parser.add_argument(
        "--backend",
        choices=["none", "ollama", "openai"],
        default=DEFAULT_BACKEND,
        help=(
            "LLM backend to answer the prompt:\n"
            "  none   → just build/print prompt (no model call)\n"
            "  ollama → use local Ollama chat model (OLLAMA_CHAT_MODEL)\n"
            "  openai → use OpenAI chat completion (OPENAI_API_KEY, OPENAI_MODEL)\n"
            f"\nDefault: {DEFAULT_BACKEND!r} (can override via OPENADONAI_DEFAULT_BACKEND)"
        ),
    )

    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=ENV_DEFAULT_TOP_K,
        help=(
            "Number of chunks to retrieve from Oracle.\n"
            "If omitted, defaults are based on --mode (short=3, deep=7, scholar=12).\n"
            "You can also set a global default via OPENADONAI_DEFAULT_TOP_K.\n"
            "CLI -k always overrides env / mode defaults."
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

    # Decide top_k: CLI > env default > mode default
    if top_k_arg is not None:
        top_k = top_k_arg
    else:
        env_k = ENV_DEFAULT_TOP_K
        if env_k is not None:
            top_k = env_k
        else:
            top_k = MODE_CONFIG[mode]["top_k"]

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

    tried = set()

    while True:
        if current_mode in tried:
            if args.json:
                print(json.dumps({
                    "success": False,
                    "error": f"Mode '{current_mode}' failed multiple times.",
                }, indent=2))
            else:
                print(f"❌ Mode '{current_mode}' failed multiple times.")
            return

        tried.add(current_mode)

        result = run_oracle_round(query, current_mode, backend, top_k_arg, print_chunks)

        if args.json:
            # augment with mode_used
            result["mode_used"] = current_mode
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

        print(f"\n🔁 Mode '{current_mode}' failed → falling back to '{next_mode}'.")
        current_mode = next_mode


if __name__ == "__main__":
    main()
