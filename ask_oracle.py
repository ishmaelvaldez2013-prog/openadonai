# ask_oracle.py

import os
import sys
import argparse
import requests

ORACLE_URL = os.getenv("ORACLE_URL", "http://localhost:9000/search")

# Mode configuration: how many chunks, and how the answer should sound
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

MODE_ORDER = ["short", "deep", "scholar"]  # used for fallback logic


def fetch_context(query: str, top_k: int = 5):
    """
    Call the Archetype Oracle API and fetch top_k relevant chunks.
    Returns: (results, context_string)
    """
    try:
        resp = requests.post(
            ORACLE_URL,
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ Error calling Oracle API at {ORACLE_URL}: {e}")
        raise

    data = resp.json()
    results = data.get("results", [])
    if not results:
        print("⚠️ Oracle returned no results.")
        return [], ""

    chunks = [r["text"] for r in results]
    context = "\n\n---\n\n".join(chunks)
    return results, context


def build_prompt(query: str, context: str, mode: str) -> str:
    """
    Build the final LLM prompt using the context returned by the Oracle.
    Mode controls the answer style hint.
    """
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
    """
    For debugging — show which chunks were selected.
    """
    if not results:
        print("\n🧩 Top Matches: (none)")
        return

    print("\n🧩 Top Matches:")
    for r in results:
        print(
            f"- [{r['score']:.4f}] {r['file_path']} (chunk {r['chunk_index']})"
        )


# ---------- LLM BACKENDS ----------

def ask_ollama(prompt: str) -> str:
    """
    Send the prompt to a local Ollama chat model.
    Env vars:
      OLLAMA_BASE_URL   (default: http://localhost:11434)
      OLLAMA_CHAT_MODEL (default: llama3.1:8b)
    """
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = base + "/api/chat"
    model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
    }

    print(f"\n🧠 Sending prompt to Ollama model: {model} @ {url}")
    resp = requests.post(url, json=payload, timeout=240)
    resp.raise_for_status()
    data = resp.json()
    # Non-streaming /api/chat response: { "message": { "role": "assistant", "content": "..." }, ... }
    message = data.get("message", {})
    return message.get("content", "").strip()


def ask_openai(prompt: str) -> str:
    """
    Send the prompt to OpenAI (requires OPENAI_API_KEY and openai package).

    Env vars:
      OPENAI_API_KEY      (required)
      OPENAI_MODEL        (default: gpt-4.1-mini)
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ openai package not installed. Run: pip install openai")
        raise

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set in environment.")
        raise RuntimeError("OPENAI_API_KEY missing")

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

    print(f"\n🧠 Sending prompt to OpenAI model: {model}")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are the OpenAdonAI Oracle, grounded strictly in the provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()


# ---------- MODE / FALLBACK HELPERS ----------

def downgrade_mode(mode: str) -> str | None:
    """
    Given a mode, return the next lower mode, or None if already at lowest.
    scholar -> deep -> short -> None
    """
    # MODE_ORDER is ascending; we want to step "down" to a lower index
    order = MODE_ORDER
    if mode not in order:
        return None
    idx = order.index(mode)
    if idx == 0:
        return None  # already at lowest (short)
    return order[idx - 1]


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask the OpenAdonAI Archetype Oracle (RAG over Obsidian)."
    )
    parser.add_argument("query", nargs="+", help="Your question to the Oracle.")
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve from Oracle. "
             "If omitted, chosen based on --mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["short", "deep", "scholar"],
        default="deep",
        help="Answer mode: affects context size and style. (default: deep)",
    )
    parser.add_argument(
        "--backend",
        choices=["none", "ollama", "openai"],
        default="none",
        help="LLM backend to answer the prompt. "
             "'none' = just print prompt (default).",
    )
    parser.add_argument(
        "--no-chunks",
        action="store_true",
        help="Do not print the list of top matched chunks.",
    )
    return parser.parse_args()


def run_oracle_round(query: str, mode: str, backend: str, top_k_arg: int | None, print_chunks: bool) -> bool:
    """
    Run one round of:
      - fetch context with top_k (from mode/top_k_arg)
      - build prompt
      - optionally send to backend

    Returns True if successful, False if an error occurred that might be fixed by downgrading mode.
    """
    # Determine top_k based on mode, unless user explicitly passed one
    if top_k_arg is not None:
        top_k = top_k_arg
    else:
        top_k = MODE_CONFIG.get(mode, MODE_CONFIG["deep"])["top_k"]

    print(f"\n⚙️ Mode: {mode} | top_k: {top_k} | backend: {backend}")

    try:
        # 1) Fetch context from Oracle API
        results, context = fetch_context(query, top_k=top_k)
    except Exception as e:
        print(f"❌ Error during context retrieval in mode '{mode}': {e}")
        # Downgrading the mode won't fix a dead server, so we treat this as non-recoverable.
        return False

    if not context:
        print("⚠️ No context returned from Oracle.")
        return False

    # 2) Show which chunks were used
    if print_chunks:
        pretty_print_results(results)

    # 3) Build final Oracle prompt
    prompt = build_prompt(query, context, mode)

    print("\n\n================= ORACLE PROMPT =================\n")
    print(prompt)
    print("\n=================================================\n")

    # 4) Optionally send to an LLM
    if backend == "none":
        print("📝 Backend: none → not sending to any LLM (prompt only).")
        return True

    try:
        if backend == "ollama":
            answer = ask_ollama(prompt)
        elif backend == "openai":
            answer = ask_openai(prompt)
        else:
            print(f"❌ Unknown backend: {backend}")
            return False
    except Exception as e:
        print(f"❌ Error during LLM call in mode '{mode}': {e}")
        # This *might* be a context-length or payload-size issue → allow fallback to lower mode.
        return False

    print("\n================= ORACLE ANSWER =================\n")
    print(answer)
    print("\n=================================================\n")

    return True


def main():
    args = parse_args()
    query = " ".join(args.query)

    current_mode = args.mode
    backend = args.backend
    top_k_arg = args.top_k  # may be None
    print_chunks = not args.no_chunks

    print(f"🔮 Querying Archetype Oracle...\n→ {query}\n")

    tried_modes = set()

    while True:
        if current_mode in tried_modes:
            # prevent accidental loops
            print(f"❌ Mode '{current_mode}' already tried and failed. Stopping.")
            break

        tried_modes.add(current_mode)

        ok = run_oracle_round(
            query=query,
            mode=current_mode,
            backend=backend,
            top_k_arg=top_k_arg,
            print_chunks=print_chunks,
        )

        if ok:
            # success, we're done
            break

        # If failure, try downgrading mode (if possible)
        next_mode = downgrade_mode(current_mode)
        if not next_mode:
            print(f"⚠️ No lower mode available to downgrade from '{current_mode}'. Giving up.")
            break

        print(f"\n🔁 Encountered an error in mode '{current_mode}'. "
              f"Falling back to lower mode: '{next_mode}'.")
        current_mode = next_mode


if __name__ == "__main__":
    main()
