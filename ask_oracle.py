# ask_oracle.py

import os
import sys
import argparse
import requests

ORACLE_URL = os.getenv("ORACLE_URL", "http://localhost:9000/search")


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
        sys.exit(1)

    data = resp.json()
    results = data.get("results", [])
    if not results:
        print("⚠️ Oracle returned no results.")
        sys.exit(0)

    chunks = [r["text"] for r in results]
    context = "\n\n---\n\n".join(chunks)
    return results, context


def build_prompt(query: str, context: str) -> str:
    """
    Build the final LLM prompt using the context returned by the Oracle.
    """
    return f"""
You are the OpenAdonAI Oracle.

Use ONLY the context below from my Archetype vault to answer the question.
If the context does not contain the answer, say so. Do not invent unrelated information.

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
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set in environment.")
        sys.exit(1)

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

    print(f"\n🧠 Sending prompt to OpenAI model: {model}")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are the OpenAdonAI Oracle, grounded strictly in the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask the OpenAdonAI Archetype Oracle (RAG over Obsidian)."
    )
    parser.add_argument("query", nargs="+", help="Your question to the Oracle.")
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve from Oracle (default: 5).",
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


def main():
    args = parse_args()
    query = " ".join(args.query)

    print(f"🔮 Querying Archetype Oracle...\n→ {query}\n")

    # 1) Fetch top chunks from Oracle API
    results, context = fetch_context(query, top_k=args.top_k)

    # 2) Show which chunks were used
    if not args.no_chunks:
        pretty_print_results(results)

    # 3) Build final Oracle prompt
    prompt = build_prompt(query, context)

    print("\n\n================= ORACLE PROMPT =================\n")
    print(prompt)
    print("\n=================================================\n")

    # 4) Optionally send to an LLM
    if args.backend == "none":
        print("📝 Backend: none → not sending to any LLM (prompt only).")
        return

    if args.backend == "ollama":
        answer = ask_ollama(prompt)
    elif args.backend == "openai":
        answer = ask_openai(prompt)
    else:
        print(f"❌ Unknown backend: {args.backend}")
        return

    print("\n================= ORACLE ANSWER =================\n")
    print(answer)
    print("\n=================================================\n")


if __name__ == "__main__":
    main()