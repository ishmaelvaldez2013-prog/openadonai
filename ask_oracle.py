import os
import sys
import argparse
import requests
import json

from dotenv import load_dotenv
load_dotenv()

# Optional integration with AnythingLLM book workspace
try:
    from anythingllm_client import query_anythingllm_books, is_anythingllm_enabled
except ImportError:
    # Graceful fallback if the client is not present; books will be disabled.
    def is_anythingllm_enabled() -> bool:
        return False

    def query_anythingllm_books(question: str, max_snippets: int = 5):
        return {"text": "", "context": "", "sources": []}

ORACLE_URL = os.getenv("ORACLE_URL", "http://localhost:9000/search")

# Env toggle for including book context by default (used as a global override if desired)
INCLUDE_BOOKS_ENV = os.getenv("OPENADONAI_INCLUDE_BOOKS", "false").strip().lower() == "true"

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

# Persona configuration
PERSONA_CONFIG = {
    "default": (
        "You are the OpenAdonAI Oracle.\n"
        "- Answer clearly, helpfully, and precisely.\n"
        "- Stay strictly grounded in the provided context."
    ),
    "scholar": (
        "You are the OpenAdonAI Oracle in SCHOLAR mode.\n"
        "- Prioritize clarity, structure, and textual precision.\n"
        "- Anchor interpretations in scripture, Hermetic/Kabbalistic pattern, and careful reasoning.\n"
        "- Use headings and layered exposition when appropriate.\n"
        "- Stay strictly grounded in the provided context."
    ),
    "mystic": (
        "You are the OpenAdonAI Oracle in MYSTIC mode.\n"
        "- Emphasize visionary, Melchizedek, and cosmic pattern insight.\n"
        "- Preserve metaphors and symbolic correspondences while remaining coherent and grounded.\n"
        "- Stay strictly grounded in the provided context; do not invent lore beyond it."
    ),
    "engineer": (
        "You are the OpenAdonAI Oracle in ENGINEER mode.\n"
        "- Focus on code, architecture, infrastructure, and practical implementation details.\n"
        "- Be precise, stepwise, and explicit.\n"
        "- Stay strictly grounded in the provided context."
    ),
}

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
    """Call Oracle RAG API (Obsidian index) and fetch chunks."""
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


def build_prompt(
    query: str,
    obsidian_context: str,
    mode: str,
    persona: str = "default",
    book_context: str = "",
) -> str:
    """
    Build the final Oracle prompt with:
      - Persona preamble
      - Obsidian Vault context (if present)
      - Embedded Book Library context (AnythingLLM), if provided
    """
    mode_cfg = MODE_CONFIG.get(mode, MODE_CONFIG["deep"])
    style_hint = mode_cfg["style_hint"]

    obsidian_context = (obsidian_context or "").strip()
    book_context = (book_context or "").strip()

    if obsidian_context and book_context:
        combined_context = f"""### Obsidian Vault Context
{obsidian_context}

### Embedded Book Library Context (AnythingLLM)
{book_context}"""
    elif obsidian_context:
        combined_context = obsidian_context
    elif book_context:
        combined_context = f"""### Embedded Book Library Context (AnythingLLM)
{book_context}"""
    else:
        combined_context = ""

    persona_key = persona if persona in PERSONA_CONFIG else "default"
    persona_text = PERSONA_CONFIG[persona_key]

    return f"""
{persona_text}

You have access to two knowledge sources:
1. Obsidian Vault – Ishmael's scrolls, notes, volt systems, and development logs.
2. Embedded Book Library – PDF/Text books indexed via AnythingLLM.

Use ONLY the provided context below to answer the question.
If the context does not contain the answer, say so. Do not invent unrelated information.

Answer style mode: {mode.upper()}
Persona: {persona_key}
Guidance: {style_hint}

======== BEGIN CONTEXT ========
{combined_context}
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
            "  2) Fetches top-k chunks from your Obsidian index\n"
            "  3) Optionally also fetches RAG context from AnythingLLM embedded book workspace\n"
            "  4) Builds an Oracle-style prompt\n"
            "  5) Optionally sends it to an LLM backend (Ollama or OpenAI)\n\n"
            "Env defaults:\n"
            "  OPENADONAI_DEFAULT_MODE        → default mode (short|deep|scholar)\n"
            "  OPENADONAI_DEFAULT_BACKEND     → default backend (none|ollama|openai)\n"
            "  OPENADONAI_DEFAULT_TOP_K       → default top_k (int > 0)\n"
            "  OPENADONAI_INCLUDE_BOOKS       → 'true' to globally include book context by default\n"
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
            "\nDefaults:\n"
            "  • OPENADONAI_DEFAULT_MODE sets global default.\n"
            "  • scholar mode will include book context by default (if available),\n"
            "    while short/deep use Obsidian-only by default."
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
            "Number of chunks to retrieve from Oracle (Obsidian).\n"
            "If omitted, defaults are based on --mode (short=3, deep=7, scholar=12).\n"
            "You can also set a global default via OPENADONAI_DEFAULT_TOP_K.\n"
            "CLI -k always overrides env / mode defaults."
        ),
    )

    parser.add_argument(
        "--source-mode",
        choices=["blend", "obsidian", "books"],
        default="blend",
        help=(
            "Where to pull context from:\n"
            "  blend    → Obsidian + Books (default)\n"
            "  obsidian → Obsidian vault only (ignore AnythingLLM)\n"
            "  books    → Books workspace only (ignore Obsidian)\n"
        ),
    )

    parser.add_argument(
        "--persona",
        choices=["default", "scholar", "mystic", "engineer"],
        default="default",
        help=(
            "Persona flavor for the Oracle's voice:\n"
            "  default  → neutral, clear, balanced.\n"
            "  scholar  → structured, academic, scriptural/Hermetic mapping.\n"
            "  mystic   → visionary, Melchizedek / cosmic pattern tone.\n"
            "  engineer → technical, code/infra/practical implementation focus.\n"
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
            "JSON includes: success, mode_used, persona, top_k, prompt, answer, results,\n"
            "               error, book_context_used, obsidian_chunk_count,\n"
            "               book_snippet_count, book_sources, source_mode."
        ),
    )

    parser.add_argument(
        "--include-books",
        action="store_true",
        help="Force-enable fetching book context from AnythingLLM for this query.",
    )

    parser.add_argument(
        "--no-books",
        action="store_true",
        help="Force-disable book context for this query, even if scholar or env says true.",
    )

    parser.add_argument(
        "--sources-only",
        action="store_true",
        help=(
            "Show which sources (Obsidian chunks + book snippets) would be used,\n"
            "without calling the LLM backend. Useful with 'oracle sources'."
        ),
    )

    return parser.parse_args()


def run_oracle_round(
    query,
    mode,
    backend,
    top_k_arg,
    print_chunks,
    include_books,
    source_mode,
    persona,
):
    """
    Returns dict:
      {
        success, mode_used, persona, top_k, prompt, answer,
        results, error, book_context_used,
        obsidian_chunk_count, book_snippet_count,
        book_sources, source_mode
      }
    """
    output = {
        "success": False,
        "mode_used": mode,
        "persona": persona,
        "top_k": None,
        "prompt": "",
        "answer": "",
        "results": [],
        "error": "",
        "book_context_used": False,
        "obsidian_chunk_count": 0,
        "book_snippet_count": 0,
        "book_sources": [],
        "source_mode": source_mode,
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

    obsidian_context = ""
    results = []

    # 1) Obsidian context via Oracle RAG (only if source_mode requires it)
    if source_mode in ("blend", "obsidian"):
        try:
            results, obsidian_context = fetch_context(query, top_k)
            output["results"] = results
            output["obsidian_chunk_count"] = len(results)
        except Exception as e:
            # For obsidian-only, this is fatal; for blend, we can still continue with books.
            if source_mode == "obsidian":
                output["error"] = f"Error fetching Obsidian context: {e}"
                return output
            else:
                sys.stderr.write(f"⚠️  Error fetching Obsidian context (blend mode): {e}\n")
                sys.stderr.flush()

    if print_chunks and results:
        pretty_print_results(results)

    # 2) Embedded book context via AnythingLLM (optional)
    book_context = ""
    if include_books and source_mode in ("blend", "books") and is_anythingllm_enabled():
        try:
            book_result = query_anythingllm_books(query, max_snippets=5)
            book_context = (book_result.get("context") or "").strip()
            book_sources = book_result.get("sources") or []
            output["book_sources"] = book_sources

            if book_context:
                output["book_context_used"] = True
                # Rough heuristic: each snippet starts with "[" and is separated by blank lines
                parts = [p for p in book_context.split("\n\n") if p.strip()]
                snippet_count = sum(1 for p in parts if p.lstrip().startswith("["))
                output["book_snippet_count"] = snippet_count if snippet_count > 0 else len(book_sources) or 1
        except Exception as e:
            # For books-only mode, this is fatal; for blend, just warn.
            if source_mode == "books":
                output["error"] = f"Error querying AnythingLLM books: {e}"
                return output
            else:
                sys.stderr.write(f"⚠️  Error querying AnythingLLM books (blend mode): {e}\n")
                sys.stderr.flush()

    # 3) Build prompt
    prompt = build_prompt(
        query=query,
        obsidian_context=obsidian_context,
        mode=mode,
        persona=persona,
        book_context=book_context,
    )
    output["prompt"] = prompt

    if backend == "none":
        # When backend is none, we still mark success if we got *some* context.
        if obsidian_context or book_context:
            output["success"] = True
        else:
            output["error"] = "No context available from selected sources."
        return output

    # 4) Call LLM backend
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
    source_mode = args.source_mode
    persona = args.persona

    tried = set()

    while True:
        # Resolve whether to include book context for *this* mode:
        # Priority: source-mode > CLI flags > scholar default > env > false

        # Start from CLI flags
        if args.no_books:
            include_books = False
        elif args.include_books:
            include_books = True
        else:
            # No explicit CLI override
            if current_mode == "scholar":
                # Scholar mode always includes books by default if available
                include_books = True
            else:
                # short/deep stick to Obsidian-only by default,
                # unless env explicitly set OPENADONAI_INCLUDE_BOOKS=true
                include_books = INCLUDE_BOOKS_ENV

        # Override based on source_mode
        if source_mode == "obsidian":
            include_books = False
        elif source_mode == "books":
            include_books = True

        if current_mode in tried:
            if args.json:
                print(json.dumps({
                    "success": False,
                    "error": f"Mode '{current_mode}' failed multiple times.",
                    "source_mode": source_mode,
                    "persona": persona,
                }, indent=2))
            else:
                print(f"❌ Mode '{current_mode}' failed multiple times.")
            return

        tried.add(current_mode)

        result = run_oracle_round(
            query=query,
            mode=current_mode,
            backend=backend,
            top_k_arg=top_k_arg,
            print_chunks=print_chunks,
            include_books=include_books,
            source_mode=source_mode,
            persona=persona,
        )

        if args.json:
            # augment with mode_used & persona
            result["mode_used"] = current_mode
            result["persona"] = persona
            print(json.dumps(result, indent=2))
            return

        # If success → done
        if result["success"]:
            # Sources-only path: do NOT call LLM or print answer beyond preview
            if args.sources_only:
                print("\n🔎 Oracle Sources Preview")
                print("-------------------------\n")

                obs_results = result.get("results", []) or []
                print(f"Obsidian chunks: {len(obs_results)}")
                if obs_results:
                    for r in obs_results:
                        fp = r.get("file_path", "?")
                        idx = r.get("chunk_index", 0)
                        score = r.get("score", 0.0)
                        print(f"  - {fp} (chunk {idx}, score {score:.4f})")
                else:
                    print("  (none)")

                book_used = result.get("book_context_used", False)
                book_sources = result.get("book_sources", []) or []

                if book_used and book_sources:
                    print(f"\nBooks snippets: {len(book_sources)}")
                    for s in book_sources:
                        title = s.get("title") or s.get("filename") or "Unknown"
                        page = s.get("page") or s.get("pageNumber")
                        source_id = s.get("source") or ""
                        line = f"  - {title}"
                        if page:
                            line += f", p.{page}"
                        if source_id:
                            line += f" [{source_id}]"
                        print(line)
                else:
                    print("\nBooks snippets: (none)")

                print("")
                return

            # Normal answer path
            if result["answer"]:
                print("\n================= ORACLE ANSWER =================\n")

                obs_count = result.get("obsidian_chunk_count", 0)
                book_used = result.get("book_context_used", False)
                book_count = result.get("book_snippet_count", 0) if book_used else 0

                # Build source report line
                if obs_count == 0 and not book_used:
                    source_line = (
                        f"Sources used ({result.get('source_mode','blend')}, persona={result.get('persona','default')}): "
                        "(none – context appears empty)"
                    )
                elif obs_count > 0 and not book_used:
                    source_line = (
                        f"Sources used ({result.get('source_mode','blend')}, persona={result.get('persona','default')}): "
                        f"Obsidian({obs_count} chunks)"
                    )
                elif obs_count == 0 and book_used:
                    source_line = (
                        f"Sources used ({result.get('source_mode','blend')}, persona={result.get('persona','default')}): "
                        f"Books({book_count} snippets)"
                    )
                else:
                    source_line = (
                        f"Sources used ({result.get('source_mode','blend')}, persona={result.get('persona','default')}): "
                        f"Obsidian({obs_count} chunks), Books({book_count} snippets)"
                    )

                print(source_line + "\n")
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
