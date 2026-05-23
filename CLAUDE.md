# CLAUDE.md — OpenAdonAI Oracle

Codebase guide for AI assistants working in this repository.

---

## What This Project Is

**OpenAdonAI Oracle** is a local RAG (Retrieval-Augmented Generation) system built around an Obsidian markdown vault. It indexes personal knowledge, performs semantic search, and routes questions to a local LLM (Ollama) or OpenAI. The system exposes a FastAPI REST endpoint, a Python CLI, a zsh shell wrapper, and a macOS system tray app.

Secondary integration: **AnythingLLM** serves as a book workspace for additional context blending.

---

## Repository Layout

```
/                         # Root — all core modules live here
├── app.py                # FastAPI application (port 9000); primary API entry point
├── ask_oracle.py         # Core Q&A engine; backends, modes, personas, source blending
├── query_index.py        # Embedding load + cosine similarity search
├── index_obsidian.py     # Obsidian vault → chunked embeddings pipeline
├── openadonai_cli.py     # Python CLI (start/index/doctor/ask/logs/stop/restart)
├── openadonai_start.py   # Startup orchestrator (Ollama + models + API launch)
├── openadonai_boot.sh    # macOS LaunchAgent boot script
├── openadonai_tray.py    # macOS system tray (rumps)
├── oracle                # Zsh shell wrapper — primary end-user CLI
├── oracle_doctor.py      # Health-check helpers (Ollama, models, index, Oracle API)
├── anythingllm_client.py # AnythingLLM HTTP client for book workspace
├── main.py               # Minimal alternate FastAPI entry (older, not primary)
├── foundations.html      # Static HTML documentation
├── pillars.{css,js}      # Static web assets
├── apps/
│   └── streamlit/
│       ├── atlas_bridge.py    # Streamlit UI — agent run-plan builder (YAML output)
│       └── _router_stub.py    # Template for Streamlit multi-page router
└── index_data/           # Runtime-generated; not committed
    ├── embeddings.npy    # NumPy embedding matrix
    ├── metadata.json     # Chunk metadata (source, text, file)
    ├── file_state.json   # Modification timestamps for incremental indexing
    └── file_cache/       # Per-file chunk cache
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x (uses `dict \| None` union syntax — requires 3.10+) |
| API framework | FastAPI + Uvicorn |
| Embeddings store | NumPy `.npy` files (cosine similarity, no external vector DB) |
| Local LLM | Ollama (chat + embeddings) |
| Remote LLM | OpenAI (optional, via `--backend openai`) |
| Book context | AnythingLLM workspace |
| Web UI | Streamlit |
| macOS integration | rumps (tray), osascript (notifications), LaunchAgent (background service) |
| Config | `.env` file loaded via `python-dotenv` |

---

## Environment Configuration

All runtime behaviour is controlled through environment variables (`.env` file at repo root, never committed).

### Required

| Variable | Default | Purpose |
|---|---|---|
| `OBSIDIAN_ROOT` | — | Absolute path to Obsidian vault directory |
| `OLLAMA_CHAT_MODEL` | — | Chat model name, e.g. `mistral:instruct`, `llama3.1:8b` |

### Common Optional

| Variable | Default | Purpose |
|---|---|---|
| `INDEX_DIR` | `./index_data` | Where embeddings and metadata are stored |
| `OLLAMA_BASE_URL` / `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` / `EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `OLLAMA_EXTRA_MODELS` | — | Comma-separated extra models to health-check |
| `ORACLE_HOST` | `127.0.0.1` | FastAPI bind host |
| `ORACLE_PORT` | `9000` | FastAPI bind port |
| `ORACLE_URL` | `http://localhost:9000/search` | URL used by `ask_oracle.py` to call the API |
| `ORACLE_RELOAD` | `true` | Uvicorn auto-reload on file changes |
| `OPENADONAI_DEFAULT_MODE` | `deep` | Default answer depth (`short\|deep\|scholar`) |
| `OPENADONAI_DEFAULT_BACKEND` | `none` | Default LLM backend (`none\|ollama\|openai`) |
| `OPENADONAI_DEFAULT_TOP_K` | — | Number of context chunks to retrieve |
| `OPENADONAI_INCLUDE_BOOKS` | `false` | Include AnythingLLM book context by default |
| `ANYTHINGLLM_BASE_URL` | `http://localhost:3001` | AnythingLLM server URL |
| `ANYTHINGLLM_API_KEY` | — | AnythingLLM API key |
| `ANYTHINGLLM_WORKSPACE_SLUG` | — | Workspace slug for book queries |
| `PYTHON_BIN` | `.venv/bin/python` | Python interpreter path |
| `OPENAI_API_KEY` | — | OpenAI key (only needed with `--backend openai`) |
| `DRY_RUN_EMBED` | `false` | Skip embedding generation (for local testing) |

---

## How to Run

### Prerequisites

1. Ollama running locally with the desired chat and embedding models pulled.
2. Python virtual environment at `.venv/` with dependencies installed.
3. `.env` file configured (see above).

### Start the full stack

```bash
openadonai start        # Launches Ollama + pulls models + starts Oracle API
# or
python openadonai_start.py
```

### Build / rebuild the index

```bash
openadonai index
# or
python index_obsidian.py
```

Indexing is incremental — only changed files are re-embedded (tracked via `index_data/file_state.json`).

### Ask a question

```bash
# Shell wrapper (preferred for interactive use)
oracle "What is the nature of consciousness?"
oracle --mode scholar --backend ollama "Explain emergence"

# Python CLI
openadonai ask "your question"

# Direct Python
python ask_oracle.py --backend ollama --mode deep "your question"
```

### Start the Streamlit UI

```bash
streamlit run apps/streamlit/atlas_bridge.py
```

### Health check

```bash
openadonai doctor
# or
oracle health
# or
python oracle_doctor.py
```

### Stop the service

```bash
openadonai stop
```

---

## CLI Reference

### `oracle` (shell script — primary interface)

```
oracle "question"                          # Ask with defaults
oracle --mode short|deep|scholar "..."    # Set answer depth
oracle --backend none|ollama|openai "..."  # Set LLM backend
oracle --persona default|scholar|mystic|engineer "..."
oracle --source obsidian|books|blend "..."  # Source routing
oracle continuum "..."                     # Conversation with continuity
oracle health                              # Health check
oracle start / stop / log / test
oracle modes                               # List available modes
oracle sources                             # List available sources
oracle index                               # Rebuild index
```

### `openadonai` (Python CLI)

```
openadonai start      # Full startup pipeline
openadonai index      # Rebuild RAG index
openadonai doctor     # Health check
openadonai ask "..."  # Ask a question
openadonai logs       # Tail Oracle logs
openadonai stop       # Stop Oracle service
openadonai restart    # Stop + start
```

---

## Answer Modes

| Mode | Chunks | Use case |
|---|---|---|
| `short` | 3 | Quick answers |
| `deep` | 7 | Default balanced depth |
| `scholar` | 12 | Comprehensive research |

## Personas

| Persona | Style |
|---|---|
| `default` | Neutral assistant |
| `scholar` | Academic, sourced |
| `mystic` | Philosophical, reflective |
| `engineer` | Precise, technical |

## Source Modes

| Source | Behaviour |
|---|---|
| `obsidian` | Obsidian vault only (RAG) |
| `books` | AnythingLLM book workspace only |
| `blend` | Both merged |

---

## Core Module Details

### `app.py` — FastAPI server

- `GET /health` — Returns `{"status": "ok"}`
- `POST /search` — Body: `{"query": str, "top_k": int}` → Returns ranked chunks with scores

### `query_index.py` — Search engine

- Loads `embeddings.npy` and `metadata.json` from `INDEX_DIR` at startup
- Embeds the query via Ollama, computes cosine similarity, returns top-k results
- Key function: `search_index(query, top_k) -> list[dict]`

### `index_obsidian.py` — Indexing pipeline

- Recursively scans `OBSIDIAN_ROOT` for `.md` files
- Chunks with configurable size (default 800 chars) and overlap (default 200 chars)
- Calls Ollama embed endpoint per chunk
- Caches per-file chunks in `index_data/file_cache/`
- Writes final `embeddings.npy` + `metadata.json`

### `ask_oracle.py` — Q&A engine

- Selects chunks via `query_index.py`
- Optionally fetches book context from `anythingllm_client.py`
- Builds a prompt with persona system message + context + question
- Streams or returns response from Ollama or OpenAI
- Supports `--continuum` flag for conversation memory

### `oracle_doctor.py` — Health checks

- Checks: Ollama reachable, required models loaded, index files present, Oracle API responding
- Used by `openadonai doctor` and `oracle health`

---

## Development Conventions

### No formal test suite

There is no pytest or unittest setup. Validation is done via:
- `openadonai doctor` — integration health check
- `oracle test` — end-to-end query self-test
- Manual `oracle "question"` invocations

When making changes, always run `openadonai doctor` before committing.

### No requirements.txt / pyproject.toml

Dependencies are managed manually in a `.venv`. When adding a new package:
1. Install it: `pip install <package>`
2. Document the addition in your PR description
3. Note the import at the top of the relevant file

### Code style

- Type hints are used extensively — preserve them
- No linter config present; follow PEP 8 conventions
- Module-level docstrings explain purpose; inline comments explain non-obvious logic
- Don't add comments that just restate what the code does

### Environment loading

Every module that needs config calls `load_dotenv()` near the top. Always check for an env var with `os.getenv("VAR", "default")` — never hardcode paths or URLs.

### Index data is runtime state

Never commit anything under `index_data/`. It is generated at runtime and should remain in `.gitignore`.

### macOS-specific files

`openadonai_tray.py` and `openadonai_boot.sh` target macOS only. Don't add cross-platform guards to existing code unless asked — the project is intentionally macOS-first.

---

## Git Workflow

- `main` — stable branch; merge feature work here
- Feature branches are prefixed (e.g. `dev-tools`, `claude/...`)
- Commits should be descriptive; no squash-merge policy currently enforced
- Push to the branch designated at session start; do not push to `main` directly without confirmation

---

## Common Pitfalls

1. **Index not built** — `oracle` queries will return empty results. Run `openadonai index` first.
2. **Ollama not running** — All embedding and chat calls will fail. Start Ollama before the Oracle API.
3. **Model not pulled** — `openadonai doctor` will report the missing model. Run `ollama pull <model>`.
4. **`.env` missing** — The app will silently use defaults; `OBSIDIAN_ROOT` is unset and indexing will fail. Always ensure `.env` exists.
5. **Port conflict on 9000** — Set `ORACLE_PORT` to a free port and update `ORACLE_URL` to match.
6. **`PYTHON_BIN` wrong path** — If `.venv` is not at repo root, set `PYTHON_BIN` to the correct interpreter.
