# OpenAdonAI — CLAUDE.md

## Project Purpose

OpenAdonAI is a **local-first RAG (Retrieval-Augmented Generation) oracle** built around an Obsidian vault. It indexes personal knowledge base notes, embeds them with Ollama, and answers queries through a FastAPI server, a zsh CLI, and a Streamlit UI. AnythingLLM is an optional second source for book/document workspaces.

## Architecture Overview

```
Obsidian Vault (.md files)
    └── index_obsidian.py     ← chunking + Ollama embedding → embeddings.npy / metadata.json

FastAPI Oracle (app.py / main.py)
    └── query_index.py        ← cosine similarity search on embeddings

ask_oracle.py                 ← RAG context + Ollama/OpenAI chat → answer
    └── anythingllm_client.py ← optional AnythingLLM book source

oracle (zsh)                  ← primary CLI: start / stop / ask / continuum / index / health
openadonai_start.py           ← Python startup orchestrator (ensures Ollama + uvicorn running)
openadonai_tray.py            ← macOS menu bar app (rumps)
apps/streamlit/atlas_bridge.py ← Streamlit UI for planning agent runs
```

## Key Commands

```bash
# Start the full stack (Ollama + FastAPI)
./oracle start
# or
python openadonai_start.py start

# Index / re-index the Obsidian vault
./oracle index
# or
python index_obsidian.py

# Ask a one-shot question
./oracle "What is the Melchizedek path?"
# or with mode/persona
python ask_oracle.py --mode scholar --persona mystic "your question"

# Stateful continuum session
./oracle continuum --mode deep --persona scholar "continue..."

# Health check / diagnostics
./oracle health
python oracle_doctor.py

# Run Streamlit app
streamlit run apps/streamlit/atlas_bridge.py
```

## Environment Variables (.env)

```
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=mistral:instruct
ORACLE_HOST=127.0.0.1
ORACLE_PORT=9000
INDEX_DIR=./index_data
OBSIDIAN_ROOT=/path/to/your/vault
PYTHON_BIN=.venv/bin/python

# Optional AnythingLLM
ANYTHINGLLM_BASE_URL=http://localhost:3001
ANYTHINGLLM_API_KEY=...
ANYTHINGLLM_WORKSPACE_SLUG=books

# Query defaults
OPENADONAI_DEFAULT_MODE=deep          # short | deep | scholar
OPENADONAI_DEFAULT_BACKEND=ollama     # ollama | openai
OPENADONAI_DEFAULT_TOP_K=7
OPENADONAI_INCLUDE_BOOKS=false
```

## File Map

| File | Role |
|------|------|
| `app.py` | FastAPI server — `/health` and `/search` endpoints |
| `main.py` | Alternative FastAPI impl (legacy, nearly identical to app.py) |
| `query_index.py` | Vector search: loads embeddings, cosine similarity, returns top-k chunks |
| `index_obsidian.py` | Indexing pipeline: reads vault → chunks → embeds → saves index |
| `ask_oracle.py` | Main query logic: RAG → LLM → answer; supports depth modes + personas |
| `anythingllm_client.py` | AnythingLLM workspace client for book/doc sources |
| `oracle` | Primary zsh CLI wrapper — use this for day-to-day commands |
| `openadonai_start.py` | Python orchestrator: starts Ollama, loads models, starts uvicorn |
| `openadonai_cli.py` | macOS-focused CLI with LaunchAgent integration |
| `openadonai_boot.sh` | Auto-start hook for zsh login shells |
| `openadonai_tray.py` | macOS menu bar tray app (rumps) |
| `oracle_doctor.py` | Full stack diagnostics: Ollama, models, index, FastAPI endpoints |
| `apps/streamlit/atlas_bridge.py` | Streamlit UI for planning and exporting agent runs |
| `apps/streamlit/_router_stub.py` | Sidebar nav template (not yet wired as central router) |

## Runtime Artifacts (not committed)

- `index_data/embeddings.npy` — numpy array of all chunk embeddings
- `index_data/metadata.json` — chunk metadata (file, heading, text snippet)
- `index_data/.cache/` — per-file embedding cache keyed by mtime/size
- `logs/oracle.log` — continuum session history

## Query Modes & Personas

**Depth modes** (controls top-k chunks pulled from RAG):
- `short` → ~3 chunks, brief answer
- `deep` → ~7 chunks, detailed answer
- `scholar` → ~12 chunks, exhaustive with sources

**Personas** (controls LLM system prompt preamble):
- `scholar` — textual precision, scriptural grounding
- `mystic` — visionary, cosmic pattern recognition
- `engineer` — technical, code-aware

## Dependencies

```
fastapi, uvicorn      # API server
numpy                 # embedding storage + similarity
requests              # HTTP client (Ollama, AnythingLLM)
pydantic              # request/response schemas
streamlit             # web UI
python-dotenv         # .env loading
rumps                 # macOS tray (openadonai_tray.py only)
```

Ollama must be installed and running separately — it is not a Python package.

## Conventions

- The `oracle` zsh script is the canonical day-to-day interface; prefer it over calling Python files directly when demonstrating usage.
- `ask_oracle.py` is the query brain — all RAG + LLM logic lives there.
- `index_obsidian.py` uses per-file caching; only changed files are re-embedded on incremental runs.
- `app.py` and `main.py` overlap significantly — `app.py` is the active entry point; `main.py` is legacy.
- The `apps/streamlit/` folder is the home for all Streamlit UIs. New apps go here.
- Atlas Bridge writes agent run plans to `.agent_runs/{run_id}/run.yaml` + `prompt.txt`.
- No test suite currently exists. Use `oracle_doctor.py` for stack validation.
