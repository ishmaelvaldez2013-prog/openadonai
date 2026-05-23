"""Atlas Bridge Streamlit page with standalone and importable modes.

Standalone:
- streamlit run apps/streamlit/atlas_bridge.py

Imported:
- import apps.streamlit.atlas_bridge as atlas_bridge
- atlas_bridge.render()

In a main app/router sidebar, add label 'Atlas Bridge' and call
atlas_bridge.render().
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import streamlit as st


DEFAULT_CONTEXT_FILES = [
    "apps/streamlit/i_ching_app.py",
    "apps/streamlit/i_ching_dx.py",
    "signal_ingest_rules.yaml",
    "obsidian_daily_ingest.py",
    "README.md",
]

DEFAULT_IGNORE_GLOBS = [
    "**/.env",
    "**/*.key",
    "**/secrets/**",
]

DEFAULT_ACCEPTANCE_ITEMS = [
    'README_ROUTER_HOOK_CODEBLOCK_ONLY:README.md|["import apps.streamlit.atlas_bridge as atlas_bridge","atlas_bridge.render()"]',
]


def generate_run_id() -> str:
    """Required UTC run_id format for bridge runs."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def parse_multiline_list(raw_text: str) -> list[str]:
    """Return non-empty trimmed lines from a textarea input."""
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def yaml_scalar(value: str) -> str:
    """Emit YAML-safe scalar using JSON quoting (valid YAML 1.2)."""
    return json.dumps(value, ensure_ascii=False)


def yaml_list_lines(items: list[str], indent: int) -> list[str]:
    prefix = " " * indent
    if not items:
        return [f"{prefix}[]"]
    return [f"{prefix}- {yaml_scalar(item)}" for item in items]


def build_run_yaml(
    *,
    run_id: str,
    project: str,
    repo_root: str,
    goal: str,
    acceptance_items: list[str],
    context_include: list[str],
    ignore_globs: list[str],
) -> str:
    lines: list[str] = []
    lines.append(f"run_id: {yaml_scalar(run_id)}")
    lines.append(f"project: {yaml_scalar(project)}")
    lines.append(f"repo_root: {yaml_scalar(repo_root)}")
    lines.append('mode: "plan_execute_verify_report"')
    lines.append("")
    lines.append("planner:")
    lines.append(f"  goal: {yaml_scalar(goal)}")
    lines.append("  constraints:")
    lines.append('    - "idempotent"')
    lines.append('    - "no network calls"')
    lines.append("  acceptance:")
    lines.extend(yaml_list_lines(acceptance_items, indent=4))
    lines.append("")
    lines.append("context:")
    lines.append("  include:")
    lines.extend(yaml_list_lines(context_include, indent=4))
    lines.append("  ignore_globs:")
    lines.extend(yaml_list_lines(ignore_globs, indent=4))
    lines.append("  max_bytes_per_file: 200000")
    lines.append("")
    lines.append("executor:")
    lines.append("  codex:")
    lines.append("    cli_cmd:")
    lines.append('      - "codex"')
    lines.append("    extra_args: []")
    lines.append("  tests:")
    lines.append("    enabled: true")
    lines.append("    cmd:")
    lines.append('      - "bash"')
    lines.append('      - "ops/run_tests.sh"')
    lines.append("")
    lines.append("reporter:")
    lines.append("  write_files:")
    lines.append('    - "README.md"')
    lines.append('    - "CHANGELOG.md"')
    lines.append("")
    return "\n".join(lines)


def discover_repo_roots() -> list[str]:
    """Collect likely repo roots for easy selection."""
    candidates: list[Path] = []
    seen: set[Path] = set()

    for seed in [Path.cwd(), Path(__file__).resolve().parent]:
        for candidate in [seed, *seed.parents]:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if (resolved / ".git").exists():
                candidates.append(resolved)

    if not candidates:
        candidates.append(Path.cwd().resolve())

    return [str(path) for path in candidates]


def render() -> None:
    st.set_page_config(page_title="Atlas Bridge", layout="wide")
    st.title("Atlas Bridge Run Builder")
    st.caption("Create `.agent_runs/<run_id>/run.yaml` and `prompt.txt` without executing the run.")

    repo_options = discover_repo_roots()
    selected_repo_root = st.selectbox("Detected repo roots", options=repo_options, index=0)
    repo_root_input = st.text_input("Repo root (editable)", value=selected_repo_root)

    if "atlas_bridge_run_id" not in st.session_state:
        st.session_state["atlas_bridge_run_id"] = generate_run_id()

    run_cols = st.columns([4, 1])
    with run_cols[0]:
        run_id = st.text_input("run_id (UTC)", value=st.session_state["atlas_bridge_run_id"])
    with run_cols[1]:
        if st.button("Generate run_id"):
            st.session_state["atlas_bridge_run_id"] = generate_run_id()
            st.rerun()

    st.session_state["atlas_bridge_run_id"] = run_id

    default_project = Path(repo_root_input).name or "project"
    project = st.text_input("Project", value=default_project)

    goal = st.text_input("Planner goal", value="Atlas planner request")

    prompt_text = st.text_area("Planner prompt (prompt.txt)", height=220, value="")

    context_text = st.text_area(
        "Context files (one per line)",
        height=140,
        value="\n".join(DEFAULT_CONTEXT_FILES),
    )
    acceptance_text = st.text_area(
        "Planner acceptance items (one per line)",
        height=120,
        value="\n".join(DEFAULT_ACCEPTANCE_ITEMS),
    )
    ignore_globs_text = st.text_area(
        "Context ignore globs (one per line)",
        height=110,
        value="\n".join(DEFAULT_IGNORE_GLOBS),
    )

    acceptance_items = parse_multiline_list(acceptance_text)
    context_include = parse_multiline_list(context_text)
    ignore_globs = parse_multiline_list(ignore_globs_text)

    run_yaml_text = build_run_yaml(
        run_id=run_id,
        project=project,
        repo_root=str(Path(repo_root_input).expanduser()),
        goal=goal,
        acceptance_items=acceptance_items,
        context_include=context_include,
        ignore_globs=ignore_globs,
    )

    run_yaml_relative = f".agent_runs/{run_id}/run.yaml"
    orchestrator_cmd = f"python -m bridge.orchestrator {run_yaml_relative}"

    st.subheader("run.yaml preview")
    st.code(run_yaml_text, language="yaml")

    st.subheader("Command to run agent")
    st.code(orchestrator_cmd, language="bash")

    repo_root_path = Path(repo_root_input).expanduser()
    repo_ok = repo_root_path.exists() and repo_root_path.is_dir()
    if not repo_ok:
        st.error(f"Repo root does not exist or is not a directory: {repo_root_path}")

    run_dir = repo_root_path / ".agent_runs" / run_id

    write_clicked = st.button("Write run.yaml + prompt.txt", disabled=not repo_ok)
    if write_clicked and repo_ok:
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "run.yaml").write_text(run_yaml_text, encoding="utf-8")
        prompt_out = prompt_text if prompt_text.endswith("\n") else f"{prompt_text}\n"
        (run_dir / "prompt.txt").write_text(prompt_out, encoding="utf-8")

        st.success(f"Wrote: {run_dir / 'run.yaml'} and {run_dir / 'prompt.txt'}")
        st.code(orchestrator_cmd, language="bash")

    run_now_clicked = st.button("Run orchestrator now (manual)", disabled=not repo_ok)
    if run_now_clicked and repo_ok:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.yaml").write_text(run_yaml_text, encoding="utf-8")
        prompt_out = prompt_text if prompt_text.endswith("\n") else f"{prompt_text}\n"
        (run_dir / "prompt.txt").write_text(prompt_out, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "-m", "bridge.orchestrator", run_yaml_relative],
            cwd=str(repo_root_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            st.success("Orchestrator run completed.")
        else:
            st.error(f"Orchestrator failed with exit code {result.returncode}.")
        st.code(result.stdout or "(no output)", language="text")


if __name__ == "__main__":
    render()
