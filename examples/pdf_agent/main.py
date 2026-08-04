"""Skill with its own tools: the ``pdf`` skill bundles scripts the agent runs.

Unlike the harness_agent example (custom ``Tool`` classes) this one mounts a
skill that carries its own tooling: the Anthropic ``pdf`` skill ships Python
scripts in ``skills/pdf/scripts/`` (``check_fillable_fields.py``,
``extract_form_field_info.py``, ``fill_fillable_fields.py``, …).  The agent
drives them through the builtin ``shell`` tool — no custom tools or MCP.

The same skill mounts on *any* LLM call, so the example shows both paths:

- a harness/ReAct loop — the agent runs the skill's scripts via the shell
  tool to inspect ``form.pdf`` (check fillable fields, extract field info);
- a plain ``LLM`` node — no tool loop, the model answers from the skill's
  instructions alone.

The example also runs the skill's fill pipeline deterministically
(``fill_fillable_fields.py``) to produce a filled ``filled.pdf``, so the full
toolchain is exercised regardless of how reliably the local model follows a
multi-step recipe.

Layout::

    examples/pdf_agent/
    ├── skills/
    │   └── pdf/                  # vendored Agent Skills pdf skill
    │       ├── SKILL.md
    │       ├── FORMS.md
    │       ├── REFERENCE.md
    │       └── scripts/          # the skill's own tools
    ├── make_form.py              # creates the sample fillable form.pdf
    └── main.py

Requires Ollama with a tool-calling model (qwen2.5:7b works well) and the PDF
deps installed into the venv used to run this example:

    uv pip install --python .venv/bin/python pypdf pdfplumber reportlab
    ollama pull qwen2.5:7b
    python examples/pdf_agent/main.py
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from draf.flow import Flow
from draf.graph import Graph
from draf.node import LLM
from draf.provider import ProviderRegistry
from draf.tool.builtin import ReadFileTool, ShellTool

# The shell tool resolves `python` through PATH.  When running inside a venv,
# make sure the subprocesses use the SAME interpreter that has the PDF deps
# installed (pypdf, pdfplumber, reportlab), not a system python.
if sys.prefix != sys.base_prefix:
    os.environ["PATH"] = (
        os.path.join(sys.prefix, "bin") + os.pathsep + os.environ.get("PATH", "")
    )

MODEL = "qwen2.5:7b"
SKILL_DIR = Path(__file__).resolve().parent / "skills"
WORKDIR = Path(__file__).resolve().parent
FORM = WORKDIR / "form.pdf"

HARNESS_SYSTEM = (
    "You analyze PDF forms using ONLY the pdf skill's bundled scripts. "
    "Never write your own Python code and never answer without running a "
    "command first. The form is `form.pdf` in the working directory.\n"
    "Workflow:\n"
    "1. Run `python skills/pdf/scripts/check_fillable_fields.py form.pdf`\n"
    "2. Run `python skills/pdf/scripts/extract_form_field_info.py "
    "form.pdf field_info.json`\n"
    "3. Read field_info.json with the read_file tool\n"
    "4. Answer listing the field ids and their types."
)


def ensure_sample_form() -> None:
    """Create ``form.pdf`` on first run so the agent has something to work on."""
    if not FORM.exists():
        subprocess.run(
            [sys.executable, str(WORKDIR / "make_form.py")],
            cwd=WORKDIR,
            check=True,
        )


def fill_form_deterministically() -> None:
    """Run the skill's fill pipeline directly (no LLM in the loop).

    Exercises the full toolchain the agent would use: check fillable fields,
    extract field info, write the values payload, fill, then verify.
    """
    scripts = SKILL_DIR / "pdf" / "scripts"
    py = sys.executable

    subprocess.run(
        [py, str(scripts / "check_fillable_fields.py"), str(FORM)],
        cwd=WORKDIR,
        check=True,
    )
    subprocess.run(
        [
            py,
            str(scripts / "extract_form_field_info.py"),
            str(FORM),
            str(WORKDIR / "field_info.json"),
        ],
        cwd=WORKDIR,
        check=True,
    )
    (WORKDIR / "field_values.json").write_text(
        json.dumps(
            [
                {"field_id": "customer_name", "page": 1, "value": "John Smith"},
                {"field_id": "quantity", "page": 1, "value": "3"},
                {"field_id": "priority", "page": 1, "value": "/Yes"},
                {"field_id": "size", "page": 1, "value": "/L"},
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            py,
            str(scripts / "fill_fillable_fields.py"),
            str(FORM),
            str(WORKDIR / "field_values.json"),
            str(WORKDIR / "filled.pdf"),
        ],
        cwd=WORKDIR,
        check=True,
    )


async def run_harness() -> None:
    """Skill mounted on a harness/ReAct loop that executes its scripts."""

    flow = Flow(
        "pdf_agent",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.harness(
        model=MODEL,
        system=HARNESS_SYSTEM,
        input_key="query",
        output_key="answer",
        skills=["pdf"],
        skill_dir=str(SKILL_DIR),
    )
    graph = flow.compile()

    result = await graph.run(
        state={
            "query": (
                "Inspect form.pdf: does it have fillable fields, and what "
                "fields does it contain? Follow the numbered workflow."
            )
        },
        tools=[ShellTool(root_dir=str(WORKDIR)), ReadFileTool()],
        max_iterations=8,
    )

    print("=== Harness / ReAct loop (skill scripts via shell tool) ===")
    print("Query:", result["query"])
    print("Answer:", result["answer"])


async def run_plain_llm() -> None:
    """Same skill, but on a bare ``LLM`` node — no tool loop.

    With ``use_tools=False`` the model never sees the tools; it answers
    straight from the skill instructions merged into its system prompt.
    """
    graph = Graph(
        nodes={
            "answer": LLM(
                {
                    "model": MODEL,
                    "system": "You answer questions about the pdf skill.",
                    "input_key": "query",
                    "output_key": "answer",
                    "skills": ["pdf"],
                    "skill_dir": str(SKILL_DIR),
                    "use_tools": False,
                }
            ),
        },
        edges=[],
        entry_point="answer",
        provider="ollama",
    )

    result = await graph.run(
        state={
            "query": (
                "According to the pdf skill instructions, what should you do "
                "first when asked to fill out a PDF form?"
            )
        }
    )
    print("\n=== Plain LLM node (skill instructions only) ===")
    print("Query:", result["query"])
    print("Answer:", result["answer"])


async def main() -> None:
    ensure_sample_form()

    fill_form_deterministically()
    print("=== Deterministic fill via the skill's scripts ===")
    print("Wrote filled.pdf from form.pdf (John Smith / 3 / priority / Large)")
    print()

    await run_harness()
    await run_plain_llm()


if __name__ == "__main__":
    asyncio.run(main())
