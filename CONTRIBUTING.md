# Contributing to Teff

Thanks for jumping in. This covers the repository layout, how to develop and
verify changes, how docs and tests work, and the conventions the project
follows. Read it before opening a PR.

## Repository layout

```
pyproject.toml          package, entry points, test/lint/mypy config
teff/                   the framework (see below)
docs/                   MkDocs site (guide + reference, see "Docs")
examples/               runnable examples under examples/*/ , one per feature
tests/                  the main pytest suite (coverage gate lives here)
examples/*/tests/       per-example offline tests (run explicitly, not in CI)
.github/workflows/ci.yml  CI: lint (ruff) · typecheck (mypy) · tests+coverage
```

Inside `teff/`, the package is layered by concern:

```
teff/flow/          chainable Flow builder (compiles to Graph) + SubFlow/Case
teff/graph/         the run scheduler: Graph, Edge (conditions), hooks
teff/node/          Node base + every built-in node type (llm, transform,
                    react_agent, interrupt, supervisor, map, retry, …)
teff/state/         State, reducers (merge rules), TypedDict helpers
teff/checkpoint/    durability backends (file, sqlite, pg, history)
teff/provider/      LLM providers + registry
teff/tool/          Tool base, registry, `@tool`, built-in tools
teff/rag/           RAGTool + vector-store adapters
teff/memory/        long-term memory tools
teff/harness/       ReAct/harness agent loops
teff/errors.py      type hierarchy (TeffError …)
teff/yaml.py / yaml_schema.py   YAML workflow load + validation
teff/cli.py         the `teff` CLI (run/graph/eval/daemon/new)
teff/scaffold/      codegen templates used by `teff new`
```

## Setting up

Requires Python ≥ 3.11.

```bash
uv sync --all-extras      # IMPORTANT: --all-extras (light deps host the tests)
source .venv/bin/activate # (uv puts the venv there)
```

`--all-extras` matters: a plain `uv sync` drops the optional vector-store deps
(`chromadb`, etc.) and a handful of tests fail locally even though CI passes.

## Verification gate (this is exactly what CI runs)

```bash
uv run ruff check .                 # lint
uv run ruff format --check .        # format (note: also covers Python blocks in docs/*.md)
uv run mypy .                        # type-check
uv run pytest -q --cov=teff --cov-report=term-missing --cov-fail-under=70   # tests + coverage
uv run mkdocs build                  # docs must build (exit 0)
```

Wide test suite that still needs an external model (or live vector store) must
**mock** that dependency; a short mocked transport for LLM calls is shown in
`examples/recipes/support_triage/tests/`. The full gate must pass: coverage
stays ≥ 70%, no lint/format/mypy issues, docs build clean.

## Docs: how they're generated

Two very different kinds of Markdown live under `docs/`:

- **Hand-written guides** (`docs/getting-started/*`, `docs/guide/*`,
  `docs/recipes/*`, `docs/reference/*`) — Markdown you edit by hand.
- **API reference, pulled from docstrings** — `docs/gen_ref_pages.py` runs at
  build time via the `gen-files` mkdocs plugin. For **every** public module
  under `teff/` it writes `docs/api/teff.<module>.md` containing a
  `::: teff.<module>` directive; the `mkdocstrings` plugin resolves each
  directive **from the module's `__doc__`, signatures, and type annotations**.

So: to fix reference docs, edit docstring/annotations in the source and rebuild;
to fix a guide, edit the Markdown. All the site is configured in `mkdocs.yml`
(`gen-files`, `mkdocstrings`, `hooks`/`docs/hooks.py`). `docs/hooks.py` only
quiets informational Griffe warnings — it does nothing to content.

```bash
uv run mkdocs build       # == CI check; renders into site/
uv run mkdocs serve       # live preview
```

## Adding a new node type

Prefer the **plugin path** (no core change): a module that imports registers
the type, wired into a workflow via the `plugins:` key — see `docs/guide/plugins.md`.

For a node that belongs in the framework itself:

```python
# teff/node/supervizor.py (example shape; register in teff/node/__init__.py)
from teff.node.node import Node


class MyNode(Node):
    type = "my_type"

    async def execute(self, ctx, state: dict) -> dict:
        ...
        return {output_key: value}
```

Register it in `teff/node/__init__.py` and add it to the
`docs/reference/nodes.md` table (and the `Cli cheat-sheet` "which node" table).
Every node keeps the same contract: `async def execute(ctx, state) -> dict`.

## Adding a tool

```python
from teff.tool.registry import tool


@tool("my_lookup")
async def my_lookup(query: str, limit: int = 10) -> str: ...
```

Register in `teff/tool/builtin/__init__.py` for a built-in and list it in
`docs/reference/tools.md`; otherwise users pass tools via the `tools=` argument.
See `docs/guide/plugins.md` for the plugin route.

## Adding an LLM provider

Providers live under `teff/provider/` and are wired into
`teff.provider.DEFAULT_PROVIDERS`. Subclass the provider base, implement the
completions/streaming calls, add pricing hooks if available, and add a config
example under `teff/provider/` as a module + a row in
`docs/reference/providers.md`.
Prefer `--mock`/offline tests that patch httpx over live endpoints.

## Commit conventions

Commits use a `<scope>: <imperative summary>` subject, then a bulleted body
with the context. Scopes seen in the repo: `docs:`, `test:`, `perf:`, `deps`,
`checkpointer:`, `tool:`, `memory:`, `observability:`, and feature names. The
summary is a command, and the body explains *why/what*, which examples/tests it
touches, and any caveats.

```text
docs: guide to core entities and how to build your own

- add docs/guide/core-entities.md (entity map, entities, glossary)
- render the flow from a single mental model (graphs over state)
- cross-link to plugins/state/durable/troubleshooting
```

Keep commits scoped: separate docs from behaviour, and never bundle secrets or
unrelated formatting with a feature.

## PR flow

- Branch, open a PR. Make it small and focused.
- Fill tests + update docs (guides AND the node/tool tables if you contribute a
  new type).
- Run the full verification gate; it must be green.
- Reviewers run the same gate, so fix lints/format/mypy before requesting a review.