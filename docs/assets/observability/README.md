# Dashboard screenshots

Drop the dashboard screenshots here. They are referenced from the README and
[`docs/guide/observability.md`](../../guide/observability.md), so the filenames
must match exactly.

| File                   | What it shows                                                    |
| ---------------------- | ---------------------------------------------------------------- |
| `runs-dark.png`        | runs list — filters, status pills, pagination (dark theme)       |
| `run-detail-dark.png`  | run detail page — graph, per-node LLM prompt/response, tags/notes|

How to capture:

1. Start a traced workflow: `teff obs-server --db ./data/traces.db` and run a
   few `workflow.yaml` turns (or `examples/observability/app.py`).
2. Open `http://localhost:8001/obs/ui`, pick a run, open its detail page.
3. Screenshot both views (viewport ~1200px wide is fine). The dark theme is the
   default.

Both files are referenced with relative links that resolve from `docs/guide/`
and `README.md` at the repo root, so no `mkdocs.yml` changes are needed.
