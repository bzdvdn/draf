"""CLI for running draf workflows from YAML files.

The app doubles as the default ``run`` command: ``draf --file wf.yaml``
and ``draf run --file wf.yaml`` are equivalent.  Additional subcommands
cover validation, inspection, evaluation, and versioning.
"""

import asyncio
import json
import os

import typer

from draf._version import __version__
from draf.checkpoint import DEFAULT_OWNER
from draf.scaffold import TEMPLATES

#: Hosts where unauthenticated trace serving is still permitted.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

app = typer.Typer(
    name="draf",
    help="Workflow as data. Agents as graphs.",
    invoke_without_command=True,
)


def _checkpointer_from_config(config: dict):
    """Build a checkpointer from a ``{type, ...}`` dict."""
    ctype = config.get("type")
    if ctype == "file":
        from draf.checkpoint import JSONFileCheckpointer

        return JSONFileCheckpointer(config.get("path", "checkpoints"))
    if ctype == "sqlite":
        from draf.checkpoint import SQLiteCheckpointer

        return SQLiteCheckpointer(config.get("path", "checkpoints.db"))
    if ctype == "pg":
        from draf.checkpoint.pg import PGCheckpointer

        dsn = config.get("dsn")
        if not isinstance(dsn, str):
            raise typer.BadParameter("checkpoint 'dsn' is required for type 'pg'")
        return PGCheckpointer(dsn, config.get("table", "checkpoints"))
    if ctype == "sqlite_history":
        from draf.checkpoint import SQLiteHistoryCheckpointer

        return SQLiteHistoryCheckpointer(config.get("path", "checkpoints.db"))
    if ctype == "pg_history":
        from draf.checkpoint import PGHistoryCheckpointer

        dsn = config.get("dsn")
        if not isinstance(dsn, str):
            raise typer.BadParameter(
                "checkpoint 'dsn' is required for type 'pg_history'"
            )
        return PGHistoryCheckpointer(dsn, config.get("table", "checkpoints"))
    raise typer.BadParameter(f"unknown checkpoint type: {ctype!r}")


def _run_workflow(
    file: str,
    *,
    output: str | None = None,
    pretty: bool = False,
    trace: bool = False,
    checkpoint: str | None = None,
    checkpoint_id: str | None = None,
    checkpoint_owner: str = DEFAULT_OWNER,
    resume: dict | None = None,
    node_timeout: float | None = None,
    max_iterations: int | None = None,
    interactive: bool = False,
) -> None:
    from draf.yaml import load_workflow

    try:
        graph, tools, initial_state, reducers = load_workflow(file)
        cfg = _load_yaml(file)
    except Exception as e:
        typer.echo(f"error: failed to load workflow: {e}", err=True)
        raise typer.Exit(1)

    checkpointer = None
    if checkpoint:
        base_dir = os.path.dirname(os.path.abspath(file))
        cfg = _resolve_checkpoint_config(json.loads(checkpoint), base_dir)
        checkpointer = _checkpointer_from_config(cfg)

    base_dir = os.path.dirname(os.path.abspath(file))
    observer_factory = _observer_factory(file, cfg, graph, base_dir)

    try:
        result = asyncio.run(
            _run_loop(
                graph,
                initial_state,
                tools=tools,
                reducers=reducers,
                checkpointer=checkpointer,
                checkpoint_id=checkpoint_id,
                checkpoint_owner=checkpoint_owner,
                resume=resume,
                node_timeout=node_timeout,
                max_iterations=max_iterations,
                interactive=interactive,
                trace=trace,
                observer_factory=observer_factory,
            )
        )
    except Exception as e:
        typer.echo(f"error: workflow failed: {e}", err=True)
        raise typer.Exit(1)

    text = json.dumps(result, indent=2 if pretty else None, default=str) + "\n"
    if output:
        with open(output, "w") as f:
            f.write(text)
    else:
        typer.echo(text)


async def _run_loop(
    graph,
    state,
    *,
    tools,
    reducers,
    checkpointer,
    checkpoint_id,
    checkpoint_owner,
    resume,
    node_timeout,
    max_iterations,
    interactive,
    trace,
    observer_factory=None,
) -> dict | None:
    """Run a graph, handling interrupts interactively or via resume."""
    from draf.node.interrupt import GraphInterrupt
    from draf.trace import RunTracer

    observer = observer_factory() if observer_factory else None
    tracer = observer.tracer if observer else (RunTracer() if trace else None)
    try:
        while True:
            try:
                result = await graph.run(
                    state,
                    tools=tools,
                    reducers=reducers,
                    checkpointer=checkpointer,
                    checkpoint_id=checkpoint_id,
                    owner=checkpoint_owner,
                    resume=resume,
                    node_timeout=node_timeout,
                    max_iterations=max_iterations,
                    tracer=tracer,
                    on_llm_payload=observer.on_llm_payload if observer else None,
                )
                if tracer is not None and observer is None:
                    typer.echo(tracer.to_json(), err=True)
                return result
            except GraphInterrupt as interrupt:
                if not interactive:
                    if tracer is not None:
                        typer.echo(tracer.to_json(), err=True)
                    raise
                typer.echo(
                    f"\n-- paused: {interrupt.prompt or interrupt.key} "
                    f"(checkpoint {interrupt.checkpoint_id!r}) --",
                    err=True,
                )
                answer = input("> ").strip()
                resume = {interrupt.key: answer}
    finally:
        if observer is not None:
            observer.export()


def _resolve_checkpoint_config(config: dict, base_dir: str) -> dict:
    """Resolve relative paths in a ``checkpoint:`` JSON block against base_dir."""
    result = dict(config)
    for key in ("path",):
        if key in result and not os.path.isabs(result[key]):
            result[key] = os.path.join(base_dir, result[key])
    return result


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise typer.BadParameter(
            f"{path}: expected a YAML mapping, got {type(data).__name__}"
        )
    return data


def _observer_factory(file: str, cfg: dict, graph, base_dir: str):
    """Build a per-run GraphObserver factory from the workflow's YAML.

    Returns ``None`` when the workflow has no ``observability:`` block, so
    existing workflows are untouched.
    """
    from draf.observability import build_observer_factory

    return build_observer_factory(
        cfg.get("observability"),
        base_dir=base_dir,
        graph=graph,
        name=cfg.get("name", "workflow"),
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    file: str = typer.Option(None, "--file", "-f", help="Path to workflow YAML file"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Write result to file"
    ),
    pretty: bool = typer.Option(
        False, "--pretty", "-p", help="Pretty-print JSON output"
    ),
    trace: bool = typer.Option(
        False, "--trace", "-t", help="Print a JSON run trace to stderr"
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        help='JSON checkpointer config, e.g. \'{"type":"file","path":"cp"}\'',
    ),
    checkpoint_id: str | None = typer.Option(
        None, "--checkpoint-id", help="Checkpoint key identifying the run"
    ),
    checkpoint_owner: str = typer.Option(
        DEFAULT_OWNER,
        "--checkpoint-owner",
        help="Owner/session scoping the checkpoint (e.g. a user id)",
    ),
    resume: str | None = typer.Option(
        None, "--resume", help='Resume values as JSON, e.g. \'{"approved":"да"}\''
    ),
    node_timeout: float | None = typer.Option(
        None, "--node-timeout", help="Max seconds per node"
    ),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations", help="Max node executions (loop guard)"
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="Prompt the operator on stdin when a workflow pauses for input",
    ),
) -> None:
    """Run a workflow from a YAML file (default command)."""
    if ctx.invoked_subcommand is not None:
        return
    if not file:
        typer.echo(ctx.get_usage(), err=True)
        raise typer.Exit(1)
    _run_workflow(
        file,
        output=output,
        pretty=pretty,
        trace=trace,
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        checkpoint_owner=checkpoint_owner,
        resume=json.loads(resume) if resume else None,
        node_timeout=node_timeout,
        max_iterations=max_iterations,
        interactive=interactive,
    )


@app.command()
def run(
    file: str = typer.Option(..., "--file", "-f", help="Path to workflow YAML file"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Write result to file"
    ),
    pretty: bool = typer.Option(
        False, "--pretty", "-p", help="Pretty-print JSON output"
    ),
    trace: bool = typer.Option(
        False, "--trace", "-t", help="Print a JSON run trace to stderr"
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        help='JSON checkpointer config, e.g. \'{"type":"file","path":"cp"}\'',
    ),
    checkpoint_id: str | None = typer.Option(
        None, "--checkpoint-id", help="Checkpoint key identifying the run"
    ),
    checkpoint_owner: str = typer.Option(
        DEFAULT_OWNER,
        "--checkpoint-owner",
        help="Owner/session scoping the checkpoint (e.g. a user id)",
    ),
    resume: str | None = typer.Option(
        None, "--resume", help='Resume values as JSON, e.g. \'{"approved":"да"}\''
    ),
    node_timeout: float | None = typer.Option(
        None, "--node-timeout", help="Max seconds per node"
    ),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations", help="Max node executions (loop guard)"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", help="Prompt on stdin when a workflow pauses for input"
    ),
) -> None:
    """Run a workflow from a YAML file."""
    _run_workflow(
        file,
        output=output,
        pretty=pretty,
        trace=trace,
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        checkpoint_owner=checkpoint_owner,
        resume=json.loads(resume) if resume else None,
        node_timeout=node_timeout,
        max_iterations=max_iterations,
        interactive=interactive,
    )


@app.command()
def daemon(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
    interval: float = typer.Option(
        60.0, "--interval", "-i", help="Seconds between ticks"
    ),
    once: bool = typer.Option(False, "--once", help="Run a single tick and exit"),
    trace: bool = typer.Option(
        False, "--trace", "-t", help="Print a JSON run trace to stderr"
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        help='JSON checkpointer config, e.g. \'{"type":"file","path":"cp"}\'',
    ),
    checkpoint_id: str = typer.Option(
        "daemon", "--checkpoint-id", help="Checkpoint key for durable daemon state"
    ),
    checkpoint_owner: str = typer.Option(
        DEFAULT_OWNER,
        "--checkpoint-owner",
        help="Owner/session scoping the checkpoint (e.g. a user id)",
    ),
    node_timeout: float | None = typer.Option(
        None, "--node-timeout", help="Max seconds per node"
    ),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations", help="Max node executions (loop guard)"
    ),
) -> None:
    """Run a workflow as a daemon: poll on an interval, keeping state between ticks.

    The workflow itself defines what a *tick* does — e.g. list open GitLab
    merge requests, review new ones, post verdicts and notify Telegram.
    Durable state (already-reviewed MRs, counters, …) is carried across ticks
    via the optional ``--checkpoint``.
    """
    from draf.yaml import load_workflow

    try:
        graph, tools, initial_state, reducers = load_workflow(file)
        cfg = _load_yaml(file)
    except Exception as e:
        typer.echo(f"error: failed to load workflow: {e}", err=True)
        raise typer.Exit(1)

    checkpointer = None
    if checkpoint:
        base_dir = os.path.dirname(os.path.abspath(file))
        cfg = _resolve_checkpoint_config(json.loads(checkpoint), base_dir)
        checkpointer = _checkpointer_from_config(cfg)

    base_dir = os.path.dirname(os.path.abspath(file))
    observer_factory = _observer_factory(file, cfg, graph, base_dir)

    try:
        asyncio.run(
            _daemon_loop(
                graph,
                initial_state,
                tools=tools,
                reducers=reducers,
                checkpointer=checkpointer,
                checkpoint_id=checkpoint_id,
                checkpoint_owner=checkpoint_owner,
                interval=interval,
                once=once,
                node_timeout=node_timeout,
                max_iterations=max_iterations,
                trace=trace,
                observer_factory=observer_factory,
            )
        )
    except Exception as e:
        typer.echo(f"error: daemon failed: {e}", err=True)
        raise typer.Exit(1)


async def _daemon_loop(
    graph,
    initial_state,
    *,
    tools,
    reducers,
    checkpointer,
    checkpoint_id,
    checkpoint_owner,
    interval,
    once,
    node_timeout,
    max_iterations,
    trace,
    observer_factory=None,
) -> None:
    """Run *graph* once per tick, persisting state between ticks.

    Each tick re-runs the workflow from its entry point (so sources like
    GitLab are re-polled), starting from the durable state of the previous
    tick.  ``GraphInterrupt`` pauses are logged and skipped rather than
    blocking the daemon.
    """
    from draf.checkpoint import Checkpoint
    from draf.node.interrupt import GraphInterrupt
    from draf.trace import RunTracer

    state = dict(initial_state)
    if checkpointer is not None:
        saved = await checkpointer.load(checkpoint_id, owner=checkpoint_owner)
        if saved is not None:
            state = dict(saved.state)

    tick = 0
    while True:
        tick += 1
        observer = observer_factory() if observer_factory else None
        tracer = observer.tracer if observer else (RunTracer() if trace else None)
        try:
            try:
                state = await graph.run(
                    state,
                    tools=tools,
                    reducers=reducers,
                    node_timeout=node_timeout,
                    max_iterations=max_iterations,
                    tracer=tracer,
                    on_llm_payload=observer.on_llm_payload if observer else None,
                )
                if tracer is not None and observer is None:
                    typer.echo(tracer.to_json(), err=True)
                typer.echo(json.dumps(state, ensure_ascii=False, default=str))
                if checkpointer is not None:
                    await checkpointer.save(
                        checkpoint_id,
                        Checkpoint(state=dict(state), next_node_id=None, iteration=0),
                        owner=checkpoint_owner,
                    )
            except GraphInterrupt as interrupt:
                if tracer is not None and observer is None:
                    typer.echo(tracer.to_json(), err=True)
                typer.echo(
                    f"tick {tick}: paused at {interrupt.node_id!r} "
                    f"({interrupt.prompt or interrupt.key}) — skipped in daemon mode",
                    err=True,
                )
        finally:
            if observer is not None:
                observer.export()
        if once:
            return
        await asyncio.sleep(interval)


@app.command()
def graph(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
    mermaid: bool = typer.Option(
        False, "--mermaid", help="Render the workflow graph as a Mermaid diagram"
    ),
) -> None:
    """Inspect a workflow graph: YAML topology or a Mermaid diagram."""
    from draf.yaml import load_workflow

    try:
        graph_, _tools, _state, _reducers = load_workflow(file)
    except Exception as e:
        typer.echo(f"error: failed to load workflow: {e}", err=True)
        raise typer.Exit(1)

    if mermaid:
        typer.echo(graph_.to_mermaid())
        return
    typer.echo(graph_.to_yaml())


@app.command()
def validate(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
) -> None:
    """Validate a workflow YAML file without running it."""
    from draf.yaml_schema import format_errors, validate_workflow_file

    try:
        errors = validate_workflow_file(file)
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if errors:
        typer.echo(format_errors(errors, source=file), err=True)
        typer.echo(f"invalid: {len(errors)} error(s)", err=True)
        raise typer.Exit(1)
    typer.echo(f"ok: {file} is a valid workflow")


@app.command("eval")
def eval_(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
    data: str = typer.Option(
        ..., "--data", "-d", help="Dataset file (.json/.jsonl/.csv)"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Write the JSON report to a file"
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Model used to score outputs (LLM judge)"
    ),
    judge_provider: str | None = typer.Option(
        None, "--judge-provider", help="Provider key for the judge model"
    ),
    exact: bool = typer.Option(
        False, "--exact", help="Score by exact (normalised) string match"
    ),
    max_examples: int | None = typer.Option(
        None, "--max-examples", help="Limit the number of examples"
    ),
    output_key: str | None = typer.Option(
        None, "--output-key", help="State key holding the answer"
    ),
) -> None:
    """Evaluate a workflow against a dataset and report pass/fail."""
    import json as _json

    from draf.eval import format_report, load_dataset, run_eval
    from draf.yaml import load_workflow

    try:
        workflow = load_workflow(file)
        dataset = load_dataset(data)
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    try:
        report = asyncio.run(
            run_eval(
                workflow,
                dataset,
                judge_model=judge_model,
                judge_provider=judge_provider,
                exact=exact,
                max_examples=max_examples,
                output_key=output_key,
            )
        )
    except Exception as e:
        typer.echo(f"error: eval failed: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(format_report(report), err=True)
    text = _json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    if output:
        with open(output, "w") as f:
            f.write(text)
    else:
        typer.echo(text)


@app.command()
def inspect(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="JSON checkpointer config"
    ),
    checkpoint_id: str = typer.Option(
        ..., "--checkpoint-id", help="Run key to inspect"
    ),
    checkpoint_owner: str = typer.Option(
        DEFAULT_OWNER,
        "--checkpoint-owner",
        help="Owner/session scoping the checkpoint (e.g. a user id)",
    ),
) -> None:
    """Print the saved state for a checkpointed run."""
    try:
        cp = _checkpointer_from_config(json.loads(checkpoint))
        saved = asyncio.run(cp.load(checkpoint_id, owner=checkpoint_owner))
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if saved is None:
        typer.echo(f"no checkpoint for {checkpoint_id!r}", err=True)
        raise typer.Exit(1)
    from draf.checkpoint import checkpoint_to_dict

    typer.echo(json.dumps(checkpoint_to_dict(saved), indent=2, default=str))


@app.command()
def prune(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="JSON checkpointer config"
    ),
    checkpoint_owner: str | None = typer.Option(
        None,
        "--checkpoint-owner",
        help="Only prune this owner (default: all owners)",
    ),
    max_age: float | None = typer.Option(
        None,
        "--max-age",
        help="Delete checkpoints older than this many seconds",
    ),
    keep_last: int | None = typer.Option(
        None, "--keep-last", help="Keep only the N most recent per owner"
    ),
) -> None:
    """Delete stale checkpoints (TTL / keep-last GC)."""
    try:
        cp = _checkpointer_from_config(json.loads(checkpoint))
        removed = asyncio.run(
            cp.cleanup(
                owner=checkpoint_owner,
                max_age=max_age,
                keep_last=keep_last,
            )
        )
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"removed {removed} checkpoint(s)")


@app.command()
def obs_server(
    db: str = typer.Option("traces.db", "--db", help="SQLite file holding the traces"),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Address to bind (use 0.0.0.0 to expose)"
    ),
    port: int = typer.Option(8001, "--port", help="Port to listen on"),
    prefix: str = typer.Option(
        "/obs", "--prefix", help="URL prefix for the dashboard and ingest"
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="DRAF_OBS_API_KEY",
        help="Shared key required in the X-API-Key header (mandatory on 0.0.0.0)",
    ),
) -> None:
    """Serve the trace dashboard + ingest endpoint (standalone obs server).

    Workflows with no API push their traces here via ``observability:``
    (``type: webhook``), and this process serves the dashboard UI::

        draf obs-server --db traces.db --host 127.0.0.1 --port 8001
        # open http://localhost:8001/obs/ui

    Traces contain full prompts/responses.  Binding to a non-loopback host
    (``0.0.0.0``) without ``--api-key`` is refused: the server refuses to
    start rather than expose them unauthenticated.
    """
    if api_key is None and host not in _LOOPBACK_HOSTS:
        raise typer.BadParameter(
            "--api-key is required when binding outside 127.0.0.1 "
            "(traces contain full prompts/responses)",
            param_hint="--host",
        )
    try:
        from draf.observability.server import serve
    except ImportError as e:
        typer.echo(
            f"error: 'draf[observability]' is required for obs-server: {e}",
            err=True,
        )
        raise typer.Exit(1)
    serve(db, host=host, port=port, prefix=prefix, api_key=api_key)


@app.command()
def new(
    name: str = typer.Argument(..., help="Project name, e.g. 'support-ai'"),
    dest: str | None = typer.Option(
        None, "--dest", help="Destination directory (default: ./<slug>)"
    ),
    template: str = typer.Option(
        "fastapi",
        "--template",
        "-t",
        help=f"App template: {', '.join(TEMPLATES)}",
    ),
    with_variants: str = typer.Option(
        "",
        "--with",
        help="Comma-separated feature variants: postgres,rag,celery",
    ),
) -> None:
    """Scaffold a new draf app from a template (fastapi|cli|daemon)."""
    from draf.scaffold import new_project

    variants = tuple(v for v in (p.strip() for p in with_variants.split(",")) if v)
    try:
        path = new_project(name, dest=dest, template=template, variants=variants)
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"created {path}")
    typer.echo(
        f"next: uv sync && uv run pytest tests/ && uv run {TEMPLATES[template].entry}"
    )
    if variants:
        typer.echo(f"variants: {', '.join(variants)}")


@app.command()
def version() -> None:
    """Print the draf version."""
    typer.echo(f"draf {__version__}")


if __name__ == "__main__":
    app()
