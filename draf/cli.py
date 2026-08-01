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
    except Exception as e:
        typer.echo(f"error: failed to load workflow: {e}", err=True)
        raise typer.Exit(1)

    checkpointer = None
    if checkpoint:
        base_dir = os.path.dirname(os.path.abspath(file))
        cfg = _resolve_checkpoint_config(json.loads(checkpoint), base_dir)
        checkpointer = _checkpointer_from_config(cfg)

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
) -> dict | None:
    """Run a graph, handling interrupts interactively or via resume."""
    from draf.node.interrupt import GraphInterrupt
    from draf.trace import RunTracer

    tracer = RunTracer() if trace else None
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
            )
            if tracer is not None:
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
def validate(
    file: str = typer.Argument(..., help="Path to workflow YAML file"),
) -> None:
    """Validate a workflow YAML file without running it."""
    from draf.yaml_schema import validate_workflow_file, format_errors

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

    from draf.yaml import load_workflow
    from draf.eval import load_dataset, run_eval, format_report

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
def version() -> None:
    """Print the draf version."""
    typer.echo(f"draf {__version__}")


if __name__ == "__main__":
    app()
