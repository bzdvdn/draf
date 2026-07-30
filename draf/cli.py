"""CLI for running draf workflows from YAML files."""

import asyncio
import json

import typer

app = typer.Typer(
    name="draf",
    help="Workflow as data. Agents as graphs.",
)


@app.command()
def run(
    file: str = typer.Option(..., "--file", "-f", help="Path to workflow YAML file"),
    output: str | None = typer.Option(None, "--output", "-o", help="Write result to file"),
    pretty: bool = typer.Option(False, "--pretty", "-p", help="Pretty-print JSON output"),
) -> None:
    """Run a workflow from a YAML file."""
    from draf.yaml import load_workflow

    try:
        graph, tools, initial_state, reducers = load_workflow(file)
    except Exception as e:
        typer.echo(f"error: failed to parse workflow: {e}", err=True)
        raise typer.Exit(1)

    try:
        result = asyncio.run(graph.run(initial_state, tools=tools, reducers=reducers))
    except Exception as e:
        typer.echo(f"error: workflow failed: {e}", err=True)
        raise typer.Exit(1)

    text = json.dumps(result, indent=2 if pretty else None, default=str) + "\n"
    if output:
        with open(output, "w") as f:
            f.write(text)
    else:
        typer.echo(text)


if __name__ == "__main__":
    app()
