"""Project templates + ``draf new`` scaffolding.

``draf new <name> --template <kind>`` renders a runnable draf app from one
of three templates:

* ``fastapi`` — the default.  A FastAPI service: supervisor ``Flow`` built
  on :meth:`draf.flow.Flow.route`, tool set, durable sessions, API-key auth
  and a debug ``cli.py``.
* ``cli`` — the same supervisor ``Flow`` with a terminal-first interface
  (``cli.py run`` / ``cli.py chat``), no HTTP server.
* ``daemon`` — a background worker: producers drop jobs into ``data/queue/``,
  the worker polls, runs each as a durable turn and writes results.

Each template shares a common core (``_common/`` — graph, nodes, tools,
service, storage, wiring tests) overlaid with template-specific entry points,
config, and API/queue layers.  Every module carries a ``HOW TO EXTEND``
comment, so a template reads as a guide for growing your own app.  See
``examples/simple_router/`` for the minimal ``route()`` example and
``examples/production_repair_ai/`` for a fully runnable, richer instance
built on the same layout.

``draf new <name>`` copies the core + chosen template into a new directory,
renames it, and renders the ``{{PROJECT_NAME}}`` / ``{{project_slug}}`` /
``{{ProjectName}}`` placeholders.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_COMMON = _ROOT / "_common"
_TEMPLATES = {
    "fastapi": _ROOT / "fastapi",
    "cli": _ROOT / "cli",
    "daemon": _ROOT / "daemon",
}

_PLACEHOLDERS = ("{{PROJECT_NAME}}", "{{project_slug}}", "{{ProjectName}}")
_SKIP = {"__pycache__", ".git"}
#: Text files that get placeholder rendering; everything else is copied as-is.
_PLACEHOLDER_EXTENSIONS = {".py", ".toml", ".md", ".txt", ".yml", ".yaml", ".json"}


def _slug(name: str) -> str:
    """Turn any project name into a safe package/identifier slug."""
    slug = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot derive a slug from {name!r}")
    if slug[0].isdigit():
        slug = f"app_{slug}"
    return slug


def _pascal(name: str) -> str:
    """Turn a name into PascalCase, e.g. ``support ai`` -> ``SupportAi``."""
    return "".join(part.capitalize() for part in re.split(r"[^a-z0-9]+", name) if part)


def _render(text: str, *, project_name: str, slug: str, pascal: str) -> str:
    return (
        text.replace("{{PROJECT_NAME}}", project_name)
        .replace("{{project_slug}}", slug)
        .replace("{{ProjectName}}", pascal)
    )


def _copy_tree(
    src_root: Path,
    target: Path,
    *,
    project_name: str,
    slug: str,
    pascal: str,
) -> None:
    """Copy one template directory into *target*, rendering placeholders."""
    for src in sorted(src_root.rglob("*")):
        if src.name in _SKIP or any(part in _SKIP for part in src.parts):
            continue
        rel = src.relative_to(src_root)
        dst = target / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in _PLACEHOLDER_EXTENSIONS or src.name == "main.py":
            rendered = _render(
                src.read_text(encoding="utf-8"),
                project_name=project_name,
                slug=slug,
                pascal=pascal,
            )
            dst.write_text(rendered, encoding="utf-8")
        else:
            shutil.copy2(src, dst)


def new_project(
    name: str, dest: str | None = None, *, template: str = "fastapi"
) -> Path:
    """Scaffold a new draf app from a template.

    Args:
        name: Project name (any case/format, e.g. ``"support-ai"`` or
            ``"Support AI"``); a slug is derived from it.
        dest: Destination directory.  Defaults to ``./<slug>``.
        template: Which template to render — ``"fastapi"`` (default),
            ``"cli"`` or ``"daemon"``.

    Returns:
        The created project directory.

    Raises:
        ValueError: If *template* is not one of the known templates.
        FileExistsError: If *dest* already exists and is non-empty.
    """
    if template not in _TEMPLATES:
        raise ValueError(
            f"unknown template {template!r}; choose from {sorted(_TEMPLATES)}"
        )
    if not _COMMON.is_dir() or not _TEMPLATES[template].is_dir():
        raise FileNotFoundError(
            f"template directories not found: {_COMMON}, {_TEMPLATES[template]} "
            "(is draf installed?)"
        )
    slug = _slug(name)
    pascal = _pascal(name)
    target = Path(dest) if dest else Path(slug)

    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"destination already exists: {target}")

    target.mkdir(parents=True, exist_ok=True)
    _copy_tree(_COMMON, target, project_name=name, slug=slug, pascal=pascal)
    _copy_tree(_TEMPLATES[template], target, project_name=name, slug=slug, pascal=pascal)
    return target
