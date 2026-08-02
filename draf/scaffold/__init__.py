"""Project templates + ``draf new`` scaffolding.

``draf new <name> --template <kind> [--with postgres,rag,celery]`` renders
a runnable draf app from one of three templates:

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
``examples/applications/repair-ai-chat/`` for a fully runnable, richer instance
built on the same layout.

Feature variants are additive overlays copied on top of the template core,
one per directory under ``variants/``:

* ``postgres`` — ``deploy/compose.yaml`` (pgvector) + ``.env.example``; the
  DSN points sessions (and RAG vectors) at Postgres.
* ``rag`` — a document catalog over ``data/documents/`` with RAG search
  tools wired into the writer agent.
* ``celery`` — a Celery worker + beat pair that re-embeds the catalog when
  the seed documents change.

``draf new <name>`` copies the core + chosen template into a new directory,
renames it, and renders the ``{{PROJECT_NAME}}`` / ``{{project_slug}}`` /
``{{ProjectName}}`` placeholders.
"""

from __future__ import annotations

import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_COMMON = _ROOT / "_common"
_TEMPLATE_DIRS = {
    "fastapi": _ROOT / "fastapi",
    "cli": _ROOT / "cli",
    "daemon": _ROOT / "daemon",
}
_VARIANTS_DIR = _ROOT / "variants"

_PLACEHOLDERS = ("{{PROJECT_NAME}}", "{{project_slug}}", "{{ProjectName}}")
_SKIP = {"__pycache__", ".git", "template.toml"}
#: Text files that get placeholder rendering; everything else is copied as-is.
_PLACEHOLDER_EXTENSIONS = {".py", ".toml", ".md", ".txt", ".yml", ".yaml", ".json"}


@dataclass(frozen=True)
class TemplateManifest:
    """Metadata from a template's ``template.toml``.

    Attributes:
        name: Template kind (``"fastapi"`` / ``"cli"`` / ``"daemon"``).
        description: One-line description shown in ``draf new --help``.
        entry: How to run the generated project (e.g. ``python main.py``).
        variants: Feature variants the template can combine with.
        path: The template's directory inside this package.
    """

    name: str
    description: str
    entry: str
    variants: tuple[str, ...]
    path: Path


def _load_manifest(name: str) -> TemplateManifest:
    """Load and validate one template manifest from its ``template.toml``."""
    path = _TEMPLATE_DIRS[name]
    with (path / "template.toml").open("rb") as fh:
        data = tomllib.load(fh)
    return TemplateManifest(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        entry=str(data.get("entry", "python main.py")),
        variants=tuple(str(v) for v in data.get("variants", [])),
        path=path,
    )


#: Every template's manifest, keyed by kind.
TEMPLATES: dict[str, TemplateManifest] = {
    name: _load_manifest(name) for name in sorted(_TEMPLATE_DIRS)
}

#: Every variant overlay shipped in this package.
VARIANTS: tuple[str, ...] = tuple(
    sorted(p.name for p in _VARIANTS_DIR.iterdir() if p.is_dir())
)


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
    """Copy one template/variant directory into *target*, rendering placeholders.

    ``template.toml`` is the registry manifest (not app content) and is never
    copied into the generated project.
    """
    for src in sorted(src_root.rglob("*")):
        if src.name in _SKIP or any(part in _SKIP for part in src.parts):
            continue
        rel = src.relative_to(src_root)
        dst = target / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (
            src.suffix in _PLACEHOLDER_EXTENSIONS
            or src.name == "main.py"
            or src.name.startswith(".env")
        ):
            rendered = _render(
                src.read_text(encoding="utf-8"),
                project_name=project_name,
                slug=slug,
                pascal=pascal,
            )
            dst.write_text(rendered, encoding="utf-8")
        else:
            shutil.copy2(src, dst)


def validate_variants(
    manifest: TemplateManifest, variants: tuple[str, ...]
) -> tuple[str, ...]:
    """Return the deduplicated *variants*, rejecting unknown or unsupported ones.

    Args:
        manifest: The selected template manifest.
        variants: Requested variant names.

    Returns:
        The variants with duplicates removed, preserving order.

    Raises:
        ValueError: If a variant is unknown, or is not offered by *manifest*.
    """
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variant(s) {unknown}; choose from {VARIANTS}")
    unsupported = [v for v in variants if v not in manifest.variants]
    if unsupported:
        raise ValueError(
            f"template {manifest.name!r} does not support variant(s) "
            f"{unsupported}; it offers {manifest.variants}"
        )
    return tuple(dict.fromkeys(variants))


def new_project(
    name: str,
    dest: str | None = None,
    *,
    template: str = "fastapi",
    variants: tuple[str, ...] = (),
) -> Path:
    """Scaffold a new draf app from a template (+ optional variants).

    Args:
        name: Project name (any case/format, e.g. ``"support-ai"`` or
            ``"Support AI"``); a slug is derived from it.
        dest: Destination directory.  Defaults to ``./<slug>``.
        template: Which template to render — ``"fastapi"`` (default),
            ``"cli"`` or ``"daemon"``.
        variants: Feature variants to enable on top of the template — any
            subset of ``postgres``, ``rag``, ``celery``.

    Returns:
        The created project directory.

    Raises:
        ValueError: If *template* is unknown, a *variant* is unknown, or a
            *variant* is not offered by *template*.
        FileExistsError: If *dest* already exists and is non-empty.
    """
    if template not in TEMPLATES:
        raise ValueError(
            f"unknown template {template!r}; choose from {sorted(TEMPLATES)}"
        )
    manifest = TEMPLATES[template]
    variants = validate_variants(manifest, variants)
    if not _COMMON.is_dir() or not manifest.path.is_dir():
        raise FileNotFoundError(
            f"template directories not found: {_COMMON}, {manifest.path} "
            "(is draf installed?)"
        )
    slug = _slug(name)
    pascal = _pascal(name)
    target = Path(dest) if dest else Path(slug)

    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"destination already exists: {target}")

    target.mkdir(parents=True, exist_ok=True)
    _copy_tree(_COMMON, target, project_name=name, slug=slug, pascal=pascal)
    _copy_tree(manifest.path, target, project_name=name, slug=slug, pascal=pascal)
    for variant in variants:
        _copy_tree(
            _VARIANTS_DIR / variant,
            target,
            project_name=name,
            slug=slug,
            pascal=pascal,
        )
    return target
