"""Generate mkdocs-gen-files reference pages for the ``teff`` package.

Every public module under ``teff/`` gets ``docs/api/<module>.md`` rendered
through mkdocstrings, plus a ``teff/**``-style listing page so the whole
public surface is documented straight from docstrings.

Run automatically by MkDocs via ``gen-files`` in ``mkdocs.yml``; invoke with
``python docs/gen_ref_pages.py`` to preview output locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "teff"
OUT = Path("api")
SKIP = {
    "__pycache__",
    ".mypy_cache",
    "_version",
    "scaffold",  # rendered by `teff new`, not a user-facing package API
}


def modules(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP for part in path.parts):
            continue
        if path.name.endswith("test_") or "tests" in path.parts:
            continue
        if path.name == "__init__.py":
            if path == root / "__init__.py":
                continue
            yield path
            continue
        if path.name.startswith("_"):
            continue
        yield path


def rel_name(path: Path) -> str:
    rel = path.relative_to(PKG)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return "teff." + ".".join(parts)


def main() -> None:
    seen: list[str] = []

    for path in modules(PKG):
        name = rel_name(path)
        out = OUT / f"{name}.md"
        with mkdocs_gen_files.open(out, "w") as fh:
            fh.write(f"# {name}\n\n")
            fh.write(f"::: {name}\n")
        seen.append(name)

    listing = "teff"
    with mkdocs_gen_files.open(OUT / f"{listing}.md", "w") as fh:
        fh.write(f"# `{listing}`\n\n::: {listing}\n")

    with mkdocs_gen_files.open(OUT / "index.md", "w") as fh:
        fh.write("# API Reference\n\n")
        fh.write("Every public module, generated from docstrings.\n\n")
        for name in seen:
            fh.write(f"- [`{name}`]({name}.md)\n")


main()
