"""Skills — reusable instruction + tool-scope bundles loaded from folders.

Skills follow the open *Agent Skills* layout: ``<name>/SKILL.md`` holds
YAML frontmatter plus markdown instructions.  A skill contributes
instructions to any LLM call (``LLM`` node, ``ReActAgent``/``harness``)
and can narrow which tools that call may use via ``allowed-tools`` /
``disallowed-tools``.

Example::

    skills/data-analysis/SKILL.md
        ---
        name: data-analysis
        description: Answer questions over tabular data
        allowed-tools: [csv_query, plot]
        ---
        You are a data analyst.  When asked about numbers, always query the
        CSV first with the csv_query tool, then answer from its output.

    flow.react(model="llama3.1:8b", skills=["data-analysis"], skill_dir="skills")
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from draf.tool.tool import Tool

if typing.TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class Skill:
    """A loadable skill bundle.

    Attributes:
        name: Skill name (defaults to the folder name).
        description: What it does and when to use it.
        when_to_use: Optional extra routing hints.
        instructions: Markdown body injected into the system prompt.
        allowed_tools: If set, only these tools are visible to the call.
        disallowed_tools: Tools removed from the visible set.
        path: Directory the skill was loaded from (``None`` if synthetic).
    """

    name: str
    description: str = ""
    when_to_use: str = ""
    instructions: str = ""
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    path: Path | None = None


def _to_list(value: typing.Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v for v in value.replace(",", " ").split() if v]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split ``---``-delimited YAML frontmatter from the markdown body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1 :])
    return "", text


def load_skill(path: str | Path) -> Skill:
    """Load a skill from a folder or a ``SKILL.md`` file.

    Args:
        path: Either a directory containing ``SKILL.md`` or the path to
            the ``SKILL.md`` file itself.

    Returns:
        A :class:`Skill`.

    Raises:
        FileNotFoundError: If no ``SKILL.md`` is found at *path*.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "SKILL.md"
    if not p.is_file():
        raise FileNotFoundError(f"skill file not found: {p}")
    text = p.read_text(encoding="utf-8")
    front, body = _split_frontmatter(text)
    meta: dict = {}
    if front.strip():
        loaded = yaml.safe_load(front)
        if isinstance(loaded, dict):
            meta = loaded

    allowed = meta.get("allowed-tools", meta.get("allowed_tools"))
    disallowed = meta.get("disallowed-tools", meta.get("disallowed_tools"))

    return Skill(
        name=str(meta.get("name") or p.parent.name),
        description=str(meta.get("description", "") or ""),
        when_to_use=str(meta.get("when_to_use", "") or ""),
        instructions=body.strip(),
        allowed_tools=_to_list(allowed) or None,
        disallowed_tools=_to_list(disallowed),
        path=p.parent,
    )


def resolve_skills(cfg: dict) -> list[Skill]:
    """Resolve ``cfg["skills"]`` into a list of loaded skills.

    Each entry may be a :class:`Skill`, a path to a skill folder or
    ``SKILL.md`` file, or a bare name resolved against ``cfg["skill_dir"]``
    (default ``"skills"`` relative to the current directory).
    """
    raw = cfg.get("skills") or []
    if isinstance(raw, (str, Path, Skill)):
        raw = [raw]
    skill_dir = Path(cfg.get("skill_dir", "skills"))

    skills: list[Skill] = []
    for item in raw:
        if isinstance(item, Skill):
            skills.append(item)
            continue
        p = Path(item)
        if p.is_file() or p.is_dir():
            skills.append(load_skill(p))
            continue
        skills.append(load_skill(skill_dir / str(item) / "SKILL.md"))
    return skills


def skills_instructions(skills: list[Skill]) -> str:
    """Render skill instructions as a single block for the system prompt."""
    parts = []
    for s in skills:
        if not s.instructions:
            continue
        header = f"### Skill: {s.name}"
        if s.description:
            header += f" — {s.description}"
        parts.append(f"{header}\n\n{s.instructions}")
    return "\n\n".join(parts)


def scope_tools(
    pool: "Mapping[str, Tool]", cfg: dict, skills: list[Skill] | None = None
) -> dict[str, Tool]:
    """Filter the tool pool to what a node / its skills may use.

    ``cfg["use_tools"]`` may be ``False`` (nothing), ``True`` (everything),
    or a list of names.  Skills narrow the set further: ``allowed_tools``
    intersects with whatever the node allows, ``disallowed_tools`` removes
    tools outright.
    """
    use = cfg.get("use_tools", False)
    if not use:
        return {}

    if isinstance(use, (list, tuple, set)):
        keys = {str(k) for k in use}
        allowed = {k for k in pool if k in keys}
    else:
        allowed = set(pool)

    for s in skills or []:
        if s.allowed_tools is not None:
            allowed &= set(s.allowed_tools)

    disallowed: set[str] = set()
    for s in skills or []:
        disallowed |= set(s.disallowed_tools)

    return {k: v for k, v in pool.items() if k in allowed and k not in disallowed}
