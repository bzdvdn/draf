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
        builtin: True for skills bundled with draf (core/``draf-`` skills),
            False for user skills loaded from disk.
    """

    name: str
    description: str = ""
    when_to_use: str = ""
    instructions: str = ""
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    path: Path | None = None
    builtin: bool = False


def _to_list(value: typing.Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v for v in value.replace(",", " ").split() if v]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


# Core meta-skills shipped with draf.  These are framework-level behaviour
# contracts for the model, orthogonal to any specific tool or domain, and
# are resolved by name when no user skill with the same name exists on
# disk.  All are namespaced with the ``draf-`` prefix to avoid colliding
# with user skills.
_CORE_SKILLS: list[Skill] = [
    Skill(
        name="draf-tool-discipline",
        description="When and how to call tools in the ReAct loop",
        when_to_use="Any agent that uses tools",
        builtin=True,
        instructions="""\
Follow the tool discipline rules:

1. Call a tool only when the answer is not already in your context or
   knowledge.  Prefer answering directly when you can.
2. Never fabricate a tool result.  Base every claim on the actual tool
   output returned to you.
3. On a tool error: retry once with the same call if the failure looks
   transient, otherwise change approach or fall back — but never ignore
   the error silently.
4. Do not repeat a tool call with identical arguments if the previous
   call already returned.  If a call would loop, stop and explain the
   situation instead.

Tool calls exist to ground your answer in facts, not to pad the
conversation.""",
    ),
    Skill(
        name="draf-structured-output",
        description="Fill JSON schemas exactly in structured-output mode",
        when_to_use="Nodes using response_format / json_object",
        builtin=True,
        instructions="""\
When producing structured output, follow these rules:

1. Return only the fields declared in the schema, with exactly the
   declared types.
2. Do not use ``null`` where the schema expects a concrete value; fall
   back to an empty string or a sensible default.
3. In ``json_object`` mode output nothing but the JSON document — no
   markdown fences, no prose before or after.
4. Never invent a value to fill a required field; if the data is missing,
   say so through the field's default/empty representation.""",
    ),
    Skill(
        name="draf-verification",
        description="Cross-check final answers against tool output",
        when_to_use="Any agent producing a final answer",
        builtin=True,
        instructions="""\
Verify your answer before delivering it:

1. State only facts that are confirmed by the tool output you actually
   received.
2. Re-read the result of your last tool call before writing the final
   answer.
3. Do not carry over details that were not present in the results.
4. When you are not certain, say so explicitly instead of presenting a
   guess as fact.""",
    ),
]


def core_skills() -> list[Skill]:
    """Return the built-in core (``draf-``) skills."""
    return list(_CORE_SKILLS)


def get_core_skill(name: str) -> Skill | None:
    """Return a core skill by name, or ``None`` if it does not exist."""
    for s in _CORE_SKILLS:
        if s.name == name:
            return s
    return None


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
        candidate = skill_dir / str(item) / "SKILL.md"
        if candidate.is_file():
            skills.append(load_skill(candidate))
            continue
        core = get_core_skill(str(item))
        if core is not None:
            skills.append(core)
            continue
        msg = f"skill not found: {item}"
        raise FileNotFoundError(msg)
    return skills


def skills_instructions(skills: list[Skill]) -> str:
    """Render skill instructions as a single block for the system prompt."""
    parts = []
    for s in skills:
        if not s.instructions:
            continue
        header = f"### Skill: {s.name}"
        if s.builtin:
            header += " [system]"
        if s.description:
            header += f" — {s.description}"
        parts.append(f"{header}\n\n{s.instructions}")
    return "\n\n".join(parts)


def scope_tools(
    pool: "Mapping[str, Tool]", cfg: dict, skills: list[Skill] | None = None
) -> dict[str, Tool]:
    """Filter the tool pool to what a node / its skills may use.

    ``cfg["use_tools"]`` may be ``None`` or an empty list (nothing),
    ``"all"`` (everything), or a list of names to allow.  (``True``/``False``
    are also honoured for backwards compatibility.)  Skills narrow the set
    further: ``allowed_tools`` intersects with whatever the node allows,
    ``disallowed_tools`` removes tools outright.
    """
    use = cfg.get("use_tools")

    def _all() -> set[str]:
        return set(pool)

    if use is None:
        allowed: set[str] = set()
    elif isinstance(use, str):
        allowed = _all() if use.strip().lower() in ("all", "*") else {use}
    elif isinstance(use, (list, tuple, set)):
        allowed = {str(k) for k in use}
    elif isinstance(use, bool):
        allowed = _all() if use else set()
    else:
        allowed = set()

    for s in skills or []:
        if s.allowed_tools is not None:
            allowed &= set(s.allowed_tools)

    disallowed: set[str] = set()
    for s in skills or []:
        disallowed |= set(s.disallowed_tools)

    return {k: v for k, v in pool.items() if k in allowed and k not in disallowed}
