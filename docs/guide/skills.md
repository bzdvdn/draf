# Skills

Bundle instructions **and** a tool scope into a reusable folder using the
open *Agent Skills* layout — `skills/<name>/SKILL.md` with YAML frontmatter
plus markdown instructions:

```markdown
---
name: city-guide
description: Answer questions about cities
allowed-tools: [city_weather, city_population]
disallowed-tools: [secret_tool]
---

You are a city guide.  When asked to compare cities, call BOTH
`city_weather` and `city_population` in the SAME turn.
```

Mount it on any LLM-capable call — the `LLM` node or `react()` / `harness()`:

```python
flow.harness(
    model="llama3.1:8b",
    input_key="query",
    output_key="answer",
    skills=["city-guide"],
    skill_dir="skills",
)

# same for a plain LLM node
flow.step(LLM(model="llama3.1:8b", skills=["city-guide"], use_tools=True))
```

A mounted skill:

- merges its instructions into the system prompt;
- narrows the visible tools: `allowed-tools` intersects with the node's set,
  `disallowed-tools` removes tools outright — so `secret_tool` above stays
  out of the model's reach even though it is registered for the run.

Bare names resolve against `skill_dir`; you can also pass skill paths or
already-loaded `Skill` objects. `use_tools` gives the same per-node scope
without skills: `True` (all), `False` (none), or a list of names.

## Core skills

Draf ships built-in system skills (prefixed `draf-`, marked `[system]`), e.g.
`draf-tool-discipline`, `draf-structured-output`, `draf-verification`. They
are loaded by name and visible in prompts as `[system]`:

```python
from draf import core_skills, get_core_skill

for skill in core_skills():
    print(skill.name, skill.description)

skill = get_core_skill("draf-verification")
```

Custom skills (with a `SKILL.md`) shadow a core skill of the same name. See
the [API reference](../api/draf.skill.md) for `resolve_skills` and
`skills_instructions`.