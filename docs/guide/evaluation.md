# Evaluation (`teff eval`)

Score a workflow against a dataset of examples — exact-match by default, or
an LLM judge for open-ended answers.

```
dataset.jsonl   # one JSON object per line
{"id": "q1", "query": "What is the mascot of Teff?", "expected": "a rocket"}
```

Every key except `id` / `expected` is merged into the workflow state as an
initial override (on top of the workflow's own `state.initial`). `.json` and
`.csv` datasets are also accepted.

## CLI

```bash
teff eval workflow.yaml --data dataset.jsonl --exact
teff eval workflow.yaml --data dataset.jsonl --judge-model gpt-4o --output report.json
```

## Python

```python
from teff.yaml import load_workflow
from teff.eval import load_dataset, run_eval, format_report

workflow = load_workflow("workflow.yaml")
dataset = load_dataset("dataset.jsonl")
report = await run_eval(workflow, dataset, exact=True)
print(format_report(report))  # total=… passed=… failed=… unscored=… errors=…
```

`--exact` scores by normalised string equality; otherwise an LLM judge
(`--judge-model`, `--judge-provider`) decides PASS/FAIL per example.
`--output-key` names the state key holding the answer (a heuristic looks
through common keys first), `--max-examples` caps the run.

The same heuristic lives in `teff.eval.extract_output(state, output_key=None)`
— pick the answer from the final state, falling back to common answer keys and
finally the whole state as JSON.