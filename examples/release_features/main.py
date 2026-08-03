"""Release-ready API tour: validation, eval, typed errors, cost, cache.

Runs without any live API keys:

1. ``draf validate`` equivalent — workflow YAML is checked before running.
2. ``run_eval`` — scores the workflow against a small dataset (exact mode).
3. Typed error hierarchy — catching ``draf.ConfigError`` instead of ``KeyError``.
4. Cost / cache demo — a mocked model transport shows caching skipping the
   network and ``RunSummary`` reporting cost + token usage.

Usage::

    uv run python examples/release_features/main.py
"""

import asyncio
import json

import httpx

import draf

WORKFLOW = "examples/release_features/workflow.yaml"
DATASET = "examples/release_features/dataset.jsonl"
PRICING = "examples/release_features/pricing.yaml"


def demo_validation_and_errors() -> None:
    print("== 1. validation + typed errors ==")
    from draf.yaml_schema import format_errors, validate_workflow_file

    errors = validate_workflow_file(WORKFLOW)
    if not errors:
        print(f"  {WORKFLOW}: valid")
    else:
        print(format_errors(errors, source=WORKFLOW))

    try:
        draf.from_yaml("name: bad\nsteps:\n  - id: s1\n    type: react_agnt\n")
    except draf.ConfigError as exc:
        print(f"  ConfigError caught (also a KeyError): {exc}")
    try:
        draf.from_yaml("steps: [\n  ::: not yaml")
    except draf.ConfigError as exc:
        print(f"  ConfigError on bad YAML: {exc}")


async def demo_eval() -> None:
    print("\n== 2. draf eval (exact mode) ==")
    from draf.eval import format_report, load_dataset, run_eval
    from draf.yaml import load_workflow

    workflow = load_workflow(WORKFLOW)
    dataset = load_dataset(DATASET)
    report = await run_eval(workflow, dataset, exact=True)
    print(format_report(report))


class _MockedTransport:
    """Serves canned OpenAI-style replies; counts network hits."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, url, headers=None, json=None):
        self.calls += 1
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "mock answer",
                    }
                }
            ],
            "usage": {"prompt_tokens": 250, "completion_tokens": 40},
        }

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return data

        return Resp()


async def demo_cost_and_cache() -> None:
    print("\n== 3. cost report + LLM response cache ==")
    from draf.harness import Harness, set_provider_concurrency
    from draf.trace import RunTracer

    # throttle parallel model traffic globally for this provider
    set_provider_concurrency("openai", 2)

    transport = _MockedTransport()
    original_post = httpx.AsyncClient.post
    httpx.AsyncClient.post = transport
    try:
        harness = Harness(
            model="gpt-4o",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            cache=True,  # idempotent resumes: same request never pays twice
            max_parallel=2,
        )
        harness._headers = {}

        tracer = RunTracer()
        messages = [{"role": "user", "content": "hello"}]
        first = await harness.call(messages)
        second = await harness.call(messages)
        tracer.llm(
            "openai",
            harness.model,
            first.usage["prompt"],
            first.usage["completion"],
            first.latency_ms,
        )
        tracer.llm(
            "openai",
            harness.model,
            second.usage["prompt"],
            second.usage["completion"],
            second.latency_ms,
        )

        print(
            f"  network calls: {transport.calls} (expected 1 — 2nd served from cache)"
        )
        print(f"  1st cached={first.cached}, 2nd cached={second.cached}")
        summary = tracer.summary()
        print("  RunSummary:", json.dumps(summary.to_dict(), indent=2))
    finally:
        httpx.AsyncClient.post = original_post
        set_provider_concurrency("openai", 0)


def demo_custom_pricing() -> None:
    print("\n== 4. custom provider/model pricing ==")
    from draf.trace import clear_pricing, load_pricing, model_pricing

    load_pricing(PRICING)
    print(
        f"  openrouter 'openai/gpt-4o'        -> {model_pricing('openai/gpt-4o', 'openrouter')}"
    )
    print(
        f"  openrouter default                 -> {model_pricing('openai/gpt-5', 'openrouter')}"
    )
    print(
        f"  kilo 'kilo/mega'                  -> {model_pricing('kilo/mega', 'kilo')}"
    )
    print(
        f"  kilo default                       -> {model_pricing('anything-else', 'kilo')}"
    )
    clear_pricing()


def main() -> None:
    demo_validation_and_errors()
    asyncio.run(demo_eval())
    asyncio.run(demo_cost_and_cache())
    demo_custom_pricing()
    print(f"\npackage version: draf {draf.__version__}")


if __name__ == "__main__":
    main()
