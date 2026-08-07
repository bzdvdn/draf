# Recipes: closed real-world cases

A recipe is a **real business case from start to finish** — not a feature
demo. Each one follows the same template so you can compare cases and lift
the structure for your own:

1. **The business problem** — what the system is for, who uses it.
2. **Graph design** — the topology (and *why* it is shaped this way).
3. **State schema** — what flows between nodes.
4. **Durability & HITL** — why this case needs checkpoints / interrupts.
5. **Testing strategy** — how it is verified offline, no LLM.
6. **Cost & observability** — what to watch, what it costs.
7. **Deploy** — how it runs in production.
8. **How to adapt** — the knobs you turn for a different business.

Each recipe points at a **runnable example** in `examples/` so you are never
reading abstract code.

| Recipe | Business case | Runnable example | Core pattern |
| ------ | ------------- | ---------------- | ------------ |
| [Fraud review](fraud-review.md) | Payment screening | [`fraud_gate`](../../examples/applications/fraud_gate/) | Dynamic `Command` routing + HITL + durable resume |
| [Release approval](release-approval.md) | Ship-gate with human sign-off | [`release_coordinator`](../../examples/release_coordinator/) | Supervisor + `route()` + `Map` + approval gate |
| [Support triage](support-triage.md) | Customer-support SLA hub | [`simple_router`](../../examples/simple_router/) + [`repair-ai-chat`](../../examples/applications/repair-ai-chat/) | `route()` supervisor + RAG + HITL escalation |
| [Ops daemon](ops-daemon.md) | Scheduled change-triage | [`repo-health`](../../examples/applications/repo-health/) | YAML-only agent-driven tick via CLI daemon |

## Reading the pattern

The four cases differ in *business*, but the *graph anatomy* repeats:

- **One entry point** — a node that builds context (ingest / reset / context
  builder).
- **A driver** — either an agent loop (`react_agent`, `ReActAgent`) or a
  supervisor (`route()` / `Supervisor`) that decides what happens next.
- **A fan-out** — `Parallel` (fixed branches) or `Map` (runtime list).
- **A human gate** — `Interrupt` / `Ask` where a wrong answer is costly.
- **A finalizer** — a node that commits, sends, or writes the outcome.

If your case is a mix, combine the recipes — e.g. a support hub that also
re-triages nightly (triage + ops daemon) or a release gate with a fraud check
(approval + fraud).
