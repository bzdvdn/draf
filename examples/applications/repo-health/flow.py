"""Repo-health daemon — programmatic Flow (same logic as workflow.yaml).

A repository health / change-triage daemon.  One tick:

1. lock acquire daemon:tick — skip the tick if another instance holds it.
2. csv_query read — the priority table (`data/priority.csv`, columns
   file,owner,priority).
3. git status + git log — what changed recently.
4. csv_query filter — owner/priority for each changed file.
5. redis exists/set — alert each file only once a day.
6. wait_for redis_key deploy:ready — wait for the maintenance window
   (a timeout is reported, not fatal).
7. send_telegram — one summary.
8. lock release daemon:tick.

The graph is built with :class:`draf.flow.Flow` — ``context_builder``,
then a ReAct loop (``react_agent`` + ``tool_exec``) with all tools scoped
in.  It is the programmatic twin of ``workflow.yaml`` in this directory.

Run one tick (needs Ollama + REDIS_URL + Telegram tokens exported):

    python examples/applications/repo-health/flow.py

Or run the identical workflow as a daemon via YAML:

    draf daemon -f examples/applications/repo-health/workflow.yaml --interval 300
"""

import asyncio
import os

from draf.flow import Flow
from draf.node import ContextBuilder
from draf.provider import ProviderRegistry
from draf.tool.builtin import (
    CsvQueryTool,
    GitTool,
    LockTool,
    RedisTool,
    SendTelegramTool,
    WaitForTool,
)

# Tool configs read the same env vars as the ${VAR} placeholders in
# workflow.yaml; a missing var keeps a usable default (offline-safe).
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

SYSTEM = """\
You are a repository-health agent running as a daemon.  One tick
checks one repository (the git tool's `path`), reports on it, and
stops — the daemon re-runs you every interval.

The "File priority table" section gives the CSV path
(`data/priority.csv`, columns `file,owner,priority`).  Follow this
order:

1. Take the tick lock: call lock with action=acquire, key=daemon:tick,
   ttl=300.  If it says "held by someone else", another instance is
   running this tick — stop immediately without sending anything.
2. Read the priority table: csv_query action=read.  Remember the
   columns (file, owner, priority).
3. Inspect the repository with git:
   - git action=status — are there uncommitted changes?
   - git action=log limit=5 — recent commits.
4. For each changed file, look up its owner and priority in the
   priority table (csv_query action=filter column=file value=<file>).
   Files marked high or medium priority are "attention items".
5. De-duplicate alerts in redis: for each attention item, check
   redis action=exists key=alerted:<file>.  If it exists, skip it;
   otherwise redis action=set key=alerted:<file> value=1 ttl=86400
   and include it in the report.
6. Wait for the maintenance window: wait_for condition=redis_key
   target=deploy:ready timeout=120.  If it reports "timed out",
   do NOT fail — just note in the report that the window was not
   open.
7. Send ONE send_telegram summary containing: the git status of the
   repo, the recent commits (git log), the list of new attention
   items with their owners, and whether the maintenance window was
   open.  If there was nothing new, send a short heartbeat line.
8. Release the tick lock only if you acquired it in step 1: lock
   action=release key=daemon:tick.  Releasing a lock you do not own
   is harmless (it is refused) but unnecessary.

Pass every argument explicitly; tool results come back as messages
you can read.  Be conservative and do not invent files that git
status did not report.
"""


def build_flow() -> Flow:
    flow = Flow(
        "repo_health",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.step(
        ContextBuilder(
            sections={"priority_csv": "File priority table (CSV: file,owner,priority)"},
            messages_key="messages",
            output_key="input",
            reset_keys=("messages", "_tool_call_name", "_tool_calls"),
        )
    )
    flow.react(
        model="llama3.1:8b",
        system=SYSTEM,
        input_key="input",
        output_key="summary",
        messages_key="messages",
        use_tools="all",
    )
    return flow


def build_tools() -> list:
    return [
        GitTool({"path": "."}),
        CsvQueryTool({"path": "data/priority.csv"}),
        RedisTool({"url": REDIS_URL}),
        LockTool({"url": REDIS_URL}),
        WaitForTool({"url": REDIS_URL, "poll_interval": 2, "timeout": 120}),
        SendTelegramTool({"token": TELEGRAM_BOT_TOKEN, "chat_id": TELEGRAM_CHAT_ID}),
    ]


async def main() -> None:
    graph = build_flow().compile()
    state = {"priority_csv": "data/priority.csv"}
    result = await graph.run(
        state,
        tools=build_tools(),
        max_iterations=40,
    )
    summary = result.get("summary", "")
    print("Summary:", summary or "(no summary produced)")


if __name__ == "__main__":
    asyncio.run(main())
