"""Make ``draf.testing`` fixtures available across the test suite.

``draf.testing`` is also registered as a ``pytest11`` entry point in
``pyproject.toml`` so downstream users get ``mock_llm`` automatically;
this conftest loads it explicitly for the in-tree (non-installed) run.
When ``draf`` is installed (editable or not) pytest loads the plugin from
the entry point, so registering it again here would raise — hence the
guard.
"""

import importlib.metadata


def _draf_plugin_installed() -> bool:
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            return any(
                ep.name == "draf" and ep.value == "draf.testing"
                for ep in eps.select(group="pytest11")
            )
        return any(
            getattr(ep, "group", None) == "pytest11"
            and ep.name == "draf"
            and ep.value == "draf.testing"
            for ep in eps
        )
    except (ImportError, TypeError):
        return False


pytest_plugins = [] if _draf_plugin_installed() else ["draf.testing"]
