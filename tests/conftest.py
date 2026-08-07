"""Make ``teff.testing`` fixtures available across the test suite.

``teff.testing`` is also registered as a ``pytest11`` entry point in
``pyproject.toml`` so downstream users get ``mock_llm`` automatically;
this conftest loads it explicitly for the in-tree (non-installed) run.
When ``teff`` is installed (editable or not) pytest loads the plugin from
the entry point, so registering it again here would raise — hence the
guard.
"""

import importlib.metadata
import sys


def _teff_plugin_installed() -> bool:
    try:
        if sys.version_info >= (3, 10):
            eps = importlib.metadata.entry_points()
            return any(
                ep.name == "teff" and ep.value == "teff.testing"
                for ep in eps.select(group="pytest11")
            )
        return False
    except (ImportError, TypeError, AttributeError):
        return False


pytest_plugins = [] if _teff_plugin_installed() else ["teff.testing"]
