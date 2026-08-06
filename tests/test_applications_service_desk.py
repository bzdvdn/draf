"""Runner that makes the ``service_desk`` example tests part of the root suite.

The example keeps its own ``tests/test_service_desk.py`` in
``examples/applications/service_desk/``; because ``examples`` is in pytest's
``norecursedirs`` that directory isn't collected by ``pytest`` at the repo
root.  This module re-exports the example's test functions so they run with
the root suite too — without duplicating their definitions.
"""

import sys
from pathlib import Path

_PATH_ROOT = (
    Path(__file__).resolve().parents[1] / "examples" / "applications" / "service_desk"
)
_PATH_TESTS = _PATH_ROOT / "tests"
for _p in (str(_PATH_TESTS), str(_PATH_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_service_desk import *  # noqa: E402,F401,F403  (re-export the example's tests)