"""Repo-level runner for the ``examples/simple_router`` offline test suite.

The example owns its granular tests (``examples/simple_router/tests/``),
which run in a **fresh interpreter**: every example exposes the same top-level
``src`` package, so importing two examples in one pytest process would collide
in ``sys.modules``.  Running the suite via subprocess keeps each example's
``src`` isolated and the whole repo suite green.
"""

import subprocess
import sys
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "simple_router"


def test_simple_router_example_suite():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(_EXAMPLE / "tests")],
        cwd=str(_EXAMPLE),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
