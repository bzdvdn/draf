"""Make ``draf.testing`` fixtures available across the test suite.

``draf.testing`` is also registered as a ``pytest11`` entry point in
``pyproject.toml`` so downstream users get ``mock_llm`` automatically;
this conftest loads it explicitly for the in-tree (non-installed) run.
"""

pytest_plugins = ["draf.testing"]
