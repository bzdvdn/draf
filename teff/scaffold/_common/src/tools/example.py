"""A minimal, stateless tool — no services, no I/O.

HOW TO EXTEND
    * Replace this with your own ``Tool`` subclass: define ``name`` and
      ``description`` (the LLM sees these), and a ``run()`` whose keyword
      arguments are the tool's JSON schema.  Return a plain dict.
    * To bind a tool to a real service / database, pass the dependency into
      ``__init__`` and build the instance in ``tools/__init__.py``.
"""

from __future__ import annotations

from datetime import date, timedelta

from teff.tool import Tool


class CurrentDate(Tool):
    name = "current_date"
    description = "Return today's date, optionally shifted by offset_days."

    def run(self, offset_days: int = 0) -> dict:  # type: ignore[override]
        return {"date": (date.today() + timedelta(days=offset_days)).isoformat()}
