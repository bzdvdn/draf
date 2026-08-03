"""MkDocs event hooks to quiet third-party logging during builds.

The heavy griffe warnings ("Parameter 'X' does not appear in the function
signature", "No type or annotation for ...") come from docstrings whose
`Args:` blocks describe runtime keyword arguments accepted via
``**kwargs`` (tool configs, harness knobs).  mkdocstrings patches griffe's
loggers onto ``mkdocs.plugins.*`` and prefixes messages with ``griffe: ``.
They are informational, not failures, so we hold those loggers at ERROR
(real collection errors still surface).
"""

from __future__ import annotations

import logging


def _quiet_griffe():
    logging.getLogger("mkdocs.plugins").setLevel(logging.ERROR)


def on_startup(command, dirty):
    _quiet_griffe()


def on_files(files, *, config):
    _quiet_griffe()
    return files
