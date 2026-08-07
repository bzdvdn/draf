"""Shell tool — run shell commands asynchronously, without a shell.

Commands are tokenized with :func:`shlex.split` and executed directly via
``execve`` (no ``/bin/sh`` involved), so shell metacharacters can never
escalate a permitted first token into arbitrary command execution.  Tokens
containing shell metacharacters are rejected outright; ``&&``, ``;``,
backticks and ``$(...)`` are never interpreted by the tool.
"""

import asyncio
import shlex

from teff.tool.tool import Tool

_DEFAULT_BLOCKED = {
    "sudo",
    "su",
    "chmod",
    "chown",
    "chattr",
    "mkfs",
    "dd",
    "fdisk",
    "parted",
    "iptables",
    "ufw",
    "systemctl",
    "service",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "init",
    "kill",
    "pkill",
    "killall",
    "passwd",
    "useradd",
    "userdel",
    "usermod",
    "visudo",
}

_SHELL_METACHARS = set("&;|`$<>(){}[]*?~#!\\\"' \t\n")


class ShellTool(Tool):
    """Run shell commands without a shell.

    The command is split into an argument vector and executed via
    ``execve`` directly — ``/bin/sh`` is never involved, so ``&&``, ``;``,
    pipes, backticks and ``$(...)`` are inert literal arguments, never
    executed.  Tokens that still contain shell metacharacters (globs,
    redirections, quotes, whitespace) are rejected.

    Args:
        root_dir: Working directory for the command.
        allowed_commands: If set, only commands whose first token is in this
            list are permitted.  ``None`` (default) allows all commands.
            A built-in blocklist of dangerous commands (``sudo``, ``dd``,
            ``reboot``, …) is always enforced.
    """

    name = "shell"
    description = "Run shell commands"

    def __init__(self, root_dir: str = ".", allowed_commands: list[str] | None = None):
        self.root_dir = root_dir
        self._allowed = allowed_commands

    async def arun(self, command: str = "") -> str:  # type: ignore[override]
        cmd = shlex.split(command)
        if not cmd:
            raise ValueError("empty command")
        prog = cmd[0]
        if prog in _DEFAULT_BLOCKED:
            raise PermissionError(f"blocked command: {prog}")
        if self._allowed is not None and prog not in self._allowed:
            raise PermissionError(
                f"command not allowed: {prog} (allowed: {self._allowed})"
            )
        for token in cmd:
            if _SHELL_METACHARS.intersection(token):
                raise PermissionError(
                    f"shell metacharacters are not allowed in: {token!r}"
                )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.root_dir,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            msg = stderr.decode().strip()
            raise RuntimeError(msg)
        return stdout.decode().strip()
