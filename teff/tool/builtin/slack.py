"""Slack tool — send messages to a Slack channel."""

from teff.tool.tool import Tool


class SlackSendTool(Tool):
    """Send a message to a Slack channel.

    Requires ``slack-sdk`` (from ``teff[tools]``) and a bot token.

    Args:
        config: Optional dict with ``token`` (bot token) and
            ``channel`` (default channel, e.g. ``#general``).
    """

    name = "slack_send"
    description = "Send a message to a Slack channel"

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.token = cfg.get("token", "")
        self.channel = cfg.get("channel", "")

    def run(self, text: str = "", channel: str = "") -> str:  # type: ignore[override]
        if not self.token:
            raise ValueError("slack_send requires 'token' in config")
        target = channel or self.channel
        if not target:
            raise ValueError("channel is required")
        try:
            from slack_sdk import WebClient
        except ImportError as e:
            msg = "slack_send requires 'slack-sdk' (pip install teff[tools])"
            raise ImportError(msg) from e

        response = WebClient(token=self.token).chat_postMessage(
            channel=target, text=text
        )
        return f"sent to {target} (ts={response['ts']})"
